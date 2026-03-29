import os
import time

import torch
import torch.nn.functional as F
# from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter

from core.loss.UnsupervisedContrastiveLoss import UspConLoss
from core.loss.SupervisedContrastiveLoss import SupConLoss
from core.loss.AutomaticWeightedLoss import AutomaticWeightedLoss
from core.loss.ClassificationLoss import ClassificationLossFactory
from core.model.MainModel import MainModel

from helper.eval import *
from helper.early_stop import EarlyStopping

class TransTVDiag(object):

    def __init__(self, args, logger, device):
        self.args = args
        self.device = device
        self.logger = logger

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.epochs = args.epochs
        self.eval_period = args.eval_period
        self.log_every_n_steps = args.log_step
        self.tau = args.temperature
        log_dir = f"logs/{args.dataset}"
        os.makedirs(log_dir, exist_ok=True)

        self.patience = args.patience

        # alias tensorboard='python3 -m tensorboard.main'
        # tensorboard --logdir=logs
        self.writer = SummaryWriter(log_dir)
        self.printParams()

    def active_modalities(self):
        alias_map = {
            "metric": "metric",
            "metrics": "metric",
            "trace": "trace",
            "traces": "trace",
            "log": "log",
            "logs": "log",
        }
        active = {
            alias_map[item.strip().lower()]
            for item in str(self.args.modalities).split(",")
            if item.strip().lower() in alias_map
        }
        if not active:
            raise ValueError("No valid modalities were provided. Use a subset of: metric, trace, log.")
        return active

    def checkpoint_path(self):
        modality_order = ("metric", "trace", "log")
        active = self.active_modalities()
        modality_suffix = "-".join([modality for modality in modality_order if modality in active])
        loss_suffix = f"root_{self.args.root_loss}_type_{self.args.type_loss}"
        return os.path.join(self.writer.log_dir, f"TransTVDiag-{modality_suffix}-{loss_suffix}.pt")

    def printParams(self):
        self.logger.info(f"Training with: {self.device}")
        self.logger.info(f"batch size: {self.args.batch_size}")
        self.logger.info(f"modalities: {self.args.modalities}")
        self.logger.info(f"root cause localization loss: {self.args.root_loss}")
        self.logger.info(f"fault type classification loss: {self.args.type_loss}")
        self.logger.info(f"focal loss gamma: {self.args.focal_gamma}")
        self.logger.info(f"Status of task-oriented learning: {self.args.TO}")
        self.logger.info(f"Status of cross-modal association: {self.args.CM}")
        self.logger.info(f"guide weight: {self.args.guide_weight}")
        self.logger.info(f"temperature: {self.args.temperature}")
        self.logger.info(f"Status of dynamic_weight: {self.args.dynamic_weight}")
        self.logger.info(f"aug percent: {self.args.aug_percent}")
        self.logger.info(f"lr: {self.args.lr}, weight_decay: {self.args.weight_decay}")

    @staticmethod
    def compute_class_weights(labels, num_classes, device):
        labels = torch.as_tensor(labels, dtype=torch.long)
        counts = torch.bincount(labels, minlength=num_classes).float()
        counts = torch.clamp(counts, min=1.0)
        weights = counts.sum() / (num_classes * counts)
        return weights.to(device)

    def get_train_label_weights(self, train_dl):
        root_labels = []
        type_labels = []

        for _, (root, failure_type) in train_dl.dataset.data:
            root_labels.append(root)
            type_labels.append(failure_type)

        root_weights = self.compute_class_weights(root_labels, self.args.N_I, self.device)
        type_weights = self.compute_class_weights(type_labels, self.args.N_T, self.device)
        return root_weights, type_weights

    def train(self, train_dl):
        model = MainModel(self.args).to(self.device)
        active_modalities = self.active_modalities()
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        awl = AutomaticWeightedLoss(3)
        supConLoss = SupConLoss(self.tau, self.device).to(self.device)
        uspConLoss = UspConLoss(self.tau, self.device).to(self.device)
        root_weights, type_weights = self.get_train_label_weights(train_dl)
        root_criterion = ClassificationLossFactory.create(
            self.args.root_loss,
            weight=root_weights,
            focal_gamma=self.args.focal_gamma,
        ).to(self.device)
        type_criterion = ClassificationLossFactory.create(
            self.args.type_loss,
            weight=type_weights,
            focal_gamma=self.args.focal_gamma,
        ).to(self.device)
        
        self.logger.info(model)
        self.logger.info(f"Root class weights: {root_weights.detach().cpu().tolist()}")
        self.logger.info(f"Type class weights: {type_weights.detach().cpu().tolist()}")
        self.logger.info(f"Start training for {self.epochs} epochs.")
        
        # Overhead
        train_times = []
        
        # early stop configuration
        earlyStop = EarlyStopping(patience=self.patience)

        for epoch in range(self.epochs):
            n_iter = 0
            start_time = time.time()
            model.train()
            epoch_loss, epoch_con_l, epoch_rcl_l, epoch_fti_l = 0, 0, 0, 0

            for batch_labels, metric_feat, trace_feat, log_feat, in_degree, out_degree, attn_mask, path_data, dist in train_dl:
                instance_labels = batch_labels[:, 0].to(self.device)
                type_labels = batch_labels[:, 1].to(self.device)

                opt.zero_grad()

                (f_m, f_t, f_l), root_logit, type_logit = model(
                    metric_feat.to(self.device), trace_feat.to(self.device), log_feat.to(self.device), 
                    in_degree.to(self.device), out_degree.to(self.device),
                    dist.to(self.device), path_data.to(self.device), attn_mask.to(self.device))

                # Task-oriented learning
                # l_to, l_cm = torch.tensor(0), torch.tensor(0)
                l_to = torch.tensor(0.0, device=self.device)
                l_cm = torch.tensor(0.0, device=self.device)
                if self.args.TO:
                    # l_to = supConLoss(f_m, instance_labels) + \
                    #     supConLoss(f_t, instance_labels) + \
                    #                supConLoss(f_l, type_labels)
                    if "metric" in active_modalities:
                        l_to = l_to + supConLoss(f_m, instance_labels)
                    if "trace" in active_modalities:
                        l_to = l_to + supConLoss(f_t, instance_labels)
                    if "log" in active_modalities:
                        l_to = l_to + supConLoss(f_l, type_labels)

                # Cross-modal association
                if self.args.CM:
                    # l_cm = uspConLoss(f_m, f_t) + \
                    #     uspConLoss(f_m, f_l)
                    if {"metric", "trace"}.issubset(active_modalities):
                        l_cm = l_cm + uspConLoss(f_m, f_t)
                    if {"metric", "log"}.issubset(active_modalities):
                        l_cm = l_cm + uspConLoss(f_m, f_l)
                    # if {"trace", "log"}.issubset(active_modalities):
                    #     l_cm = l_cm + uspConLoss(f_t, f_l)
                
                # Failure Diagnosis
                gamma = self.args.guide_weight
                l_con = gamma * (l_to + l_cm)

                # l_rcl = F.cross_entropy(root_logit, instance_labels)
                # l_fti = F.cross_entropy(type_logit, type_labels)
                l_rcl = root_criterion(root_logit, instance_labels)
                l_fti = type_criterion(type_logit, type_labels)
                if self.args.dynamic_weight:
                    total_loss = awl(l_rcl, l_fti, l_con)
                else:
                    total_loss = l_con + l_rcl + l_fti

                self.logger.debug("con_loss: {:.3f}, RCL_loss: {:.3f}, FTI_loss: {:.3f}"
                      .format(l_con, l_rcl, l_fti))

                total_loss.backward()
                opt.step()
                epoch_loss += total_loss.detach().item()
                epoch_con_l += l_con.detach().item()
                epoch_rcl_l += l_rcl.detach().item()
                epoch_fti_l += l_fti.detach().item()
                # epoch_loss += total_loss.detach().item()
                n_iter += 1
                
            mean_epoch_loss = epoch_loss / n_iter
            mean_con_loss = epoch_con_l / n_iter
            mean_rcl_loss = epoch_rcl_l / n_iter
            mean_fti_loss = epoch_fti_l / n_iter
            end_time = time.time()
            time_per_epoch = (end_time - start_time)
            train_times.append(time_per_epoch)
            self.logger.info("Epoch {} done. Loss: {:.3f}, Time per epoch: {:.3f}[s]"
                         .format(epoch, mean_epoch_loss, time_per_epoch))

            top1, top3, top5 = HR(root_logit, instance_labels, topk=(1, 3, 5))
            NDCG_3 = NDCG(root_logit.cpu(), instance_labels.cpu(), topk=(3,))[0]
            pre = precision(type_logit, type_labels, k=5)
            rec = recall(type_logit, type_labels, k=5)
            f1 = f1score(type_logit, type_labels, k=5)
            self.writer.add_scalar('loss/mean total loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean con loss', mean_con_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean RCL loss', mean_rcl_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean FTI loss', mean_fti_loss, global_step=epoch)
            self.writer.add_scalar('train/top1', top1, global_step=epoch)
            self.writer.add_scalar('train/top3', top3, global_step=epoch)
            self.writer.add_scalar('train/top5', top5, global_step=epoch)
            self.writer.add_scalar('train/NDCG_3', NDCG_3, global_step=epoch)
            self.writer.add_scalar('train/precision', pre, global_step=epoch)
            self.writer.add_scalar('train/recall', rec, global_step=epoch)
            self.writer.add_scalar('train/f1-score', f1, global_step=epoch)

            # early break
            stop = earlyStop.should_stop(mean_epoch_loss, epoch)
            if stop:
                self.logger.info(f"Early stop at epoch {epoch} due to lack of improvement.")
                break

            
        state = {
            'epoch': self.epochs,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        checkpoint_path = self.checkpoint_path()
        torch.save(state, checkpoint_path)
        
        # torch.save(state, os.path.join(self.writer.log_dir, 'TransTVDiag.pt'))

                 
        self.logger.info("Training has finished.")
        self.logger.debug(f"The training time is {np.sum(train_times)}[s]")
        self.logger.debug(f"The training time per epoch is {np.mean(train_times)}[s]")
        self.logger.info(f"Model checkpoint and metadata has been saved at {checkpoint_path}.")


    def evaluate(self, test_dl):
        checkpoint_path = self.checkpoint_path()
        checkpoint = torch.load(checkpoint_path)
        # checkpoint = torch.load(os.path.join(self.writer.log_dir, 'TransTVDiag.pt'))
        model = MainModel(self.args).to(self.device)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        root_logits, type_logits = [], []
        roots, types = [], []
        inference_times = []
        for labels, metric_feat, trace_feat, log_feat, in_degree, out_degree, attn_mask, path_data, dist in test_dl:
            root, failure_type = labels.flatten().tolist()
            roots.append(root)
            types.append(failure_type)
        
            start_time = time.time()
            with torch.no_grad():
                _, root_logit, type_logit = model(
                    metric_feat.to(self.device), trace_feat.to(self.device), log_feat.to(self.device), 
                    in_degree.to(self.device), out_degree.to(self.device),
                    dist.to(self.device), path_data.to(self.device), attn_mask.to(self.device))
                root_logits.append(root_logit.flatten())
                type_logits.append(type_logit.flatten())
            end_time = time.time()
            inference_times.append(end_time - start_time)

        root_logits = torch.vstack(root_logits).cpu()
        type_logits = torch.vstack(type_logits).cpu()
        roots = torch.tensor(roots)
        types = torch.tensor(types)

        top1, top2, top3, top4, top5 = HR(root_logits, roots, topk=(1, 2, 3, 4, 5))
        NDCG_3 = NDCG(root_logits, roots, topk=(3,))[0]
        avg_5 = np.mean([top1, top2, top3, top4, top5])
        avg_3 = np.mean([top1, top2, top3])
        pre = precision(type_logits, types, k=5)
        rec = recall(type_logits, types, k=5)
        f1 = f1score(type_logits, types, k=5)

        self.logger.info("[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%},avg@3: {:.3f}, avg@5: {:.3f}, NDCG@3: {:.3f}".format(top1, top2, top3, top4, top5,avg_3, avg_5, NDCG_3))
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}".format(pre, rec, f1))
        self.logger.info(f"Evaluated checkpoint: {checkpoint_path}")
        self.logger.info(f"The average test time is {np.mean(inference_times)}[s]")
        self.logger.info(f"The total test time is {np.sum(inference_times)}[s]")