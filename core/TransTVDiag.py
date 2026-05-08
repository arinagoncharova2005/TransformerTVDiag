import os
import time

import pandas as pd
import torch
import torch.nn.functional as F
# from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import classification_report, confusion_matrix

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
        if self.args.root_loss == "focal" or self.args.type_loss == "focal" or self.args.root_loss == "weighted_focal" and self.args.type_loss == "weighted_focal":
            return os.path.join(self.writer.log_dir, f"TransTVDiag-{modality_suffix}-{loss_suffix}-focal_gamma_{self.args.focal_gamma}.pt")
        return os.path.join(self.writer.log_dir, f"TransTVDiag-{modality_suffix}-{loss_suffix}.pt")

    def evaluation_artifacts_dir(self):
        artifacts_dir = getattr(self.args, 'eval_artifacts_dir', None)
        if artifacts_dir:
            return artifacts_dir
        label = getattr(self.args, 'experiment_label', '')
        if label:
            return os.path.join(self.writer.log_dir, 'evaluation_artifacts', label)        
        return os.path.join(self.writer.log_dir, 'evaluation_artifacts')

    @staticmethod
    def safe_artifact_name(name):
        return str(name).replace(os.sep, '_').replace('.', '_')

    def save_evaluation_artifacts(self, root_logits, type_logits, roots, types, checkpoint_path):
        artifacts_dir = self.evaluation_artifacts_dir()
        os.makedirs(artifacts_dir, exist_ok=True)
        run_name = self.safe_artifact_name(os.path.splitext(os.path.basename(checkpoint_path))[0])

        root_pred = torch.argmax(root_logits, dim=1).numpy()
        type_pred = torch.argmax(type_logits, dim=1).numpy()
        root_true = roots.numpy()
        type_true = types.numpy()

        instance_names = getattr(
            self.args,
            'instance_label_names',
            getattr(self, 'instance_label_names', None),
        )
        type_names = getattr(
            self.args,
            'type_label_names',
            getattr(self, 'type_label_names', None),
        )
        if instance_names is None:
            instance_names = [str(index) for index in range(self.args.N_I)]
        if type_names is None:
            type_names = [str(index) for index in range(self.args.N_T)]
        instance_names = list(instance_names)
        type_names = list(type_names)
        root_label_count = max(len(instance_names), root_logits.shape[1], int(root_pred.max()) + 1)
        type_label_count = max(len(type_names), type_logits.shape[1], int(type_pred.max()) + 1)
        if len(instance_names) < root_label_count:
            instance_names.extend(str(index) for index in range(len(instance_names), root_label_count))
        if len(type_names) < type_label_count:
            type_names.extend(str(index) for index in range(len(type_names), type_label_count))

        root_labels = list(range(len(instance_names)))
        type_labels = list(range(len(type_names)))

        predictions_df = pd.DataFrame({
            'target_root': root_true,
            'pred_root': root_pred,
            'target_root_name': [instance_names[index] for index in root_true],
            'pred_root_name': [instance_names[index] for index in root_pred],
            'target_type': type_true,
            'pred_type': type_pred,
            'target_type_name': [type_names[index] for index in type_true],
            'pred_type_name': [type_names[index] for index in type_pred],
        })
        predictions_path = os.path.join(artifacts_dir, f'{run_name}_test_predictions.csv')
        predictions_df.to_csv(predictions_path, index=False)

        root_report = classification_report(
            root_true,
            root_pred,
            labels=root_labels,
            target_names=instance_names,
            output_dict=True,
            zero_division=0,
        )
        type_report = classification_report(
            type_true,
            type_pred,
            labels=type_labels,
            target_names=type_names,
            output_dict=True,
            zero_division=0,
        )
        root_report_df = pd.DataFrame(root_report).transpose()
        type_report_df = pd.DataFrame(type_report).transpose()

        root_report_path = os.path.join(artifacts_dir, f'{run_name}_classification_report_rcl.csv')
        type_report_path = os.path.join(artifacts_dir, f'{run_name}_classification_report_fti.csv')
        root_report_df.to_csv(root_report_path)
        type_report_df.to_csv(type_report_path)

        root_cm = confusion_matrix(root_true, root_pred, labels=root_labels)
        type_cm = confusion_matrix(type_true, type_pred, labels=type_labels)
        root_cm_path = os.path.join(artifacts_dir, f'{run_name}_confusion_matrix_rcl.csv')
        type_cm_path = os.path.join(artifacts_dir, f'{run_name}_confusion_matrix_fti.csv')
        pd.DataFrame(root_cm, index=instance_names, columns=instance_names).to_csv(root_cm_path)
        pd.DataFrame(type_cm, index=type_names, columns=type_names).to_csv(type_cm_path)

        self.logger.info(f"Evaluation predictions saved to {predictions_path}")
        self.logger.info(f"RCL classification report saved to {root_report_path}")
        self.logger.info(f"FTI classification report saved to {type_report_path}")
        self.logger.info(f"RCL confusion matrix saved to {root_cm_path}")
        self.logger.info(f"FTI confusion matrix saved to {type_cm_path}")

        return {
            'predictions_path': predictions_path,
            'rcl_classification_report_path': root_report_path,
            'fti_classification_report_path': type_report_path,
            'rcl_confusion_matrix_path': root_cm_path,
            'fti_confusion_matrix_path': type_cm_path,
        }


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

    def get_train_label_stats(self, train_dl):
        root_labels = []
        type_labels = []

        for _, (root, failure_type) in train_dl.dataset.data:
            root_labels.append(root)
            type_labels.append(failure_type)

        root_labels_t = torch.as_tensor(root_labels, dtype=torch.long)
        type_labels_t = torch.as_tensor(type_labels, dtype=torch.long)

        root_counts = torch.bincount(root_labels_t, minlength=self.args.N_I).float()
        type_counts = torch.bincount(type_labels_t, minlength=self.args.N_T).float()

        root_weights = self.compute_class_weights(root_labels, self.args.N_I, self.device)
        type_weights = self.compute_class_weights(type_labels, self.args.N_T, self.device)
        return root_weights, type_weights, root_counts, type_counts

    def train(self, train_dl):
        model = MainModel(self.args).to(self.device)
        active_modalities = self.active_modalities()
        # in official version, but weights are not learning
        # opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        
        awl = AutomaticWeightedLoss(3)
        # corrected official implementation in order for weights to be learned while training
        # opt = torch.optim.Adam(
        #     list(model.parameters()) + list(awl.parameters()),
        #     lr=self.lr, weight_decay=self.weight_decay,
        # )
        opt = torch.optim.Adam([
            {"params": model.parameters(), "weight_decay": self.weight_decay},
            {"params": awl.parameters(), "weight_decay": 0.0},
        ], lr=self.lr)
        
        supConLoss = SupConLoss(self.tau, self.device).to(self.device)
        uspConLoss = UspConLoss(self.tau, self.device).to(self.device)
        root_weights, type_weights, root_counts, type_counts = self.get_train_label_stats(train_dl)

        cb_beta = getattr(self.args, 'cb_beta', 0.999)
        root_criterion = ClassificationLossFactory.create(
            self.args.root_loss,
            weight=root_weights,
            focal_gamma=self.args.focal_gamma,
            samples_per_class=root_counts,
            cb_beta = cb_beta,
        ).to(self.device)
        type_criterion = ClassificationLossFactory.create(
            self.args.type_loss,
            weight=type_weights,
            focal_gamma=self.args.focal_gamma,
            samples_per_class=type_counts,
            cb_beta = cb_beta,
        ).to(self.device)

        # DRW: warmup with CE, then switch to target loss on early stopping
        drw_enabled = getattr(self.args, 'drw', False)
        drw_switched = False
        if drw_enabled:
            root_criterion_target = root_criterion
            type_criterion_target = type_criterion
            root_criterion = ClassificationLossFactory.create('ce').to(self.device)
            type_criterion = ClassificationLossFactory.create('ce').to(self.device)
            self.logger.info(
                f"DRW enabled: CE warmup until early stop trigger, "
                f"then switch to {self.args.root_loss}/{self.args.type_loss}"
            )
        
        self.logger.info(model)
        self.logger.info(f"Root class weights: {root_weights.detach().cpu().tolist()}")
        self.logger.info(f"Type class weights: {type_weights.detach().cpu().tolist()}")
        self.logger.info(f"Start training for {self.epochs} epochs.")
        
        # Overhead
        train_times = []
        
        # early stop configuration
        earlyStop = EarlyStopping(patience=self.patience)
        final_train_metrics = {}

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
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

            # TODO: return back
            top1, top3, top5 = HR(root_logit, instance_labels, topk=(1, 3, 5))
            # top1, top3 = HR(root_logit, instance_labels, topk=(1, 3))
            NDCG_3 = NDCG(root_logit.cpu(), instance_labels.cpu(), topk=(3,))[0]
            pre = precision(type_logit, type_labels, k=1)
            rec = recall(type_logit, type_labels, k=1)
            f1 = f1score(type_logit, type_labels, k=1)
            self.writer.add_scalar('loss/mean total loss', mean_epoch_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean con loss', mean_con_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean RCL loss', mean_rcl_loss, global_step=epoch)
            self.writer.add_scalar('loss/mean FTI loss', mean_fti_loss, global_step=epoch)
            self.writer.add_scalar('train/top1', top1, global_step=epoch)
            self.writer.add_scalar('train/top3', top3, global_step=epoch)
            # TODO: return back
            self.writer.add_scalar('train/top5', top5, global_step=epoch)
            self.writer.add_scalar('train/NDCG_3', NDCG_3, global_step=epoch)
            self.writer.add_scalar('train/precision', pre, global_step=epoch)
            self.writer.add_scalar('train/recall', rec, global_step=epoch)
            self.writer.add_scalar('train/f1-score', f1, global_step=epoch)
            final_train_metrics = {
                'epoch': epoch,
                'train_loss': mean_epoch_loss,
                'train_con_loss': mean_con_loss,
                'train_rcl_loss': mean_rcl_loss,
                'train_fti_loss': mean_fti_loss,
                'train_top1': top1,
                'train_top3': top3,
                'train_NDCG_3': NDCG_3,
                'train_precision': pre,
                'train_recall': rec,
                'train_f1': f1,
            }

            # early break or DRW phase switch
            stop = earlyStop.should_stop(mean_epoch_loss, epoch)
            if stop:
                if drw_enabled and not drw_switched:
                    # Phase 1 done — switch from CE to target loss
                    drw_switched = True
                    root_criterion = root_criterion_target
                    type_criterion = type_criterion_target
                    earlyStop = EarlyStopping(patience=self.patience)
                    self.logger.info(
                        f"DRW phase switch at epoch {epoch}: "
                        f"CE -> {self.args.root_loss}/{self.args.type_loss}"
                    )
                else:
                    self.logger.info(f"Early stop at epoch {epoch} due to lack of improvement.")
                    break

            
        state = {
            'epoch': self.epochs,
            'model': model.state_dict(),
            'opt': opt.state_dict(),
        }
        if self.args.model_filename is None:
            checkpoint_path = self.checkpoint_path()
        else:
            checkpoint_path = self.args.model_filename
        os.makedirs(os.path.dirname(checkpoint_path) or '.', exist_ok=True)
        torch.save(state, checkpoint_path)
        
        # torch.save(state, os.path.join(self.writer.log_dir, 'TransTVDiag.pt'))

                 
        self.logger.info("Training has finished.")
        self.logger.debug(f"The training time is {np.sum(train_times)}[s]")
        self.logger.debug(f"The training time per epoch is {np.mean(train_times)}[s]")
        self.logger.info(f"Model checkpoint and metadata has been saved at {checkpoint_path}.")
        final_train_metrics['checkpoint_path'] = checkpoint_path
        final_train_metrics['train_time_total'] = float(np.sum(train_times))
        final_train_metrics['train_time_per_epoch'] = float(np.mean(train_times))
        return final_train_metrics


    def evaluate(self, test_dl):
        if self.args.model_filename is None:
            checkpoint_path = self.checkpoint_path()
        else:
            checkpoint_path = self.args.model_filename
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

        self.instance_label_names = getattr(test_dl.dataset, 'instance_label_names', None)
        self.type_label_names = getattr(test_dl.dataset, 'type_label_names', None)
        
        # TODO: return back
        top1, top2, top3, top4, top5 = HR(root_logits, roots, topk=(1, 2, 3, 4, 5))
        # top1, top2, top3, top4 = HR(root_logits, roots, topk=(1, 2, 3, 4))
        NDCG_3 = NDCG(root_logits, roots, topk=(3,))[0]
        # TODO: return back
        avg_5 = np.mean([top1, top2, top3, top4, top5])
        avg_3 = np.mean([top1, top2, top3])
        pre = precision(type_logits, types, k=1)
        rec = recall(type_logits, types, k=1)
        f1 = f1score(type_logits, types, k=1)
        # TODO: return back
        self.logger.info("[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%}, top5: {:.3%},avg@3: {:.3f}, avg@5: {:.3f}, NDCG@3: {:.3f}".format(top1, top2, top3, top4, top5,avg_3, avg_5, NDCG_3))
        # self.logger.info("[Root localization] top1: {:.3%}, top2: {:.3%}, top3: {:.3%}, top4: {:.3%},avg@3: {:.3f}, NDCG@3: {:.3f}".format(top1, top2, top3, top4,avg_3, NDCG_3))
        self.logger.info("[Failure type classification] precision: {:.3%}, recall: {:.3%}, f1-score: {:.3%}".format(pre, rec, f1))
        self.logger.info(f"Evaluated checkpoint: {checkpoint_path}")
        self.logger.info(f"The average test time is {np.mean(inference_times)}[s]")
        self.logger.info(f"The total test time is {np.sum(inference_times)}[s]")
        artifact_paths = self.save_evaluation_artifacts(root_logits, type_logits, roots, types, checkpoint_path)
        return {
            'test_top1': top1,
            'test_top2': top2,
            'test_top3': top3,
            'test_top4': top4,
            'test_avg_3': avg_3,
            'test_NDCG_3': NDCG_3,
            'test_precision': pre,
            'test_recall': rec,
            'test_f1': f1,
            'test_time_avg': float(np.mean(inference_times)),
            'test_time_total': float(np.sum(inference_times)),
            'checkpoint_path': checkpoint_path,
            **artifact_paths,
        }