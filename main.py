import argparse
import copy
from datetime import datetime
import os
import random
from helper.logger import get_logger
import dgl
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from core.TransTVDiag import TransTVDiag
from prepare_data import prepare_for_graphormer
from process.EventProcess import EventProcess
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()

# common
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--log_step', type=int, default=20)
parser.add_argument('--eval_period', type=int, default=10)
parser.add_argument('--reconstruct', type=bool, default=True)
parser.add_argument('--gpu_devices', type=str, default='0')

parser.add_argument('--train', type=bool, default=True)
parser.add_argument('--evaluate', type=bool, default=True)

parser.add_argument('--no_train', dest='train', action='store_false')
parser.add_argument('--no_evaluate', dest='evaluate', action='store_false')
parser.add_argument('--no_reconstruct', dest='reconstruct', action='store_false')

parser.add_argument('--experiment_label', type=str, default='')

# dataset
parser.add_argument('--dataset', type=str, default='gaia', help='name of dataset')
parser.add_argument('--labels_file', type=str, default='label_15_85.csv', help="path to labels file")
parser.add_argument('--N_T', type=int, default=5, help='number of failure types')
parser.add_argument('--N_I', type=int, default=10, help='number of instances')

# TVDiag
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--patience', type=int, default=10)

parser.add_argument('--embedding_dim', type=int, default=128)
parser.add_argument('--seq_hidden', type=int, default=128)
parser.add_argument('--linear_hidden', type=list, default=[64])
parser.add_argument('--graph_hidden', type=int, default=64)
parser.add_argument('--num_heads', type=int, default=8)
parser.add_argument('--num_layers', type=int, default=2)
parser.add_argument('--graph_out', type=int, default=32)
parser.add_argument('--feat_drop', type=float, default=0)
parser.add_argument('--attn_drop', type=float, default=0)
parser.add_argument('--aggregator', type=str, default='lstm')
parser.add_argument('--TO', default=False, action='store_true')
parser.add_argument('--CM', default=False, action='store_true')
parser.add_argument('--temperature', type=float, default=0.3)
parser.add_argument('--guide_weight', type=float, default=0.1)
parser.add_argument('--dynamic_weight', action='store_true')
parser.add_argument('--epochs', type=int, default=3000)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--aug_percent', type=float, default=0.2)
parser.add_argument('--drw', action='store_true',
    help='Enable Deferred Re-Weighting: CE warmup until early stop, then switch to target loss')
parser.add_argument('--cb_beta', type=float, default=0.999,
    help='Beta for class-balanced loss (Cui et al. CVPR 2019). Higher = stronger re-weighting')
parser.add_argument(
    '--root_loss',
    type=str,
    default='ce',
    help='Loss for root localization: ce, weighted_ce, focal, weighted_focal, cb_focal, cb_ce',
)
# TO DO
parser.add_argument(
    '--type_loss',
    type=str,
    default='ce',
    help='Loss for failure type classification: ce, weighted_ce, focal, weighted_focal, cb_focal, cb_ce',
)
parser.add_argument(
    '--focal_gamma',
    type=float,
    default=2.0,
    help='Gamma parameter for focal loss',
)
parser.add_argument(
    '--focal_gamma_sweep',
    type=str,
    default='',
    help='Comma-separated focal gamma values for CV sweep, e.g. 0.25,0.5,0.75,1.0',
)
parser.add_argument(
    '--cv_folds',
    type=int,
    default=0,
    help='Run stratified k-fold cross-validation when greater than 1.',
)
parser.add_argument(
    '--cv_stratify_by',
    type=str,
    default='anomaly_type',
    choices=['anomaly_type', 'instance', 'combined'],
    help='Column(s) used for stratified CV. combined means anomaly_type + instance.',
)
parser.add_argument(
    '--cv_results_file',
    type=str,
    default='',
    help='Optional path for per-fold CV/sweep CSV results.',
)
parser.add_argument(
    '--cap_per_instance',
    type=int,
    default=0,
    help='Matched-N experiment: cap each RCL instance to at most this many '
         'TRAIN samples per fold (0 = off). Cuts the majority instances down to '
         'the rare size so every class is small and equal. Test set is untouched. '
         'Lets the encoder retrain on few samples of every class, to test whether '
         'rare decodability is limited by sample size or by the signal.',
)
parser.add_argument(
    '--eval_artifacts_dir',
    type=str,
    default='',
    help='Directory for predictions, classification reports, and confusion matrices.',
)
parser.add_argument(
    '--modalities',
    type=str,
    default='metric,trace,log',
    help='Comma-separated subset of modalities to use: metric,trace,log',
)
parser.add_argument(
    '--use_graph_priors',
    type=bool,
    default=True,
    help='Whether Graphormer uses degree/spatial/path/virtual-distance encoders.'
         ' Default True. Pass --no_graph_priors to zero them out for ablation.',
)
parser.add_argument(
    '--no_graph_priors',
    dest='use_graph_priors',
    action='store_false',
    help='Ablation: zero out degree/spatial/path/virtual-distance encoders in'
         ' Graphormer. Reduces the encoder to plain multi-head self-attention'
         ' over node features + virtual graph token. Used to test whether graph'
         ' structure contributes to TransTVDiag (Section 4.5/4.6 ablation).',
)
parser.add_argument(
    '--model_filename',
    type=str
)
args = parser.parse_args()
print('parsed modalities: ', args.modalities)

def set_seed(seed):
    dgl.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(labels)

    return prepare_for_graphormer(graphs, labels)

def build_dataloader(args, logger, train_indices=None, test_indices=None):
    reconstruct = args.reconstruct
    processor = EventProcess(args, logger)
    print('reconstruct', reconstruct)
    train_data, test_data = processor.process(
        reconstruct=reconstruct, 
        train_indices=train_indices, 
        test_indices=test_indices
    )

    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate) if args.train else None
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False, collate_fn=collate) if args.evaluate else None

    return train_dataloader, test_dataloader

def parse_focal_gamma_values(args):
    if not args.focal_gamma_sweep:
        return [args.focal_gamma]
    return [float(value.strip()) for value in args.focal_gamma_sweep.split(',') if value.strip()]


def build_stratify_labels(labels_df, stratify_by):
    if stratify_by == 'combined':
        return labels_df['anomaly_type'].astype(str) + '|' + labels_df['instance'].astype(str)
    return labels_df[stratify_by].astype(str)
    
def downsample_train_indices(labels_df, train_indices, cap, seed):
    """Matched-N control: cap each RCL instance to at most `cap` training samples.

    Majority instances (mobservice1/2) are cut down to the rare size, so every
    class is small and roughly equal. Instances that already have fewer than
    `cap` samples are kept as-is. Only the train split is touched; the test
    split stays full. Returns the reduced, sorted train indices.
    """
    train_indices = np.asarray(train_indices)
    instances = labels_df['instance'].values[train_indices]
    rng = np.random.default_rng(seed)
    kept = []
    for inst in np.unique(instances):
        idx = train_indices[instances == inst]
        if cap > 0 and len(idx) > cap:
            idx = rng.choice(idx, size=cap, replace=False)
        kept.append(idx)
    kept = np.concatenate(kept)
    kept.sort()
    return kept

def validate_stratified_folds(y, n_splits):
    counts = y.value_counts()
    min_count = counts.min()
    if min_count < n_splits:
        rare_classes = counts[counts < n_splits].to_dict()
        raise ValueError(
            f"Cannot run {n_splits}-fold stratified CV: some classes have fewer than "
            f"{n_splits} samples for the selected stratification: {rare_classes}"
        )


def cv_results_path(args):
    if args.cv_results_file:
        return args.cv_results_file
    timestamp = datetime.now().isoformat().replace(':', '-')
    if args.experiment_label:
        return f"logs/{args.dataset}/cv_results_{args.experiment_label}_{timestamp}.csv"
    return f"logs/{args.dataset}/cv_results_{timestamp}.csv"


def gamma_to_name(gamma):
    return str(gamma).replace('.', 'p').replace('-', 'm')


def run_single_experiment(args, logger, device):
    logger.info("Load dataset")
    train_dl, test_dl = build_dataloader(args, logger)

    logger.info("Training...")
    model = TransTVDiag(args, logger, device)

    train_metrics = model.train(train_dl) if args.train else {}
    eval_metrics = model.evaluate(test_dl) if args.evaluate else {}
    return {**train_metrics, **eval_metrics}


def run_cv_sweep(args, logger, device):
    if not args.train or not args.evaluate:
        raise ValueError("CV sweep requires both training and evaluation. Do not use --no_train or --no_evaluate.")

    label_path = f"data/{args.dataset}/{args.labels_file}"
    labels_df = pd.read_csv(label_path)
    stratify_labels = build_stratify_labels(labels_df, args.cv_stratify_by)
    validate_stratified_folds(stratify_labels, args.cv_folds)

    gamma_values = parse_focal_gamma_values(args)
    splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    results = []
    results_file = cv_results_path(args)
    os.makedirs(os.path.dirname(results_file) or '.', exist_ok=True)
    # checkpoint_dir = f"logs/{args.dataset}/cv_checkpoints"
    checkpoint_dir = f"logs/{args.dataset}/cv_checkpoints/{args.experiment_label}" if args.experiment_label else f"logs/{args.dataset}/cv_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(
        f"Run stratified {args.cv_folds}-fold CV sweep for focal_gamma={gamma_values}; "
        f"stratify_by={args.cv_stratify_by}"
    )

    for gamma in gamma_values:
        for fold, (train_indices, test_indices) in enumerate(splitter.split(labels_df, stratify_labels)):
            fold_args = copy.copy(args)
            fold_args.focal_gamma = gamma
            fold_args.seed = args.seed + fold
            fold_args.experiment_label = f"{args.experiment_label}-gamma_{gamma}-fold_{fold}" if args.experiment_label else f"gamma_{gamma}-fold_{fold}"
            fold_args.model_filename = os.path.join(
                checkpoint_dir,
                f"TransTVDiag-root_{args.root_loss}-type_{args.type_loss}-gamma_{gamma_to_name(gamma)}-fold_{fold}.pt",
            )
            set_seed(fold_args.seed)

            if args.cap_per_instance > 0:
                before = len(train_indices)
                train_indices = downsample_train_indices(
                    labels_df, train_indices, args.cap_per_instance, fold_args.seed
                )
                inst_counts = (
                    labels_df.iloc[train_indices]['instance'].value_counts().to_dict()
                )
                logger.info(
                    f"Matched-N downsample: train {before} -> {len(train_indices)} "
                    f"(cap={args.cap_per_instance}/instance); per-instance train counts: "
                    f"{inst_counts}"
                )

            logger.info(
                f"CV run: gamma={gamma}, fold={fold + 1}/{args.cv_folds}, "
                f"train={len(train_indices)}, test={len(test_indices)}"
            )
            train_dl, test_dl = build_dataloader(fold_args, logger, train_indices=train_indices, test_indices=test_indices)
            model = TransTVDiag(fold_args, logger, device)
            train_metrics = model.train(train_dl)
            eval_metrics = model.evaluate(test_dl)

            row = {
                'focal_gamma': gamma,
                'fold': fold,
                'seed': fold_args.seed,
                'train_size': len(train_indices),
                'test_size': len(test_indices),
                **train_metrics,
                **eval_metrics,
            }
            results.append(row)
            pd.DataFrame(results).to_csv(results_file, index=False)

    results_df = pd.DataFrame(results)
    metric_columns = [
        column for column in results_df.columns
        if column.startswith('train_') or column.startswith('test_')
    ]
    summary = results_df.groupby('focal_gamma')[metric_columns].agg(['mean', 'std'])
    summary_file = results_file.replace('.csv', '_summary.csv') if results_file.endswith('.csv') else f"{results_file}_summary.csv"
    summary.to_csv(summary_file)
    logger.info(f"CV per-fold results saved to {results_file}")
    logger.info(f"CV summary saved to {summary_file}")
    return results_df, summary


if __name__ == '__main__':
    task_name = 'TransTVDiag' if len(args.experiment_label) == 0 else f'TransTVDiag-{datetime.now().isoformat()}-{args.experiment_label}' 
    logger = get_logger(f'logs/{args.dataset}', task_name)
    use_gpu = torch.cuda.is_available()
    set_seed(args.seed)

    if use_gpu:
        logger.info("Currently using GPU {}".format(args.gpu_devices))
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_devices
        device='cuda'
    else:
        logger.info("Currently using CPU (GPU is highly recommended)")
        device = 'cpu'

    # logger.info("Load dataset")
    # train_dl, test_dl = build_dataloader(args, logger)
    
    # logger.info("Training...")
    # model = TransTVDiag(args, logger, device)

    # if args.train:
    #     model.train(train_dl)
    
    # if args.evaluate:
    #     model.evaluate(test_dl)
    if args.cv_folds > 1:
        run_cv_sweep(args, logger, device)
    else:
        run_single_experiment(args, logger, device)