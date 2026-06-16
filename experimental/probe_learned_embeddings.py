"""Linear probe + fold-artifact sanity on the learned embeddings.

Reads embeddings.npz from export_learned_embeddings.py and answers two
questions the t-SNE/silhouette figure cannot:

1. FOLD ARTIFACT? The pooled t-SNE mixes 5 folds, each with its own
   FastText refit and its own model => unaligned 96-d spaces. If the
   t-SNE islands are really folds (not classes), the separability story
   is bogus. We recolour the same t-SNE by fold and also report a probe
   that *predicts the fold* from f: high fold-accuracy = the geometry is
   contaminated by fold identity and the pooled view is misleading.

2. LINEAR DECODABILITY per task. For each outer fold separately (so the
   probe never crosses unaligned spaces) we run an internal StratifiedKFold
   logistic-regression probe on f and collect out-of-fold predictions.
   Reported: macro-F1 for FTI and for RCL, plus per-class RCL recall.
   Expectation matching the figure: the two majority instances
   (mobservice1/2) are decodable, the 8 minority instances are not ->
   separability tracks sample size, not loss choice (RQ2).

Probe: Pipeline(StandardScaler, LogisticRegression(class_weight='balanced')).
Per-class recall is pooled over the within-fold OOF predictions of all folds.

Run:
    python experimental/probe_learned_embeddings.py --npz logs/gaia/learned_embeddings/cb_ce_awl/embeddings.npz
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import (
    f1_score, recall_score, precision_score, accuracy_score, confusion_matrix,
)
from sklearn.model_selection import (
    StratifiedKFold, cross_val_predict, train_test_split,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TSNE_PERPLEXITY = 30
TSNE_SEED = 42
LOGIN_SUBSAMPLE = 400
FOLD_PALETTE = ['#4E79A7', '#F28E2B', '#59A14F', '#E15759', '#B07AA1']


def make_probe():
    # saga + higher max_iter: lbfgs did not converge on the 96-d balanced
    # problem; saga is robust here and silences the ConvergenceWarning so the
    # reported F1 is the converged value, not an early-stopped one.
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, class_weight='balanced',
                           solver='saga', C=1.0),
    )


def plot_rcl_recall(rcl_names, rcl_present, counts, rcl_rec, chance, out_path):
    """Bar chart of per-instance linear decodability: majority vs minority."""
    order = sorted(rcl_present, key=lambda c: -counts[c])
    names = [rcl_names[c] for c in order]
    recs = [rcl_rec[rcl_present.index(c)] for c in order]
    # mobservice1/2 are the two majority instances (~96% of samples)
    colors = ['#4E79A7' if counts[c] > 1000 else '#E15759' for c in order]
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    bars = ax.bar(range(len(order)), recs, color=colors,
                  edgecolor='black', linewidth=0.4)
    ax.axhline(chance, ls='--', lw=1.0, color='grey',
               label=f'chance = {chance:.2f}')
    for b, c in zip(bars, order):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f'n={counts[c]}', ha='center', va='bottom', fontsize=8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(names, rotation=40, ha='right')
    ax.set_ylabel('linear-probe recall')
    ax.set_ylim(0, 1.0)
    ax.set_title('Per-instance linear decodability of the learned representation\n'
                 '(blue = majority, red = minority)')
    ax.legend(loc='upper right')
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def safe_splits(y, max_splits=5):
    """Largest n_splits <= max that every class can support (>= n_splits)."""
    _, counts = np.unique(y, return_counts=True)
    return max(2, min(max_splits, int(counts.min())))


def probe_within_folds(X, y, fold):
    """Per outer fold: internal StratifiedKFold OOF logreg. Returns pooled
    OOF (y_true, y_pred) across folds and the list of per-fold macro-F1."""
    y_true_all, y_pred_all = [], []
    per_fold_f1 = []
    for fk in np.unique(fold):
        m = fold == fk
        Xf, yf = X[m], y[m]
        if len(np.unique(yf)) < 2:
            continue
        ns = safe_splits(yf, 5)
        skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=0)
        pred = cross_val_predict(make_probe(), Xf, yf, cv=skf)
        per_fold_f1.append(f1_score(yf, pred, average='macro'))
        y_true_all.append(yf)
        y_pred_all.append(pred)
    return (np.concatenate(y_true_all), np.concatenate(y_pred_all),
            np.asarray(per_fold_f1))


def fold_identity_probe(X, fold):
    """Predict the fold from f. High accuracy => geometry leaks fold id."""
    ns = safe_splits(fold, 5)
    skf = StratifiedKFold(n_splits=ns, shuffle=True, random_state=0)
    pred = cross_val_predict(make_probe(), X, fold, cv=skf)
    return accuracy_score(fold, pred), 1.0 / len(np.unique(fold))


def balanced_subset(fti_short, rng):
    keep = []
    for c in np.unique(fti_short):
        idxs = np.where(fti_short == c)[0]
        if c == 'login_failure' and len(idxs) > LOGIN_SUBSAMPLE:
            idxs = rng.choice(idxs, size=LOGIN_SUBSAMPLE, replace=False)
        keep.append(idxs)
    keep = np.concatenate(keep); keep.sort()
    return keep


def plot_fold_tsne(X_sub, fold_sub, out_path):
    Xs = StandardScaler().fit_transform(X_sub)
    coords = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init='pca',
                  learning_rate='auto', random_state=TSNE_SEED,
                  metric='cosine').fit_transform(Xs)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(6.8, 5.8))
    for fk in sorted(np.unique(fold_sub)):
        m = fold_sub == fk
        ax.scatter(coords[m, 0], coords[m, 1], s=14,
                   color=FOLD_PALETTE[int(fk) % len(FOLD_PALETTE)],
                   alpha=0.75, edgecolor='black', linewidth=0.25,
                   label=f'fold {int(fk)} (n={int(m.sum())})')
    ax.set_title('Same t-SNE, coloured by CV fold\n'
                 '(islands = folds => pooled view is a fold artifact)')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(loc='best', framealpha=0.9, markerscale=1.4)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def learning_curve_majority(X, y, fold, target, caps, rng):
    """Subsampling control (separates probe sample-size from representation).

    The encoder representation is FIXED (it saw all ~7800 of `target`). We
    only starve the PROBE: within each fold, a stratified 70/30 split, then
    the target instance's TRAIN positives are capped to `cap`, the rest of
    the 10-way training set is kept full, and we read the target's recall on
    the untouched test split. Test set is identical across caps.

    Reading: if the majority instance's recall collapses to ~chance when its
    probe-train count is cut to a rare instance's size, the rare failure is
    explained by sample starvation. If it stays high at that size, the rare
    failure is NOT probe sample-size -> it is in the representation / data.
    Recall alone can be inflated by a class_weight='balanced' probe that over-
    predicts the target, so precision is tracked too: the "still high" reading
    is only safe if BOTH stay high at the rare-instance training size.
    Returns {cap: (mean recall, mean precision) across folds}.
    """
    rec_out = {c: [] for c in caps}
    prec_out = {c: [] for c in caps}
    for fk in np.unique(fold):
        m = fold == fk
        Xf, yf = X[m], y[m]
        if (yf == target).sum() < 10:
            continue
        idx = np.arange(len(yf))
        tr, te = train_test_split(idx, test_size=0.3, stratify=yf, random_state=0)
        tr_target = tr[yf[tr] == target]
        tr_other = tr[yf[tr] != target]
        for cap in caps:
            if cap is None or cap >= len(tr_target):
                keep_t = tr_target
            else:
                keep_t = rng.choice(tr_target, size=cap, replace=False)
            tr_cap = np.concatenate([keep_t, tr_other])
            clf = make_probe()
            clf.fit(Xf[tr_cap], yf[tr_cap])
            pred = clf.predict(Xf[te])
            rec_out[cap].append(recall_score(yf[te], pred, labels=[target],
                                             average=None, zero_division=0)[0])
            prec_out[cap].append(precision_score(yf[te], pred, labels=[target],
                                                 average=None, zero_division=0)[0])
    return {c: (float(np.mean(rec_out[c])) if rec_out[c] else float('nan'),
                float(np.mean(prec_out[c])) if prec_out[c] else float('nan'))
            for c in caps}


def plot_learning_curve(curve, rare_recall, target_name, out_path):
    caps = [c for c in curve if c is not None]
    xs = caps + [max(caps) * 2]            # plot 'all' at 2x the largest cap
    rec_ys = [curve[c][0] for c in caps] + [curve[None][0]]
    prec_ys = [curve[c][1] for c in caps] + [curve[None][1]]
    plt.rcParams.update({'font.family': 'serif', 'font.size': 11})
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    ax.plot(xs, rec_ys, 'o-', color='#4E79A7', label=f'{target_name} recall')
    ax.plot(xs, prec_ys, 's--', color='#59A14F', label=f'{target_name} precision')
    ax.axhline(rare_recall, ls='--', color='#E15759',
               label=f'mean minority recall = {rare_recall:.2f}')
    ax.set_xscale('log')
    ax.set_xlabel(f'probe training examples of {target_name} (log)')
    ax.set_ylabel('recall')
    ax.set_ylim(0, 1.0)
    ax.set_title('Subsampling control: majority decodability vs probe sample size')
    ax.legend(loc='best')
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_rcl_confusion(rcl_true, rcl_pred, rcl_present, rcl_names, counts, out_path):
    """Row-normalised RCL confusion (recall view): for each true instance, the
    fraction routed to each predicted instance. Answers where the rare ones go
    -> absorbed by mobservice1/2 vs scattered among other minority instances."""
    cm = confusion_matrix(rcl_true, rcl_pred, labels=rcl_present).astype(float)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)
    order = sorted(range(len(rcl_present)), key=lambda i: -counts[rcl_present[i]])
    cm_norm = cm_norm[np.ix_(order, order)]
    labels = [f'{rcl_names[rcl_present[i]]} (n={counts[rcl_present[i]]})' for i in order]
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    im = ax.imshow(cm_norm, cmap='magma', vmin=0, vmax=1)
    ax.set_xticks(range(len(order))); ax.set_yticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=40, ha='right')
    ax.set_yticklabels(labels)
    ax.set_xlabel('predicted instance'); ax.set_ylabel('true instance')
    ax.set_title('RCL probe confusion (row-normalised)\nwhere does each instance get routed?')
    for i in range(len(order)):
        for j in range(len(order)):
            v = cm_norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=7, color='white' if v < 0.6 else 'black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(); fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return cm_norm, order


def per_modality_probe(npz, fold, rcl, rcl_names):
    """RCL macro-F1 + per-class recall using each modality's f alone."""
    rcl_present = sorted(np.unique(rcl))
    counts = {c: int((rcl == c).sum()) for c in rcl_present}
    rows = []
    for key in ('f_metric', 'f_trace', 'f_log'):
        Xm = npz[key].astype(np.float32)
        y_true, y_pred, _ = probe_within_folds(Xm, rcl, fold)
        macro = f1_score(y_true, y_pred, average='macro')
        rec = recall_score(y_true, y_pred, labels=rcl_present,
                           average=None, zero_division=0)
        rows.append((key.replace('f_', ''), macro, dict(zip(rcl_present, rec))))
    return rcl_present, counts, rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True)
    p.add_argument('--config', default=None)
    cli = p.parse_args()

    npz = np.load(cli.npz, allow_pickle=True)
    config = cli.config or Path(cli.npz).parent.name
    out_dir = Path(cli.npz).resolve().parent

    X = npz['f'].astype(np.float32)
    fold = npz['fold'].astype(int)
    fti = npz['fti_true'].astype(int)
    rcl = npz['rcl_true'].astype(int)
    rcl_names = [str(x) for x in npz['rcl_names']]
    fti_names = [str(x) for x in npz['fti_names']]
    print(f'N={len(X)}, dim={X.shape[1]}, folds={sorted(np.unique(fold))}')

    # --- 1. fold-artifact checks ---
    fold_acc, fold_chance = fold_identity_probe(X, fold)
    print(f'\n[fold-identity probe] accuracy={fold_acc:.3f}  (chance={fold_chance:.3f})')
    print('  -> high above chance = f geometry is contaminated by fold id;'
          ' pooled t-SNE separability is partly a fold artifact.')

    rng = np.random.default_rng(42)
    fti_short = np.array([fti_names[i] for i in fti])
    keep = balanced_subset(fti_short, rng)
    plot_fold_tsne(X[keep], fold[keep], out_dir / f'fold_tsne_{config}.png')

    # --- 2. within-fold linear probe ---
    fti_true, fti_pred, fti_f1s = probe_within_folds(X, fti, fold)
    rcl_true, rcl_pred, rcl_f1s = probe_within_folds(X, rcl, fold)
    fti_macro = f1_score(fti_true, fti_pred, average='macro')
    rcl_macro = f1_score(rcl_true, rcl_pred, average='macro')

    rcl_present = sorted(np.unique(rcl_true))
    rcl_rec = recall_score(rcl_true, rcl_pred, labels=rcl_present,
                           average=None, zero_division=0)
    rcl_prec = precision_score(rcl_true, rcl_pred, labels=rcl_present,
                               average=None, zero_division=0)
    rcl_f1 = f1_score(rcl_true, rcl_pred, labels=rcl_present,
                      average=None, zero_division=0)
    counts = {c: int((rcl == c).sum()) for c in rcl_present}
    chance = 1.0 / len(rcl_present)
    plot_rcl_recall(rcl_names, rcl_present, counts, rcl_rec, chance,
                    out_dir / f'rcl_recall_{config}.png')
    cm_norm, cm_order = plot_rcl_confusion(
        rcl_true, rcl_pred, rcl_present, rcl_names, counts,
        out_dir / f'rcl_confusion_{config}.png')

    lines = []
    lines.append('Linear probe on POST-ENCODER learned embeddings f '
                 '(within-fold OOF logreg, class-balanced).')
    lines.append(f'Config: {config}.  N={len(X)}, dim={X.shape[1]}.\n')
    lines.append(f'[fold-identity probe] accuracy={fold_acc:.3f}  '
                 f'(chance={fold_chance:.3f})')
    lines.append('  high above chance => pooled t-SNE islands are partly folds, '
                 'not classes.\n')
    lines.append(f'FTI macro-F1: {fti_macro:.3f}  '
                 f'(per-fold {fti_f1s.mean():.3f} ± {fti_f1s.std():.3f})')
    lines.append(f'RCL macro-F1: {rcl_macro:.3f}  '
                 f'(per-fold {rcl_f1s.mean():.3f} ± {rcl_f1s.std():.3f})\n')
    lines.append('RCL per-class decodability (recall / precision / F1):')
    order = sorted(rcl_present, key=lambda c: -counts[c])
    lines.append(f'  {"instance":<16}{"n":>7}{"recall":>9}{"prec":>9}{"f1":>9}')
    for c in order:
        j = rcl_present.index(c)
        lines.append(f'  {rcl_names[c]:<16}{counts[c]:>7}{rcl_rec[j]:>9.3f}'
                     f'{rcl_prec[j]:>9.3f}{rcl_f1[j]:>9.3f}')

    # mean recall over the minority instances (n < 1000), used as the
    # reference line for the subsampling control.
    minority = [c for c in rcl_present if counts[c] < 1000]
    rare_recall = float(np.mean([rcl_rec[rcl_present.index(c)] for c in minority]))

    # where do the rare instances get routed? read the row-normalised confusion
    # matrix: majority absorption (predicted as mobservice1/2) vs scatter.
    cls_to_row = {rcl_present[cm_order[i]]: i for i in range(len(cm_order))}
    lines.append('\nWhere rare instances get routed (top-2 predicted, row-normalised):')
    for c in sorted(minority, key=lambda c: -counts[c]):
        row = cm_norm[cls_to_row[c]]
        top = np.argsort(row)[::-1][:2]
        dest = ', '.join(f'{rcl_names[rcl_present[cm_order[j]]]} {row[j]:.2f}'
                         for j in top if row[j] > 0)
        lines.append(f'  {rcl_names[c]:<16}-> {dest}')

    # --- 3. subsampling control: H1 (sample size) vs H2 (representation) ---
    target = rcl_names.index('mobservice1')
    caps = [50, 100, 250, 500, 1000, None]
    curve = learning_curve_majority(X, rcl, fold, target, caps, rng)
    plot_learning_curve(curve, rare_recall, 'mobservice1',
                        out_dir / f'learning_curve_{config}.png')
    lines.append('\nSubsampling control (mobservice1 recall/precision vs probe train size):')
    lines.append(f'  mean minority recall (reference) = {rare_recall:.3f}')
    for c in caps:
        rec_c, prec_c = curve[c]
        lines.append(f'  n_train={"all" if c is None else c:<5} '
                     f'recall={rec_c:.3f}  precision={prec_c:.3f}')
    lines.append('  if recall at n~70 ~= minority recall -> sample size (H1); '
                 'if recall AND precision stay high -> representation/data (H2).')

    # --- 4. per-modality decodability ---
    _, _, mod_rows = per_modality_probe(npz, fold, rcl, rcl_names)
    lines.append('\nPer-modality RCL probe (which modality carries instance id):')
    lines.append(f'  {"modality":<10}{"macroF1":>9}'
                 f'{"  mob1":>8}{"  mob2":>8}{"  minority(mean)":>16}')
    for name, macro, rec in mod_rows:
        mob1 = rec[rcl_names.index('mobservice1')]
        mob2 = rec[rcl_names.index('mobservice2')]
        mino = float(np.mean([rec[c] for c in minority]))
        lines.append(f'  {name:<10}{macro:>9.3f}{mob1:>8.3f}{mob2:>8.3f}'
                     f'{mino:>16.3f}')

    report = '\n'.join(lines)
    print('\n' + report)
    txt_path = out_dir / f'probe_{config}.txt'
    txt_path.write_text(report + '\n')
    print(f'\nSaved: {txt_path}')


if __name__ == '__main__':
    main()
