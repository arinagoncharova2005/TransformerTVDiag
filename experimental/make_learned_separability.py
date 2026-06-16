"""Learned-representation class separability (corrected figure for Section 4.4).

Counterpart to make_embedding_separability.py. That figure projected the
*pre-encoder* FastText features and selected the root-cause node by its RCL
label -- and FastText bakes the node index into every node's supervised tag
(EventProcess.py:77-80), so its RCL silhouette (~0.75) is a leakage artifact,
not evidence the telemetry separates root causes.

This figure instead projects the fused *post-encoder* representation the model
actually learned, f = cat(f_m, f_t, f_l), exported out-of-sample by
experimental/export_learned_embeddings.py (each sample encoded by the
checkpoint of the fold in which it was held out). No node is picked by its
label, so there is no RCL leakage and the silhouette is a fair statement about
what the trained model generalises.

Inputs:
    logs/gaia/learned_embeddings/{config}/embeddings.npz   (from the export script)

Outputs (thesis/figures/):
    learned_tsne_fti_rcl_{config}.png
    learned_silhouette_{config}.txt

Run:
    python experimental/make_learned_separability.py --npz logs/gaia/learned_embeddings/cb_ce_awl/embeddings.npz
"""
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

OUT_DIR = Path(__file__).resolve().parent

FTI_SHORT = {
    'access permission denied exception': 'access_perm',
    'file moving program':                'file_moving',
    'login failure':                      'login_failure',
    'memory_anomalies':                   'memory_anomalies',
    'normal memory freed label':          'normal_mem_freed',
}

FTI_COLOR = {
    'access_perm':      '#E15759',
    'file_moving':      '#F28E2B',
    'login_failure':    '#BAB0AC',
    'memory_anomalies': '#4E79A7',
    'normal_mem_freed': '#59A14F',
}

RCL_PALETTE = [
    '#4E79A7', '#A0CBE8', '#F28E2B', '#FFBE7D', '#59A14F',
    '#8CD17D', '#E15759', '#FF9D9A', '#B07AA1', '#D4A6C8',
]

# Subsample the dominant login_failure class so t-SNE is not swamped by it.
LOGIN_SUBSAMPLE = 400
TSNE_PERPLEXITY = 30
TSNE_SEED = 42


def setup_style():
    plt.rcParams.update({
        'font.family': 'serif', 'font.size': 11,
        'axes.labelsize': 12, 'axes.titlesize': 13,
        'xtick.labelsize': 10, 'ytick.labelsize': 10, 'legend.fontsize': 9,
    })


def short_fti(name):
    key = str(name).strip().strip('[]')
    return FTI_SHORT.get(key, key)


def balanced_subset(fti_short, rng):
    """Cap login_failure at LOGIN_SUBSAMPLE; keep every other class in full."""
    keep = []
    for c in np.unique(fti_short):
        idxs = np.where(fti_short == c)[0]
        if c == 'login_failure' and len(idxs) > LOGIN_SUBSAMPLE:
            idxs = rng.choice(idxs, size=LOGIN_SUBSAMPLE, replace=False)
        keep.append(idxs)
    keep = np.concatenate(keep)
    keep.sort()
    return keep


def sil_highdim(X, labels):
    if len(np.unique(labels)) < 2:
        return float('nan')
    Xs = StandardScaler().fit_transform(X)
    return silhouette_score(Xs, labels, metric='cosine')


def run_tsne(X):
    Xs = StandardScaler().fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init='pca',
                learning_rate='auto', random_state=TSNE_SEED, metric='cosine')
    return tsne.fit_transform(Xs)


def plot_panels(coords, fti_short, rcl_name, rcl_order, out_path, sil_fti, sil_rcl):
    rcl_color = dict(zip(rcl_order, RCL_PALETTE))
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8))

    ax = axes[0]
    for c, color in FTI_COLOR.items():
        m = fti_short == c
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=14, color=color, alpha=0.75,
                   edgecolor='black', linewidth=0.25,
                   label=f'{c} (n={int(m.sum())})')
    ax.set_title(f'Coloured by failure type (FTI)\nsilhouette = {sil_fti:.3f}')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', framealpha=0.9, markerscale=1.4)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    for c in rcl_order:
        m = rcl_name == c
        if not m.any():
            continue
        ax.scatter(coords[m, 0], coords[m, 1], s=14, color=rcl_color[c], alpha=0.75,
                   edgecolor='black', linewidth=0.25,
                   label=f'{c} (n={int(m.sum())})')
    ax.set_title(f'Coloured by root-cause instance (RCL)\nsilhouette = {sil_rcl:.3f}')
    ax.set_xlabel('t-SNE 1'); ax.set_ylabel('t-SNE 2')
    ax.legend(loc='best', framealpha=0.9, markerscale=1.4, ncol=2)
    ax.set_xticks([]); ax.set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--npz', required=True, help='embeddings.npz from export_learned_embeddings.py')
    p.add_argument('--config', default=None, help='label for output filenames; default: parent dir name')
    cli = p.parse_args()

    setup_style()
    rng = np.random.default_rng(42)

    npz = np.load(cli.npz, allow_pickle=True)
    config = cli.config or Path(cli.npz).parent.name

    X_all = npz['f'].astype(np.float32)
    rcl_names = [str(x) for x in npz['rcl_names']]
    fti_names = [str(x) for x in npz['fti_names']]
    rcl_name_all = np.array([rcl_names[i] for i in npz['rcl_true']])
    fti_short_all = np.array([short_fti(fti_names[i]) for i in npz['fti_true']])

    keep = balanced_subset(fti_short_all, rng)
    X = X_all[keep]
    fti_short = fti_short_all[keep]
    rcl_name = rcl_name_all[keep]
    rcl_order = [r for r in rcl_names if (rcl_name == r).any()]
    print(f'N total={len(X_all)}, plotted={len(keep)} (login_failure capped at {LOGIN_SUBSAMPLE}), dim={X.shape[1]}')

    sil_fti_hd = sil_highdim(X, fti_short)
    sil_rcl_hd = sil_highdim(X, rcl_name)
    print(f'Silhouette (high-dim cosine): FTI={sil_fti_hd:.3f}  RCL={sil_rcl_hd:.3f}')

    coords = run_tsne(X)
    sil_fti_2d = silhouette_score(coords, fti_short)
    sil_rcl_2d = silhouette_score(coords, rcl_name)
    print(f'Silhouette (2-D t-SNE):       FTI={sil_fti_2d:.3f}  RCL={sil_rcl_2d:.3f}')

    fig_path = OUT_DIR / f'learned_tsne_fti_rcl_{config}.png'
    plot_panels(coords, fti_short, rcl_name, rcl_order, fig_path, sil_fti_2d, sil_rcl_2d)

    txt_path = OUT_DIR / f'learned_silhouette_{config}.txt'
    with open(txt_path, 'w') as f:
        f.write('Silhouette on fused POST-ENCODER learned embeddings '
                'f = cat(f_metric, f_trace, f_log), out-of-sample.\n')
        f.write(f'Config: {config}.  N total={len(X_all)}, plotted={len(keep)} '
                f'(login_failure capped at {LOGIN_SUBSAMPLE}).\n\n')
        f.write('High-dim cosine:\n')
        f.write(f'  FTI (5 classes):  {sil_fti_hd:.3f}\n')
        f.write(f'  RCL (10 classes): {sil_rcl_hd:.3f}\n\n')
        f.write(f'2-D t-SNE (perplexity={TSNE_PERPLEXITY}, seed={TSNE_SEED}):\n')
        f.write(f'  FTI (5 classes):  {sil_fti_2d:.3f}\n')
        f.write(f'  RCL (10 classes): {sil_rcl_2d:.3f}\n')
    print(f'Saved: {txt_path}')


if __name__ == '__main__':
    main()
