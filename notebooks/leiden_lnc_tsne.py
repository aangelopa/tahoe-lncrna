python -c "
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

INPUT_DIR = os.path.expanduser('~/Thesis/Data/exploratory_lncrna')
LEIDEN_DIR = os.path.expanduser('~/Thesis/Results/leiden_lncrna')

pcs = np.load(f'{INPUT_DIR}/lncrna_pca_50.npy')
meta = pd.read_parquet(f'{LEIDEN_DIR}/lncrna_leiden_metadata.parquet')

print('Running t-SNE on', pcs.shape[0], 'cells...')
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
tsne_coords = tsne.fit_transform(pcs.astype(np.float32))
np.save(f'{LEIDEN_DIR}/lncrna_tsne_embedding.npy', tsne_coords)
print('t-SNE done')

scatter_kw = dict(s=0.5, alpha=0.3, rasterized=True, edgecolors='none')
all_colors = np.vstack([plt.cm.tab20(np.linspace(0,1,20)), plt.cm.tab20b(np.linspace(0,1,20)), plt.cm.tab20c(np.linspace(0,1,20))])

# Clusters + Cell lines
clusters = meta['leiden_lncrna'].values
unique_clusters = sorted(set(clusters), key=int)
cluster_cmap = plt.cm.tab20(np.linspace(0,1,20))
cluster_colors = {c: cluster_cmap[int(c)%20] for c in unique_clusters}

cell_lines = meta['Cell_Name_Vevo'].values
unique_cls = sorted(set(cell_lines))
cl_colors = {cl: all_colors[i%len(all_colors)] for i,cl in enumerate(unique_cls)}

fig, axes = plt.subplots(1, 2, figsize=(24, 9))
axes[0].scatter(tsne_coords[:,0], tsne_coords[:,1], c=[cluster_colors[c] for c in clusters], **scatter_kw)
axes[0].set_title(f'lncRNA t-SNE — Leiden Clusters (n={len(unique_clusters)})', fontsize=13)
axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')
handles = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=cluster_colors[c],markersize=6,label=c) for c in unique_clusters]
axes[0].legend(handles=handles, title='Cluster', bbox_to_anchor=(1.02,1), loc='upper left', fontsize=6, markerscale=1.5, ncol=2, title_fontsize=8)

axes[1].scatter(tsne_coords[:,0], tsne_coords[:,1], c=[cl_colors[cl] for cl in cell_lines], **scatter_kw)
axes[1].set_title('lncRNA t-SNE — Cell Line', fontsize=13)
axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
handles_cl = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=cl_colors[cl],markersize=5,label=cl) for cl in unique_cls]
axes[1].legend(handles=handles_cl, title='Cell line', bbox_to_anchor=(1.02,1), loc='upper left', fontsize=4, markerscale=1.5, ncol=3, title_fontsize=7)

plt.suptitle('t-SNE on lncRNA expression (Leiden clusters)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{LEIDEN_DIR}/tsne_leiden_cellline.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('Saved tsne_leiden_cellline.pdf')

# Clusters + Plate + Cell cycle
fig, axes = plt.subplots(1, 3, figsize=(28, 8))
axes[0].scatter(tsne_coords[:,0], tsne_coords[:,1], c=[cluster_colors[c] for c in clusters], **scatter_kw)
axes[0].set_title(f'Leiden Clusters (n={len(unique_clusters)})', fontsize=13)
axes[0].set_xlabel('t-SNE 1'); axes[0].set_ylabel('t-SNE 2')

plates = meta['plate'].values
unique_plates = sorted(set(plates))
plate_cmap = plt.cm.Set3(np.linspace(0,1,max(len(unique_plates),12)))
plate_colors = {p: plate_cmap[i%len(plate_cmap)] for i,p in enumerate(unique_plates)}
axes[1].scatter(tsne_coords[:,0], tsne_coords[:,1], c=[plate_colors[p] for p in plates], **scatter_kw)
axes[1].set_title('Plate (Batch)', fontsize=13)
axes[1].set_xlabel('t-SNE 1'); axes[1].set_ylabel('t-SNE 2')
handles_plate = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=plate_colors[p],markersize=6,label=p) for p in unique_plates]
axes[1].legend(handles=handles_plate, title='Plate', bbox_to_anchor=(1.02,1), loc='upper left', fontsize=7, markerscale=1.2, title_fontsize=8)

phases = meta['phase'].values
phase_colors = {'G1':'#1f77b4', 'S':'#ff7f0e', 'G2M':'#2ca02c'}
axes[2].scatter(tsne_coords[:,0], tsne_coords[:,1], c=[phase_colors.get(p,'gray') for p in phases], **scatter_kw)
axes[2].set_title('Cell Cycle Phase', fontsize=13)
axes[2].set_xlabel('t-SNE 1'); axes[2].set_ylabel('t-SNE 2')
handles_phase = [plt.Line2D([0],[0],marker='o',color='w',markerfacecolor=phase_colors[p],markersize=8,label=p) for p in ['G1','S','G2M']]
axes[2].legend(handles=handles_phase, title='Phase', loc='upper right', fontsize=9, markerscale=1.5, title_fontsize=10)

plt.suptitle('lncRNA t-SNE — Factors driving clustering', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(f'{LEIDEN_DIR}/tsne_leiden_plate_cycle.pdf', dpi=150, bbox_inches='tight')
plt.close()
print('Saved tsne_leiden_plate_cycle.pdf')
"