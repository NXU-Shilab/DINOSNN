import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import gridspec
from scipy.stats import wilcoxon
plt.rcParams['pdf.fonttype'] = 42  、
plt.rcParams['ps.fonttype'] = 42

peak_df = pd.read_csv('/mnt/data0/users/lisg/Data/brain/result_load/cmn_peak.csv')
D_peak_auc = peak_df['D_peak_auc'].values
S_peak_auc = peak_df['s_peak_auc'].values
your_model_auc = D_peak_auc
compare_model_auc = S_peak_auc






better_ratio = (your_model_auc > compare_model_auc).mean()

sns.set_theme(style="whitegrid")


fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[1, 4], hspace=0.01, wspace=0.05)


ax0 = plt.subplot(gs[0, 0])


sns.kdeplot(your_model_auc, color='#F66F69', label='DINOSNN', fill=True, alpha=0.7, ax=ax0) 
sns.kdeplot(compare_model_auc, color='#6DADF0', label='scBasset', fill=True, alpha=0.7, ax=ax0)  
ax0.legend(loc='upper right', frameon=False)
ax0.set_ylabel('')
ax0.tick_params(left=False, bottom=True, labelleft=False, labelbottom=True)
ax0.set_xticks([])
ax0.grid(False)
ax0.tick_params(axis='both', which='both', length=0)
ax0.spines['top'].set_visible(False)
ax0.spines['left'].set_visible(False)
ax0.spines['right'].set_visible(False)


ax1 = plt.subplot(gs[1, 0])
scatter_colors = sns.color_palette("coolwarm", as_cmap=True) 
scatter_plot = ax1.scatter(
    your_model_auc, compare_model_auc, alpha=0.6, c=your_model_auc - compare_model_auc, cmap=scatter_colors
)
ax1.plot([0.2, 0.9], [0.2, 0.9], color='gray', linestyle='--')  
ax1.set_xlabel('DINOSNN AUROC', fontsize=12)
ax1.set_ylabel('scBasset AUROC', fontsize=12)


ax1.text(0.95, 0.05, f'Better Ratio: {better_ratio:.2%}', transform=ax1.transAxes,fontsize=12, color='black', verticalalignment='bottom', horizontalalignment='right')
ax1.grid(False)
ax1.tick_params(axis='both', which='both', length=0)
ax1.spines['top'].set_visible(False)

cax = plt.subplot(gs[1, 1]) 
cbar = plt.colorbar(scatter_plot, cax=cax)
cbar.set_label('Performance Difference', fontsize=10)

plt.tight_layout()
# plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig2czuo.pdf', format='pdf', bbox_inches='tight')
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/supplefig2/supplement2_cmn_zuo.pdf', format='pdf', bbox_inches='tight')
plt.show()


