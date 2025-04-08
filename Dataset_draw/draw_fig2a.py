import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['ps.fonttype'] = 42
root = [
    ['/mnt/data0/users/lisg/Data/brain/result_load/acc_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/acc_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/cbl_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/cbl_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/cmn_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/cmn_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/ic_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/ic_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/pn_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/pn_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/pul_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/pul_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/sub_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/sub_cell.csv'],
    ['/mnt/data0/users/lisg/Data/brain/result_load/buen_2018_peak.csv','/mnt/data0/users/lisg/Data/brain/result_load/buen_2018_cell.csv'],
]


result_peak = pd.DataFrame()
for i in root:

    peak = pd.read_csv(i[0])
    mean_values = peak.mean().to_frame().T
    file_name = i[0].split('/')[-1].split('_')[0] 
    mean_values['source'] = file_name
    result_peak = pd.concat([result_peak, mean_values], ignore_index=True)


result_cell = pd.DataFrame()
for i in root:
    peak = pd.read_csv(i[1])
    mean_values = peak.mean().to_frame().T
    file_name = i[1].split('/')[-1].split('_')[0]  
    mean_values['source'] = file_name
    result_cell = pd.concat([result_cell, mean_values], ignore_index=True)




df2 = pd.DataFrame({'model': ['DINOSNN', 'scBasset']})


n_datasets = 8

# all_ACC = result_peak[['s_peak_acc', 'D_peak_acc']].to_numpy()
# all_AUROC = result_peak[['s_peak_auc', 'D_peak_auc']].to_numpy()
# all_F1 = result_peak[['s_peak_f1', 'D_peak_f1']].to_numpy()
# all_averages = (all_ACC + all_AUROC + all_F1) / 3

all_ACC = result_cell[['s_cell_acc', 'D_cell_acc']].to_numpy()
all_AUROC = result_cell[['s_cell_auc', 'D_cell_auc']].to_numpy()
all_F1 = result_cell[['s_cell_f1', 'D_cell_f1']].to_numpy()
all_averages = (all_ACC + all_AUROC + all_F1) / 3



index_name = ['ACC', 'AUROC', 'F1']
datasets = ['ACC', 'CBL', 'CMN', 'IC', 'PN', 'PUL', 'SUB', 'Buen_2018']
print(datasets)


fig = plt.figure(figsize=(5.5, 9.3))
gs = GridSpec(n_datasets, 3, width_ratios=[1, 3, 1], wspace=0, hspace=0)


norm = plt.Normalize(0.6, 0.9)  
cmap = plt.cm.Reds  


for row in range(n_datasets):
    
    ax1 = fig.add_subplot(gs[row, 0])
    ax2 = fig.add_subplot(gs[row, 1])
    ax3 = fig.add_subplot(gs[row, 2])


    data = np.array([
        all_ACC[row],
        all_AUROC[row],
        all_F1[row]
    ]).T

   
    
    table = ax1.table(
        cellText=df2['model'].values.reshape(-1, 1),
        loc='center',
        cellLoc='center',
        colWidths=[1],
        bbox=[0, 0, 1, 1]
    )

  
    for key, cell in table.get_celld().items():
        cell.set_fontsize(11)
        cell.set_edgecolor('none')

    n_rows = len(df2)
    cell_height = 1 / n_rows


    q = 1
    s = []
    for i in range(2):
        q -= cell_height
        s.append(q)


    for i in range(1, n_rows):
        ax1.plot([0, 1], [s[i - 1], s[i - 1]], linestyle=':', color='gray', linewidth=1.5, transform=ax1.transAxes)


    if row == 0:
        ax1.text(0.5, 1.05, 'Method', ha='center', va='bottom', transform=ax1.transAxes, fontsize=13)


    ax1.text(-0.55, 0.5, datasets[row], ha='center', va='center', fontsize=12, transform=ax1.transAxes)


    Q = 1 / 3 / 2
    S = [Q*1.15, Q * 3.12, Q * 4.8]


    if row == 0:
        for i, col in enumerate(index_name):
            ax2.text(S[i], 1.05, col, ha='center', va='bottom', transform=ax2.transAxes, fontsize=13)


    for i in range(1, n_rows):
        ax2.plot([0, 1], [s[i - 1], s[i - 1]], linestyle=':', color='gray', linewidth=1.5, transform=ax2.transAxes)


    ax2.set_aspect('equal', adjustable='datalim')


    Y = [0.565, 0.815]
    S1 =[Q*0.73, Q * 2.98, Q * 5.12]
    for i, row_data in enumerate(data):
        for j, val in enumerate(row_data):
            x = S1[j]
            y = Y[i]
            circle_color = cmap(norm(val))  
            circle = patches.Circle((x, y), radius=0.10, color=circle_color, transform=ax2.transData)
            ax2.add_patch(circle)
            ax2.text(x, y-0.001, f'{val:.3f}', ha='center', va='center', fontsize=10.5, transform=ax2.transData)


    if row == 0:
        ax3.text(0.5, 1.05, 'Average', ha='center', va='bottom', transform=ax3.transAxes, fontsize=12)


    for i in range(1, n_rows):
        ax3.plot([0, 1], [s[i - 1], s[i - 1]], linestyle=':', color='gray', linewidth=1.5, transform=ax3.transAxes)


    bar_positions = [0.25, 0.75]
    bar_values = all_averages[row]
    bars= ax3.barh(bar_positions, bar_values, height=0.25)


    for bar, val in zip(bars, bar_values):
        bar.set_color(cmap(norm(val)))  


    for i, (pos, val) in enumerate(zip(bar_positions, bar_values)):
        ax3.text(val - 0.05, pos, f'{val:.3f}', ha='right', va='center', fontsize=10, color='white')


    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)


    for ax in [ax1, ax2, ax3]:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        ax.spines['top'].set_linewidth(0.7)
        ax.spines['bottom'].set_linewidth(0.7)

        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(True)


plt.tight_layout()
fig.subplots_adjust(top=0.94, hspace=-0.3, bottom=0.05)  


line = plt.Line2D([0.39, 0.75], [0.975, 0.975], color='black', linewidth=1, transform=fig.transFigure)
fig.add_artist(line)
line2 = plt.Line2D([0.02, 0.1992], [0.94, 0.94], color='black', linewidth=0.7, transform=fig.transFigure)
fig.add_artist(line2)
line3 = plt.Line2D([0.02, 0.1992], [0.05002, 0.05002], color='black', linewidth=0.7, transform=fig.transFigure)
fig.add_artist(line3)
fig.text(0.58, 0.98, 'Performance', ha='center', va='bottom', fontsize=13, transform=fig.transFigure)


fig.text(0.58, 0.03, 'per cell',
         ha='center',  
         va='center',  
         fontsize=13,  
         transform=fig.transFigure)

plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig2a.pdf', format='pdf', bbox_inches='tight')
plt.show()
