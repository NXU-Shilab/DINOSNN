import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
import numpy as np
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42
def cal_auc(true, pred):
    cell_auc, peak_auc = [], []
    for i in range(true.shape[0]):
        peak_auc.append(roc_auc_score(y_true=true[i, :], y_score=pred[i, :]))
    for i in range(true.shape[1]):
        cell_auc.append(roc_auc_score(y_true=true[:, i], y_score=pred[:, i]))


    print('auROC per peak:', round(sum(peak_auc) / len(peak_auc),3))
    print('auROC per cell:', round(sum(cell_auc) / len(cell_auc),3))
    return peak_auc,cell_auc

'''20%'''
peak_auc20 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate20/train_output/peak_auc.npy')
cell_auc20 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate20/train_output/cell_auc.npy')
'''40%'''
peak_auc40 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate40/train_output/peak_auc.npy')
cell_auc40 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate40/train_output/cell_auc.npy')
'''60%'''
peak_auc60 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate60/train_output/peak_auc.npy')
cell_auc60 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate60/train_output/cell_auc.npy')
'''80%'''
peak_auc80 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate80/train_output/peak_auc.npy')
cell_auc80 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/dinosnn/rate80/train_output/cell_auc.npy')
'''100%'''
peak_auc100 = np.load('/mnt/data0/users/lisg/Data/brain/ic/train_output/peak_auc.npy')
cell_auc100 = np.load('/mnt/data0/users/lisg/Data/brain/ic/train_output/cell_auc.npy')
'''Read the result of scbasset'''
'''20%'''
scpeak_auc20 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate20/peak_auc.npy')
sccell_auc20 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate20/cell_auc.npy')
'''40%'''
scpeak_auc40 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate40/peak_auc.npy')
sccell_auc40 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate40/cell_auc.npy')
'''60%'''
scpeak_auc60 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate60/peak_auc.npy')
sccell_auc60 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate60/cell_auc.npy')
'''80%'''
scpeak_auc80 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate80/peak_auc.npy')
sccell_auc80 = np.load('/mnt/data0/users/lisg/Data/brain/downsample_ic/scbasset/rate80/cell_auc.npy')
'''100%'''
scpeak_auc100 = np.load('/mnt/data0/users/lisg/scBasset/final_output/ic/peak_auc.npy')
sccell_auc100 = np.load('/mnt/data0/users/lisg/scBasset/final_output/ic/cell_auc.npy')




downsample_rates = ['100%', '80%', '60%', '40%', '20%']
models = ['DINOSNN', 'scbasset']
colors = ['#F66F69', '#6DADF0']  
total_sections = 5 
models_per_section = 2  #




data = [
    [cell_auc100, sccell_auc100], 
    [cell_auc80, sccell_auc80], 
    [cell_auc60, sccell_auc60],  
    [cell_auc40, sccell_auc40], 
    [cell_auc20, sccell_auc20]  
]

# data = [
#     [peak_auc100, scpeak_auc100],  
#     [peak_auc80, scpeak_auc80],  
#     [peak_auc60, scpeak_auc60],  
#     [peak_auc40, scpeak_auc40],  
#     [peak_auc20, scpeak_auc20]  
# ]
print('cell100:',np.round(np.mean(cell_auc100), 3),np.round(np.mean(sccell_auc100), 3))
print('cell80:',np.round(np.mean(cell_auc80), 3),np.round(np.mean(sccell_auc80), 3))
print('cell60:',np.round(np.mean(cell_auc60), 3),np.round(np.mean(sccell_auc60), 3))
print('cell40:',np.round(np.mean(cell_auc40), 3),np.round(np.mean(sccell_auc40), 3))
print('cell20:',np.round(np.mean(cell_auc20), 3),np.round(np.mean(sccell_auc20), 3))
print('----------------------------')
print('peak100:',np.round(np.mean(peak_auc100), 3),np.round(np.mean(scpeak_auc100), 3))
print('peak80:',np.round(np.mean(peak_auc80), 3),np.round(np.mean(scpeak_auc80), 3))
print('peak60:',np.round(np.mean(peak_auc60), 3),np.round(np.mean(scpeak_auc60), 3))
print('peak40:',np.round(np.mean(peak_auc40), 3),np.round(np.mean(scpeak_auc40), 3))
print('peak20:',np.round(np.mean(peak_auc20), 3),np.round(np.mean(scpeak_auc20), 3))



fig = plt.figure(figsize=(13, 6))


from matplotlib.gridspec import GridSpec
gs = GridSpec(1, 2, width_ratios=[1.5, 1])  

ax1 = fig.add_subplot(gs[0])  
ax2 = fig.add_subplot(gs[1]) 

section_height = 4.0  
model_spacing = section_height / models_per_section 


positions = []
for section_idx in range(total_sections):
    base_y = section_idx * section_height
    positions += [base_y + (j + 0.5) * model_spacing for j in range(models_per_section)]


for idx, pos in enumerate(positions):

    section_idx = idx // models_per_section
    model_idx = idx % models_per_section

    box = ax1.boxplot(data[section_idx][model_idx],
                     vert=False,
                     positions=[pos],
                     widths=0.5,
                     patch_artist=True,
                     showfliers=False)


    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(box[element], color='black', linewidth=1)
    for box in box['boxes']:
        box.set_facecolor(colors[model_idx])
        box.set_alpha(0.8)


ax1.set_yticks([i * section_height + section_height / 2 for i in range(total_sections)])
ax1.set_yticklabels(downsample_rates)
ax1.set_ylim(0, total_sections * section_height)
ax1.set_xlim(0.45, 0.85)
ax1.set_xlabel('AUROC', fontsize=12)
ax1.set_ylabel('Downsample Rate', fontsize=12)



ax1.spines['right'].set_color('lightgrey')  
ax1.spines['right'].set_linewidth(3)  
ax1.spines['right'].set_visible(True)  




xticksax1 = ax1.get_xticks()

new_xticksax1 = xticksax1[1:-1]
ax1.set_xticks(new_xticksax1)



for i in range(total_sections):
    y_start = i * section_height
    ax1.axhspan(y_start, y_start + section_height,
               facecolor='lightblue' if i % 2 == 0 else 'white', alpha=0.2)


means_dinosnn = [np.mean(data[i][0]) for i in range(total_sections)]
means_scbasset = [np.mean(data[i][1]) for i in range(total_sections)]


line_y_positions = [i * section_height + section_height/2 for i in range(total_sections)]


ax2.plot(means_dinosnn, line_y_positions,
        color=colors[0], marker='o', linestyle='--', linewidth=2, markersize=8)
ax2.plot(means_scbasset, line_y_positions,
        color=colors[1], marker='s', linestyle='-.', linewidth=2, markersize=8)


ax2.set_ylim(ax1.get_ylim())
ax2.set_yticks(line_y_positions)
ax2.get_yaxis().set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.set_xlim(0.60, 0.75)
ax2.set_yticklabels([])  
ax2.set_xlabel('Average AUROC', fontsize=12)



ax2.grid(axis='y', visible=False) 
ax2.grid(axis='x', linestyle=':', alpha=0.5)

xticksax2 = ax2.get_xticks()

new_xticksax2 = xticksax2[1:-1]
ax2.set_xticks(new_xticksax2)

for i in range(total_sections):
    y_start = i * section_height
    ax2.axhspan(y_start, y_start + section_height,
               facecolor='lightblue' if i % 2 == 0 else 'white', alpha=0.2)


legend_elements = [
    plt.Line2D([0], [0], color=colors[0], lw=5, label='DINOSNN'),
    plt.Line2D([0], [0], color=colors[1], lw=5, label='scBasset'),
    plt.Line2D([0], [0], color='black', linestyle='--', marker='o', label='Mean Line (DINOSNN)'),
    plt.Line2D([0], [0], color='black', linestyle='-.', marker='s', label='Mean Line (scBasset)')
]


plt.legend(legend_elements, models,loc='upper center',
          bbox_to_anchor=(0.55, 1.11),ncol=8,
          borderaxespad=0.85,frameon=False,fontsize=14)


plt.suptitle("AUROC distribution for per cell in the IC dataset", y=0.95,fontsize=14)


plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)  
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig2e_cell.pdf', format='pdf', bbox_inches='tight')
plt.show()



