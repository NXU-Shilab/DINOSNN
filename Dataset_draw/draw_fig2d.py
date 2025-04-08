import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['ps.fonttype'] = 42

# df = pd.read_csv('/mnt/data0/users/lisg/Data/brain/pul_cell_performance.csv')

# print(df.head())
# print(df.columns)
# rename_dict = {'cell_type':'Cell Type'}
# df = df.rename(columns=rename_dict)
# df['cell_class'] = df['cell_class'].replace('no', 'Non-neurons')

# df_sorted = df.sort_values(['cell_class', 'count_D'], ascending=[True, False])
#

# auc_data = pd.DataFrame({
#     'DINOSNN': df_sorted['mean_auc_D'].values,
#     'scBasset': df_sorted['mean_auc_S'].values
# }, index=df_sorted['Cell Type'])

# auc_data_t = auc_data.T
#

# unique_classes = df_sorted['cell_class'].unique()
#
# color_dict = {
#     'GABA': '#3191ed',      
#     'GLUT': '#cb6751',     
#     'Non-neurons': '#da8dc9'
# }
#
# fig = plt.figure(figsize=(18, 3.2))
# gs = fig.add_gridspec(3, 1, height_ratios=[3, 8, 0.5], hspace=0)
# ax_bar = fig.add_subplot(gs[0])
# ax_heat = fig.add_subplot(gs[1])
# ax_class = fig.add_subplot(gs[2])
#

# x = np.arange(len(df_sorted)) + 0.5
# ax_bar.bar(x, df_sorted['count_D'],
#            width=0.6,
#            color='#E1E6E1',
#            edgecolor='none',
#            align='center')
#

# ax_bar.set_yscale('log')  
#

# ax_bar.spines['top'].set_visible(False)
# ax_bar.spines['right'].set_visible(False)
# ax_bar.spines['left'].set_visible(False)
# ax_bar.set_xticks([])
# ax_bar.set_ylabel('Cell Count(log)') 
# ax_bar.set_xlim(0, len(df_sorted))
#

# min_count = df_sorted['count_D'].min()
# ax_bar.set_ylim(bottom=max(1, min_count/2)) 
#
#
#

# # colors = ["#F02C77", "#FD673D", "#FFCF14"]
# # positions = [0, 0.5648854970932007, 1] 
# #

# # cmap = LinearSegmentedColormap.from_list("custom_gradient", list(zip(positions, colors)))
# colors = ["#A9D6E5", "#89C2D9", "#61A5C2", "#468FAF", "#2C7DA0",
#           "#2A6F97", "#014F86", "#01497C", "#013A63", "#012A4A"]
#

# cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
#

# sns.heatmap(auc_data_t,
#             cmap=cmap,
#             annot=True,
#             fmt='.3f',
#             cbar=False,
#             ax=ax_heat,
#             xticklabels=auc_data_t.columns,
#             yticklabels=auc_data_t.index,
#             linewidths=0)

# ax_heat.set_yticklabels(ax_heat.get_yticklabels(),
#                         rotation=0,
#                         va='center')  
#

# norm = plt.Normalize(auc_data_t.min().min(), auc_data_t.max().max())
# sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
# cbar_ax = fig.add_axes([0.92, 0.4, 0.008, 0.4])  # [x, y, width, height]
# cbar = fig.colorbar(sm, cax=cbar_ax)
# cbar.set_label('AUROC', rotation=270, labelpad=15)
#
#
#
#

# for idx, cell_class in enumerate(df_sorted['cell_class']):
#     ax_class.add_patch(plt.Rectangle((idx, 0), 1, 1,
#                                    facecolor=color_dict[cell_class],
#                                    edgecolor='none'))
#

# ax_class.set_xlim(0, len(df_sorted))
# ax_class.set_ylim(0, 1)
# ax_class.set_xticks([])
# ax_class.set_yticks([])
# ax_class.spines['top'].set_visible(False)
# ax_class.spines['right'].set_visible(False)
# ax_class.spines['bottom'].set_visible(False)
# ax_class.spines['left'].set_visible(False)
#

# legend_elements = [plt.Rectangle((0,0),1,1, facecolor=color_dict[c]) for c in unique_classes]
# ax_class.legend(legend_elements,
#                 unique_classes,
#                 loc='center left',
#                 bbox_to_anchor=(1.02, 0.5),
#                 title='Cell Classes')

# ax_heat.set_xticklabels(ax_heat.get_xticklabels(),
#                         rotation=35,
#                         ha='right',
#                         va='top',
#                         position=(0, 0))
#

# plt.tight_layout()
# fig.subplots_adjust(top=0.94, hspace=-0.3, bottom=0.2)  
# plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/supplement2_pul_you.pdf', format='pdf', bbox_inches='tight')

# plt.show()





'''---------------------------------------------------------------------------------------------------------------'''

df = pd.read_csv('/mnt/data0/users/lisg/Data/brain/buen_2018_cell_performance.csv')
values_to_remove = ['BM1137', 'BM1214', 'BM0828', 'PB1022', 'BM1077', 'BM0106', 'singles']
df = df[~df.iloc[:, 0].isin(values_to_remove)]
print(df)
rename_dict = {'cell_type':'Cell Type'}
df = df.rename(columns=rename_dict)
df_sorted=df
print(df_sorted)


auc_data = pd.DataFrame({
    'DINOSNN': df_sorted['mean_auc_D'].values,
    'scBasset': df_sorted['mean_auc_S'].values
}, index=df_sorted['Cell Type'])

auc_data_t = auc_data.T





fig = plt.figure(figsize=(20, 3.2))
gs = fig.add_gridspec(2, 1, height_ratios=[3, 8], hspace=0)  
ax_bar = fig.add_subplot(gs[0])
ax_heat = fig.add_subplot(gs[1])


x = np.arange(len(df_sorted)) + 0.5
ax_bar.bar(x, df_sorted['count_D'],
           width=0.6,
           color='#E1E6E1',
           edgecolor='none',
           align='center')
ax_bar.set_yscale('log')
ax_bar.spines['top'].set_visible(False)
ax_bar.spines['right'].set_visible(False)
ax_bar.spines['left'].set_visible(False)
ax_bar.set_xticks([])
ax_bar.set_ylabel('Cell Count(log)')
ax_bar.set_xlim(0, len(df_sorted))
min_count = df_sorted['count_D'].min()
ax_bar.set_ylim(bottom=max(1, min_count/2))


colors = ["#A9D6E5", "#89C2D9", "#61A5C2", "#468FAF", "#2C7DA0",
          "#2A6F97", "#014F86", "#01497C", "#013A63", "#012A4A"]
cmap = LinearSegmentedColormap.from_list("custom_gradient", colors)
sns.heatmap(auc_data_t,
            cmap=cmap,
            annot=True,
            fmt='.3f',
            cbar=False,
            ax=ax_heat,
            xticklabels=auc_data_t.columns,
            yticklabels=auc_data_t.index,
            linewidths=0)
ax_heat.set_yticklabels(ax_heat.get_yticklabels(),
                        rotation=0,
                        va='center')


norm = plt.Normalize(auc_data_t.min().min(), auc_data_t.max().max())
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
cbar_ax = fig.add_axes([0.92, 0.4, 0.008, 0.4])
cbar = fig.colorbar(sm, cax=cbar_ax)
cbar.set_label('AUROC', rotation=270, labelpad=15)


ax_heat.set_xticklabels(ax_heat.get_xticklabels(),
                        rotation=35,
                        ha='right',
                        va='top',
                        position=(0, 0))



fig.subplots_adjust(top=0.94, hspace=-0.3, bottom=0.1) 
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/supplement2_buen_you.pdf',format='pdf',bbox_inches='tight')
plt.show()
