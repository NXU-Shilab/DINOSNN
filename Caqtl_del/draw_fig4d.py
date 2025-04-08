import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
plt.rcParams['pdf.fonttype'] = 42  
plt.rcParams['ps.fonttype'] = 42

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


shap = pd.read_csv('/mnt/data0/users/lisg/Data/brain/caqtl_example/cblfinal_shap.csv')
shap_row = shap.iloc[0:1]


shap_row_sorted = shap_row.T.sort_values(by=0, ascending=False).T


plt.figure(figsize=(10, 6))
sns.set_style("white")


bar_colors = []
label_colors = []
for col in shap_row_sorted.columns:
    if 'MGC' in col:
        bar_colors.append('#e63946')  
        label_colors.append('red')    
    else:
        bar_colors.append('#BEBCDF') 
        label_colors.append('black')

ax = sns.barplot(
    x=shap_row_sorted.values.flatten(),
    y=shap_row_sorted.columns,
    palette=bar_colors 
)
ax.invert_yaxis() 


for ytick, color in zip(ax.get_yticklabels(), label_colors):
    ytick.set_color(color)


for i, v in enumerate(shap_row_sorted.values.flatten()):
    ax.text(v + 0.2, i, f"{v:.2f}", color='black', ha='left', va='center')


ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


ax.set_xlabel("(CBL)SHAP value(positive impact on model output)", fontsize=13)
ax.set_ylabel("Cell Type", fontsize=13)


plt.tight_layout()
# plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig4d/sub_bar.pdf', format='pdf',bbox_inches='tight')
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/supplefig5/cbl_bar.pdf', format='pdf',bbox_inches='tight')
plt.show()