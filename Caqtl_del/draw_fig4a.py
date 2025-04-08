import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import ConnectionPatch
import pandas as pd
plt.rcParams['pdf.fonttype'] = 42 
plt.rcParams['ps.fonttype'] = 42

df = pd.read_csv('/mnt/data0/users/lisg/Data/brain/caqtl_ml_result.csv')
print(df)
left_data = ['ACC', 'CBL', 'CMN', 'IC', 'PN', 'PUL', 'SUB']
left_auc = df.iloc[:7, 1].to_numpy()
print(left_auc)
right_data = ['CADD', 'Sei']
right_auc = df.iloc[7:9, 1].to_numpy()
print(right_auc)


fig, (ax1, ax2) = plt.subplots(
    1, 2,
    sharey=True,
    gridspec_kw={
        'width_ratios': [7, 2],
        'wspace': 0
    },
    figsize=(8, 3)
)


ax1.set_facecolor('#f8f8f8')  
ax1.bar(left_data, left_auc, color='#C8CBE5')  
ax1.tick_params(axis='x', rotation=45)
ax1.set_ylabel('AUROC', fontsize=12)
ax1.set_ylim(0.4, 0.7)


ax2.set_facecolor('#f0f8ff') 
bars = ax2.bar(right_data, right_auc, color=['#43BDD9', '#F2C6C2'])  
ax2.tick_params(axis='x', rotation=45)
ax2.tick_params(axis='y', which='both', left=False, labelleft=False)  



ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)


for bar in bars:
    color = bar.get_facecolor()
    x_pos = bar.get_x() + bar.get_width()
    y_top = bar.get_height()

    con = ConnectionPatch(
        xyA=(x_pos, y_top),
        xyB=(-0.4, y_top),
        coordsA=ax2.transData,
        coordsB=ax1.transData,
        linestyle='--',
        color=color,
        linewidth=1.5,
        alpha=0.8
    )
    fig.add_artist(con)


plt.tight_layout()
plt.savefig('/mnt/data0/users/lisg/Data/brain/plot_results/fig4d/fig4a.pdf', format='pdf', bbox_inches='tight')
plt.show()
