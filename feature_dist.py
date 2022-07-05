# @Author  : Andrian Lee
# @Time    : 2021/12/12 15:13
# @File    : plotting.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

sns.set_theme(style="darkgrid")

df = pd.read_excel(r"D:\pre_grad_research\MantleWater\HDiff-XAI\all_data_20220606.xlsx", sheet_name='train')
# df = df[df['train_test'] == 0]

# 0, 1
color_list = ['#EF90B9', '#85B4FF']

fig = plt.figure(figsize=(10, 10))  # figsize=(2, 4)

for idx, i in enumerate(df.columns[10:19]):
    print(idx, (int((idx) / 3) * 4 + ((idx) % 3) + 1))
    ax = fig.add_subplot(3, 5, (int((idx) / 3) * 5 + ((idx) % 3) + 1))
    sns.violinplot(x="label", y=i, data=df, ax=ax, palette=color_list, )  # width=0.2, linewidth=1

ax = plt.subplot2grid((3, 5), (0, 3), colspan=2, rowspan=2)
sns.violinplot(x="label", y=df.columns[19], data=df, ax=ax, palette=color_list)
ax.set_ylabel("H$_{2}$O (ppm)")

plt.tight_layout()

plt.savefig("./feature_dist.png")
plt.show()

# figure,ax=plt.subplots(2,2)
