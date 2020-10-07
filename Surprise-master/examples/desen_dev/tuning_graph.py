import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import numpy as np

from matplotlib.patches import Rectangle
import operator
import ast

from os import listdir, getcwd


plt.style.use('ggplot')

plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 20


tune_zscore = pd.read_csv('archive/parameters_Desen_ZScores.csv')

tune_zscore['param_sim_options'] = tune_zscore[['param_sim_options']].applymap(ast.literal_eval)
tune_zscore['min_support'] = tune_zscore[['param_sim_options']].applymap(operator.itemgetter('min_support'))

g1 = sns.boxplot(x="param_k", y="mean_test_rmse", data=tune_zscore)
g2 = sns.boxplot(x="param_min_k", y="mean_test_rmse", data=tune_zscore)

g1 = sns.boxplot(x="min_support", y="mean_test_rmse", data=tune_zscore)

tune_zscore_pivot = pd.pivot_table(tune_zscore, values="mean_test_rmse", index="param_k", columns="param_min_k")
ax = sns.heatmap(tune_zscore_pivot, annot=True)
ax.add_patch(Rectangle((2, 7), 1, 1, fill=False, edgecolor='white', lw=1.5))

tune_inter = pd.read_csv('archive/parameters_Desen_ZScores.csv')

g3 = sns.boxplot(x="param_k", y="mean_test_rmse", data=tune_inter)
g4 = sns.boxplot(x="param_min_k", y="mean_test_rmse", data=tune_inter)


all_files = listdir(getcwd())
all_files = [filename for filename in all_files if filename.endswith('item.csv')]

df_list = {file_path.split('.', 1)[0]: pd.read_csv(file_path) for file_path in all_files}
df_all = pd.concat(df_list).reset_index(level=0)
sns.lineplot(x='param_k', y='mean_test_rmse', hue='level_0', data=df_all)
