import pandas as pd
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
import nni
import matplotlib.pyplot as plt
# from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

file1_1_path = "/users/k21116947/ABCD/data/mri_y_rsfmr_cor_gp_aseg.csv"
file1_2_path = "/users/k21116947/ABCD/data/mri_y_rsfmr_cor_gp_gp.csv"
file1_3_path = "/users/k21116947/ABCD/data/mri_y_rsfmr_var_aseg.csv"
file1_4_path = "/users/k21116947/ABCD/data/mri_y_rsfmr_var_dsk.csv"
file1_5_path = "/users/k21116947/ABCD/data/mri_y_rsfmr_var_dst.csv"
file1_6_path = "/users/k21116947/ABCD/data/mri_y_rsfmr_var_gp.csv"

file2_path = "/users/k21116947/ABCD/data/mh_p_cbcl.csv"


file1_1 = pd.read_csv(file1_1_path)
file1_2 = pd.read_csv(file1_2_path)
file1_3 = pd.read_csv(file1_3_path)
file1_4 = pd.read_csv(file1_4_path)
file1_5 = pd.read_csv(file1_5_path)
file1_6 = pd.read_csv(file1_6_path)

dfs = [file1_1, file1_2, file1_3, file1_4, file1_5, file1_6]

dfs = [df[df['eventname'] == 'baseline_year_1_arm_1'].drop(columns=['eventname']) for df in dfs]

# 分别将筛选后的数据框赋值回原来的变量
file1_1, file1_2, file1_3, file1_4, file1_5, file1_6 = dfs


file1 = pd.merge(file1_1, file1_2, on='src_subject_id', how='inner')
file1 = pd.merge(file1, file1_3, on='src_subject_id', how='inner')
file1 = pd.merge(file1, file1_4, on='src_subject_id', how='inner')
file1 = pd.merge(file1, file1_5, on='src_subject_id', how='inner')
file1 = pd.merge(file1, file1_6, on='src_subject_id', how='inner')


file2 = pd.read_csv(file2_path)

file2_baseline_filtered = file2[file2['eventname'] == 'baseline_year_1_arm_1']

file2_drop_scores = file2_baseline_filtered.loc[:, ~file2_baseline_filtered.columns.str.endswith(('_r', '_t', '_m', '_nm', '_nm_2', '___1'))]
file2_final_cleaned = file2_drop_scores.dropna(axis=0)

file1.to_csv('features.csv')
file2_final_cleaned.to_csv('labels.csv')