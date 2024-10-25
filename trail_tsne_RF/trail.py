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
from sklearn.multioutput import MultiOutputRegressor


features =  pd.read_csv("/users/k21116947/ABCD/trail4/features.csv")
features = features.drop(features.columns[0],axis=1)
labels_raw =  pd.read_csv("/users/k21116947/ABCD/trail4/labels.csv")
labels_raw = labels_raw.drop(labels_raw.columns[0],axis=1)

params = nni.get_next_parameter()
# params = get_next_parameter(search_space)

# Apply t-SNE on the reduced dataset
n_components = params['n_components']


#apply t-SNE
tsne = TSNE(
        n_components=params['n_components'],
        perplexity=params['perplexity'],
        learning_rate=params['learning_rate'],
        metric=params['metric'],
        random_state=42,
    )
tsne_results = tsne.fit_transform(labels_raw.drop(columns = ['src_subject_id', 'eventname']))

# Create a DataFrame with the t-SNE results
tsne_columns = [f'tsne_dim_{i+1}' for i in range(n_components)]


file2_tsne_df_clean_reduced = pd.DataFrame(tsne_results, columns=tsne_columns)

file2_tsne_df_clean_reduced['src_subject_id'] = labels_raw.loc[labels_raw.index, 'src_subject_id'].values

file2_tsne_baseline_reduced = file2_tsne_df_clean_reduced

# Merging both datasets on 'src_subject_id'
merged_data_reduced = pd.merge(features, file2_tsne_baseline_reduced, on='src_subject_id', how='inner')

merged_data_final= merged_data_reduced.dropna(axis=1, thresh = 3).dropna(axis=0)


# Extracting features and labels for the random forest model
X = merged_data_final.drop(columns=['src_subject_id'] + tsne_columns)
# X = X.drop(X.columns[0],axis=1)
# y = merged_data_final[tsne_columns]
y = merged_data_final[tsne_columns].mean(axis=1)

# Create a pipeline that scales and then applies the model

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('rf', RandomForestRegressor(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_split=params['min_samples_split'],
        min_samples_leaf=params['min_samples_leaf'],
        random_state=42
    ))
])

# Perform 10-fold cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=10, scoring='r2')

# Report the mean R2 score from cross-validation to NNI
mean_cv_r2 = cv_scores.mean()
nni.report_final_result(mean_cv_r2)

# Print cross-validation scores
print(f'10-Fold Cross-Validation R2 Scores: {cv_scores}')
print(f'Mean R2 Score from 10-Fold Cross-Validation: {mean_cv_r2}')
