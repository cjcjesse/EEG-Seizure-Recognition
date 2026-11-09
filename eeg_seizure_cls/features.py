import numpy as np
import pandas as pd

from sklearn.preprocessing import PowerTransformer

def extract_stat_features(raw_df):
    df_arr = raw_df.loc[:, ~raw_df.columns.isin(['patient_id', 'binary_y'])].copy().values
    min_vals = df_arr.copy().min(axis=1).astype(np.float32)
    max_vals = df_arr.copy().max(axis=1).astype(np.float32)
    range_vals = (max_vals - min_vals).astype(np.float32)
    std_vals = df_arr.copy().std(axis=1).astype(np.float32)
    median = np.median(df_arr, axis=1, keepdims=True)
    mad_vals = np.median(np.abs(df_arr.copy() - median), axis=1).astype(np.float32)
    p25_vals = np.quantile(df_arr.copy(), 0.25, axis=1).astype(np.float32)
    p75_vals = np.quantile(df_arr.copy(), 0.75, axis=1).astype(np.float32)
    iqr_vals = (p75_vals - p25_vals).astype(np.float32)
    length = np.abs(np.diff(df_arr.copy(), axis=1)).sum(axis=1).astype(np.float32)

    return {
        'min_vals':min_vals,
        'max_vals':max_vals,
        'range_vals':range_vals,
        'std_vals':std_vals,
        'mad_vals':mad_vals,
        'p25_vals':p25_vals,
        'p75_vals':p75_vals,
        'iqr_vals':iqr_vals,
        'length':length
    }

def build_clean_df(raw_df, stat_features):
    # create clean dataframe with no transformations applied outside of element-wise log-transformation on length feature
    clean_df = pd.DataFrame({
        'min_vals':stat_features['min_vals'],
        'max_vals':stat_features['max_vals'],
        'range_vals':stat_features['range_vals'],
        'std_vals':stat_features['std_vals'],
        'mad_vals':stat_features['mad_vals'],
        'p25_vals':stat_features['p25_vals'],
        'p75_vals':stat_features['p75_vals'],
        'iqr_vals':stat_features['iqr_vals'],
        'log_length':np.log(stat_features['length'])
    })
    clean_df['patient_id'] = raw_df['patient_id'].copy()
    clean_df['binary_y'] = raw_df['binary_y'].copy()
    clean_df = clean_df[[
        'patient_id',
        'min_vals',
        'max_vals',
        'range_vals',
        'std_vals',
        'mad_vals',
        'p25_vals',
        'p75_vals',
        'iqr_vals',
        'log_length',
        'binary_y'
    ]]

    return clean_df

def build_yj_df(raw_df, stat_features):
    # EDA on statistical features/descriptors WITH Yeo-Johnson transformation
    new_df = pd.DataFrame({
        'min_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['min_vals'].reshape((-1, 1))).flatten(),
        'max_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['max_vals'].reshape((-1, 1))).flatten(),
        'range_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['range_vals'].reshape((-1, 1))).flatten(),
        'std_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['std_vals'].reshape((-1, 1))).flatten(),
        'mad_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['mad_vals'].reshape((-1, 1))).flatten(),
        'p25_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['p25_vals'].reshape((-1, 1))).flatten(),
        'p75_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['p75_vals'].reshape((-1, 1))).flatten(),
        'iqr_vals_pt':PowerTransformer(method='yeo-johnson').fit_transform(stat_features['iqr_vals'].reshape((-1, 1))).flatten(),
        'log_length':np.log(stat_features['length']) # compress large range vs all other features
    })
    new_df['patient_id'] = raw_df['patient_id'].copy()
    new_df['binary_y'] = raw_df['binary_y'].copy()
    new_df = new_df[[
        'patient_id',
        'min_vals_pt',
        'max_vals_pt',
        'range_vals_pt',
        'std_vals_pt',
        'mad_vals_pt',
        'p25_vals_pt',
        'p75_vals_pt',
        'iqr_vals_pt',
        'log_length',
        'binary_y'
    ]]

    return new_df

def summarize_features(new_df):
    features_df = new_df.loc[:, ~new_df.columns.isin(['patient_id', 'binary_y'])].copy()
    summ_stats = features_df.describe()
    summ_stats.loc['dtypes'] = new_df.dtypes
    summ_stats.loc['sparsity'] = (new_df == 0).sum()
    summ_stats.loc['range'] = summ_stats.loc['max'] - summ_stats.loc['min']
    summ_stats.loc['min (z-norm)'] = (summ_stats.loc['min'] - summ_stats.loc['mean']) / summ_stats.loc['std']
    summ_stats.loc['max (z-norm)'] = (summ_stats.loc['max'] - summ_stats.loc['mean']) / summ_stats.loc['std']
    summ_stats.loc['range (z-norm)'] = summ_stats.loc['max (z-norm)'] - summ_stats.loc['min (z-norm)']
    summ_stats.loc['min (min-max norm)'] = (summ_stats.loc['min'] - summ_stats.loc['min']) / summ_stats.loc['range']
    summ_stats.loc['max (min-max norm)'] = (summ_stats.loc['max'] - summ_stats.loc['min']) / summ_stats.loc['range']
    summ_stats.loc['range (min-max norm)'] = summ_stats.loc['max (min-max norm)'] - summ_stats.loc['min (min-max norm)']

    return features_df, summ_stats
