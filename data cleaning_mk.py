#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install polars
#!pip install optuna
#!pip install numpy polars pandas scikit-learn optuna scipy matplotlib missingno colorama tqdm IPython lightgbm catboost xgboost


# In[2]:


import numpy as np
import polars as pl
import pandas as pd
from sklearn.base import clone
from copy import deepcopy
import optuna
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import missingno as msno
import re
from colorama import Fore, Style

from tqdm import tqdm
from IPython.display import clear_output

import warnings
warnings.filterwarnings('ignore')
pd.options.display.max_columns = None

import lightgbm as lgb
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import xgboost as xgb
from sklearn.ensemble import VotingRegressor
from sklearn.model_selection import *
from sklearn.metrics import *

SEED = 42
n_splits = 5


# In[3]:


get_ipython().run_cell_magic('time', '', "\ntrain = pd.read_csv('C:/Users/kelly/OneDrive - University of Iowa/STAT5400/Computing_project/train.csv')\ntest = pd.read_csv('C:/Users/kelly/OneDrive - University of Iowa/STAT5400/Computing_project/test.csv')\n#sample = pd.read_csv('/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv')\n\ntrain = train.drop('id',axis=1)\ntest = test.drop('id',axis=1)\n\nfeaturesCols = ['Basic_Demos-Enroll_Season', 'Basic_Demos-Age', 'Basic_Demos-Sex',\n       'CGAS-Season', 'CGAS-CGAS_Score', 'Physical-Season', 'Physical-BMI',\n       'Physical-Height', 'Physical-Weight', 'Physical-Waist_Circumference',\n       'Physical-Diastolic_BP', 'Physical-HeartRate', 'Physical-Systolic_BP',\n       'Fitness_Endurance-Season', 'Fitness_Endurance-Max_Stage',\n       'Fitness_Endurance-Time_Mins', 'Fitness_Endurance-Time_Sec',\n       'FGC-Season', 'FGC-FGC_CU', 'FGC-FGC_CU_Zone', 'FGC-FGC_GSND',\n       'FGC-FGC_GSND_Zone', 'FGC-FGC_GSD', 'FGC-FGC_GSD_Zone', 'FGC-FGC_PU',\n       'FGC-FGC_PU_Zone', 'FGC-FGC_SRL', 'FGC-FGC_SRL_Zone', 'FGC-FGC_SRR',\n       'FGC-FGC_SRR_Zone', 'FGC-FGC_TL', 'FGC-FGC_TL_Zone', 'BIA-Season',\n       'BIA-BIA_Activity_Level_num', 'BIA-BIA_BMC', 'BIA-BIA_BMI',\n       'BIA-BIA_BMR', 'BIA-BIA_DEE', 'BIA-BIA_ECW', 'BIA-BIA_FFM',\n       'BIA-BIA_FFMI', 'BIA-BIA_FMI', 'BIA-BIA_Fat', 'BIA-BIA_Frame_num',\n       'BIA-BIA_ICW', 'BIA-BIA_LDM', 'BIA-BIA_LST', 'BIA-BIA_SMM',\n       'BIA-BIA_TBW', 'PAQ_A-Season', 'PAQ_A-PAQ_A_Total', 'PAQ_C-Season',\n       'PAQ_C-PAQ_C_Total', 'SDS-Season', 'SDS-SDS_Total_Raw',\n       'SDS-SDS_Total_T', 'PreInt_EduHx-Season',\n       'PreInt_EduHx-computerinternet_hoursday','sii']\n\ntrain = train[featuresCols]\n#train = train.dropna(subset='sii')\n\ncat_c = ['Basic_Demos-Enroll_Season','CGAS-Season','Physical-Season','Fitness_Endurance-Season','FGC-Season',\n 'BIA-Season','PAQ_A-Season','PAQ_C-Season','SDS-Season','PreInt_EduHx-Season']\n\ndef update(df):\n    global cat_c\n    for c in cat_c : \n        df[c] = df[c].fillna('Missing')\n        df[c] = df[c].astype('category')\n        \n    return df\n        \n#train = update(train)\n#test = update(test)\n\ndef create_mapping(column, dataset):\n    unique_values = dataset[column].unique()\n    return {value: idx for idx, value in enumerate(unique_values)}\n\n    \nfor col in cat_c:\n    all_values = pd.concat([train[col], test[col]]).unique()\n    mapping = {value: idx for idx, value in enumerate(all_values)}\n\n    train[col] = train[col].replace(mapping).astype(int)\n    test[col] = test[col].replace(mapping).astype(int)")


# In[4]:


get_ipython().run_cell_magic('time', '', '\ntrain')


# In[5]:


import pandas as pd
import numpy as np
from xgboost import XGBRegressor, XGBClassifier

def fill_missing_with_xgboost(train, test, target_column, n_estimators=1000, random_state=42):
    """
    Use XGBoost model to fill in missing values in a specific feature of train and test:
    1. First merge train and test,
    2. Train the model to fill in missing values,
    3. Then split the filled train and test.
    
    Parameters:
    train (pd.DataFrame): Training dataset
    test (pd.DataFrame): Test dataset
    target_column (str): Target feature (column name) to fill missing values
    model_type (str): 'regression' or 'classification' to specify the model type
    n_estimators (int): Number of XGBoost base learners
    random_state (int): Random seed
    
    Returns:
    train_filled, test_filled: Two DataFrames (train and test with missing values filled)
    """
    global cat_c
    if target_column in cat_c:
        model_type = 'classification'
    else:
        model_type = 'regression'
    
    # 1. Add a marker column to identify train and test
    train['is_train'] = 1
    test['is_train'] = 0

    # Merge train and test datasets
    df = pd.concat([train, test], ignore_index=True)

    # 2. Find feature columns excluding the target column
    features_columns = df.columns[df.columns != target_column].tolist()

    # Extract rows with and without missing values
    df_missing = df[df[target_column].isnull()]  # Rows with missing values
    df_not_missing = df[~df[target_column].isnull()]  # Rows without missing values

    if df_missing.empty or df_not_missing.empty:
        print(f"No missing data in '{target_column}' column or all data are missing.")
        return train, test

    # 3. Prepare training set and features
    X_train = df_not_missing[features_columns]  # Features of rows without missing values
    y_train = df_not_missing[target_column]     # Target column of rows without missing values

    # 4. Initialize XGBoost model
    if model_type == 'regression':
        model = XGBRegressor(n_estimators=n_estimators, random_state=random_state)
    elif model_type == 'classification':
        model = XGBClassifier(n_estimators=n_estimators, random_state=random_state)
    else:
        raise ValueError("model_type should be either 'regression' or 'classification'.")

    # 5. Train the model
    model.fit(X_train, y_train)

    # 6. Predict missing values using the model
    X_pred = df_missing[features_columns]
    y_pred = model.predict(X_pred)

    # 7. Fill missing values with predictions
    df.loc[df[target_column].isnull(), target_column] = y_pred

    # 8. Split the dataset back into original train and test
    train_filled = df[df['is_train'] == 1].drop(columns=['is_train'])
    test_filled = df[df['is_train'] == 0].drop(columns=['is_train'])

    return train_filled, test_filled

tianbu_cols = ['CGAS-CGAS_Score','Physical-BMI','Physical-Height','Physical-Weight', 
               'Physical-Diastolic_BP','Physical-HeartRate','Physical-Systolic_BP', 
               'SDS-SDS_Total_Raw','SDS-SDS_Total_T','PreInt_EduHx-computerinternet_hoursday']

for col in tianbu_cols:
    print("Starting to fill feature: " + col)
    train, test = fill_missing_with_xgboost(train, test, col)



# In[6]:


FGC_cols = [
  'FGC-FGC_CU',
  'FGC-FGC_CU_Zone',
  'FGC-FGC_GSND',
  'FGC-FGC_GSND_Zone',
  'FGC-FGC_GSD',
  'FGC-FGC_GSD_Zone',
  'FGC-FGC_PU',
  'FGC-FGC_PU_Zone',
  'FGC-FGC_SRL',
  'FGC-FGC_SRL_Zone',
  'FGC-FGC_SRR',
  'FGC-FGC_SRR_Zone',
  'FGC-FGC_TL',
  'FGC-FGC_TL_Zone'
]
for col in FGC_cols:
    print("Starting to fill feature: " + col)
    train, test = fill_missing_with_xgboost(train, test, col)

BIA = ["BIA-BIA_TBW","BIA-BIA_TBW","BIA-BIA_DEE","BIA-BIA_BMC","BIA-BIA_Fat", "BIA-BIA_BMI"]
for col in BIA:
    print("Starting to fill feature: " + col)
    train, test = fill_missing_with_xgboost(train, test, col)


# In[7]:


train = update(train)
test = update(test)


# In[8]:


get_ipython().run_cell_magic('time', '', '\ntrain.hist(figsize=(15, 10), bins=20, xlabelsize=8, ylabelsize=8)\n\nplt.tight_layout()\nplt.show()')


# In[9]:


if 'sii' in train.columns:
    train = train.dropna(subset=['sii'])
    train["sii"].hist()
else:
    print("Column 'sii' does not exist in the DataFrame.")


# In[10]:


missing_percent_train = train.isnull().mean() * 100
missing_percent_train


# In[11]:


missing_percent_test = test.isnull().mean() * 100
missing_percent_test


# In[12]:


msno.matrix(train)
plt.show()


# In[13]:


msno.matrix(test)
plt.show()


# In[ ]:





# In[14]:


# interaction feature
def create_interaction_features(df, feature_pairs):
    global cat_c
    for feature1, feature2 in feature_pairs:
        if feature1 not in cat_c or feature2 not in cat_c:
            print("feature1: " + feature1 + ", feature2: " + feature2)
            new_feature_name = f"{feature1}_x_{feature2}"
            df[new_feature_name] = df[feature1] * df[feature2]
    return df

def create_division_features(df, feature_pairs):
    global cat_c
    for feature1, feature2 in feature_pairs:
        if feature1 not in cat_c or feature2 not in cat_c:
            print(f"feature1: {feature1}, feature2: {feature2}")
            
            # Create A/B feature, handle division by zero or NaN
            new_feature_name1 = f"{feature1}_div_{feature2}"
            df[new_feature_name1] = df[feature1] / df[feature2]
            
            # Mask NaN or zero values
            df[new_feature_name1] = df[new_feature_name1].mask((df[feature1] == 0) | (df[feature2] == 0) | 
                                                               df[feature1].isna() | df[feature2].isna())
            
            # Create B/A feature, handle similarly
            new_feature_name2 = f"{feature2}_div_{feature1}"
            df[new_feature_name2] = df[feature2] / df[feature1]
            df[new_feature_name2] = df[new_feature_name2].mask((df[feature1] == 0) | (df[feature2] == 0) | 
                                                               df[feature1].isna() | df[feature2].isna())

    return df

feature_pairs = [
    ('PreInt_EduHx-computerinternet_hoursday', 'Basic_Demos-Age'),
    ('Basic_Demos-Age', 'SDS-SDS_Total_T'),
    ('FGC-FGC_SRR_Zone', 'SDS-SDS_Total_T'),
    ('BIA-BIA_BMC', 'Physical-HeartRate'),
    #('Fitness_Endurance-Season', 'Physical-Waist_Circumference'),
    ('BIA-BIA_Fat', 'Physical-BMI'),
    ('PreInt_EduHx-Season', 'Fitness_Endurance-Season'),
    ('SDS-SDS_Total_T', 'Physical-Systolic_BP'),
    ('Basic_Demos-Sex', 'FGC-FGC_PU_Zone')
]

train = create_interaction_features(train, feature_pairs)
test = create_interaction_features(test, feature_pairs)
train = create_division_features(train, feature_pairs)
test = create_division_features(test, feature_pairs)


# In[15]:


train
# Save the train DataFrame to a CSV file
train.to_csv('C:/Users/kelly/OneDrive - University of Iowa\STAT5400\Computing_project/train_xgboost.csv', index=False)


# In[16]:


import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

# Prepare the data
X = train.drop(['sii'], axis=1)
y = train['sii']

# Train the XGBoost model
XGBoost = xgb.XGBRegressor(random_state=SEED, enable_categorical=True)
XGBoost.fit(X, y)

# Get feature importance
importance = XGBoost.feature_importances_

# Get feature names
features = X.columns

# Create a DataFrame for feature importance
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 20))
plt.barh(importance_df['Feature'], importance_df['Importance'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('Feature Importance in XGBoost')
plt.gca().invert_yaxis()  # Invert y-axis to have the most important features at the top
plt.show()


# In[17]:


##### this code is added to check the code performance 
# code performance check 
# performance metrics 

from sklearn.model_selection import train_test_split

# Assuming 'train' is your original DataFrame
X = train.drop(['sii'], axis=1)
y = train['sii']

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Train the XGBoost model on the training set
XGBoost = xgb.XGBRegressor(random_state=SEED, enable_categorical=True)
XGBoost.fit(X_train, y_train)

# Predict on the validation set
y_pred = XGBoost.predict(X_val)

# Calculate MSE
mse = mean_squared_error(y_val, y_pred)
print(f"Mean Squared Error: {mse}")


# In[18]:


##### this code is added to check the code performance 

from sklearn.model_selection import cross_val_score

# Perform cross-validation
scores = cross_val_score(XGBRegressor(random_state=SEED, enable_categorical=True), X, y, cv=5, scoring='neg_mean_squared_error')
print(f"Cross-Validation MSE: {-scores.mean()}")


# In[19]:


# 特徴量の名前を取得
features = X.columns

# データフレームとして整理（特徴量重要度と欠損値の割合を結合）
importance_df = pd.DataFrame({'Feature': features, 'Importance': importance})
missing_df = pd.DataFrame({'Feature': missing_percent_train.index, 'MissingPercent': missing_percent_train.values})
combined_df = pd.merge(importance_df, missing_df, on='Feature')

# 散布図を作成
plt.figure(figsize=(10, 6))
plt.scatter(combined_df['MissingPercent'], combined_df['Importance'], alpha=0.7)
plt.xlabel('Missing Percentage (%)')
plt.ylabel('Feature Importance')
plt.title('Feature Importance vs Missing Percentage')
plt.grid(True)
plt.show()


# In[20]:


# Get the set of columns for each DataFrame
train_columns = set(train.columns)
test_columns = set(test.columns)

# Find columns only in train but not in test
train_only = train_columns - test_columns

# Find columns only in test but not in train
test_only = test_columns - train_columns

# Output the different columns
print(f"Columns only in train: {train_only}")
print(f"Columns only in test: {test_only}")


# In[22]:


get_ipython().run_cell_magic('time', '', '\ndef quadratic_weighted_kappa(y_true, y_pred):\n    return cohen_kappa_score(y_true, y_pred, weights=\'quadratic\')\n\ndef threshold_Rounder(oof_non_rounded, thresholds):\n    return np.where(oof_non_rounded < thresholds, 0,\n                    np.where(oof_non_rounded < thresholds, 1,\n                             np.where(oof_non_rounded < thresholds, 2, 3)))\n\ndef evaluate_predictions(thresholds, y_true, oof_non_rounded):\n    rounded_p = threshold_Rounder(oof_non_rounded, thresholds)\n    return -quadratic_weighted_kappa(y_true, rounded_p)\n\ndef TrainML(model_class, test_data):\n    \n    X = train.drop([\'sii\'], axis=1)\n    y = train[\'sii\']\n    test_data = test_data.drop([\'sii\'], axis=1)\n    SKF = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)\n    \n    train_S = []\n    test_S = []\n    \n    oof_non_rounded = np.zeros(len(y), dtype=float) \n    oof_rounded = np.zeros(len(y), dtype=int) \n    test_preds = np.zeros((len(test_data), n_splits))\n\n    for fold, (train_idx, test_idx) in enumerate(tqdm(SKF.split(X, y), desc="Training Folds", total=n_splits)):\n        X_train, X_val = X.iloc[train_idx], X.iloc[test_idx]\n        y_train, y_val = y.iloc[train_idx], y.iloc[test_idx]\n\n        model = clone(model_class)\n        model.fit(X_train, y_train)\n\n        y_train_pred = model.predict(X_train)\n        y_val_pred = model.predict(X_val)\n\n        oof_non_rounded[test_idx] = y_val_pred\n        y_val_pred_rounded = y_val_pred.round(0).astype(int)\n        oof_rounded[test_idx] = y_val_pred_rounded\n\n        train_kappa = quadratic_weighted_kappa(y_train, y_train_pred.round(0).astype(int))\n        val_kappa = quadratic_weighted_kappa(y_val, y_val_pred_rounded)\n\n        train_S.append(train_kappa)\n        test_S.append(val_kappa)\n        \n        test_preds[:, fold] = model.predict(test_data)\n        \n        print(f"Fold {fold+1} - Train QWK: {train_kappa:.4f}, Validation QWK: {val_kappa:.4f}")\n        clear_output(wait=True)\n\n    print(f"Mean Train QWK --> {np.mean(train_S):.4f}")\n    print(f"Mean Validation QWK ---> {np.mean(test_S):.4f}")\n\n    KappaOPtimizer = minimize(evaluate_predictions,\n                              x0=[0.5, 1.5, 2.5], args=(y, oof_non_rounded), \n                              method=\'Nelder-Mead\') # Nelder-Mead | # Powell\n    assert KappaOPtimizer.success, "Optimization did not converge."\n    \n    oof_tuned = threshold_Rounder(oof_non_rounded, KappaOPtimizer.x)\n    tKappa = quadratic_weighted_kappa(y, oof_tuned)\n\n    print(f"----> || Optimized QWK SCORE :: {Fore.CYAN}{Style.BRIGHT} {tKappa:.3f}{Style.RESET_ALL}")\n\n    tpm = test_preds.mean(axis=1)\n    tpTuned = threshold_Rounder(tpm, KappaOPtimizer.x)\n    \n    submission = pd.DataFrame({\n        \'id\': sample[\'id\'],\n        \'sii\': tpTuned\n    })\n\n    return submission, tKappa')


# In[ ]:


from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import lightgbm as lgb

# Define the parameter grid for LightGBM
param_grid_lgbm = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [6, 12, 18],
    'num_leaves': [31, 127, 255],
    'min_data_in_leaf': [10, 20, 30],
    'feature_fraction': [0.8, 0.9, 1.0],
    'bagging_fraction': [0.8, 0.9, 1.0],
    'bagging_freq': [1, 4, 7],
    'lambda_l1': [0, 1, 10],
    'lambda_l2': [0, 0.01, 0.1]
}

# Initialize the LightGBM model
lgbm_model = lgb.LGBMRegressor(random_state=SEED)

# Initialize GridSearchCV or RandomizedSearchCV
grid_search_lgbm = RandomizedSearchCV(estimator=lgbm_model, param_distributions=param_grid_lgbm, n_iter=100, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=SEED)

# Fit the search
grid_search_lgbm.fit(X_train, y_train)

# Get the best parameters
best_params_lgbm = grid_search_lgbm.best_params_
print(f"Best parameters for LightGBM: {best_params_lgbm}")


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import scipy.stats as st

# Define the parameter distribution for XGBoost
param_dist_xgb = {
    'learning_rate': st.uniform(0.01, 0.2),
    'max_depth': st.randint(3, 10),
    'n_estimators': st.randint(100, 1000),
    'subsample': st.uniform(0.6, 1.0),
    'colsample_bytree': st.uniform(0.6, 1.0),
    'reg_alpha': st.uniform(0, 10),
    'reg_lambda': st.uniform(0, 10)
}

# Initialize the XGBoost model
xgb_model = xgb.XGBRegressor(random_state=SEED, enable_categorical=True)

# Initialize RandomizedSearchCV
random_search_xgb = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dist_xgb, n_iter=100, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=SEED)

# Fit the search
random_search_xgb.fit(X_train, y_train)

# Get the best parameters
best_params_xgb = random_search_xgb.best_params_
print(f"Best parameters for XGBoost: {best_params_xgb}")


# In[ ]:


from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
import scipy.stats as st

# Define the parameter distribution for CatBoost
param_dist_catboost = {
    'iterations': st.randint(100, 1000),
    'learning_rate': st.uniform(0.01, 0.1),
    'l2_leaf_reg': st.uniform(1, 10),
    'subsample': st.uniform(0.5, 1.0),
    'random_strength': st.uniform(1, 2),
    'bagging_temperature': st.uniform(0, 1),
    'border_count': st.randint(5, 20)
}

# Initialize the CatBoost model
catboost_model = CatBoostRegressor(random_state=SEED, silent=True)

# Initialize RandomizedSearchCV
random_search_catboost = RandomizedSearchCV(estimator=catboost_model, param_distributions=param_dist_catboost, n_iter=100, cv=5, scoring='neg_mean_squared_error', verbose=1, n_jobs=-1, random_state=SEED)

# Fit the search
random_search_catboost.fit(X_train, y_train)

# Get the best parameters
best_params_catboost = random_search_catboost.best_params_
print(f"Best parameters for CatBoost: {best_params_catboost}")


# In[ ]:


best_params_lgbm = {
    'learning_rate': 0.046,
    'max_depth': 12,
    'num_leaves': 478,
    'min_data_in_leaf': 13,
    'feature_fraction': 0.893,
    'bagging_fraction': 0.784,
    'bagging_freq': 4,
    'lambda_l1': 10,
    'lambda_l2': 0.01
}
best_params_xgb = {
    'learning_rate': 0.05,
    'max_depth': 6,
    'n_estimators': 200,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 1,
    'reg_lambda': 5
}
best_params_catboost = {
    'iterations': 804,
    'learning_rate': 0.007849710402582562, 
    'l2_leaf_reg': 7.31183636902306, 
    'subsample': 0.5630297785016092, 
    'random_strength': 1.7097065892440113, 
    'bagging_temperature': 0.026593521316435192,
    'border_count': 12
}


# In[ ]:


get_ipython().run_cell_magic('time', '', '# LightGBM\nLight = lgb.LGBMRegressor(**best_params_lgbm, random_state=SEED, verbose=-1)\nSubmission_LGBM, k_lgbm = TrainML(Light, test)')

