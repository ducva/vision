import datetime

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import time
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


def preprocess_transaction(new_trans: pd.DataFrame, merchants: pd.DataFrame):
    new_trans['purchase_period'] = new_trans['purchase_date'].dt.year.astype(str).str.cat(new_trans['purchase_date'].dt.week.astype(str), sep="_")
    new_trans['purchase_period'] = new_trans['purchase_period'].astype('category').cat.codes
    new_trans['purchase_month'] = new_trans['purchase_date'].dt.month

    new_trans = pd.get_dummies(new_trans, columns=['category_2', 'category_3'])
    for col in ['authorized_flag', 'category_1']:
        new_trans[col] = new_trans[col].map({'Y': 1, 'N': 0})

    drop_columns = ['city_id', 'state_id', 'subsector_id', 'merchant_group_id', 'merchant_category_id', 'category_2', 'active_months_lag3', 'active_months_lag6', 'active_months_lag12']
    merchants.drop(drop_columns, axis=1, inplace=True)
    for col in ['category_1', 'category_4']:
        merchants[col] = merchants[col].map({'Y': 1, 'N': 0})
    merchants = pd.get_dummies(merchants, columns=['most_recent_sales_range', 'most_recent_purchases_range'])

    return pd.merge(new_trans, merchants, on="merchant_id", how="left", suffixes=('_nt', '_mc'))


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def train_lgbm(train: pd.DataFrame, test: pd.DataFrame, target, features, categorical_feats):
    param = {'num_leaves': 100,
             'min_data_in_leaf': 30,
             'objective': 'regression',
             'max_depth': 6,
             'learning_rate': 0.005,
             "min_child_samples": 20,
             "boosting": "gbdt",
             "feature_fraction": 0.9,
             "bagging_freq": 1,
             "bagging_fraction": 0.9,
             "bagging_seed": 11,
             "metric": 'rmse',
             "lambda_l1": 0.1,
             "verbosity": -1}
    folds = KFold(n_splits=5, shuffle=True, random_state=15)
    oof = np.zeros(len(train))
    predictions = np.zeros(len(test))
    start = time.time()
    feature_importance_df = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train.values, target.values)):
        print("fold n°{}".format(fold_))
        trn_data = lgb.Dataset(train.iloc[trn_idx][features], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
        val_data = lgb.Dataset(train.iloc[val_idx][features], label=target.iloc[val_idx], categorical_feature=categorical_feats)

        num_round = 10000
        clf = lgb.train(param, trn_data, num_round, valid_sets=[trn_data, val_data], verbose_eval=100, early_stopping_rounds=200)
        oof[val_idx] = clf.predict(train.iloc[val_idx][features], num_iteration=clf.best_iteration)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["feature"] = features
        fold_importance_df["importance"] = clf.feature_importance()
        fold_importance_df["fold"] = fold_ + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

        predictions += clf.predict(test[features], num_iteration=clf.best_iteration) / folds.n_splits

    print("CV score: {:<8.5f}".format(mean_squared_error(oof, target) ** 0.5))


def create_submission(test: pd.DataFrame, predictions):
    sub_df = pd.DataFrame({"card_id": test["card_id"].values})
    sub_df["target"] = predictions
    sub_df.to_csv("submit.csv", index=False)


def read_data(input_file: str):
    df = pd.read_csv(input_file, parse_dates=['first_active_month'])
    df['elapsed_time'] = (datetime.date(2018, 2, 1) - df['first_active_month'].dt.date).dt.days
    return df


class SampleDataset(Dataset):

    def __init__(self, train_df: pd.DataFrame, trans_df: pd.DataFrame):
        self.train_df: pd.DataFrame = train_df
        self.trans_df: pd.DataFrame = trans_df

    def __len__(self):
        return self.train_df.shape(0)

    def __getitem__(self, idx):
        card_id = self.train_df.iloc[idx, 0]
        card_features = self.train_df.iloc[idx, 1:].as_matrix()

        return {
            'card_id': card_id,
            'features': card_features
        }
