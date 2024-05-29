from copy import copy
from datetime import datetime

import pandas as pd
import sklearn

import preprocessing_df
from create_dataset import create_dataset
from model_creation import run_greed_search_cv, percentage_no_winner_change, \
    percentage_no_winner_change_binary_classification, find_last_minute_in_data


def pipeline(config, models_config, dataset_config):
    start_time = datetime.now()
    print(f"Starting pipeline.... Time is {start_time}", )
    create_datasets(config, dataset_config)
    create_model(config, models_config, dataset_config)
    end_time = datetime.now()
    print(f"Done. Execution time: {end_time - start_time}")


def create_datasets(config, dataset_config):
    if config['create_train_dataset']:
        last_date = datetime.strptime(config['last_date_in_train_dataset'], '%Y-%m-%d')
        create_dataset(dataset_config, "data.csv", config=config, last_date=last_date)
    if config['create_test_dataset']:
        last_date = datetime.strptime(config['last_date_in_test_dataset'], '%Y-%m-%d')
        first_date = datetime.strptime(config['last_date_in_train_dataset'], '%Y-%m-%d')
        create_dataset(dataset_config, config=config, last_date=last_date, first_date=first_date,
                       file_name="test_data.csv")


def create_model(config, models_config, dataset_config):
    configuration = {"config": config, "models_config": models_config, "dataset_config": dataset_config}
    dataset_name, drop_goals, is_binary_classification, normalize_mode, save_min_dataset = get_config(config)
    df = pd.read_csv(f"datasets/{dataset_name}")
    test_df = pd.read_csv(f"datasets/test_{dataset_name}")
    minute = find_last_minute_in_data(test_df)
    base_model_result = get_base_model_result(is_binary_classification, test_df, minute)
    print(f"Base model result: {base_model_result}")
    df, test_df = preprocessing_data(dataset_name, drop_goals, is_binary_classification, normalize_mode, df, test_df)
    select_features(df, test_df, config)
    print(f"iterate_over_features for df")
    iterate_over_features(df)
    print(f"iterate_over_features for test df")
    iterate_over_features(test_df)
    save_min_datasets(save_min_dataset, df, test_df)
    run_greed_search_cv(models_config, df, test_df, configuration, base_model_result=base_model_result,
                        minute=minute)


def save_min_datasets(save_min_dataset, df, test_df):
    if save_min_dataset:
        df.to_csv(f"datasets/df_min.csv", index=False)
        test_df.to_csv(f"datasets/test_df_min.csv", index=False)


def preprocessing_data(dataset, drop_goals, is_binary_classification, normalize_mode, df, test_df, save_df=False):
    df = preprocessing_df.preprocess_df(df, downsampled=True, drop_goals=drop_goals,
                                        binary_classification=is_binary_classification)
    test_df = preprocessing_df.preprocess_df(test_df, downsampled=False, drop_goals=drop_goals,
                                             binary_classification=is_binary_classification)
    if normalize_mode:
        df, test_df = preprocessing_df.normalize(df, test_df, normalize_mode)
    if save_df:
        df.to_csv(f"datasets/{dataset}_preprocessed.csv", index=False)
        test_df.to_csv(f"datasets/test_{dataset}_preprocessed.csv", index=False)
    return df, test_df


def get_base_model_result(is_binary_classification, test_df, minute):
    if is_binary_classification:
        base_model_result = percentage_no_winner_change_binary_classification(copy(test_df), minute)
    else:
        base_model_result = percentage_no_winner_change(copy(test_df), minute)
    return base_model_result


def get_config(config):
    dataset = config['dataset']
    drop_goals = config['drop_goals']
    is_binary_classification = config['binary_classification']
    normalize_mode = config['normalize_mode']
    save_min_dataset = config['save_min_dataset']
    return dataset, drop_goals, is_binary_classification, normalize_mode, save_min_dataset


def iterate_over_features(df):
    columns = df.columns
    summary = {}
    print("Existing columns in dataset:")
    for column in columns:
        if df[column].dtype == 'bool':  # Handle boolean columns
            value_counts = df[column].value_counts().to_dict()
            summary[column] = {'True': value_counts.get(True, 0), 'False': value_counts.get(False, 0)}
            print(f"{column} summary: {summary[column]}")
            df[column] = df[column].map({True: 1, False: 0})
        elif df[column].dtype == "object" or pd.api.types.is_string_dtype(df[column]):  # for categorical columns
            value_counts = df[column].value_counts().to_dict()
            summary[column] = {"Unique Values Count": len(value_counts), "Values": value_counts}
        else:  # for numeric columns
            summary[column] = {"Min": df[column].min(), "Max": df[column].max()}
        print(f"{column} {summary[column]}")
    print(f"Total columns:{len(columns)}")
    print(summary)


def select_features(df, test_df, config):
    features_to_select = get_feature_to_select(config)
    if features_to_select:
        preprocessing_df.feature_selection(df, features_to_select)
        preprocessing_df.feature_selection(test_df, features_to_select)


def get_feature_to_select(config):
    if "features_to_select" in config:
        features_to_select = config['features_to_select']
    else:
        features_to_select = None
    return features_to_select
