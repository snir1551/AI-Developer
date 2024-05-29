import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, precision_score, f1_score, \
    classification_report
from sklearn.model_selection import GridSearchCV, cross_val_predict, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC


def run_greed_search_cv(config, df_train, df_validation, configuration, base_model_result=None, minute=None,
                        sample_weight=None):
    results_folder = config["results_folder"]
    models_greed_search_sv = build_greed_search_scv(config)
    if 'proba' in config and config['proba']:
        proba = config['proba']
    else:
        proba = None
    if 'lower_proba' in config and config['lower_proba']:
        lower_proba = config['lower_proba']
    else:
        lower_proba = None
    for grid_search in models_greed_search_sv:
        start_time = datetime.now()
        print(f"Running grid search for {grid_search.estimator}...")
        x_train = df_train.drop("target", axis=1)
        y_train = df_train["target"]
        if sample_weight:
            grid_search.fit(x_train, y_train, sample_weight=sample_weight)
        else:
            grid_search.fit(x_train, y_train)
        params = grid_search.best_params_
        score = grid_search.best_score_
        best_model = grid_search.best_estimator_
        best_model_retrain = retrain_best_model(grid_search, x_train, y_train, sample_weight=sample_weight)
        validation_results = test_model(best_model_retrain, df_validation.drop("target", axis=1),
                                        df_validation["target"], proba=proba, lower_proba=lower_proba)
        other_results = []
        for i in range(len(grid_search.cv_results_['rank_test_score'])):
            other_results.append({key: convert(grid_search.cv_results_[key][i]) for key in grid_search.cv_results_})
        end_time = datetime.now()
        end_time_str = end_time.strftime("%Y-%m-%d_%H-%M-%S")
        result = {
            "comment": configuration['config']["comment"],
            "minute": minute,
            "base_model_result": base_model_result,
            "params": params,
            "score": score,
            "validation": validation_results,
            "configuration": configuration,
            "time": str({end_time - start_time})
            # "other_results": other_results
        }
        print(f"Finished grid search for {grid_search.estimator} in {end_time - start_time}")
        if not os.path.exists(results_folder):
            os.makedirs(results_folder)
        estimator_dir = f"{results_folder}/{grid_search.estimator}"
        estimator_dir = estimator_dir.replace("()", "")
        if not os.path.exists(estimator_dir):
            os.makedirs(estimator_dir)
        with open(f"{estimator_dir}/{end_time_str}.json", "w") as file:
            json.dump(result, file, default=convert)


def test_model(model, x_test, y_test, proba=None, lower_proba=None):
    if proba:
        y_pred = model.predict_proba(x_test)
        y_pred = np.where(y_pred[:, 1] > 1 - proba, 1, 0)
    elif lower_proba:
        y_pred = model.predict_proba(x_test)
        y_pred = np.where(y_pred[:, 1] > lower_proba, 1, 0)  # lower_proba < original threshold
    else:
        y_pred = model.predict(x_test)
    try:
        feature_importances = model.feature_importances_
        features_df = pd.DataFrame({'Feature': x_Xtest.columns, 'Importance': feature_importances})
        features_df = features_df.sort_values(by='Importance', ascending=False)
        features_df_top_20 = features_df.head(100)
        features_json = features_df_top_20.to_json(orient='records')
        feature_importances = json.loads(features_json)
    except Exception as e:
        print(f"Error while extracting feature importances: {e}")
        feature_importances = None
    try:
        calculate_metrics_for_each_class_value = calculate_metrics_for_each_class(confusion_matrix(y_test, y_pred))
    except Exception as e:
        print(f"Error while calculating metrics for each class: {e}")
        calculate_metrics_for_each_class_value = None
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "metrics_for_each_class": calculate_metrics_for_each_class_value,
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
        "f1": f1_score(y_test, y_pred, average="macro"),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importances": feature_importances
    }


def calculate_metrics_for_each_class(confusion_matrix_data):
    tn, fp = confusion_matrix_data[0]
    fn, tp = confusion_matrix_data[1]

    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0

    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_score_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0

    return {
        "precision_1": precision_1,
        "recall_1": recall_1,
        "f1_score_1": f1_score_1,
        "precision_0": precision_0,
        "recall_0": recall_0,
        "f1_score_0": f1_score_0,
    }


def convert(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def build_greed_search_scv(config):
    models_configs = config["models_configs"]
    results = []
    for model_config in models_configs:
        model_type = model_config["type"]
        params = model_config["params"]
        if model_type == "RandomForestClassifier":
            model = RandomForestClassifier()
        elif model_type == "LinearSVC":
            model = LinearSVC()
        elif model_type == "SVC":
            model = SVC()
        elif model_type == "GradientBoostingClassifier":
            model = GradientBoostingClassifier()
        elif model_type == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        elif model_type == "LogisticRegression":
            model = LogisticRegression()
        else:
            raise ValueError(f"Model type {model_type} is not supported")
        results.append(
            create_search_cv(model, params, cv=config['cv'], n_jobs=config['n_jobs'], verbose=config['verbose']))
    return results


def create_search_cv(model, params, cv, n_jobs, verbose):
    return GridSearchCV(model, params, cv=cv, n_jobs=n_jobs, verbose=verbose)


def percentage_no_winner_change(data, minute):
    goals_local_team_col = f'goalsLocalteam_{minute}'
    goals_visitor_team_col = f'goalsVisitorTeam_{minute}'
    if goals_local_team_col not in data.columns or goals_visitor_team_col not in data.columns:
        raise Exception(f"Data for minute {minute} is missing.")
    conditions = [
        (data[goals_local_team_col] > data[goals_visitor_team_col]),
        (data[goals_local_team_col] < data[goals_visitor_team_col]),
    ]
    choices = ['LocalTeamWin', 'VisitorTeamWin']
    data['leader_at_minute'] = np.select(conditions, choices, default='Draw')
    data['change_in_winner'] = data['leader_at_minute'] != data['target']
    no_change_count = data.shape[0] - data['change_in_winner'].sum()
    percentage_no_change = (no_change_count / data.shape[0]) * 100
    return percentage_no_change


def percentage_no_winner_change_binary_classification(data, minute):
    goals_local_team_col = f'goalsLocalteam_{minute}'
    goals_visitor_team_col = f'goalsVisitorTeam_{minute}'
    if goals_local_team_col not in data.columns or goals_visitor_team_col not in data.columns:
        raise Exception(f"Data for minute {minute} is missing.")
    conditions = [
        (data[goals_local_team_col] > data[goals_visitor_team_col])
    ]
    choices = ['LocalTeamWin']
    data['leader_at_minute'] = np.select(conditions, choices, default='NotLocalWin')
    data['target_merged'] = data['target'].replace({'VisitorTeamWin': 'NotLocalWin', 'Draw': 'NotLocalWin'})
    data['change_in_winner'] = data['leader_at_minute'] != data['target_merged']
    no_change_count = data.shape[0] - data['change_in_winner'].sum()
    percentage_no_change = (no_change_count / data.shape[0]) * 100
    return percentage_no_change


def find_last_minute_in_data(data):
    goal_columns = [col for col in data.columns if
                    col.startswith('goalsLocalteam_') or col.startswith('goalsVisitorTeam_')]
    minutes = [int(col.split('_')[-1]) for col in goal_columns if col.split('_')[-1].isdigit()]
    last_minute = max(minutes) if minutes else None
    return last_minute


def retrain_best_model(grid_search, x_train, y_train, sample_weight=None):
    print(f"Retraining the best model...")
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    new_model = best_model.__class__(**best_params)
    if sample_weight:
        new_model.fit(x_train, y_train, sample_weight=sample_weight)
    else:
        new_model.fit(x_train, y_train)
    print(f"Done retraining the best model.")
    return best_model
