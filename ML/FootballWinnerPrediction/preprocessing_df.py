import pandas as pd
from matplotlib.image import resample
from sklearn.utils import resample


def normalize(train_df, test_df, normalize_mode):
    if normalize_mode == 'min_max':
        for col in train_df.columns:
            if col != 'target':
                mean = train_df[col].mean()
                std = train_df[col].std()
                train_df[col] = (train_df[col] - mean) / std
                test_df[col] = (test_df[col] - mean) / std
        return train_df, test_df
    elif normalize_mode == 'z_score':
        for col in train_df.columns:
            if col != 'target':
                min_value = train_df[col].min()
                max_value = train_df[col].max()
                train_df[col] = (train_df[col] - min_value) / (max_value - min_value)
                test_df[col] = (test_df[col] - min_value) / (max_value - min_value)
        return train_df, test_df
    elif normalize_mode == 'advanced':
        for col in train_df.columns:
            if col != 'target':
                if "possession" in col:
                    train_df[col] = train_df[col] / 100
                    test_df[col] = test_df[col] / 100
                elif "teamWinsBalance" in col:
                    train_df[col] = train_df[col] / 10
                    test_df[col] = test_df[col] / 10
                elif "Balance" in col:
                    percentiles, train_df, max_value_to_normalize = normalize_categorize_and_dummy_balance(train_df,
                                                                                                           col)
                    percentiles, test_df, max_value_to_normalize = normalize_categorize_and_dummy_balance(test_df, col,
                                                                                                          percentiles=percentiles,
                                                                                                          max_value_to_normalize=max_value_to_normalize)
                elif "Wins" in col:
                    train_df[col] = train_df[col] / 10
                    test_df[col] = test_df[col] / 10
        return train_df, test_df
    else:
        raise ValueError(f"Unknown normalize mode: {normalize_mode}")


def normalize_categorize_and_dummy_balance(data, column, percentiles=None, max_value_to_normalize=None):
    if percentiles is None:
        percentiles = data[column].quantile([0.10, 0.25, 0.75, 0.90]).to_dict()
    print(f"percentiles for {column}: {percentiles}")

    def categorize_value(x):
        if x < percentiles[0.10]:
            return 'visitor_team_total_domination'
        elif x < percentiles[0.25]:
            return 'visitor_team_domination'
        elif x <= percentiles[0.75]:
            return 'no_domination'
        elif x <= percentiles[0.90]:
            return 'local_team_domination'
        else:
            return 'local_team_total_domination'

    categorized_column_name = f'categorized_{column}'
    data[categorized_column_name] = data[column].apply(categorize_value)
    dummies = pd.get_dummies(data[categorized_column_name], prefix=categorized_column_name)
    data = data.join(dummies)
    mapping = {'visitor_team_total_domination': -1, 'visitor_team_domination': -0.5, 'no_domination': 0,
               'local_team_domination': 0.5, 'local_team_total_domination': 1}
    data[categorized_column_name] = data[categorized_column_name].map(mapping)
    data[column], max_value_to_normalize = normalize_balance_column(data[column], column_name=column,
                                                                    max_value_to_normalize=max_value_to_normalize)
    return percentiles, data, max_value_to_normalize


def normalize_balance_column(column, column_name, max_value_to_normalize=None):
    if max_value_to_normalize is None:
        mean = column.mean()
        std = column.std()
        max_value_to_normalize = mean + std * 2
        print(f"max_value_to_normalize for before changes {column_name}: {max_value_to_normalize}")
        if max_value_to_normalize < 10:
            max_value_to_normalize = 10
        elif max_value_to_normalize > 50:
            max_value_to_normalize = 50
        print(f"max_value_to_normalize for after changes {column_name}: {max_value_to_normalize}")
    column_clipped = column.clip(lower=-max_value_to_normalize, upper=max_value_to_normalize)
    result = column_clipped / max_value_to_normalize
    return result, max_value_to_normalize


def preprocess_df(data, downsampled=False, drop_goals=False, binary_classification=False):
    if drop_goals:
        score_features = [col for col in data.columns if 'score' in col]
        goals_features = [col for col in data.columns if 'goals' in col]
        data.drop(score_features, axis=1, inplace=True)
        data.drop(goals_features, axis=1, inplace=True)
    is_winner_change_column = [col for col in data.columns if 'isWinnerChange' in col]
    if is_winner_change_column:
        data.drop(is_winner_change_column, axis=1, inplace=True)
    if binary_classification:
        target_classes = [0, 1]
        target_mapping = {'LocalTeamWin': 1, 'VisitorTeamWin': 0, 'Draw': 0}
    else:
        target_mapping = {'LocalTeamWin': 0, 'VisitorTeamWin': 1, 'Draw': 2}
        target_classes = [0, 1, 2]
    data['target'] = data['target'].map(target_mapping)
    if downsampled:
        minority_size = data['target'].value_counts().min()
        x_y = pd.concat([data.drop('target', axis=1), data['target']], axis=1)
        downsampled = pd.DataFrame()
        for target_class in target_classes:
            class_subset = x_y[x_y.target == target_class]
            downsampled_subset = resample(class_subset, replace=False, n_samples=minority_size, random_state=123)
            downsampled = pd.concat([downsampled, downsampled_subset])
        data = downsampled
    return data


def feature_selection(df, features_to_select):
    features_to_select.append('target')
    all_features = df.columns
    features_to_drop = set(all_features) - set(features_to_select)
    df.drop(features_to_drop, axis=1, inplace=True)
