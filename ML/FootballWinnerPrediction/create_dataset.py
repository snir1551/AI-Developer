import csv
import json
import os
from pymongo import MongoClient

with open('db_config.json', 'r') as file:
    config = json.load(file)
client = MongoClient(config['connectionString'])
db = client[config['dbName']]
collection = db[config['collectionName']]
diff_target = 0


def get_target(match):
    if 'localteamScore' not in match or 'visitorteamScore' not in match:
        raise Exception(f"No data about score in match {match['_id']}")
    local_team_score = match['localteamScore']
    visitor_team_score = match['visitorteamScore']
    if local_team_score == visitor_team_score:
        return "Draw"
    elif local_team_score > visitor_team_score:
        return "LocalTeamWin"
    else:
        return "VisitorTeamWin"


def copy_target(match):
    global diff_target
    target_legacy = get_target(match)
    target_from_db = match['winner']
    if target_legacy != target_from_db:
        print(
            f"Target legacy {target_legacy} and target from db {target_from_db} are different for match {match['_id']}")
        diff_target = diff_target + 1
    return target_from_db


def get_statistic_feature(match, minute, statistic_name):
    if statistic_name not in match:
        raise Exception(f"No data about {statistic_name} in match {match['_id']}")
    amount = 0
    if match[statistic_name] is None:
        return amount
    for data in match[statistic_name]:
        if minute == data['minute']:
            return data['amount']
        if data['minute'] > minute:
            return amount
        else:
            amount = data['amount']
    return amount


def create_csv(csv_header, dataset_rows, file_path):
    with open(file_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_header)
        writer.writeheader()
        writer.writerows(dataset_rows)


def create_csv_headers(input_data):
    csv_header = []
    for statistic in input_data:
        if "minutes" in statistic:
            for minute in statistic['minutes']:
                csv_header.append(f"{statistic['name']}_{minute}")
        else:
            csv_header.append(statistic['name'])
    csv_header.append("target")
    return csv_header


def create_balance_feature(match, local_team_statistic, visitor_team_statistic, minute):
    if local_team_statistic not in match or visitor_team_statistic not in match:
        raise Exception(
            f"No data about score in match {match['_id']} for {local_team_statistic} or {visitor_team_statistic}")
    local_team_score = get_statistic_feature(match, minute, local_team_statistic)
    visitor_team_score = get_statistic_feature(match, minute, visitor_team_statistic)
    return local_team_score - visitor_team_score


def create_previous_games_balance_feature(match):
    local_team_score = match['localTeamWins']
    visitor_team_score = match['visitorTeamWins']
    return local_team_score - visitor_team_score


def add_special_feature(match, current_line, statistic):
    if statistic['name'] == 'isWinnerChange':
        final_result = get_target(match)
        for minute in statistic['minutes']:
            current_local_team_score = get_statistic_feature(match, minute, 'goalsLocalteam')
            current_visitor_team_score = get_statistic_feature(match, minute, 'goalsVisitorTeam')
            if current_local_team_score == current_visitor_team_score:
                current_result = "Draw"
            elif current_local_team_score > current_visitor_team_score:
                current_result = "LocalTeamWin"
            else:
                current_result = "VisitorTeamWin"
            if final_result != current_result:
                current_line[f"{statistic['name']}_{minute}"] = 1
            else:
                current_line[f"{statistic['name']}_{minute}"] = 0
    elif statistic['name'] == 'teamWinsBalance':
        current_line[f"{statistic['name']}"] = create_previous_games_balance_feature(match)


def create_csv_rows(documents, statistic_config):
    dataset_rows = []
    count = 0
    errors = 0
    for match in documents:
        count = count + 1
        target = get_target(match)
        print(f"Target for match {match['_id']} is {target} total documents {count} done. errors {errors}")
        current_line = {}
        try:
            for statistic in statistic_config:
                if 'special' in statistic and statistic['special']:
                    add_special_feature(match, current_line, statistic)
                elif "copyValue" in statistic and statistic["copyValue"]:
                    current_line[f"{statistic['name']}"] = match[statistic['name']]
                elif "balanceFeature" in statistic and statistic["balanceFeature"]:
                    for minute in statistic['minutes']:
                        amount = create_balance_feature(match, statistic['localTeamStatistic'],
                                                        statistic['visitorTeamStatistic'], minute)
                        current_line[f"{statistic['name']}_{minute}"] = amount
                elif match[statistic['name']] is None:
                    for minute in statistic['minutes']:
                        current_line[f"{statistic['name']}_{minute}"] = 0
                else:
                    for minute in statistic['minutes']:
                        amount = get_statistic_feature(match, minute, statistic['name'])
                        current_line[f"{statistic['name']}_{minute}"] = amount
            current_line['target'] = target
            dataset_rows.append(current_line)
        except Exception as e:
            errors = errors + 1
            print(e)
    return dataset_rows


def create_dataset(dataset_config, file_name, config=None, first_date=None, last_date=None):
    try:
        datasets_dir = "datasets"
        if not os.path.exists(datasets_dir):
            os.makedirs(datasets_dir)
        file_path = os.path.join(datasets_dir, file_name)
        documents = get_data_cursor(10, config, first_date, last_date)
        dataset_rows = create_csv_rows(documents, dataset_config)
        csv_header = create_csv_headers(dataset_config)
        print("Starting save data to csv")
        create_csv(csv_header, dataset_rows, file_path)
        print(f"Data saved to {file_path}")
        print(f"Total diff target {diff_target}")
    except Exception as e:
        print(e)


def get_data_cursor(number_of_last_games, config=None, first_date=None, last_date=None):
    query = {}
    if first_date is not None and last_date is not None:
        query = {"$and": [{"matchTime": {"$gte": first_date}}, {"matchTime": {"$lte": last_date}}]}
    elif first_date is not None and last_date is None:
        query = {"matchTime": {"$gte": first_date}}
    elif first_date is None and last_date is not None:
        query = {"matchTime": {"$lte": last_date}}
    if config is not None and 'data_query' in config:
        if 'leagueName' in config['data_query']:
            query['leagueName'] = config['data_query']['leagueName']
    pipeline = [
        {"$match": query},
        add_stage_lookup_previous_visitor_team_games(number_of_last_games),
        add_stage_lookup_previous_local_team_games(number_of_last_games),
        {
            "$addFields": {
                "localTeamWins": {
                    "$size": {
                        "$filter": {
                            "input": "$previousLocalTeamGames",
                            "as": "game",
                            "cond": {
                                "$eq": [
                                    "$$game.winner",
                                    "LocalTeam"
                                ]
                            }
                        }
                    }
                },
                "visitorTeamWins": {
                    "$size": {
                        "$filter": {
                            "input": "$previousVisitorTeamGames",
                            "as": "game",
                            "cond": {
                                "$eq": [
                                    "$$game.winner",
                                    "VisitorTeam"
                                ]
                            }
                        }
                    }
                }
            }
        },
        {
            "$match": {
                "$expr": {
                    "$and": [
                        {
                            "$gte": [{"$size": "$previousLocalTeamGames"}, number_of_last_games]
                        },
                        {
                            "$gte": [{"$size": "$previousVisitorTeamGames"}, number_of_last_games]
                        }
                    ]
                }
            }
        },
        {"$unset": ["previousLocalTeamGames", "previousVisitorTeamGames"]}
    ]
    print(pipeline)
    cursor = collection.aggregate(pipeline)
    return cursor


def add_stage_lookup_previous_visitor_team_games(number_of_games):
    return {
        "$lookup": {
            "from": "matchFilteredWithNamesExport",
            "let": {"visitorTeam": "$visitorteamId", "matchDate": "$matchTime"},
            "pipeline": [
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                {"$eq": ["$visitorteamId", "$$visitorTeam"]},
                                {"$lt": ["$matchTime", "$$matchDate"]}
                            ]
                        }
                    }
                },
                get_sort_for_last_team_games(),
                get_limit_for_last_team_games(number_of_games),
                get_project_for_last_team_games()
            ],
            "as": "previousVisitorTeamGames"
        }
    }


def add_stage_lookup_previous_local_team_games(number_of_games):
    return {
        "$lookup": {
            "from": "matchFilteredWithNamesExport",
            "let": {"localTeam": "$localteamId", "matchDate": "$matchTime"},
            "pipeline": [
                {
                    "$match": {
                        "$expr": {
                            "$and": [
                                {"$eq": ["$localteamId", "$$localTeam"]},
                                {"$lt": ["$matchTime", "$$matchDate"]}
                            ]
                        }
                    }
                },
                get_sort_for_last_team_games(),
                get_limit_for_last_team_games(number_of_games),
                get_project_for_last_team_games()
            ],
            "as": "previousLocalTeamGames"
        }
    }


def get_project_for_last_team_games():
    return {"$project": {"ftScore": 1, "winner": 1, "matchTime": 1}}


def get_sort_for_last_team_games():
    return {"$sort": {"matchTime": -1}}


def get_limit_for_last_team_games(number_of_games):
    return {"$limit": number_of_games}
