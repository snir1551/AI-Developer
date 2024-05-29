import json
from pipeline import pipeline

if __name__ == '__main__':
    with open('config/config.json', 'r') as file:
        config = json.load(file)
    with open('config/models_config.json', 'r') as file:
        models_config = json.load(file)
    with open('config/dataset_config.json', 'r') as file:
        dataset_config = json.load(file)
    pipeline(config, models_config, dataset_config)
