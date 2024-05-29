from flask import Flask, request, jsonify

from pipeline import pipeline

from fastapi import FastAPI

app = FastAPI()


@app.post('/run-pipeline', response_description='Start new pipeline')
def run_pipeline(data: dict):
    print(data)
    config = data['config']
    models_config = data['models_config']
    dataset_config = data['dataset_config']
    pipeline(config, models_config, dataset_config)
    return jsonify({"status": "Pipeline executed", "result": "OK"})

