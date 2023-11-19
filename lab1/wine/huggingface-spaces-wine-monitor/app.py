import gradio as gr
from PIL import Image
import hopsworks as hw

project = hw.login()
fs = project.get_feature_store()

dataset_api = project.get_dataset_api()
dataset_api.download("Resources/data.csv")
dataset_api.download("Resources/confusion_matrix.png")
dataset_api.download("Resources/history.png")

import pandas as pd
df = pd.read_csv('data.csv')

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            gr.Label("Today's Predicted Wine quality")
            input_img = gr.Number(df['prediction'], elem_id="predicted-qual")
        with gr.Column():
            gr.Label("Actual Wine Quality")
            input_img = gr.Number(df['ground_truth'], elem_id="actual-qual")
    with gr.Row():
        with gr.Column():
            gr.Label("Past predictions")
            input_img = gr.Image("history.png", elem_id="past-predict")
        with gr.Column():
            gr.Label("Confusion Maxtrix with Historical Prediction Performance")
            input_img = gr.Image("confusion_matrix.png", elem_id="confusion-matrix")