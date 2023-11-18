import gradio as gr
from PIL import Image
import requests
import hopsworks as hw
import joblib
import pandas as pd

project = hw.login()
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("wine_model", version=1)
model_dir = model.download()

model = joblib.load(model_dir+'/wine_model.pkl')
print("Model Loaded...")

def wine(type,
         fixed_acidity,
         volatile_acidity,
         citric_acid,
         residual_sugar,
         chlorides,
         free_sulfur_dioxide,
         total_sulfur_dioxide,
         density,
         ph,
         sulphates,
         alchol):
    
    