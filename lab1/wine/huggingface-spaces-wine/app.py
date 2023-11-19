import gradio as gr
from PIL import Image
import requests
import hopsworks as hw
import joblib
import pandas as pd

project = hw.login(project="jayeshv")
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
         alcohol):
    print("Lets taste wine?")
    
    df = pd.DataFrame([[type, fixed_acidity, volatile_acidity,
                        citric_acid, residual_sugar,
                        chlorides, free_sulfur_dioxide,
                        total_sulfur_dioxide,
                        density, ph, sulphates, alcohol]],
                      columns = ['type', 'fixed_acidity', 'volatile_acidity',
                                 'citric_acid', 'residual_sugar', 'chlroides',
                                 'free_sulfur_dioxide', 'total_sulfur_dioxide',
                                 'ph', 'sulphates', 'alcohol'])
    
    print("Predicting...")
    print(df)
    
    res = model.predict(df)
    print(res)
    
    return int(res.round()) + 3

demo = gr.Interface(
    fn = wine,
    title = 'Wine Quality prediction',
    description = '',
    allow_flagging = 'never',
    inputs = [
        gr.inputs.Number(default=0, label="type"),
        gr.inputs.Number(default=0.67, label="fixed_acidity"),
        gr.inputs.Number(default=-0.38, label="volatile_acidity"),
        gr.inputs.Number(defalt=0.55, label="citric_acid"),
        gr.inputs.Number(default=0.4, label="residual_sugar"),
        gr.inputs.Number(default=-0.18, label="chlroides"),
        gr.inputs.Number(default=-0.0033, label="free_sulfur_dioxide"),
        gr.inputs.Number(default=-0.3, label="total_sulfur_dioxide"),
        gr.inputs.Number(default=0.19, label="density"),
        gr.inputs.Number(default=0.22, label="ph"),
        gr.inputs.Number(default=-0.62, label="sulphates"),
        gr.inputs.Number(default=-0.37, label="alcohol")
    ],
    outputs = ['Quality']# output's an integer from 3-9
)

demo.launch(debug=True)
