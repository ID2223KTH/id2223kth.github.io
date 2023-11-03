import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("iris_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/iris_model.pkl")
print("Model downloaded")

def iris(sepal_length, sepal_width, petal_length, petal_width):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[sepal_length,sepal_width,petal_length,petal_width]], 
                      columns=['sepal_length','sepal_width','petal_length','petal_width'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    flower_url = "https://raw.githubusercontent.com/featurestoreorg/serverless-ml-course/main/src/01-module/assets/" + res[0] + ".png"
    img = Image.open(requests.get(flower_url, stream=True).raw)            
    return img
        
demo = gr.Interface(
    fn=iris,
    title="Iris Flower Predictive Analytics",
    description="Experiment with sepal/petal lengths/widths to predict which flower it is.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default=2.0, label="sepal length (cm)"),
        gr.inputs.Number(default=1.0, label="sepal width (cm)"),
        gr.inputs.Number(default=2.0, label="petal length (cm)"),
        gr.inputs.Number(default=1.0, label="petal width (cm)"),
        ],
    outputs=gr.Image(type="pil"))

demo.launch(debug=True)

