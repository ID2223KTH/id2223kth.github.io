import os
import modal
    
LOCAL=False

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"])
   secret = modal.Secret.from_name("jim-hopsworks-ai")
   period = modal.Period(days=1)
   
   @stub.function(image=hopsworks_image, schedule=period, secret=secret)
   def f():
       g()
else:
   hopsworks_image=""
   secret=""
   period=""
   class stub:
    def function(**kwargs):
        print("Not loading Modal")

def g():
    import pandas as pd
    import hopsworks
    import joblib
    import datetime
    from PIL import Image
    from datetime import datetime
    import dataframe_image as dfi
    from sklearn.metrics import confusion_matrix
    from matplotlib import pyplot
    import seaborn as sns

    project = hopsworks.login()
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("iris", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/iris_model.pkl")
    
    feature_view = fs.get_feature_view(name="iris", version=1)
    
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    
    flower = y_pred[y_pred.size-1]
    flower_img = "assets/" + flower + ".png"
    print("Flower predicted: " + flower)
    img = Image.open(flower_img)            
    img.save("./assets/latest_iris.png")
    dataset_api = project.get_dataset_api()    
    dataset_api.upload("./assets/latest_iris.png", "Resources/images", overwrite=True)
    
    iris_fg = fs.get_feature_group(name="iris", version=1)
    df = iris_fg.read()
    label = df.iloc[-1]["variety"]
    label_flower = "assets/" + label + ".png"
    print("Flower actual: " + label)
    img = Image.open(label_flower)            
    img.save("./assets/actual_iris.png")
    dataset_api.upload("./assets/actual_iris.png", "Resources/images", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="iris_predictions",
                                                version=1,
                                                primary_key=["datetime"],
                                                description="Iris flower Prediction/Outcome Monitoring"
                                                )
    
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [flower],
        'label': [label],
        'datetime': [now],
       }
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    df_recent = history_df.tail(5)
    dfi.export(df_recent, './assets/df_recent.png', table_conversion = 'matplotlib')
    dataset_api.upload("./assets/df_recent.png", "Resources/images", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    results = confusion_matrix(labels, predictions)
    
    # Only create the confusion matrix when our iris_predictions feature group has examples of all 3 iris flowers
    if results.shape == (3,3):
        df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                             ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])
    
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./assets/confusion_matrix.png")
        dataset_api.upload("./assets/confusion_matrix.png", "Resources/images", overwrite=True)
    else:
        print("You need to run the batch inference pipeline more times until you get 3 different iris flowers")  


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()

