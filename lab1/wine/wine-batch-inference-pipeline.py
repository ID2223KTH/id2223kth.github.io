import os
import modal

LOCAL = True

if LOCAL == False:
   stub = modal.Stub()
   hopsworks_image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn==1.1.1","dataframe-image"])
   @stub.function(image=hopsworks_image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalable-ml"))
   def f():
       g()


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
    import requests
    
    project = hopsworks.login(project="jayeshv")
    fs = project.get_feature_store()
    
    mr = project.get_model_registry()
    model = mr.get_model("wine_model", version=1)
    model_dir = model.download()
    model = joblib.load(model_dir + "/wine_model.pkl")
    
    feature_view = fs.get_feature_group(name="wine", version=2)
    batch_data = feature_view.get_batch_data()
    
    y_pred = model.predict(batch_data)
    seed_idx = 5
    y_pred_ = y_pred[seed_idx].round().int() + 3
    
    iris_fg = fs.get_feature_group(name="wine", version=2)
    df = iris_fg.read()
    actual_pred = df[seed_idx]['quality'].astype('int64')
    
    op_dict = {'prediction': y_pred_,
               'ground_truth': actual_pred}
    op_df = pd.DataFrame(op_dict)
    
    op_df.to_csv("./data.csv")
    dataset_api.upload("./actual_iris.png", "Resources/images", overwrite=True)

    # upload data to resources for huggingface moinitoring
    dataset_api = project.get_dataset_api()
    dataset_api.upload("./data.csv", "Resources", overwrite=True)
    
    monitor_fg = fs.get_or_create_feature_group(name="wine_predictions",
                                            version=1,
                                            primary_key=["datetime"],
                                            description="Wine Quality Prediction/Outcome Monitoring"
                                            )
    now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
    data = {
        'prediction': [y_pred_],
        'label': [actual_pred],
        'datetime': [now],
       }
    
    monitor_df = pd.DataFrame(data)
    monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})
    
    history_df = monitor_fg.read()
    # Add our prediction to the history, as the history_df won't have it - 
    # the insertion was done asynchronously, so it will take ~1 min to land on App
    history_df = pd.concat([history_df, monitor_df])

    df_recent = history_df.tail(4)
    dfi.export(df_recent, './history.png', table_conversion = 'matplotlib')
    dataset_api.upload("./history.png", "Resources", overwrite=True)
    
    predictions = history_df[['prediction']]
    labels = history_df[['label']]
    
    if predictions.value_counts().count() == 3:
        results = confusion_matrix(predictions, labels)
        df_cm = pd.DataFrame(results, [f'True {i}' for i in range(3, 9)],
                            [f'Pred {i}' for i in range(3, 9)])
        
        cm = sns.heatmap(df_cm, annot=True)
        fig = cm.get_figure()
        fig.savefig("./confusion_matrix.png")
        dataset_api.upload("./confusion_matrix.png", "Resources", overwrite=True)
    else:
        print("You need 3 different flower predictions to create the confusion matrix.")


if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()