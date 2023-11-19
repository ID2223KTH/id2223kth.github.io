import os
import modal

LOCAL=False

if LOCAL == False:
    stub = modal.Stub("wine-daily")
    image = modal.Image.debian_slim().pip_install(["hopsworks"])

    @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("hopsworks.ai"))
    def f():
        g()

def generate_wine():
    """
    Returns a single wine as a single row in a DataFrame
    """
    import pandas as pd
    import hopsworks
    import numpy as np

    project = hopsworks.login(project="jayeshv")
    fs = project.get_feature_store()

    wine_fg = fs.get_feature_group(name="wine", version=2)
    wine_df = pd.DataFrame(wine_fg.read())

    indices = np.random.choice(wine_df.index, size=(1, wine_df.shape[1]), replace=True)
    df = pd.DataFrame(data=wine_df.to_numpy()[indices, np.arange(len(wine_df.columns))], 
                       columns=wine_df.columns)

    df['type'] = df['type'].astype('int64')
    df['quality'] = df['quality'].astype('int64')   
    return df


def get_random_wine():
    """
    Returns a DataFrame containing one random wine
    """
    import pandas as pd
    import random

    wine_df = generate_wine()

    print(wine_df)
    print(f"Added wine with quality {wine_df['quality'][0]}")

    return wine_df   


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(project="jayeshv")
    fs = project.get_feature_store()

    wine_df = get_random_wine()
    
    wine_fg = fs.get_feature_group(name="wine",version=2)
    print(wine_fg)
    wine_fg.insert(wine_df)


if __name__ == "__main__":
    if LOCAL == True:
        g()
    else:
        stub.deploy("wine-daily")
        with stub.run():
            f()