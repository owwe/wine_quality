import os


def get_random_iris_flower():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random
    import numpy as np
    #read the wine quality data set
    df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv")
    df.drop(['type','pH'],inplace= True,axis = 1)
    df.columns = df.columns.str.replace(' ',"_")
    result = df.groupby('quality').agg(['mean', 'std']).T
    random_quality = np.random.randint(3,10)
    distinct_features  = set()
    for i in result.index:
        distinct_features.add(i[0])
    made_up_data = {}
    for feature in distinct_features:
        mean,std = result.iloc[:,0][feature]
        value = np.random.normal(mean,std,1)
        made_up_data[feature] = value
    return pd.DataFrame(made_up_data)


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login()
    fs = project.get_feature_store()

    wine_df = get_random_iris_flower()
    print('get feature group ')
    wine_fg = fs.get_feature_group(name="wine_quality",version=1)
    print(f'get feature group {wine_fg}')

    wine_fg.insert(wine_df)

if __name__ == "__main__":
    g()
    