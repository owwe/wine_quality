import pandas as pd
import hopsworks
import joblib
import datetime
from PIL import Image
from datetime import datetime
#import dataframe_image as dfi
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot
import seaborn as sns
import requests
import os
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
api_key = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=api_key, project="DD2223_lab1")
fs = project.get_feature_store()

mr = project.get_model_registry()
model = mr.get_model("winequality_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/winequality_model.pkl")

feature_view = fs.get_feature_view(name="wine_quality", version=1)
batch_data = feature_view.get_batch_data()

y_pred = model.predict(batch_data)
#print(y_pred)
offset = 1
wine = y_pred[y_pred.size-offset]
print("Predicted wine quality: " + str(int(wine) + 3))

wine_latest="latest_wine.txt"
if os.path.exists(wine_latest):
    print(f"The file 'wine_latest' already exists. Overwriting...")

# Open the file in write mode, this will create the file if it doesn't exist
with open(wine_latest, 'w') as file:
    file.write(str(int(wine) + 3))

dataset_api = project.get_dataset_api()    
dataset_api.upload("./latest_wine.txt", "Resources/images", overwrite=True)

wine_fg = fs.get_feature_group(name="wine_quality", version=1)
df = wine_fg.read() 
#print(df)
label = df.iloc[-offset]["quality"]
print("Actual wine quality: " + str(int(label)))

wine_actual="actual_wine.txt"
if os.path.exists(wine_latest):
    print(f"The file 'wine_latest' already exists. Overwriting...")

# Open the file in write mode, this will create the file if it doesn't exist
with open(wine_actual, 'w') as file:
    file.write(str(int(label)))

dataset_api.upload("./actual_wine.txt", "Resources/images", overwrite=True)

monitor_fg = fs.get_or_create_feature_group(name="wine_quality_predictions",
                                            version=1,
                                            primary_key=["datetime"],
                                            description="Wine Quality Prediction/Outcome Monitoring"
                                            )

now = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")

data = {
    'prediction': [int(wine) + 3],
    'label': [int(label)],
    'datetime': [now],
    }


monitor_df = pd.DataFrame(data)
monitor_fg.insert(monitor_df, write_options={"wait_for_job" : False})

history_df = monitor_fg.read()
# Add our prediction to the history, as the history_df won't have it - 
# the insertion was done asynchronously, so it will take ~1 min to land on App
history_df = pd.concat([history_df, monitor_df])

df_recent = history_df.tail(6)
fig, ax = plt.subplots(figsize=(8, 4))
ax.axis('tight')
ax.axis('off')
ax.table(cellText=df_recent.values, colLabels=df_recent.columns, cellLoc = 'center', loc='center')

# Save the plot as an image
plt.savefig('./df_recent.png')

# Optionally, you can also display the plot
plt.show()
#dfi.export(df_recent, './df_recent.png', table_conversion = 'matplotlib')
dataset_api.upload("./df_recent.png", "Resources/images", overwrite=True)

predictions = history_df[['prediction']]
labels = history_df[['label']]

# Only create the confusion matrix when our wine_quality_predictions feature group has examples of all 7 qualities
print("Number of different wine quality predictions to date: " + str(predictions.value_counts().count()))
if predictions.value_counts().count() == 7:
    results = confusion_matrix(labels, predictions)

    df_cm = pd.DataFrame(results, [3, 4, 5, 6, 7, 8, 9],
                            [3, 4, 5, 6, 7, 8, 9])

    cm = sns.heatmap(df_cm, annot=True)
    fig = cm.get_figure()
    fig.savefig("./wine_confusion_matrix.png")
    dataset_api.upload("./wine_confusion_matrix.png", "Resources/images", overwrite=True)
else:
    print("You need 7 different wine quality predictions to create the confusion matrix.")
    print("Run the batch inference pipeline more times until you get 7 different wine quality predictions") 