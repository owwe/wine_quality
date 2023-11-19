import gradio as gr
from PIL import Image
import requests
import hopsworks
import joblib
import pandas as pd

project = hopsworks.login(project="DD2223_lab1")
fs = project.get_feature_store()


mr = project.get_model_registry()
model = mr.get_model("winequality_model", version=1)
model_dir = model.download()
model = joblib.load(model_dir + "/winequality_model.pkl")
print("Model downloaded")

def winequality(fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, sulphates, alcohol):
    print("Calling function")
#     df = pd.DataFrame([[sepal_length],[sepal_width],[petal_length],[petal_width]], 
    df = pd.DataFrame([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
       chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, sulphates, alcohol]], 
                      columns=['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'sulphates', 'alcohol'])
    print("Predicting")
    print(df)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(df) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
#     print("Res: {0}").format(res)
    print(res)
    return str(res[0]+3)
        
demo = gr.Interface(
    fn=winequality,
    title="Wine Quality Predictive Analytics",
    description="Experiment to predict wine quality.",
    allow_flagging="never",
    inputs=[
        gr.inputs.Number(default= 6.0, label='fixed acidity'), 
        gr.inputs.Number(default= 0.3, label='volatile acidity'),                
        gr.inputs.Number(default= 0.44, label='citric acid') ,               
        gr.inputs.Number(default= 1.5, label='residual sugar'),              
        gr.inputs.Number(default= 0.046, label='chlorides'),
        gr.inputs.Number(default= 15.0, label='free sulfur dioxide') ,                
        gr.inputs.Number(default= 182.0, label='total sulfur dioxide') ,                
        gr.inputs.Number(default= 0.99455, label='density'),                
        gr.inputs.Number(default= 0.52, label='sulphates'),                
        gr.inputs.Number(default= 10.4, label='alcohol'),
        ],
    outputs="text")     #gr.TextArea())

demo.launch(debug=True)

