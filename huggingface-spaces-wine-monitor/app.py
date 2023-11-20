import gradio as gr
import hopsworks

project = hopsworks.login(project="DD2223_lab1")
fs = project.get_feature_store()

dataset_api = project.get_dataset_api() 

dataset_api.download("Resources/images/latest_wine.txt")
dataset_api.download("Resources/images/actual_wine.txt")
dataset_api.download("Resources/images/df_recent.png")
dataset_api.download("Resources/images/wine_confusion_matrix.png")

# Function to read text from a file and return it
def read_text_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

with gr.Blocks() as demo:
    with gr.Row():
      with gr.Column():
          gr.Label("Today's Predicted Wine Quality")
          input = gr.Text(read_text_file("latest_wine.txt"), elem_id="recent-text")
      with gr.Column():          
          gr.Label("Today's Actual Wine Quality")
          input = gr.Text(read_text_file("actual_wine.txt"), elem_id="recent-text")      
    with gr.Row():
      with gr.Column():
          gr.Label("Recent Prediction History")
          input = gr.Image("df_recent.png", elem_id="recent-predictions")
      with gr.Column():          
          gr.Label("Confusion Maxtrix with Historical Prediction Performance")
          input_img = gr.Image("wine_confusion_matrix.png", elem_id="confusion-matrix")        

demo.launch()
