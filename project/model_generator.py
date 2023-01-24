import visualkeras
import os

from pathlib import Path

from tensorflow.keras.models import Sequential

class ModelGenerator:
    
    def __init__(self):
        self.model = Sequential()
    
    def create_model(self, layers):
        for layer in layers:
            self.model.add(layer)

    def show_model(self, model_creation_parameters, number_model):
        if model_creation_parameters.show_model:
            visualkeras.layered_view(self.model, legend=True).show()
        if model_creation_parameters.save_model:
            path_folder = model_creation_parameters.path_save_models + "/model" + str(number_model) + "/model_img"
            
            path = Path(path_folder)
            path.mkdir(parents=True, exist_ok=True)
            
            filename = "layers_model.png"
            
            final_path = path_folder + "/" + filename
            visualkeras.layered_view(self.model, legend=True,  to_file=final_path)