import visualkeras

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
            path = 'model_img/model'+ str(number_model+1) +'.png'
            visualkeras.layered_view(self.model, legend=True,  to_file=path)