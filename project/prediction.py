import h5py
from keras.models import load_model

class Prediction:
    
    def __init__(self, path_hdf5_file):
        # Prédire les sorties à partir de l'image
        self.model = load_model(path_hdf5_file)
    
    def predict_from_hdf5(self, image):
        prediction = self.model.predict(image)
        
        return prediction