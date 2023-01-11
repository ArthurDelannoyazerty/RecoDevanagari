import os

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

from matplotlib import pyplot as plt

from pathlib import Path


class ModelTrainOrganisator:
    
    def __init__(self, models, model_creation_parameters):
        self.models = models
        self.model_creation_parameters = model_creation_parameters
    
    def train_all_models(self):
        for i in range(len(self.models)):
            model = self.models[i]
            
            trainGenerator = keras.utils.image_dataset_from_directory(
                directory='dataset/DevanagariHandwrittenCharacterDataset/Train/',
                labels='inferred',
                label_mode='categorical',
                color_mode = "grayscale",
                batch_size=32,
                image_size=(32, 32))
            validationGenerator = keras.utils.image_dataset_from_directory(
                directory='dataset/DevanagariHandwrittenCharacterDataset/Test/',
                labels='inferred',
                label_mode='categorical',
                color_mode = "grayscale",
                batch_size=32,
                image_size=(32, 32))

            model.compile(optimizer = Adam(learning_rate = 1e-3, decay = 1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])

            path_folder = self.model_creation_parameters.path_save_models + "/model" + str(i) + "/"
            path = Path(path_folder)
            path.mkdir(parents=True, exist_ok=True)

            callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                                        patience = 7, min_lr = 1e-5),
                        EarlyStopping(monitor = 'val_loss', patience = 9, # Patience should be larger than the one in ReduceLROnPlateau
                                    min_delta = 1e-5),
                        CSVLogger(os.path.join(path_folder, 'training.log'), append = True),
                        ModelCheckpoint(os.path.join(path_folder, 'backup_last_model.hdf5')),
                        ModelCheckpoint(os.path.join(path_folder, 'best_val_acc.hdf5'), monitor = 'val_accuracy', mode = 'max', save_best_only = True),
                        ModelCheckpoint(os.path.join(path_folder, 'best_val_loss.hdf5'), monitor = 'val_loss', mode = 'min', save_best_only = True)]

            history = model.fit(trainGenerator, epochs = 50, validation_data = validationGenerator, callbacks = callbacks)
            if self.model_creation_parameters.save_history:
                self.save_model_history(history, path_folder)


            model = load_model(os.path.join(path_folder, 'best_val_loss.hdf5'))
            loss, acc = model.evaluate(validationGenerator)

            print('Loss on Validation Data : ', loss)
            print('Accuracy on Validation Data :', '{:.4%}'.format(acc))
    
    
    def save_model_history(self, history, path_folder):
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path_folder + '/accuracy.png')
        
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path_folder + '/loss.png')
        