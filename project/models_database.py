from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, LSTM, Attention

class ModelsDatabase():
    
    def __init__(self, model_creation_parameters):
        self.layers_models = []

        self.layers_model1 = [Conv2D(128, (3, 3), strides = 1, activation = 'relu', input_shape = (32, 32, 1)),
                              MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'),
                              Conv2D(128, (3, 3), strides = 1, activation = 'relu'),
                              MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'),
                              Conv2D(128, (3, 3), strides = 1, activation = 'relu'),
                              MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'),
                              Flatten(),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dropout(0.1),
                              Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dropout(0.1),
                              Dense(model_creation_parameters.number_output, activation = 'softmax')]
        self.layers_models.append(self.layers_model1)

        self.layers_model2 = [Dense(128, activation = 'relu', kernel_initializer = 'he_uniform', input_shape = (32, 32, 1)),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dense(model_creation_parameters.number_output, activation = 'softmax')]
        self.layers_models.append(self.layers_model2)

        self.layers_model3 = [LSTM(128, input_shape=(32, 32)),
                              Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'),
                              Dense(model_creation_parameters.number_output, activation='softmax')]
        self.layers_models.append(self.layers_model3)





