from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

class ModelsDatabase():
    
    def __init__(self):
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
                              Dense(46, activation = 'softmax')]
        self.layers_models.append(self.layers_model1)

        self.layers_model2 = [Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                              MaxPooling2D((2, 2)),
                              Conv2D(128, (3, 3), activation='relu'),
                              MaxPooling2D((2, 2)),
                              Conv2D(256, (3, 3), activation='relu'),
                              Flatten(),
                              Dense(512, activation='relu'),
                              Dense(92, activation='softmax')]
        self.layers_models.append(self.layers_model2)

        self.layers_model3 = [LSTM(128, input_shape=(None, 28, 28, 1)),
                              Dense(92, activation='softmax')]
        self.layers_models.append(self.layers_model3)

        self.layers_model4 = [Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
                              Attention(128),
                              Dense(92, activation='softmax')]
        self.layers_models.append(self.layers_model4)



