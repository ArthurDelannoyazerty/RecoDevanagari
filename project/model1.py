import os

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


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

model = Sequential()

model.add(Conv2D(128, (3, 3), strides = 1, activation = 'relu', input_shape = (32, 32, 1)))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(128, (3, 3), strides = 1, activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Conv2D(128, (3, 3), strides = 1, activation = 'relu'))
model.add(MaxPooling2D((2, 2), strides = (2, 2), padding = 'same'))

model.add(Flatten())

model.add(Dense(128, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dropout(0.1))

model.add(Dense(100, activation = 'relu', kernel_initializer = 'he_uniform'))
model.add(Dropout(0.1))

model.add(Dense(46, activation = 'softmax'))

model.compile(optimizer = Adam(learning_rate = 1e-3, decay = 1e-5), loss = 'categorical_crossentropy', metrics = ['accuracy'])

if not os.path.isdir('Model_1'):
    os.mkdir('Model_1')

callbacks = [ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1,
                              patience = 7, min_lr = 1e-5),
             EarlyStopping(monitor = 'val_loss', patience = 9, # Patience should be larger than the one in ReduceLROnPlateau
                          min_delta = 1e-5),
             CSVLogger(os.path.join('Model_1', 'training.log'), append = True),
             ModelCheckpoint(os.path.join('Model_1', 'backup_last_model.hdf5')),
             ModelCheckpoint(os.path.join('Model_1', 'best_val_acc.hdf5'), monitor = 'val_accuracy', mode = 'max', save_best_only = True),
             ModelCheckpoint(os.path.join('Model_1', 'best_val_loss.hdf5'), monitor = 'val_loss', mode = 'min', save_best_only = True)]

model.fit(trainGenerator, epochs = 50, validation_data = validationGenerator, callbacks = callbacks)

model = load_model(os.path.join('Model_1', 'best_val_loss.hdf5'))
loss, acc = model.evaluate(validationGenerator)

print('Loss on Validation Data : ', loss)
print('Accuracy on Validation Data :', '{:.4%}'.format(acc))