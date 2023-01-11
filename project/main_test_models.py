from prediction import Prediction
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2

path_hdf5_file = "Model_1/backup_last_model.hdf5"
prediction = Prediction(path_hdf5_file)

img_path = "dataset/DevanagariHandwrittenCharacterDataset/chaa.png"
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
img = np.reshape(img, (1, 32, 32, 1))
# img_array = image.img_to_array(img)
# img_batch = np.expand_dims(img_array, axis=0)

prediction_result = prediction.predict_from_hdf5(img)
print("Prediction matrix : ", prediction_result)
print("Prediction max value : ", np.max(prediction_result))
print("Prediction max value index : ", np.argmax(prediction_result))
