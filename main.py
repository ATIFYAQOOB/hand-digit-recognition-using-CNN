import cv2
import numpy as np
from keras.models import load_model
img = cv2.imread("5.png",cv2.IMREAD_GRAYSCALE)
model = load_model("model.h5")
img = cv2.resize(img,(28,28))
data = np.array(img)
img_data = data
img_data = img_data/255.0
data = img_data.reshape(-1,28,28,1)
model_out = model.predict([data])
print(np.argmax(model_out))
