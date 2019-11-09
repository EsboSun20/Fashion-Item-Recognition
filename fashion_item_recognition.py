##Disclaimer: Made with assistance from Tensorflow Website.

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-Shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

train_images = train_images/255.0
test_images = test_images/255.0

#print(train_images[7])
#plt.imshow(train_images[7])
#plt.show()

## MODEL TRAINING
# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28,28)),
#     keras.layers.Dense(128, activation="relu"),
#     keras.layers.Dense(16, activation="sigmoid"),
#     keras.layers.Dense(10, activation="softmax")
# ])
#
# model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
#
# model.fit(train_images, train_labels, epochs=40)
#
# model.save("fashion_recognition.h5")

##END MODEL TRAINING

model = keras.models.load_model("fashion_recognition.h5")

# test_loss, test_acc = model.evaluate(test_images, test_labels)
#
# print("tested acc:", test_acc)




##TESTING
prediction = model.predict(test_images)

for i in range(10):
    print("Predict: " + class_names[np.argmax(prediction[i])])
    print("Actual: " + class_names[test_labels[i]])

## Visual testing
# for i in range(5):
#     plt.grid(False)
#     plt.imshow(test_images[i], cmap=plt.cm.binary)
#     plt.xlabel("Actual: " + class_names[test_labels[i]])
#     plt.title("Prediction:" + class_names[np.argmax(prediction[i])])
#     plt.show()
##END TESTING
