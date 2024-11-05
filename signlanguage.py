import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

dataTrain = pd.read_csv(r'archive/sign_mnist_train/sign_mnist_train.csv')
dataTest = pd.read_csv(r'archive/sign_mnist_test/sign_mnist_test.csv')

trainX = dataTrain.iloc[:,1:].values
trainY = dataTrain.iloc[:,0].values
testX = dataTest.iloc[:,1:].values
testY = dataTest.iloc[:,0].values

scaler = StandardScaler()
trainX = scaler.fit_transform(trainX)
testX = scaler.fit_transform(testX)

trainX = trainX / 255.0
testX = testX / 255.0
trainX = trainX.reshape(-1, 28, 28, 1)
testX = testX.reshape(-1, 28, 28, 1)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(25, activation='softmax')  # Output layer for classification
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history = model.fit(trainX, trainY, epochs=100, validation_data=(testX, testY))

loss, accuracy = model.evaluate(testX, testY)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
model.save('asl_image_classifier.h5')
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()