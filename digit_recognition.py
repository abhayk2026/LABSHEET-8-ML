"""MNIST digit recognition using a Keras dense neural network."""
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
model = keras.Sequential([layers.Input((28, 28)), layers.Flatten(), layers.Dense(128, activation='relu'), layers.Dropout(0.2), layers.Dense(64, activation='relu'), layers.Dense(10, activation='softmax')])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=5, batch_size=128, validation_split=0.1)
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.4f}')
model.save('mnist_model.keras')
plt.plot(history.history['accuracy'], label='Training')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('MNIST Accuracy'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend(); plt.tight_layout();plt.savefig('mnist_learning_curve.png');plt.close()
plt.imshow(x_test[0], cmap='gray');plt.title(f'Prediction: {np.argmax(model.predict(x_test[:1], verbose=0))}');plt.axis('off');plt.savefig('mnist_prediction.png')
