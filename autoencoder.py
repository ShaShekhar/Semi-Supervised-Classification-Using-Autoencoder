import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle
import os
import random

def load_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_data_wrapper(name):
    f_name = './Pickled-data/{}'.format(name)
    train_data = load_data(f_name)
    #print(train_data1.shape) # (70210, 28, 28, 1)
    train_data = np.array(train_data, dtype=np.float32) # Convert into float
    train_data /= 255                                    # Normalize it
    random.shuffle(train_data)

    validation_data = load_data('./Pickled-data/val_data.pkl')
    #print(validation_data.shape) # (1458, 28, 28, 1)
    validation_data = np.array(validation_data, dtype=np.float32)
    validation_data /= 255

    test_data = load_data('./Pickled-data/test_data.pkl')
    #print(test_data.shape) # (3393, 28, 28, 1)
    test_data = np.array(test_data, dtype=np.float32)
    test_data /= 255
    return train_data, validation_data, test_data

train_data, validation_data, test_data = load_data_wrapper('train_data.pkl')

input_shape = (28, 28, 1)
input_layer = tf.keras.Input(shape=input_shape)                   # [None, 28, 28, 1]
x = tf.keras.layers.Conv2D(10, 5, activation='relu')(input_layer) # [None, 24, 24, 10]
x = tf.keras.layers.MaxPooling2D(2)(x)                            # [None, 12, 12, 10]
x = tf.keras.layers.Conv2D(20, 2, activation='relu')(x)           # [None, 11, 11, 20]
x = tf.keras.layers.MaxPooling2D(2)(x)                            # [None, 5, 5, 20]
# Decoder
x = tf.keras.layers.UpSampling2D(2)(x)                            # [None, 10, 10, 20]
x = tf.keras.layers.Conv2DTranspose(20, 2, activation='relu')(x)  # [None, 11, 11, 20]
x = tf.keras.layers.UpSampling2D(2)(x)                            # [None, 22, 22, 20]
x = tf.keras.layers.Conv2DTranspose(10, 5, activation='relu')(x)  # [None, 26, 26, 10]
x = tf.keras.layers.Conv2DTranspose(1, 3, activation='sigmoid')(x)# [None, 28, 28, 1]

model = tf.keras.Model(inputs=input_layer, outputs=x)
model.summary()

if os.path.exists('autoencoder.h5'):
    model = tf.keras.models.load_model('autoencoder.h5')
    # After training 1st time train multiple time with increased batch size e.g., 128, 256, 512, 1024, 2048
    #model.compile(loss='binary_crossentropy', optimizer='adam')
    #model.fit(train_data, train_data, batch_size=1024, epochs=5, validation_data=(validation_data, validation_data))
    #model.save('autoencoder.h5')
else:
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_data, train_data, batch_size=64, epochs=5, validation_data=(validation_data, validation_data))
    model.save('autoencoder.h5')

n = len(test_data)
random_index = np.random.randint(0, n, 10)

for i in range(len(random_index)):
    test_sample = test_data[random_index[i]]
    test_sample = np.expand_dims(test_sample, 0)
    test_prediction = model.predict(test_sample)
    # Convert the sample and prediction into 2d for plotting using matplotlib
    test_sample = np.squeeze(np.array((test_sample * 255), dtype=np.uint8))
    test_prediction = np.squeeze(np.array((test_prediction * 255), dtype=np.uint8))

    plt.subplot(1, 2, 1)
    plt.imshow(test_sample, cmap='gray')
    plt.title('Original')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.imshow(test_prediction, cmap='gray')
    plt.title('Predicted')
    plt.xticks([]), plt.yticks([])
    plt.show()
