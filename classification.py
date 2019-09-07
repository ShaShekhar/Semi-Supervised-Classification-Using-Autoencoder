from imageio import imread
import numpy as np
import tensorflow as tf
import pickle
import h5py
import csv
import os

def load_data(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_data_wrapper():
    train_data = load_data('train-data.pkl')
    val_data = load_data('val-data.pkl')
    test_data = load_data('test-data.pkl')
    return train_data, val_data, test_data

(train_images, train_labels), (val_images, val_labels), (test_images, test_labels) = load_data_wrapper()

input_shape = (28, 28, 1)
input_layer = tf.keras.Input(shape=input_shape)
x = tf.keras.layers.Conv2D(10, 5, activation='relu')(input_layer)
x = tf.keras.layers.MaxPooling2D(2)(x)
x = tf.keras.layers.Conv2D(20, 2, activation='relu')(x)
x = tf.keras.layers.MaxPooling2D(2)(x)

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
prediction = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=input_layer, outputs=prediction)
model.summary()

#layer_dict = dict([(layer.name, layer) for layer in model.layers])
layer_names = [layer.name for layer in model.layers]

weight_path = 'autoencoder.h5'
f = h5py.File(weight_path)

for i in layer_names[0 : 5]:
    weight_names = f['model_weights'][i].attrs['weight_names']
    weights = [f['model_weights'][i][j] for j in weight_names]
    index = layer_names.index(i)
    model.layers[index].set_weights(weights)

# Freeze the first 5 layers and train only the last 2 fully connected layers
for layer in model.layers[0:5]:
    layer.trainable = False

if os.path.exists('classification.h5'):
    model = tf.keras.models.load_model('classification.h5')
else:
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_images, train_labels, batch_size=2048, epochs=100,
              validation_data=(val_images, val_labels))
    model.save('classification.h5')


predicted_label = model.predict(test_images) # return 2d array of number

# If the output probability is greater than 0.5, it means helmet is present in the image
# so label it as 1 and if output probability is less than 0.5, it means helmet is not present
# in the image so label it as 0.
predicted_label[predicted_label > 0.5] = 1
predicted_label[predicted_label < 0.5] = 0
predicted_label = np.squeeze(predicted_label)

right_guess = sum(predicted_label == test_labels)

accuracy = (right_guess / len(test_labels)) * 100

print('The accuracy on test data is {}'.format(accuracy))
