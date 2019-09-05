from imageio import imread
import numpy as np
import tensorflow as tf
import pickle
import h5py
import csv
import os

with open('classification_data.pkl', 'rb') as f:
    classification_data = pickle.load(f)
    train_data = classification_data[0]
    train_label = classification_data[1]

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
    # comment below lines after training is done
    #model.compile(loss='binary_crossentropy', optimizer='adam')
    #model.fit(train_data, train_label, batch_size=2096, epochs=1000)
    #model.save('classification.h5')
else:
    model.compile(loss='binary_crossentropy', optimizer='adam')
    model.fit(train_data, train_label, batch_size=2048, epochs=100)
    model.save('classification.h5')

test_image_label = []
with open('test_labels.csv', 'r') as csv_reader:
    csv_reader = csv.DictReader(csv_reader)
    for line in csv_reader:
        test_image_label.append((line['image_id'], line['label']))

test_batch = len(test_image_label)
test_data = np.zeros((test_batch, 28, 28, 1), dtype=np.float32)
label = np.array([x[1] for x in test_image_label], dtype=np.float32)

for i in range(test_batch):
    path = './Extracted-patches/Test-data/{}'.format(test_image_label[i][0])
    test_sample = imread(path)
    test_sample = test_sample.astype(np.float32) / 255.0
    test_sample = np.expand_dims(test_sample, 2)
    test_data[i, :, :, :] = test_sample
predicted_label = model.predict(test_data) # return 2d array of number

predicted_label[predicted_label > 0.5] = 1
predicted_label[predicted_label < 0.5] = 0
predicted_label = np.squeeze(predicted_label)

right_guess = sum(predicted_label == label)

accuracy = (right_guess/test_batch) * 100
print('The accuracy on test data is {}'.format(accuracy))

