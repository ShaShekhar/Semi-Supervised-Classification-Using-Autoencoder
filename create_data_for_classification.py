from imageio import imread
import numpy as np
import random
import csv
import os
import sys
import pickle

if os.path.exists('classification_data.pkl'):
    print('File exists!')
    sys.exit()

print('Dumping...')
image_label_pair = []
with open('train_labels.csv', 'r') as csv_reader:
    csv_reader = csv.DictReader(csv_reader)
    for line in csv_reader:
        image_label_pair.append((line['image_id'], line['label']))
random.shuffle(image_label_pair)

batch_size = len(image_label_pair)
data = np.zeros((batch_size, 28, 28, 1), dtype=np.float32)
labels = np.zeros(batch_size, dtype=np.int64)

for i in range(batch_size):
    image, label = image_label_pair[i]
    full_path = os.path.join('./Extracted-patches/Train-data', image)
    read_image = imread(full_path)
    img = np.array(read_image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, 2)
    data[i, :, :, :] = img
    labels[i] = int(label)

classification_data = (data, labels)
f_name = 'classification_data.pkl'
pickle.dump(classification_data, open(f_name, 'wb'))
print('Dumped {}'.format(f_name))
