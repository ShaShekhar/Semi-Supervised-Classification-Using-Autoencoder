from imageio import imread, imwrite
from tqdm import tqdm #Instantly make your loops show a smart progress meter
import numpy as np
import pickle
import random
import csv
import os
import glob
import re

def create_pickled_object(image_path):
    m = re.search(r'.*/(.*)/.*', image_path)
    file_name = m.group(1)
    csv_file = file_name.lower() + '.csv'

    image_label_pair = []
    with open(csv_file, 'r') as csv_reader:
        csv_reader = csv.DictReader(csv_reader)
        for line in csv_reader:
            image_label_pair.append((line['image_id'], line['label']))
    random.shuffle(image_label_pair)

    batch_size = len(image_label_pair)
    data = np.zeros((batch_size, 28, 28, 1), dtype=np.float32)
    labels = np.zeros(batch_size, dtype=np.int64)

    match = re.search(r'(.*)/.*', image_path)
    dir_name = match.group(1)
    print('Dumping...')

    for i in tqdm(range(batch_size), ascii=True):
        image, label = image_label_pair[i]
        full_path = os.path.join(dir_name, image)
        read_image = imread(full_path)
        img = np.array(read_image, np.float32) / 255.0
        img = np.expand_dims(img, 2)
        data[i, :, :, :] = img
        labels[i] = int(label)

    classification_data = (data, labels)
    pkl_file = file_name.lower() + '.pkl'
    open_file = open(pkl_file, 'wb')
    pickle.dump(classification_data, open_file)
    open_file.close()
    print('Dumped {}'.format(pkl_file))


def if_not_exist_then_make(file_name, image_path):
    if os.path.exists(file_name):
        print('pickled file already exists.')
    else:
        create_pickled_object(image_path)

if_not_exist_then_make('train-data.pkl', './Extracted-patches/Train-data/*.jpg')
if_not_exist_then_make('test-data.pkl', './Extracted-patches/Test-data/*.jpg')
if_not_exist_then_make('val-data.pkl', './Extracted-patches/Val-data/*.jpg')
