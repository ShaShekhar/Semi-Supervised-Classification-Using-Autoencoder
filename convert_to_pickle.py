from imageio import imread, imwrite
from tqdm import tqdm #Instantly make your loops show a smart progress meter 
import numpy as np
import pickle
import random
import os
import glob

def create_pickled_object(data_name, image_path):
    images = glob.glob(image_path)
    random.shuffle(images)
    batch_size = len(images)
    #print(batch_size) # for train size is 70210
    image_shape = imread(images[0]).shape # (28, 28)
    data = np.zeros((batch_size, *image_shape, 1), dtype=np.uint8)

    print('Dumping...')
    for i in tqdm(range(batch_size)):
        load_img = imread(images[i])
        # i'm resizing the image so that i don't have to do during the loading of data for training
        resized_img = np.expand_dims(load_img, 2)
        data[i, :, :, :] = resized_img
    f_name = './Pickled-data/{}.pkl'.format(data_name)
    pickle.dump(data, open(f_name, 'wb'))
    print('Dumped {}'.format(f_name))



def if_not_exist_then_make(data_name, file_path):
    if os.path.exists(file_path):
        print('labels file already exists.')
    else:
        create_pickled_object(data_name, file_path)

# Check if directory in which file is created is exists or not, if not then create the directory
if os.path.exists('./Pickled-data'):
    # check the individual file is exist or not, if not then create the file
    if_not_exist_then_make('train_data', './Extracted-patches/Augmented-data/*.jpg')
    if_not_exist_then_make('test_data', './Extracted-patches/Test-data/*.jpg')
    if_not_exist_then_make('val_data', './Extracted-patches/val-data/*.jpg')

else:
    os.mkdir('Pickled-data')
    if_not_exist_then_make('train_data', './Extracted-patches/Augmented-data/*.jpg')
    if_not_exist_then_make('test_data', './Extracted-patches/Test-data/*.jpg')
    if_not_exist_then_make('val_data', './Extracted-patches/Val-data/*.jpg')
