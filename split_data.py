import numpy as np
import glob
import shutil
import os
import re
import random

path = './Data/JPEGImages/*.jpg'
all_images = glob.glob(path)
random.shuffle(all_images)
#print(len(all_images))
test_images = all_images[0 : 500]
val_images = all_images[500 : 774]
train_images = all_images[774:]

def move(images, path):
    for each_image in images:
        # Use Regex standard libray to extract image_id_number.
        # './Data/JPEGImages/03125.jpg' --> 03125
        m = re.search(r'.*/(.*)\.\w+', each_image)
        image_number = m.group(1)
        destination_path = '{}/{}.jpg'.format(path,image_number)
        print(destination_path)
        os.rename(each_image, destination_path)
        # If you want your original data should not tempered then you should use the copy command.
        #shutil.copyfile(each_image, destination_path)

def if_not_exist_then_make(images, dir_path):
    if os.path.exists(dir_path):
        print('Directory Already exists!')
    else:
        os.mkdir(dir_path)     # Create dir_path directory
        move(images, dir_path) # move the images into dir_path directory

if os.path.exists('./Splited-data'):
    if_not_exist_then_make(test_images, './Splited-data/Test-data')
    if_not_exist_then_make(val_images, './Splited-data/Val-data')
    if_not_exist_then_make(train_images, './Splited-data/Train-data')
else:
    os.mkdir('Splited-data')
    if_not_exist_then_make(test_images, './Splited-data/Test-data')
    if_not_exist_then_make(val_images, './Splited-data/Val-data')
    if_not_exist_then_make(train_images, './Splited-data/Train-data')
