import numpy as np
import matplotlib.pyplot as plt
from imageio import imread, imwrite
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import re
import glob
import sys

if os.path.exists('./Extracted-patches/Augmented-data'):
    print('Augmented-data directory exists!')
    sys.exit()
else:
    os.mkdir('./Extracted-patches/Augmented-data')

gen = ImageDataGenerator(rotation_range=10, brightness_range=[0.5,1.1],
                         shear_range=0.2, horizontal_flip=True)

images = './Extracted-patches/Train-data/*.jpg'
images_name = glob.glob(images)

for image in images_name:
    # Use Regex standard libray to extract image_id_number.
     # './Extracted-patches/Augmented-data/03125.jpg' --> 03125
    m = re.search(r'.*/(.*)\.\w+', image)
    image_number = m.group(1)
    # input to ImageDataGenerator should be 4-dimensional
    img = np.expand_dims(imread(image), 2)
    img = np.expand_dims(img, 0)
    aug_iter = gen.flow(img)
    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(4)]
    #plt.subplot(1, 5, 1)
    #plt.imshow(np.squeeze(img), cmap='gray')
    for i in range(4):
        name = './Extracted-patches/Augmented-data/{}_{}.jpg'.format(image_number,i)
        print('Created {}'.format(name))
        #plt.subplot(1,5,i+2)
        #plt.imshow(np.squeeze(aug_images[i]), cmap='gray')
        imwrite(name, np.squeeze(aug_images[i]))
    #plt.show()
