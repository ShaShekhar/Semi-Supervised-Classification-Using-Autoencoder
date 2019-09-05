import xml.etree.ElementTree as et
import numpy as np
import cv2
import os
import re
import glob

def extract_patch(data_dir):
    m = re.search(r'.*/(.*)', data_dir)
    data_dir = m.group(1) # './Extracted-patches/Train-data' --> Train-data same for test and validation
    image_path = './Splited-data/{}/*.jpg'.format(data_dir)

    images = glob.glob(image_path)
    #print(len(images))
    for each_image in images:
        # Use Regex standard libray to extract image_id_number.
        # './Splited-data/Train-data/03125.jpg' --> 03125
        m = re.search(r'.*/(.*)\.\w+', each_image)
        image_number = m.group(1)
        xml_file = './Data/Annotations/{}.xml'.format(image_number)
        # take the xml file and save it into memory for us to start working with
        tree = et.parse(xml_file)
        root = tree.getroot()
        coordinates = []
        for child in root:
            for element in child:
                for b_box in element:
                    coordinates.append(int(b_box.text))
        no_of_coordinate = len(coordinates)
        heads = [coordinates[h:h+4] for h in range(0, no_of_coordinate, 4)]
        img = cv2.imread(each_image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # shape becomes 2-dimensional (height, width)
        #for each_head in heads:
            #img = cv2.rectangle(img, (each_head[0], each_head[1]), (each_head[2], each_head[3]), 255, 1)
        #cv2.namedWindow('image')
        #cv2.imshow('image', img)

        for patch_num, each_head in enumerate(heads):
            x_min = each_head[0]
            x_max = each_head[2]
            y_min = each_head[1]
            y_max = each_head[3]

            get_patch = img[y_min : y_max, x_min : x_max]
            #print(get_patch)
            resize_patch = cv2.resize(get_patch, (28, 28), interpolation=cv2.INTER_AREA)
            patch_name = './Extracted-patches/{}/{}_{}.jpg'.format(data_dir, image_number, patch_num)
            cv2.imwrite(patch_name, resize_patch)
            print('Image saved: {}'.format(patch_name))
            #cv2.namedWindow('resize_image')
            #cv2.imshow('resize_image', resize_patch)
            #cv2.waitKey(0)
            #cv2.destroyWindow('resize_image')
        #cv2.destroyAllWindows()

def if_not_exists_then_make(data_dir):
    if os.path.exists(data_dir):
        print('Data directory exists!')
    else:
        os.mkdir(data_dir)      # Make that data directory
        extract_patch(data_dir) # Extract patches into that data directory

if os.path.exists('./Extracted-patches'):
    if_not_exists_then_make('./Extracted-patches/Train-data')
    if_not_exists_then_make('./Extracted-patches/Test-data')
    if_not_exists_then_make('./Extracted-patches/Val-data')
else:
    os.mkdir('./Extracted-patches')
    if_not_exists_then_make('./Extracted-patches/Train-data')
    if_not_exists_then_make('./Extracted-patches/Test-data')
    if_not_exists_then_make('./Extracted-patches/Val-data')
