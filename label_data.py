import xml.etree.ElementTree as et
import numpy as np
import csv
import cv2
import os
import glob
import re

def create_labels(image_path, label_path):
    with open(label_path, 'w') as labels_file:
        fieldnames = ['image_id', 'label']
        csv_writer = csv.DictWriter(labels_file, fieldnames=fieldnames, delimiter=',')
        csv_writer.writeheader()

        data_path = image_path

        images = glob.glob(data_path)
        patch_counter = 0 # for counting the number of images we labelled

        #print(len(images))
        for each_image in images:
            # Use Regex standard libray to extract image_id_number.
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
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for each_head in heads:
                img = cv2.rectangle(img, (each_head[0], each_head[1]), (each_head[2], each_head[3]), (0, 0, 255), 1)

            match_object = re.search(r'.*/(.*)', each_image)
            image_name = match_object.group(1)
            cv2.namedWindow(image_name)
            cv2.imshow(image_name, img)

            for patch_num, each_head in enumerate(heads):
                x_min = each_head[0]
                x_max = each_head[2]
                y_min = each_head[1]
                y_max = each_head[3]

                get_patch = img[y_min : y_max, x_min : x_max]
                #print(get_patch)
                resize_patch = cv2.resize(get_patch, (28, 28), interpolation=cv2.INTER_AREA)
                patch_name = '{}_{}.jpg'.format(image_number, patch_num)

                cv2.namedWindow(patch_name)
                cv2.imshow(patch_name, resize_patch)
                cv2.waitKey(0)

                label = input('If helmet present, enter 1 else 0 : ')
                row = {'image_id' : patch_name, 'label' : label}
                csv_writer.writerow(row)
                patch_counter += 1

                cv2.destroyWindow(patch_name)
            print('------------------------------------------')
            print('{} images are labelled!'.format(patch_counter))
            print('For break out of loop press Ctrl-C')
            print('------------------------------------------')
            cv2.destroyAllWindows()

def if_not_exist_then_make(image_path, file_path):
    if os.path.exists(file_path):
        print('File already exists.')
    else:
        create_labels(image_path, file_path)

if_not_exist_then_make('./Splited-data/Train-data/*.jpg', 'train_labels.csv')
if_not_exist_then_make('./Splited-data/Val-data/*.jpg',   'val_labels.csv')
if_not_exist_then_make('./Splited-data/Test-data/*.jpg',  'test_labels.csv')
