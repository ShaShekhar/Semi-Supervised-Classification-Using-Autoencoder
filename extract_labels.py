import xml.etree.ElementTree as et
from tqdm import tqdm
import csv
import os
import re
import glob

def extract_label(data_dir):
    m = re.search(r'.*/(.*)', data_dir)
    data_dir = m.group(1) # './Extracted-patches/Train-data' --> Train-data same for test and validation
    image_path = './Splited-data/{}/*.jpg'.format(data_dir)
    images = glob.glob(image_path)

    csv_path = data_dir.lower() + '.csv'
    csv_file = open(csv_path, 'w')
    fieldnames = ['image_id', 'label']
    csv_writer = csv.DictWriter(csv_file, fieldnames)
    csv_writer.writeheader()

    for each_image in tqdm(images, ascii=True):
        # Use Regex standard libray to extract image_id_number.
        # './Splited-data/Train-data/03125.jpg' --> 03125
        m = re.search(r'.*/(.*)\.\w+', each_image)
        image_number = m.group(1)
        xml_file = './Data/Annotations/{}.xml'.format(image_number)
        # take the xml file and save it into memory for us to start working with
        tree = et.parse(xml_file)
        root = tree.getroot()

        coordinates = []

        labels = []
        for child in root:
            for element in child:
                if (element.tag == 'name'):
                    if (element.text == 'none'):
                        labels.append(0)
                    else:
                        labels.append(1)

        for patch_num, label in enumerate(labels):

            patch_name = '{}_{}.jpg'.format(image_number, patch_num)

            csv_dict = {'image_id' : patch_name, 'label' : label}
            csv_writer.writerow(csv_dict)

    csv_file.close()

def if_not_exists_then_make(data_dir):
    m = re.search(r'.*/(.*)', data_dir)
    file_name = m.group(1)
    file_name = file_name.lower() + '.csv'
    if os.path.exists(file_name):
        print('File already exists!')
    else:
        extract_label(data_dir)

if_not_exists_then_make('./Extracted-patches/Train-data')
if_not_exists_then_make('./Extracted-patches/Test-data')
if_not_exists_then_make('./Extracted-patches/Val-data')
