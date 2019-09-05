import shutil
import glob
import re

images = glob.glob('./Extracted-patches/Train-data/*.jpg')

for each_image in images:
    # './Extracted-patches/Train-data/03125.jpg' --> 03125
    m = re.search(r'.*/(.*)\.\w+', each_image)
    num = m.group(1)
    copy_to = './Extracted-patches/Augmented-data/{}.jpg'.format(num)
    shutil.copyfile(each_image, copy_to)
    print('Copied : {}'.format(copy_to))
