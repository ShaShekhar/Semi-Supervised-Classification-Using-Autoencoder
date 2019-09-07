# Semi Supervised Classification using AutoEncoders

# Introduction

By definition, machine learning can be defined as a complex process of learning the best possible and most relevant patterns, relationships, or associations from a dataset which can be used to predict the outcomes on unseen data. Broadly, their exists three different machine learning processes:

**1. Supervised Learning** is a process of training a machine learning model on a labelled dataset ie. a dataset in which the target variable is known. In this technique, the model aims to find the relationships among the independent and dependent variable. Examples of supervised learning are classification, regression and forecasting.

**2. Unsupervised Learning** is a process of training a machine learning model on a dataset in which target variable is not known. In this technique, the model aims to find the most relevant patterns in the data or the segments of data. Examples of unsupervised learning are clustering, segmentations, dimensionality reduction etc.

**3. Semi-Supervised Learning** is combination of supervised and unsupervised learning processes in which the unlabelled data is used for training a model as well. In this approach, the properties of unspervised learning are used to learn the best possible representation of data and the properties of supervised learning are used to learn the relationships in the representations which are then used to make predictions.

# Problem Statement
 Given the GrayScale image of size 28x28 classify that, in that image helmet is present or not.

 For representation purpose the images shown below are not of size 28x28.

 Hover the mouse over the image to get the classification label.

 ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00002_0.jpg "Helmet Present")       ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/01390_2.jpg "Helmet not Present")      ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00022_0.jpg "Helmet Present")        ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/02680_0.jpg "Helmet not Present")      ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00167_0.jpg "Helmet Present")      ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00080_0.jpg "Helmet Present")

# Dataset
Download the dataset from this [link](https://drive.google.com/open?id=1SUBraBUovros2qTt20LYPkRlgmsElVxg "Dataset"). It contains two folder **JPEGImages** and **Annotation** folder.The JPEGImages folder contains the actual images and Annotation folder contains .xml file which contains the Bounding Box for those images.

# Steps involved
 1. Clone this repository then place the downloaded Dataset with the name **Data** folder into the cloned repository.

    `git clone https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder.git`

 2. Now Split the images from **Data/JPEGImages** folder into _Test_, _Val_ and _Train_ Datasets using **split_data.py** file.

    run in terminal
    >python split_data.py

  It will create 3 folder named **Train-data, Val-data, Test-data inside the Splited-data folder**.

 3. By using BoundingBox coordinates given in .xml file, which are located inside the **Data/Annotations** folder extract the small patches of .jpg images from the images present in **Splited-data**.

 >python extract_patches.py

  It will create 3 folder named **Train-data, Val-data, Test-data inside the Extracted-patches folder**.

    ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00063.jpg "Image")   ------->  ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00063_0.jpg  "Extracted Patch")

 4. For training of autoencoder the _Train-data_ folder inside _Extracted-patches_ contains nearly 14000 patches, which is very less as compared to number of images required to train **Autoencoder**. Here i'm using data augmentation technique to increase the patches.

  >python data_augment.py

  It will create images inside **Augmented-data** folder and this folder is located inside the **Extracted-patches** folder.

 5. copy the **Train-data** of **Extracted-patches** folder to **Augmented-data** by using

 >python copy_train_to_aug.py

 6. Now i'm going to convert the Augmented-data, Val-data, Test-data of **Extracted-patches** folder into pickled file for efficient loading for training of autoencoder.

 >python convert_to_pickle.py

 It will create **train_data.pkl, test_data.pkl and val_data.pkl inside the Pickled-data** directory.

 7. run
 >python autoencoder.py

    and train with different batch_size.
    It will generate **autoencoder.h5** file, keep this file for initializing the weight of classification layer.

 8. It's time to train the classification layer, but to train it we need label for each sample. The .xml file which is located inside the **Data/Annotations** folder contain label for each patch. For each .xml file if the helmet present then tag name is 'name' and data is 'color of helmet' e.g., white, blue etc. **.** I've extract the labels using

 >python extract_labels.py

 It will create **train-data.csv, val-data.csv, test-data.csv**, which contain image_id and its corresponding label.

 9. For efficient loading for training classifier i've pickled the data.

 >python pickle_data_for_classification.py

 It will genrate **train-data.pkl, val-data.pkl, test-data.pkl** file.

 10. Now It's time to train the classifier

  >python classification.py

  It will create **classification.h5** which if used for initializing the model and testing on new dataset.
  The accuracy i get on test data is _93.5%_.
