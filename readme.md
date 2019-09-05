# Semi Supervised Classification using AutoEncoders

# Introduction

By definition, machine learning can be defined as a complex process of learning the best possible and most relevant patterns, relationships, or associations from a dataset which can be used to predict the outcomes on unseen data. Broadly, their exists three different machine learning processes:

**1. Supervised Learning** is a process of training a machine learning model on a labelled dataset ie. a dataset in which the target variable is known. In this technique, the model aims to find the relationships among the independent and dependent variable. Examples of supervised learning are classification, regression and forecasting.

**2. Unsupervised Learning** is a process of training a machine learning model on a dataset in which target variable is not known. In this technique, the model aims to find the most relevant patterns in the data or the segments of data. Examples of unsupervised learning are clustering, segmentations, dimensionality reduction etc.

**3. Semi-Supervised Learning** is combination of supervised and unsupervised learning processes in which the unlabelled data is used for training a model as well. In this approach, the properties of unspervised learning are used to learn the best possible representation of data and the properties of supervised learning are used to learn the relationships in the representations which are then used to make predictions.

# Problem Statement
 Given the image of size 28x28 classify that in that image helmet is present or not.
 
 ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00002_0.jpg "Helmet Present")    ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/01390_2.jpg "Helmet not Present")    ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00022_0.jpg "Helmet Present")     ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/02680_0.jpg "Helmet not Present")   ![](https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder/blob/master/fig/00167_0.jpg "Helmet Present")

# Dataset
Download the dataset from this [link](https://drive.google.com/open?id=1SUBraBUovros2qTt20LYPkRlgmsElVxg "Dataset"). It contains two folder **JPEGImages** and **Annotation** folder.The JPEGImages folder contains the actual images and Annotation folder contains .xml file which contains the Bounding Box for those images.

# Steps involved
 1. Clone this repository then place the downloaded Dataset with the name **Data** folder into the cloned repository.

    `git clone https://github.com/ShaShekhar/Semi-Supervised-Classification-Using-Autoencoder.git`

 2. Now Split the images from **Data/JPEGImages** folder into _Test_, _Val_ and _Train_ Datasets using **split_data.py** file.
    run **python split_data.py**. It will create 3 folder named **Train-data, Val-data, Test-data inside the Splited-data folder**.

 3. By using BoundingBox coordinates given in .xml file, which are located inside the **Data/Annotations** folder extract the small patches of .jpg images from the images present in **Splited-data** folder by using **extract_patches.py** file. run **python extract_patches.py**. It will create 3 folder named **Train-data, Val-data, Test-data inside the Extracted-patches folder**.

 4. For training of autoencoder the _Train-data_ folder inside _Extracted-patches_ contains nearly 14000 patches, which is very less as compared to number of images required to train **Autoencoder**. Here i'm using data augmentation technique to increase the patches. run **python data_augment.py**. It will create images inside *Augmented-data* folder and this folder is inside the **Extracted-patches** folder.

 5. copy the **Train-data** of **Extracted-patches** folder to **Augmented-data** by using **copy_train_to_aug.py**. Here I am also using images inside _Val-data_ as validation data and _Test-data_
 as test data of Extracted-patches folder for **autoencoder** training.

 6. Now i'm going to convert the Augmented-data, Val-data, Test-data of **Extracted-patches** folder into pickled file for efficient loading for training of autoencoder. run **python convert_to_pickle.py**.It will create **train_data.pkl, test_data.pkl and val_data.pkl inside the Pickled-data** directory.

 7. run **python autoencoder.py** and train with different batch_size.
    It will generate **autoencoder.h5** file, keep this file for initializing the weight of classification layer.

 8. It's time to train the classification layer, but to train it we need label for each sample. I have implemented data labelling file for this purpose. run **python label_data.py** and generate csv file. when you run this file it will show you image and its patch, press 'q' from keyboard and then terminal will get activate and enter a value of 1 or 0 based on patch, image which contain helmet label as 1 and image which doesn't contain helmet label as 0 and keep labelling for nearly 3000 images for training data and also label some images for validation and testing of classifier.

 9. I only labelled 3000 patches for training classifier. This csv file contain 'image_id' and 'label'. Load that csv file and create the pickled data for classification. I'm using csv file for pickle train_data because it contains image_ids whose labels are known.
run **python create_data_for_classification.py**. It will genrate **classification_data.pkl** file.

 10. I've also labelled some test_data ( nearly 200) to check accuracy of classifier. run **python classification.py**
The accuracy on test data is _89.81%_.
