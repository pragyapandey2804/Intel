# -*- coding: utf-8 -*-

import PIL
import csv
import pickle
import cv2
import glob, os
import numpy as np



h=2078
w=594
N_CLASSES = 2 # total number of classes
IMG_HEIGHT = 252 # the image height to be resized to
IMG_WIDTH = 28 # the image width to be resized to
CHANNELS = 1 # 1 color channel for grayscale  
batch_size = 40
    


#Return the value associated to an image in a csv file
def get_label_csv(path,name):
    csv_file = csv.reader(open(name, "r"), delimiter=",")

    for row in csv_file:
        
        #if current rows 2nd value is equal to input, print that row
        if path == row[0]:
            return(row[1])
    return -1


def get_amount_of_1(name):
    csv_file = csv.reader(open(name, "r"), delimiter=",")
    n=0
    un=0
    for row in csv_file:
        
        #if current rows 2nd value is equal to input, print that row
        if row[1] == '1':
            un=un+1
        n=n+1
    print(n)
    print(un)
    
label=get_amount_of_1('y_train.csv')



def create_training_dataset():
  
    labels = [0, 1] # Two possible labels (binary classifications)
    images=[]
    labels=[]
    
    #If the dataset has not been saved yet, we create it and save it with pickle
           
    os.chdir(os.path.abspath("Training_images"))

    
    for img in glob.glob("*.jpg"):
        try:
            img_arr = np.asarray(PIL.Image.open(img))
            img_arr = cv2.resize(img_arr, (IMG_HEIGHT, IMG_WIDTH)) # Reshaping images to preferred size
            label=get_label_csv(img,'../y_train.csv')
            if(label=='0' or label=='1'):
                img_arr=np.array(img_arr)
                images.append(img_arr)
                labels.append(int(label))
                
        except Exception as e:
            print(e)
    
    
    images=np.array(images)
    labels=np.array(labels)
    images = images.reshape(len(images), IMG_WIDTH, IMG_HEIGHT, 1)/255.0

  
    os.chdir(os.path.abspath(".."))
    dataset=[images,labels]
    
    pickle.dump(dataset, open('training_dataset.sav', 'wb'))
    
    return dataset

def create_test_dataset():
    images=[]
    labels=[]
    list_path=[]
    os.chdir(os.path.abspath("Test_images"))
    for img in glob.glob("*.jpg"):
        try:
            img_arr = np.asarray(PIL.Image.open(img))
            img_arr = cv2.resize(img_arr, (IMG_HEIGHT, IMG_WIDTH)) # Reshaping images to preferred size
            label=get_label_csv(img,"../Y_Benchmark.csv")
            if(label=='0' or label=='1'):
                img_arr=np.array(img_arr)
                images.append(img_arr)
                labels.append(int(label))
                list_path.append(img)

        except Exception as e:
            print(e)


    images=np.array(images)
    labels=np.array(labels)
    images = images.reshape(len(images), IMG_WIDTH, IMG_HEIGHT, 1)/255.0
    
    os.chdir(os.path.abspath(".."))

    dataset=[images,labels,list_path]
    pickle.dump(dataset, open('test_dataset.sav', 'wb'))
    
    return dataset
   

def getting_training_dataset():
    if (not(os.path.isfile("training_dataset.sav")) ):
        dataset=create_training_dataset()
    else:
        dataset = pickle.load(open('training_dataset.sav', 'rb'))
    images=dataset[0]
    labels=dataset[1]
    return images,labels


def getting_test_dataset():
    if (not(os.path.isfile("test_dataset.sav")) ):
        dataset=create_test_dataset()
    else:
        dataset = pickle.load(open('test_dataset.sav', 'rb'))
    images=dataset[0]
    labels=dataset[1]
    list_path=dataset[2]
    return images,labels,list_path