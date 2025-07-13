# -*- coding: utf-8 -*-


import matplotlib.pyplot as plt

import PIL

import tensorflow as tf
import csv
import pickle
import cv2
import glob, os
from metrics import f1_m, matthews_correlation
import dataset_handler as dh
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import matthews_corrcoef





def create_model():
    images,labels=dh.getting_training_dataset()
    
        
    train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.10, random_state=42)
    
    train_images = train_images.reshape(len(train_images), dh.IMG_WIDTH, dh.IMG_HEIGHT, 1)
    test_images = test_images.reshape(len(test_images), dh.IMG_WIDTH, dh.IMG_HEIGHT, 1)    
    
    cnn = tf.keras.models.Sequential()
 
    """### Convolution"""
    
    cnn.add(tf.keras.layers.Conv2D(filters=16, kernel_size=3, activation='relu', input_shape=[dh.IMG_WIDTH,dh.IMG_HEIGHT,1  ],padding='same'))
    
    
    """### Pooling"""
    
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    
    """### Adding other convolutional layers"""
    
    cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
    
    
    cnn.add(tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu'))

    cnn.add(tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu'))

    cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))

    
    """### Flattening"""
    
    cnn.add(tf.keras.layers.Flatten())
    
    
    """### Full Connection"""
    
    cnn.add(tf.keras.layers.Dense(units=100, activation='relu'))
    
    
    """### Output Layer"""
    
    cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    
    
    
    """### Training of the CNN"""
    
    cnn.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                  metrics=[matthews_correlation])
    
    history = cnn.fit(train_images, train_labels, epochs=4, 
                        validation_data=(test_images, test_labels))
    
    
    return cnn,history



def plotting_history(history):
    
    plt.plot(history.history['matthews_correlation'], label='matthews_correlation')
    plt.plot(history.history['loss'], label = 'loss')
    plt.xlabel('Epoch')
    plt.ylabel('F1_score and loss')
    plt.ylim([0, 1])
    plt.legend(loc='lower right')

def round_to(n, precision):
   correction = 0.5 if n >= 0 else -0.5
   return int( n/precision+correction ) * precision

def evaluation_model(cnn):
        
    
    X=[0 for k in range(101)]
    """### Evaluation of the model"""
    images,labels,list_path=dh.getting_test_dataset()
 
       

    test_loss, test_acc = cnn.evaluate(images,labels, verbose=2)
    print(test_loss,test_acc)
    
    ynew=cnn.predict(images)
    seuil=0.98
    X=cnn.predict(images)
    
    for k in range (len(ynew)):
        
        y_round=round_to(ynew[k]*100, 1)
        
        if ynew[k]<seuil:
            ynew[k]=0
        else:
            ynew[k]=1
    res = tf.math.confusion_matrix(labels,ynew[:,0])
    print(res)
    
    fp=int(res[0][1])
    fn=int(res[1][0])
    vp=int(res[1][1])
    vn=int(res[0][0])
    
    print("C Value: "+str((fp*100+fn)/(fp+fn+vp+vn)))
    
    return (list_path,labels,ynew[:,0],X)
    
    
    
    


import numpy as np
from sklearn.metrics import confusion_matrix

def valeo_custom_metric(dataframe_y_true, dataframe_y_pred):
    """
        Example of custom metric function.
        NOTA: the order (dataframe_y_true, dataframe_y_pred) matters if the metric is
        non symmetric.

    Args
        dataframe_y_true: Pandas Dataframe
            Dataframe containing the true values of y.
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

        dataframe_y_pred: Pandas Dataframe
            This dataframe was obtained by reading a csv file with following instruction:
            dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

    Returns
        score: Float
            The metric evaluated with the two dataframes. This must not be NaN.
    """
   
    missing_image_name = 'AE00072_145326_00_1_2_2001.jpg'

    if missing_image_name in dataframe_y_true.index:
        dataframe_y_pred.loc[missing_image_name]['labels'] = dataframe_y_true.loc[missing_image_name]['labels']

    tn, fp, fn, tp = confusion_matrix(dataframe_y_true, dataframe_y_pred).ravel()
    #print( confusion_matrix(dataframe_y_true, dataframe_y_pred))
    lambda_ = 100
    score = 1 / len(dataframe_y_true) * (fn + lambda_ * fp)
    return score


a=0
for k in range (2):
    cnn,history=create_model()
    #plotting_history(history)
    list_path,y_true,y_pred,X=evaluation_model(cnn)
    a=a+valeo_custom_metric(pd.DataFrame(y_true),pd.DataFrame(y_pred.astype(int)))
    
print(a/5)


# for k in range(100):
#     y_new=np.array(y_pred)
#     seuil=0.01*k
#     for k in range (len(y_pred)):
#         #On prend seuil plus petit que 0.5 car on souhaite limiter les faux positifs
#         if y_pred[k]<seuil:
#             y_new[k]=0
#         else:
#             y_new[k]=1
#     print(valeo_custom_metric(pd.DataFrame(y_true),pd.DataFrame(y_new.astype(int))))


with open('submission2.csv', 'w',newline='') as csvfile: 
    csvwriter = csv.writer(csvfile) 
    csvwriter.writerow(['images','labels'])
    for k in range(len(y_true)):
        csvwriter.writerow([list_path[k],int(y_pred[k])])



Y=[]
for k in range (200):
    seuil=k/200
    ynew2=[0 for k in range(len(X))]
    for i in range (len(X)):
        if float(X[i])<seuil:
            ynew2[i]=0
        else:
            ynew2[i]=1
    print(k)
    Y.append(valeo_custom_metric(pd.DataFrame(y_true),pd.DataFrame(ynew2)))



