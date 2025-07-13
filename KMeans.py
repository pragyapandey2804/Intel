# -*- coding: utf-8 -*-

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os, glob
import cv2
import PIL
import csv
import dataset_handler as dh


from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

"""### Training"""

images,labels=dh.getting_training_dataset()
wcss=[]
for i in range(2,15):
    
    
    X_train=images
    X_train = X_train.reshape(len(X_train),-1)
    
    nb=i
    kmeans = KMeans(n_clusters = nb, n_init=2)
    
    kmeans.fit(X_train)
    
    
    """### Evaluation of the model"""
    
    images,labels,list_path=dh.getting_test_dataset()
    X_test=np.array(images)
    y_test=np.array(labels)
    X_test = X_test.reshape(len(X_test), -1)
    
    Z = kmeans.predict(X_test)
    
    bottom = 0.35
    
    per1=[] #number of 1 for each cluster
    per0=[] #number of 0 for each cluster
    
    
    # Count of the amount of 1 and 0 for each cluster
    for k in range(nb):
        row=np.where(Z==k)[0]
        num = row.shape[0]
        if(num==0):
            per0.append(0)
            per1.append(0)
            print("akbcejobcaeobcuac")
            
        else:
            nb_0=0
            nb_1=0
            for i in range(0, num):
                if(y_test[row[i], ] == 0):
                    nb_0=nb_0+1
                if(y_test[row[i], ] == 1):
    
                    nb_1=nb_1+1
            per0.append(nb_0)
            per1.append(nb_1)
            
    vp=0
    vn=0
    fp=0
    fn=0
    nbf0,nbf1=0,0
    F=[]
    P=[]
    
    
    #If a cluster has more positives than negatives, we count it as completely
    #positive, otherwise we count it as completely negativee
    #We can then count the amount of false positives and false negatives
    for k in range(nb):
        if(per0[k]==0 and per1[k]==0):
            print(".")
        elif(per0[k]<per1[k]):
            F.append(k)
            vn=vn+per1[k]
            fn=fn+per0[k]
        else:
            P.append(k)
            vp=vp+per0[k]
            fp=fp+per1[k]
        
    for k in range(len(Z)):
        if Z[k] in P:
            Z[k]=0
        else:
            Z[k]=1
    M=[[vp,fn],[fp,vn]]
    
    print("Confusion Matrix: ")
    print(M)
    print(k)
    print("C Value: "+str((fn*100+fp)/(fp+fn+vp+vn)))
    wcss.append(silhouette_score(X_train, kmeans.labels_, metric='euclidean'))


plt.plot(range(2, 15), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()    
    



# import numpy as np
# from sklearn.metrics import confusion_matrix
# import pandas as pd

# def valeo_custom_metric(dataframe_y_true, dataframe_y_pred):
#     """
#         Example of custom metric function.
#         NOTA: the order (dataframe_y_true, dataframe_y_pred) matters if the metric is
#         non symmetric.

#     Args
#         dataframe_y_true: Pandas Dataframe
#             Dataframe containing the true values of y.
#             This dataframe was obtained by reading a csv file with following instruction:
#             dataframe_y_true = pd.read_csv(CSV_1_FILE_PATH, index_col=0, sep=',')

#         dataframe_y_pred: Pandas Dataframe
#             This dataframe was obtained by reading a csv file with following instruction:
#             dataframe_y_pred = pd.read_csv(CSV_2_FILE_PATH, index_col=0, sep=',')

#     Returns
#         score: Float
#             The metric evaluated with the two dataframes. This must not be NaN.
#     """
#     # the image 'AE00072_145326_00_1_2_2001.jpg' is missing in test input so
#     # the participant prediction for this image is set to the correct prediction
#     missing_image_name = 'AE00072_145326_00_1_2_2001.jpg'

#     if missing_image_name in dataframe_y_true.index:
#         dataframe_y_pred.loc[missing_image_name]['labels'] = dataframe_y_true.loc[missing_image_name]['labels']

#     tn, fp, fn, tp = confusion_matrix(dataframe_y_true, dataframe_y_pred).ravel()
#     print( confusion_matrix(dataframe_y_true, dataframe_y_pred))
#     lambda_ = 100
#     score = 1 / len(dataframe_y_true) * (fn + lambda_ * fp)
#     return score




# print(valeo_custom_metric(pd.DataFrame(labels),pd.DataFrame(Z.astype(int))))

# with open('submission_kmeans.csv', 'w',newline='') as csvfile: 
#     csvwriter = csv.writer(csvfile) 
#     csvwriter.writerow(['images','labels'])
#     for k in range(len(Z)):
#         csvwriter.writerow([list_path[k],int(Z[k])])

    # """### Saving the clusters images in different folders"""
    
    
    # for k in range(nb):    
    #     path="cluster"+str(k+1)
    #     if not os.path.exists(path):
    #         os.mkdir(path)
            
    # L=["cluster1"]+["../cluster"+str(i+1) for i in range(1,nb)]
    
    
    # for i in range(0,nb):
    #     nb_0=0
    #     os.chdir(os.path.abspath(L[i]))
    
    
    #     row = np.where(Z==i)[0]  # row in Z for elements of cluster i
    #     num = row.shape[0]       #  number of elements for each cluster
    #     r = np.floor(num/10.)    # number of rows in the figure of the cluster 
    
    #     # print("cluster "+str(i))
    #     # print(str(num)+" elements")
    
    #     for k in range(0, num):
     
    #         images = X_test[row[k], ]
    #         images = images.reshape(252, 28)
    
    #         file = PIL.Image.fromarray(images)
    #         file = file.convert("L")
    
    #         file.save('image'+str(k)+'_'+str(y_test[row[k, ]])+'.jpg','JPEG')
    
    #     plt.show()
    
    
    
    
