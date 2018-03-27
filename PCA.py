#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 20 00:30:47 2017

@author: pengsihan
"""

import os  
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import unicodedata

# Get data
# Data from http://qwone.com/~jason/20Newsgroups/
path = "/Users/pengsihan/Downloads/20news-bydate/20news-bydate-train"
pathes= os.listdir(path)[1:]
to_pathes = []


for i in range(len(pathes)):
    to_pathes.append("/Users/pengsihan/Downloads/20news-bydate/20news-bydate-train-normalized/"+pathes[i])
    pathes[i] = "/Users/pengsihan/Downloads/20news-bydate/20news-bydate-train/" + pathes[i]
    
    
s = []
error = 0
vocabulary = {}
dele = []

#Get data and normalize
for i in range(20): #Each type of email
    for file in os.listdir(pathes[i]):
        if not os.path.isdir(file):
            with open(pathes[i]+"/"+file,'rb') as f:
                st = ''
                for line in f.readlines():
                    try:
                        st = st + line.decode('utf-8').lower()
                    except:
                        error +=1
                        dele.append(pathes[i]+"/"+file)
                st = unicodedata.normalize('NFC', st)
                st = st.translate(str.maketrans(',@>[]()\'\".:$&<>?*|-_`^=!#%;~/+','                              '))
                st = st.split()
                for voc in st:
                    if voc in vocabulary.keys():
                        vocabulary[voc] += 1
                    else:
                        vocabulary[voc] = 1
                with open(to_pathes[i]+"/"+file,'w') as of:
                    for j in st:
                        of.write(j+' ')
                s.append(st)
dic = []

#Kick Noise Vocabulary
for i in vocabulary.keys():
    if vocabulary[i] >= 15:
        dic.append(i)

#Feature Vector
def feature_vector(text):
    vector = dict().fromkeys(dic,0)
    for n in text:
        if n in vector:
            vector[n] += 1
    return list(vector.values())

count = 0
fv = []
for i in range(20):
    for file in os.listdir(to_pathes[i]): 
        if not os.path.isdir(file):
            with open(to_pathes[i]+"/"+file,'r') as f:
                line = f.readline()
                text = line.split()
                f = feature_vector(text)
                fv.append(f)
                count += 1


    
    
fv = np.reshape(fv,(count,len(dic)))

mean = np.mean(fv,axis = 0)

mean_fv = np.repeat(mean,count).reshape((count,len(dic)))
centered_fv = fv - mean_fv

covariance_matrix = np.matrix(centered_fv) * np.matrix(centered_fv).transpose()
eigenvalues, eigenvectors = LA.eig(covariance_matrix)
eigenvectors =  eigenvectors * np.matrix(centered_fv)


def centered(fv,count,len_dic):
    mean = np.mean(fv,axis = 0) #get the mean feature vectors
    mean_fv = np.repeat(mean,count).reshape((count,len_dic))
    return fv - mean_fv #return the centered feature vecotrs
    
def eigen(centered_fv):
    covariance_matrix = np.matrix(centered_fv) * np.matrix(centered_fv).transpose() #get the convariance matrix
    eigenvalues, eigenvectors = LA.eig(covariance_matrix) #get the eigenvalues
    eigenvectors =  eigenvectors * np.matrix(centered_fv) 
    return eigenvalues, eigenvectors #return the eigenvalues and eigenvectors
    
def principal_components(eigenvalues,eigenvectors):
    pc = np.argsort(-eigenvalues)[:10] #pick the top 10 eigenvalues
    principal_components = eigenvectors[pc] #select the eigenvectors with the top 10 eigenvalues
    return principal_components #return the top 10 eigenvalues


image_count = 0


image_count+=1
plt.figure(image_count)
plt.title('Proportion_Variance')
proportion_variance = eigenvalues / np.sum(eigenvalues)
plt.plot(range(500),proportion_variance[:500])
plt.show()

image_count+=1
plt.figure(image_count)
plt.title('Accumulated_Proportion_Variance')
k = 0
accumlate = []
for i in eigenvalues:
    k = k + i
    accumlate.append(k)
accumlate /= np.sum(eigenvalues)
plt.plot(range(500),accumlate[:500])
plt.show()


pc = np.argsort(-eigenvalues)[:10]
principal_components = eigenvectors[pc]


#==============================================================================
# output = []
# for i in range(11314):
#     temp = fv[i]
#     k = np.dot(principal_components,temp.transpose())
#==============================================================================
    
number = 0
for i in range(20):
    for file in os.listdir(to_pathes[i]):
        #print(i)
        if not os.path.isdir(file):
            with open(to_pathes[i]+"/"+file,'w') as f:
                temp = centered_fv[number]
                number += 1
                k = np.dot(principal_components,temp.transpose())
                k = k.tolist()[0]
                for j in k:
                    f.write(str(j)+' ')

                

    



  
