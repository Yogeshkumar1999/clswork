# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 11:14:46 2018

@author: yogesh
"""

from sklearn import datasets
import numpy as np
#loading the iris dataset
iris=datasets.load_iris()
X=iris.data[:,[2,3]]
y=iris.target#target means class
#spliting the data into training and testing
from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.3)
#defining the classifer
from sklearn.svm import SVC
classifier=SVC(kernel='linear',C=1,random_state=0)
classifier.fit(X_train,y_train)
#predicting
y_pred=classifier.predict(X_test)
#confusion matrix
from sklearn.metrics import confusion_matrix
cu=confusion_matrix(y_test,y_pred)

from sklearn.metrics import accuracy_score
print('Accuracy %.2f'%accuracy_score(y_test,y_pred))
#%%
#this is for back propagation neural networks
import numpy as np
#input array
X=np.array([[1,0,1,0],[1,0,1,1],[0,1,0,1]])
#output
y=np.array([[1],[1],[0]])

#sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))

#derivation of sigmoid function
def derivatives_sigmoid(x):
    return x*(1-x)

#variabel init
epoch=50
lr=0.1
inputlayer_neurons=X.shape[1]
hiddenlayer_neurons=3
output_neurons=1

#weight and bias inti
wh=np.random.uniform(size=(inputlayer_neurons,hiddenlayer_neurons))
bh=np.random.uniform(size=(1,hiddenlayer_neurons))
wout=np.random.uniform(size=(hiddenlayer_neurons,output_neurons))
bout=np.random.uniform(sixe=(1,output_neurons))

for i in range(epoch):
    #forward propagation
    hidden_layer_input1=np.dot(X,wh)
    hidden_layer_input1=hidden_layer_input1+bh
    hiddenlayer_activations=sigmoid(hidden_layer_input)
    output_layer_input1=np.dot(hiddenlayer_activation,wout)
    output_layer_input=output_layer_input1+bout
    output=sigmoid(output_layer_input)
    
    #backword propagation
    E=y-output
    slope_output_layer=derivatives_sigmoid(output)
    slope_hidden_layer=derivatives_sigmoid(hiddenlayer_activations)
    d_output=E*slope_output_layer
    Error_at_hidden_layer=d_output.dot(wout.T)
    d_hiddenlayer=Error_at_hidden_layer*slope_hidden_layer
    wout+=hiddenlayer_activations.T.dot(d_output)*lr
    bout+=np.sum(d_output, axis=0, keepdims=True)*lr
    wh+=X.T.dot(d_hiddenlayer)*lr
    bh+=np.sum(d_hiddenlayer, axis=0, keepdims=True)*lr
print(output)                                   
                                       
                                       
                                       
                                       
            
    
























