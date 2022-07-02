import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
#%matplotlib inline

import sklearn as sk
from sklearn.neural_network import MLPClassifier as mlp
from sklearn.metrics import accuracy_score as acc
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier as knn
from sklearn.linear_model import LogisticRegression as LR
from sklearn.metrics import confusion_matrix as cnf
from sklearn.ensemble import  GradientBoostingClassifier as GBC
#from xgboost import XGBClassifier, XGBRFClassifier
#from xgboost import plot_tree, plot_importance

import librosa as lb
import librosa.display as ldi
import IPython.display as ipd
import librosa.feature as lbf


dir = './GTZAN'
print(  list(os.listdir(f'{dir}/')) )

df_3 = pd.read_csv(f'{dir}/features_3_sec.csv')
df_3 = df_3.iloc[0:, 1:]
df_3.head()
print('Number of rows:', df_3.shape[0])
print('Number of columns:', df_3.shape[1])

counter=0
for i in df_3.columns:
    if i!='label': #target Variable that list the Genre Labels
        counter+=1
    print(i)
print("The Total number of Features in this Set :",counter )


y = df_3['label']
X = df_3.loc[:, df_3.columns != 'label']
# Breaking Up  X and Y   Independent  and Target Variables )

# MinMAX Scaling implementation:
cols = X.columns
scaler = sk.preprocessing.MinMaxScaler()
np_scaled = scaler.fit_transform(X)

x = pd.DataFrame(np_scaled, columns = cols)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

print('Number of rows in train:', x_train.shape[0])
print('Number of rows in test:', x_test.shape[0])


#Classifying using Suppoort Vector Machine
def pred_svm(x_train,y_train,x_test,y_test):
    
    print('Starting SVM Classifier')
    dfs = 'ovo'
    c = [1,2,5,10,50,500]
    kernel = ['linear','poly','rbf','sigmoid']
    best_accuracy_svm = 0
    best_C = 0
    best_kernel = ''
    best_pred = pd.DataFrame()
    for i in tqdm(c):
        for j in tqdm(kernel):
            model_svc = SVC(decision_function_shape = dfs, C = i, kernel = j)
            model_svc.fit(x_train,y_train)
            y_pred = model_svc.predict(x_test)
            prediction_accuracy = round(acc(y_test, y_pred),4)
            if best_accuracy_svm < prediction_accuracy:
                best_accuracy_svm = prediction_accuracy
                best_C = i
                best_kernel = j
                best_pred = y_pred
            print('Accuracy: ',prediction_accuracy ,'for C: ',i, 'and kernel: ',j)

    print('Best accuracy of Support Vector Machine:',best_accuracy_svm,'for regularization parameter C:',best_C, 'and kernel:',best_kernel)
    
    conf = cnf(y_test, best_pred)

    plt.figure(figsize = (16, 9))
    plt.title('CONFUSION MATRIX FOR SVM', y=1.05, size=19)
    sns.heatmap(conf, cmap="bwr", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
            yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);    
    
    
#Classifyiing using Multi Layer Perceptron

def pred_mlp(x_train,y_train,x_test,y_test):
    
    print('Starting MLP Classifier')
    S = 'lbfgs'
    A = 1e-5
    rs = [1,5,10,20]
    best_accuracy_mlp = 0
    best_random_state = 0
    best_pred = pd.DataFrame()

    for i in tqdm(rs):
        model_mlp = mlp(solver = S, alpha = A , random_state = i, max_iter = 5000)
        model_mlp.fit(x_train,y_train)
        y_pred = model_mlp.predict(x_test)
        prediction_accuracy = round(acc(y_test, y_pred),4)
        if best_accuracy_mlp < prediction_accuracy:
            best_accuracy_mlp = prediction_accuracy
            best_random_state = i
            best_pred = y_pred

        print('Accuracy: ',prediction_accuracy ,'for random state: ',i)
    
    print('Best accuracy of Multi Layer Perceptron:',best_accuracy_mlp,'for Random State:',best_random_state)
    
    conf = cnf(y_test, best_pred)

    plt.figure(figsize = (16, 9))
    plt.title('CONFUSION MATRIX FOR MLP', y=1.05, size=19)
    sns.heatmap(conf, cmap="bwr", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);  

#Classifyiing using K-Nearest Neighbor
   
def pred_knn(x_train,y_train,x_test,y_test):
    
    print('Starting KNN Classifier')
    nn = [5,10,15,20]
    best_accuracy_knn = 0
    best_nn = 0
    best_pred = pd.DataFrame()
 
    for i in tqdm(nn):
        model_knn = knn(n_neighbors = i)
        model_knn.fit(x_train,y_train)
        y_pred = model_knn.predict(x_test)
        prediction_accuracy = round(acc(y_test, y_pred),4)
        if best_accuracy_knn < prediction_accuracy:
            best_accuracy_knn = prediction_accuracy
            best_nn = i
            best_pred = y_pred
            
        print('Accuracy: ',prediction_accuracy ,'for nearest neighbor: ',i)
    
    print('Best accuracy of K-Nearest Neighbor:',best_accuracy_knn,'for Nearest Neighbor:',best_nn)
    
    conf = cnf(y_test, best_pred)

    plt.figure(figsize = (16, 9))
    plt.title('CONFUSION MATRIX FOR KNN', y=1.05, size=19)
    sns.heatmap(conf, cmap="bwr", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);  
#Classifyiing using Logistic Regression

def pred_lr(x_train,y_train,x_test,y_test):
    
    print('Starting LR Classifier')
    rs = [0,5,10,20]
    s = 'lbfgs'
    mc = 'multinomial'
    best_accuracy_lr = 0
    best_rs = 0
    c = [1,2,5,10,50,500]
    best_C = 0
    best_pred = pd.DataFrame()

    for i in tqdm(c):
        for j in tqdm(rs):
            model_lr = LR(C = i,random_state = i,solver = s,max_iter = 5000,multi_class = mc, )
            model_lr.fit(x_train,y_train)
            y_pred = model_lr.predict(x_test)
            prediction_accuracy = round(acc(y_test, y_pred),4)
            if best_accuracy_lr < prediction_accuracy:
                best_accuracy_lr = prediction_accuracy
                best_rs = j
                best_C = i
                best_pred = y_pred

            print('Accuracy: ',prediction_accuracy ,'for C :',i,' and for random state: ',j)

    print('Best accuracy of Logistic Regression:',best_accuracy_lr,'for C: ',best_C,' and for Random State:',best_rs)
    
    conf = cnf(y_test, best_pred)

    plt.figure(figsize = (16, 9))
    plt.title('CONFUSION MATRIX FOR LR', y=1.05, size=19)
    sns.heatmap(conf, cmap="bwr", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
           yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);  
    
    
pred_svm(x_train,y_train,x_test,y_test)
pred_knn(x_train,y_train,x_test,y_test)
pred_lr(x_train,y_train,x_test,y_test)
pred_mlp(x_train,y_train,x_test,y_test)
