import pandas as pd
import os
from tqdm import tqdm
import numpy as np
from scipy.stats import kurtosis
from scipy.stats import skew
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pydub import AudioSegment 
from pydub.utils import make_chunks 

import sklearn as sk
from sklearn.metrics import accuracy_score as acc
from sklearn.svm import SVC 
from sklearn.metrics import confusion_matrix as cnf

import librosa as lb
import librosa.display as ldi
import IPython.display as ipd
import librosa.feature as lbf
import math 
import sys

eps = sys.float_info.epsilon


def energy(frame):
    """Computes signal energy of frame"""
    return np.sum(frame ** 2) / np.float64(len(frame))


def energy_entropy(frame, n_short_blocks=10):
    """Computes entropy of energy"""
    # total frame energy
    frame_energy = np.sum(frame ** 2)
    frame_length = len(frame)
    sub_win_len = int(np.floor(frame_length / n_short_blocks))
    if frame_length != sub_win_len * n_short_blocks:
        frame = frame[0:sub_win_len * n_short_blocks]

    # sub_wins is of size [n_short_blocks x L]
    sub_wins = frame.reshape(sub_win_len, n_short_blocks, order='F').copy()

    # Compute normalized sub-frame energies:
    s = np.sum(sub_wins ** 2, axis=0) / (frame_energy + eps)

    # Compute entropy of the normalized sub-frame energies:
    entropy = -np.sum(s * np.log2(s + eps))
    return entropy

# Get statistics from the vectors
def get_feature_stats(features):
    result = {}
    for k, v in features.items():
           result['{}_max'.format(k)] = np.max(v)
           result['{}_min'.format(k)] = np.min(v)
           result['{}_mean'.format(k)] = np.mean(v)
           result['{}_std'.format(k)] = np.std(v)
           result['{}_kurtosis'.format(k)] = kurtosis(v)
           result['{}_skew'.format(k)] = skew(v)
    return result


def extract_features_time(y,sr=22050,n_fft=1024,hop_length=512):
    features = {
                'energy': energy(y),
                'entropy': energy_entropy(y, n_short_blocks=math.ceil(len(y)/1024)),
                'rmse': lbf.rms(y, frame_length=n_fft, hop_length=hop_length).ravel(),
                'zcr': lbf.zero_crossing_rate(y, frame_length=n_fft, hop_length=hop_length).ravel(),
                }


    dict_agg_features = get_feature_stats(features)
    dict_agg_features['tempo'] = lb.beat.tempo(y=y,sr=sr,hop_length=hop_length)[0]

    return dict_agg_features

def extract_features_frequency(y,sr=22050,n_fft=1024,hop_length=512):
    features = {'centroid': lbf.spectral_centroid(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'flux': lb.onset.onset_strength(y=y, sr=sr).ravel(),                
                'contrast': lbf.spectral_contrast(y, sr=sr).ravel(),
                'bandwidth': lbf.spectral_bandwidth(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'flatness': lbf.spectral_flatness(y, n_fft=n_fft, hop_length=hop_length).ravel(),
                'rolloff': lbf.spectral_rolloff(y, sr=sr, n_fft=n_fft, hop_length=hop_length).ravel(),
                'chroma': lbf.chroma_stft(y=y, sr=sr,n_fft=n_fft, hop_length=hop_length).ravel()}


    # MFCC treatment
    mfcc = lbf.mfcc(y, n_fft=n_fft, hop_length=hop_length, n_mfcc=20)
    for idx, v_mfcc in enumerate(mfcc):
        features['mfcc_{}'.format(idx)] = v_mfcc.ravel()

    dict_agg_features = get_feature_stats(features)
   
    return dict_agg_features

def make_time_domain():
    arr_features=[]
    os.chdir('GTZAN/Data/genres_original')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(genre):
            print('\n',genre,'/',fname,'\n')
            y, sr = lb.load(genre+'/'+fname, duration=30)
            dict_features=extract_features_time(y=y,sr=sr)
            dict_features['label']=genre
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    print(df.head())
    print(df.shape)
    os.chdir('..')
    os.chdir('..')
    df.to_csv('time_data.csv',index=False)

def make_frequency_domain():
    arr_features=[]
    os.chdir('GTZAN/Data/genres_original')
    genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
    for idx,genre in tqdm(enumerate(genres),total=len(genres)):
        for fname in os.listdir(genre):
            print('\n',genre,'/',fname,'\n')
            y, sr = lb.load(genre+'/'+fname, duration=30)
            dict_features=extract_features_frequency(y=y,sr=sr)
            dict_features['label']=genre
            arr_features.append(dict_features)

    df=pd.DataFrame(data=arr_features)
    print(df.head())
    print(df.shape)
    os.chdir('..')
    df.to_csv('frequency_data.csv',index=False)
    
#Classifying using Suppoort Vector Machine which is the best parameters from Audio_Classification, i.e. C = 500, kernel = rbf
def pred_svm(x_train,y_train,x_test,y_test):
    
    print('Starting SVM Classifier')
    
    model_svc = SVC(decision_function_shape = 'ovo', C = 5000, kernel = 'rbf')
    model_svc.fit(x_train,y_train)
    y_pred = model_svc.predict(x_test)
    
    prediction_accuracy = round(acc(y_test, y_pred),4)
    print('Accuracy of Support Vector Machine:',prediction_accuracy)
    
    conf = cnf(y_test, y_pred)

    plt.figure(figsize = (16, 9))
    plt.title('CONFUSION MATRIX FOR SVM', y=1.05, size=19)
    sns.heatmap(conf, cmap="bwr", annot=True, 
            xticklabels = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"],
            yticklabels=["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "reggae", "rock"]);    

def predict_time():
    dir = './GTZAN'
    print(  list(os.listdir(f'{dir}/')) )

    df_T = pd.read_csv(f'{dir}/time_data.csv')
    df_T = df_T.iloc[0:, 1:]
    df_T.head()
    print('Number of rows:', df_T.shape[0])
    print('Number of columns:', df_T.shape[1])

    counter=0
    for i in df_T.columns:
        if i!='label': #target Variable that list the Genre Labels
            counter+=1
        print(i)
    print("The Total number of Features in this Set :",counter )


    y = df_T['label']
    X = df_T.loc[:, df_T.columns != 'label']
    # Breaking Up  X and Y   Independent  and Target Variables )

    # MinMAX Scaling implementation:
    cols = X.columns
    scaler = sk.preprocessing.MinMaxScaler()
    np_scaled = scaler.fit_transform(X)

    x = pd.DataFrame(np_scaled, columns = cols)

    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

    print('Number of rows in train:', x_train.shape[0])
    print('Number of rows in test:', x_test.shape[0])
    
    pred_svm(x_train,y_train,x_test,y_test)

def predict_frequency():
    dir = './GTZAN'
    print(  list(os.listdir(f'{dir}/')) )

    df_F = pd.read_csv(f'{dir}/frequency_data.csv')
    df_F = df_F.iloc[0:, 1:]
    df_F.head()
    print('Number of rows:', df_F.shape[0])
    print('Number of columns:', df_F.shape[1])

    counter=0
    for i in df_F.columns:
        if i!='label': #target Variable that list the Genre Labels
            counter+=1
        print(i)
    print("The Total number of Features in this Set :",counter )


    y = df_F['label']
    X = df_F.loc[:, df_F.columns != 'label']
    # Breaking Up  X and Y   Independent  and Target Variables )

    # MinMAX Scaling implementation:
    cols = X.columns
    scaler = sk.preprocessing.MinMaxScaler()
    np_scaled = scaler.fit_transform(X)

    x = pd.DataFrame(np_scaled, columns = cols)

    x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size=0.3, random_state=42)

    print('Number of rows in train:', x_train.shape[0])
    print('Number of rows in test:', x_test.shape[0])
    
    pred_svm(x_train,y_train,x_test,y_test)
    
    
    
if __name__=='__main__':
    make_time_domain()
    make_frequency_domain()
    predict_time()
    predict_frequency()