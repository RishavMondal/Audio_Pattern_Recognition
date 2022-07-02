import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
#%matplotlib inline

import sklearn as sk
#from xgboost import XGBClassifier, XGBRFClassifier
#from xgboost import plot_tree, plot_importance

import librosa as lb
import librosa.display as ldi
import IPython.display as ipd
import librosa.feature as lbf


#import eli5
#from eli5.sklearn import PermutationImportance


import warnings
warnings.filterwarnings('ignore')

def load_audio(audio_path):
#Extracting features of an audio file
#    audio_path = f'{dir}/hiphop.00010.wav'
    x , sr = lb.load(audio_path) 

# x = no. of data points & sr = sample rate
    print(type(x), type(sr))
    print('x:', x, '\n')
    print('x shape:', np.shape(x), '\n')
    print('Sample Rate (KHz):', sr)

#length/duration of audio = len(x)/sr
    print('Check Len of Audio:', x.shape[0]/sr) 

#removing blank spaces from the datapoints
    X, _ = lb.effects.trim(x)
    print('Audio File:', X, '\n')
    print('Audio File shape:', np.shape(X))
       
#Visualizing the wave pattern of the audio
    ipd.Audio(audio_path)
    plt.figure(figsize=(20, 5))
    ldi.waveshow(X, sr=sr)
    
    return X,sr

def fourier_extract(X , sr):

#Fourier Transform visualisation
    fft = 2048
    hl = 512
    X_ = np.abs(lb.stft(X, n_fft = fft, hop_length = hl))
    
    print('Fourier transform shape:','\t',np.shape(X_))
    plt.figure(figsize = (16, 8))
    plt.plot(X_);


#display Spectrogram
    Xdb = lb.amplitude_to_db(X_, ref = np.max)
    plt.figure(figsize=(16, 8))
    ldi.specshow(Xdb, sr=sr, hop_length=hl, x_axis='time', y_axis='log')
    plt.colorbar()
    plt.title("Spectrogram", fontsize = 23)
    plt.xlabel("Time")
    plt.ylabel("Frequency in HZ")


#Extracting mfcc of each type of musice genre

def mel_extract(X,sr,hl = 256):
        
    #mfccs = lbf.mfcc(X, sr=sr)
    mel = lbf.melspectrogram(X, sr=sr)
    print('mel shape:', mel.shape)
    mel_db = lb.amplitude_to_db(mel, ref=np.max)
    #print(mfccs.shape)
    print('mfcc shape:', mel_db.shape)
    plt.figure(figsize = (16, 8))
    ldi.specshow(mel_db, sr=sr, hop_length=hl, x_axis = 'time', y_axis = 'log',
                            cmap = 'bwr');
    plt.colorbar();
    plt.title("Blues Mel Spectrogram", fontsize = 24);
    
def h_p_component(X):
    
    y_harm, y_perc = lb.effects.hpss(X)
    
    print('Sound waves in Blue is Harmonic and Red is Percussive')
    
    plt.figure(figsize = (16, 6))
    plt.plot(y_harm);
    plt.plot(y_perc);

#Spectral Centroid:The spectral centroid is a measure used in digital signal processing to characterise a spectrum. 
#                  It indicates where the center of mass of the spectrum is located. Perceptually, it has a robust 
#                  connection with the impression of brightness of a sound. It is sometimes called center of spectral mass
def normalize(x, axis=0):
    return sk.preprocessing.minmax_scale(x, axis=axis)

def spec_centroid(X , sr):
    sc = lbf.spectral_centroid(X, sr=sr)[0]

    print('Centroids:', sc, '\n')
    print('Shape of Spectral Centroids:', sc.shape, '\n')
    frames = range(len(sc))

    t = lb.frames_to_time(frames)#Converts frame counts to time (seconds).

    print('frames:', frames, '\n')
    print('t:', t)
    
    plt.figure(figsize = (16, 6))
    ldi.waveshow(X, sr=sr, alpha=0.4, color = '#028A0F');
    plt.plot(t, normalize(sc), color='#141414');
    
    #spectral rolloff
    spectral_r = lbf.spectral_rolloff(X, sr=sr)[0]
    print('Spectral rolloff is the frequency below which a specified percentage of the total spectral energy, e.g. 85%, lies.')
    
        # The plot
    plt.figure(figsize = (16, 6))
    ldi.waveshow(X, sr=sr, alpha=0.4, color = 'r');
    plt.plot(t, normalize(spectral_r), color='#141414');

    
def mfcc_extract(X,sr):
    mfcc = lbf.mfcc(X, sr=sr)
    print(mfcc.shape)
    print(mfcc)
    plt.figure(figsize = (16, 6))
    ldi.specshow(mfcc, sr=sr, x_axis='time', cmap = 'cool');
    return mfcc
    
def scaled_mfcc(mfcc,sr):
    s_mfcc = sk.preprocessing.scale(mfcc, axis=1)
    print('Mean:', s_mfcc.mean(), '\n')
    print('Var:', s_mfcc.var())

    plt.figure(figsize = (16, 6))
    ldi.specshow(s_mfcc, sr=sr, x_axis='time', cmap = 'bwr');
    
def chroma_gram(X,sr):
    hl = 5000

    chromagram = lbf.chroma_stft(X, sr=sr, hop_length=hl)
    print('Chromogram shape:', chromagram.shape)

    plt.figure(figsize=(16, 6))
    ldi.specshow(chromagram, x_axis='time', y_axis='chroma', hop_length=hl, cmap='bwr');    

################################################################################################################################################    
#accessing the dataset directory
dir = './GTZAN'
print(  list(os.listdir(f'{dir}/')) )

#Blues Music
audio_path = f'{dir}/country.00022.0.wav'

X , sr = load_audio(audio_path)

fourier_extract(X, sr)

mel_extract(X, sr)

zero_cross = lb.zero_crossings(X, pad=False)
print('The zero-crossing rate (ZCR) is ',sum(zero_cross))

h_p_component(X)
spec_centroid(X, sr)
mfcc = mfcc_extract(X, sr)
scaled_mfcc(mfcc, sr)
chroma_gram(X, sr)


#Rock Music
audio_path = f'{dir}/Audio_Feature_Set/rock.00000.wav'

X , sr = load_audio(audio_path)

fourier_extract(X, sr)

mel_extract(X, sr)

zero_cross = lb.zero_crossings(X, pad=False)
print('The zero-crossing rate (ZCR) is ',sum(zero_cross))

h_p_component(X)
spec_centroid(X, sr)
mfcc = mfcc_extract(X, sr)
scaled_mfcc(mfcc, sr)
chroma_gram(X, sr)


