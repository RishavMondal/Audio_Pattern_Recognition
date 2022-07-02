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


dir = './GTZAN'
print(  list(os.listdir(f'{dir}/')) )

db_30 = pd.read_csv(f'{dir}/features_30_sec.csv')
db_30.head()
print('Number of rows:', db_30.shape[0])
print('Number of columns:', db_30.shape[1])

counter=0
for i in db_30.columns:
    if i!='label': #target Variable that list the Genre Labels
        counter+=1
    print(i)
print("The Total number of Features in this Set :",counter )


#Pearson correlation with mean
spike = [col for col in db_30.columns if 'mean' in col]
corr = db_30[spike].corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(20, 17));
plt.title('Pearson Correlation of Features with Mean Values', y=1.05, size=19)
sns.heatmap(corr, mask=mask, cmap="bwr", vmax=.3, center=0,
            square=True, linewidths=.7, cbar_kws={"shrink": .5},annot=True)

plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);

#Pearson Correlation with Variance
spike = [col for col in db_30.columns if 'var' in col]
corr = db_30[spike].corr()

mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(20, 17));
colormap = plt.cm.viridis
plt.title('Pearson Correlation of Features with variance', y=1.05, size=19)
sns.heatmap(corr, mask=mask, cmap=colormap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5},annot=True)
  
plt.xticks(fontsize = 10)
plt.yticks(fontsize = 10);

#Bar box plot of the genres
x = db_30[["label", "tempo"]]

f, ax = plt.subplots(figsize=(16, 9));
sns.boxplot(x = "label", y = "tempo", data = x, palette = 'Paired');

plt.title('BPM Boxplot for Genres', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Genres", fontsize = 15)
plt.ylabel("Beats Per Minute-Tempo", fontsize = 15)

