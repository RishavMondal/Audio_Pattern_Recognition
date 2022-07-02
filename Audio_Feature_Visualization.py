import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt

import sklearn as sk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler

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


db_30 = db_30.iloc[0:, 1:]
y = db_30['label']
X = db_30.loc[:, db_30.columns != 'label']
cols = X.columns

##############################################################################################################################

#K-means with MIN-MAX scaling

#PCA visualization
scaler = MinMaxScaler()
np_scaled = scaler.fit_transform(X)
X = pd.DataFrame(np_scaled, columns = cols)

pca = sk.decomposition.PCA(n_components=2)
scaled_df = pca.fit_transform(X)
df_p = pd.DataFrame(data = scaled_df, columns = ['pca1', 'pca2'])

fdf = pd.concat([df_p, y], axis = 1)
pca.explained_variance_ratio_

plt.figure(figsize = (16, 9))
sns.scatterplot(x = "pca1", y = "pca2", data = fdf, hue = "label", alpha = 0.7,
               s = 100);

plt.title('Genres with MIN-MAX scale PCA', fontsize = 25)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 10);
plt.xlabel("Principal Component 1", fontsize = 15)
plt.ylabel("Principal Component 2", fontsize = 15)

#cluster data
inertias = []
k_list1 = range(1, 10)
for k in k_list1:
    km = KMeans(n_clusters=k)
    km.fit(X)
    inertias.append([k, km.inertia_])

oca_results_scale = pd.DataFrame({'Cluster': range(1,10), 'SSE': inertias})
plt.figure(figsize=(16,9))
plt.plot(pd.DataFrame(inertias)[0], pd.DataFrame(inertias)[1], marker='o')
plt.title('Optimal Number of Clusters using Elbow Method (Scaled Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

# scaled silhoutte
X2 = X
kmeans_scale = KMeans(n_clusters=10, n_init=100, max_iter=1000, init='k-means++', random_state=50).fit(X2)
print('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(X2, kmeans_scale.labels_, metric='euclidean')))
labels_scale = kmeans_scale.labels_
clusters_scale = pd.concat([X2, pd.DataFrame({'cluster_scaled':labels_scale})], axis=1)
#print(clusters_scale)

##########################################################################################################################

#K-Means with standard scaling

#cluster data   
scaler_standard = StandardScaler()
np_standard = scaler_standard.fit_transform(X)
X_s = pd.DataFrame(np_standard, columns = cols)
X_s.head()

sse = []
k_list = range(1, 10)
for k in k_list:
    km = KMeans(n_clusters=k)
    km.fit(X_s)
    sse.append([k, km.inertia_])
    
oca_results_scale = pd.DataFrame({'Cluster': range(1,10), 'SSE': sse})
plt.figure(figsize=(16,9))
plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1], marker='o')
plt.title('Optimal Number of Clusters using Elbow Method (Scaled Data)')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')

#scaled silhoutte
X_s2 =X_s
kmeans_scale = KMeans(n_clusters=10, n_init=100, max_iter=1000, init='k-means++', random_state=50).fit(X_s2)
print('KMeans Scaled Silhouette Score: {}'.format(silhouette_score(X_s2, kmeans_scale.labels_, metric='euclidean')))
labels_scale = kmeans_scale.labels_
clusters_scale = pd.concat([X_s2, pd.DataFrame({'cluster_scaled':labels_scale})], axis=1)
#print(clusters_scale)

#PCA visualization
pca2 = sk.decomposition.PCA(n_components=2)
pca2d = pca2.fit_transform(X_s2)
df_p2 = pd.DataFrame(data = pca2d, columns = ['pca1', 'pca2'])

fdf2 = pd.concat([df_p2, y], axis = 1)
pca.explained_variance_ratio_

plt.figure(figsize = (16,9))
sns.scatterplot(x = 'pca1', y = 'pca2', data = fdf2, 
                hue="label", 
                s=100, alpha=0.7).set_title('Genres with Standard scale PCA', fontsize=15);
plt.legend()
plt.ylabel('PC2')
plt.xlabel('PC1')
plt.show()




