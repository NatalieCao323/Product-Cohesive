#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn import metrics
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
seed = 42
np.random.seed(seed)

# In[2]:

import os  
  
# Set the directory path  
directory = './emb'
# Get the names of all CSV files in the directory  
csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]    
# Create an empty DataFrame to store all data  
all_data = pd.DataFrame()  
# Loop through each CSV file and add its data to the all_data DataFrame  
for file in csv_files:  
    file_path = os.path.join(directory, file)  
    df = pd.read_csv(file_path)  
    all_data = all_data.append(df, ignore_index=True)  

# In[3]:

# Rename columns
all_data.columns = ['index','sents_vec']

# In[4]:

# Load restaurant menus data
menus = pd.read_csv('./archive/restaurant-menus.csv')[['restaurant_id','name']]

# In[5]:

# Add index column to menus
menus['index'] = menus.index

# In[6]:

# Merge menus with all_data on 'index'
menus = pd.merge(all_data,menus,on='index')

# In[7]:

# Display the first few rows of menus
menus.head()

# In[8]:

# Load restaurants data and filter out rows with missing 'score' or 'ratings'
restaurant = pd.read_csv('./archive/restaurants.csv')
restaurant = restaurant.dropna(subset=['score','ratings'])
restaurant_id = restaurant['id'].unique()
# Randomly select elements
random_id = np.random.choice(restaurant_id, size=20, replace=False)
print('restaurant_id',len(random_id),'shape',random_id.shape)
restaurant.head()

# In[9]:

# Set the device to 'cuda' if available, otherwise 'cpu'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
def K_cluster_analysis(K, X):
    print("K-means begin with clusters: {}".format(K))
    # Train K-means on the dataset and return clustering scores
    mb_kmeans = MiniBatchKMeans(n_clusters=K, init="k-means++", random_state=seed)
    y_pred = mb_kmeans.fit_predict(X)
    
    sse = mb_kmeans.inertia_ # Sum of squared distances of samples to their closest cluster center
    print("sse_score: {}".format(sse))
    return sse

# In[10]:

bert_features = menus[['sents_vec']]

# In[11]:

import json
# Convert string representations of lists to actual lists
bert_features['sents_vec'] = bert_features['sents_vec'].apply(json.loads)
# Expand the list into multiple columns
bert_features[[str(i) for i in range(1536)]] = pd.DataFrame(bert_features['sents_vec'].tolist(), index=bert_features.index)  
# Drop the original 'sents_vec' column
bert_features.drop(['sents_vec'], axis=1, inplace=True) 

# In[12]:

# Determine the optimal number of clusters
Ks = list(range(2,30))
sse_scores = []
for K in Ks:
    sse = K_cluster_analysis(K, bert_features)
    sse_scores.append(sse)

plt.plot(Ks, np.array(sse_scores), 'b-', label='SSE scores')
plt.savefig("sse_score.png")
plt.xlabel("K", fontsize=12)
plt.ylabel("SSE", fontsize=12)
plt.xticks(Ks, Ks)
plt.show();

# In[13]:

Best_K = 5

# In[14]:

print("The best cluster number found is {}".format(Best_K))
mb_kmeans = MiniBatchKMeans(n_clusters=Best_K,init="k-means++", random_state=seed)
y_pred = mb_kmeans.fit_predict(bert_features)
# Save clustering results
menus["Kmeans_" + str(Best_K)] = y_pred
menus['label'] = y_pred

# In[15]:

# Define a function for cluster result visualization
def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    color = ['oy', 'or', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^', 'g^']
    for i in range(numClass):
        x1 = []
        y1 = []
        for ind1 in newData[Lab[i]]:
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, color[i], label=f'Cluster-{i}')

    plt.legend(loc=1)
    plt.show();

# Use PCA for dimensionality reduction before TSNE
from sklearn.manifold import TSNE
newData = TSNE(n_components=2).fit_transform(bert_features)
clf = mb_kmeans
result = y_pred
plot_cluster(result, newData, Best_K)

# In[16]:

# Optional: Filter rows based on certain conditions, uncomment and adjust as necessary
# train_kmeans = train_kmeans[~train_kmeans['Kmeans_7'].isna()==True]
# train_kmeans = train_kmeans[~train_kmeans['Comments'].isna()==True]
# data = train_kmeans

# In[17]:

# Example for generating a word cloud, adjust as needed
# from wordcloud import WordCloud
# import matplotlib.pyplot as plt

# f = train_kmeans['name'].tolist()
# f = ' '.join(f)
# font = r'C:\Windows\Fonts\simfang.ttf'
# wordcloud = WordCloud(background_color="white", width=2000, height=2000, margin=2, font_path=font).generate(f)
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show();
# wordcloud.to_file('test.png')

# In[18]:

# Gini coefficient calculation function
def gini_coefficient(group):
    labels = group['label'].values 
    counts = np.bincount(labels)  
    probs = counts / len(labels)  
    gini = 1 - np.sum(probs**2)  
    return gini

# In[19]:

# Functions to calculate Shannon index and entropy
def calculate_shannon_index(group):
    labels = group['label'].values
    if len(set(labels)) == 1:
        return 0  
    counts = np.bincount(labels)  
    counts[counts == 0] = 1  
    probs = counts / len(labels)  
    return -np.sum(probs * np.log2(probs)) 

def calculate_entropy(group):
    labels = group['label'].values
    counts = np.bincount(labels)  
    counts[counts == 0] = 1  
    probs = counts / np.sum(counts)  
    entropy = -np.sum(probs * np.log2(probs))  
    return entropy

# In[20]:

gini_scores = train_kmeans.groupby('restaurant_id').apply(gini_coefficient)

# In[21]:

shannon_indexes = train_kmeans.groupby('restaurant_id').apply(calculate_shannon_index)

# In[22]:

entropies = train_kmeans.groupby('restaurant_id').apply(calculate_entropy)

# In[23]:

score = pd.concat([gini_scores, shannon_indexes, entropies], axis=1).reset_index()

# In[24]:

score.columns = ['id', 'gini', 'shannon', 'entropy']

# In[25]:

# Merge score with restaurant data
merge_score = pd.merge(restaurant, score, on='id')

# In[26]:

# Check the length of merged data
len(merge_score)

# In[27]:

# Display column names of merged data
merge_score.columns

# In[28]:

# Correlation analysis
merge_score.corr()

# In[29]:

# Calculate a new score
merge_score['new_score'] = merge_score['score'] * merge_score['ratings']

# In[30]:

# Regression model fitting
from statsmodels.formula.api import ols

# In[31]:

# Fit the model
model = ols('position~gini+shannon+entropy+lat+lng', data=merge_score).fit()
print(model.summary())
