#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import required libraries
from transformers import AutoTokenizer, AutoModel
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

# Set seed for reproducibility and initialize tokenizer and model
seed = 42
np.random.seed(seed)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

# In[2]:

# Load restaurant data, drop rows with missing 'score' or 'ratings', and randomly select 100 restaurants
restaurant = pd.read_csv('./archive/restaurants.csv')
restaurant = restaurant.dropna(subset=['score','ratings'])
restaurant_id = restaurant['id'].unique()
random_id = np.random.choice(restaurant_id, size=100, replace=False)
print('Number of restaurant_ids:', len(random_id), 'Shape:', random_id.shape)
restaurant.head()

# In[3]:

# Load menus, filter by selected restaurant IDs, and combine 'category' and 'name' into 'category_name'
menus = pd.read_csv('./archive/restaurant-menus.csv')
menus = menus[menus['restaurant_id'].isin(random_id)]
menus = menus[~menus['name'].isna()]
menus['category_name'] = menus['category'] + '.' + menus['name']

# In[4]:

# Display shape of the menus DataFrame
menus.shape

# In[5]:

# Display the first few rows of the menus DataFrame
menus.head()

# In[6]:

# Display statistics on the length of menu item names
menus['name'].str.len().describe([0.9,0.99])

# In[7]:

# Display statistics on the length of combined category and menu item names
menus['category_name'].str.len().describe([0.9,0.99])

# In[8]:

# Display shape of the menus DataFrame again
menus.shape

# In[9]:

# Display the number of missing values in each column of the menus DataFrame
menus.isna().sum()

# In[10]:

# Set the computing device based on availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
# Define a function for encoding sentences using the model and tokenizer
def encoding(model, tokenizer, sentences):
    model.eval()
    model.to(device)
    max_char_len = 100
    sents_inputs = tokenizer(sentences, return_tensors='pt', max_length=max_char_len, padding="max_length", truncation=True)
    input_ids = sents_inputs['input_ids']
    dataloader = DataLoader(input_ids, batch_size=32, shuffle=False)
    sents_vec = []
    tqdm_batch_iterator = tqdm(dataloader, desc='Sentence encoding')
    for index, batch in enumerate(tqdm_batch_iterator):
        input_ids = batch.to(device)
        sents_vec.append(model(input_ids)['pooler_output'].detach().cpu().numpy().tolist())
    torch.cuda.empty_cache()
    sents_vec = [np.array(xi) for x in sents_vec for xi in x]
    return sents_vec

# Define a function for K-means cluster analysis
def K_cluster_analysis(K, X):
    print("K-means begins with clusters:", K)
    mb_kmeans = MiniBatchKMeans(n_clusters=K, init="k-means++", random_state=seed)
    y_pred = mb_kmeans.fit_predict(X)
    sse = mb_kmeans.inertia_  # Total sum of squared distances to the nearest cluster center
    print("SSE score:", sse)
    return sse

# In[11]:

# Check if it's the first run to generate sentence vectors; otherwise, read from a CSV file
get_vec = True 
sentence_list = menus['name'].tolist()
if get_vec:
    sents_vec = encoding(model, tokenizer, sentence_list)
    bert_features = pd.DataFrame(sents_vec)
    bert_features.to_csv('bert_features.csv', index=False)
else:
    bert_features = pd.read_csv('bert_features.csv')
    sents_vec = bert_features.values

menus['sents_vec'] = sents_vec

# In[12]:

# Display bert_features DataFrame
bert_features

# In[13]:

# Find the optimal number of clusters and display SSE scores
Ks = list(range(2,15))
sse_scores = []
for K in Ks:
    sse = K_cluster_analysis(K, bert_features)
    sse_scores.append(sse)
plt.plot(Ks, np.array(sse_scores), 'b-', label='SSE scores')
plt.savefig("sse_score.png")
plt.xlabel("K", fontsize=12)
plt.ylabel("SSE", fontsize=12)
plt.xticks(Ks, Ks)
plt.show()

# In[14]:

# Set the best number of clusters
Best_K = 9

# In[15]:

# Fit K-means with the best number of clusters and save the results
print("The best cluster found is", Best_K)
mb_kmeans = MiniBatchKMeans(n_clusters=Best_K, init="k-means++", random_state=seed)
y_pred = mb_kmeans.fit_predict(bert_features)
menus["Kmeans_" + str(Best_K)] = y_pred
menus['label'] = y_pred
train_kmeans = menus

# In[16]:

# Define a function to visualize clustering results
def plot_cluster(result, newData, numClass):
    plt.figure(2)
    Lab = [[] for i in range(numClass)]
    index = 0
    for labi in result:
        Lab[labi].append(index)
        index += 1
    colors = ['oy', 'or', 'og', 'cs', 'ms', 'bs', 'ks', 'ys', 'yv', 'mv', 'bv', 'kv', 'gv', 'y^', 'm^', 'b^', 'k^', 'g^']
    for i in range(numClass):
        x1, y1 = [], []
        for ind1 in newData[Lab[i]]:
            try:
                y1.append(ind1[1])
                x1.append(ind1[0])
            except:
                pass
        plt.plot(x1, y1, colors[i], label=f'Cluster-{i}')
    plt.legend(loc=1)
    plt.show()

# Reduce dimensionality before using t-SNE for visualization
from sklearn.manifold import TSNE
newData = TSNE(n_components=2).fit_transform(bert_features)
clf = mb_kmeans
result = y_pred
plot_cluster(result, newData, Best_K)

# In[17 to 42]:

# Additional analysis including diversity metrics calculation (Gini coefficient, Shannon index, entropy),
# merging scores, calculating average cosine distances, and fitting regression models

