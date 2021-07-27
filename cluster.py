from sentence_transformers import SentenceTransformer, util
import numpy as np
import pylcs
import pandas as pd

from helpers import *

# Read Dataset

data_path = 'data/I2_1_1000.txt'

with open(data_path,'r') as txtfile:
    biz_ideas = [line.rstrip('\n') for line in txtfile]
    
topics_unprocessed = []

for idea in biz_ideas:

    topics_unprocessed.append(topic(idea))
    
ideas = dict(zip())

processed = preprocess(topics_unprocessed,biz_ideas)

df = pd.DataFrame(zip(processed.values(), processed.keys()))
df.columns = ['Text','Topic']

# Create a list of topics from the corpus

topics = list(df['Topic'])

# Generate embeddings using a pre-trained SentenceTransformer model

model = SentenceTransformer('paraphrase-mpnet-base-v2')
topics_embeddings_unprocessed = model.encode(topics_unprocessed)
topics_embeddings = model.encode(topics)

# K-Means Pipeline

# Define number of clusters or auto estimate optimum using intertia
num_clusters = optimize_k(topics_embeddings)

# Reduce Dimentions using TSNE or PCA
topics_red = reduce_dims(topics_embeddings,alg='tsne',num_components=2)

# Apply K-Means Clustering
cluster_assignment = kmeans(num_clusters=num_clusters, data=topics_embeddings) # data=topics_embeddings, topics_red
df['Cluster'] = cluster_assignment

# Refine the clusters using LCS
df_refined = refine_clusters(df, num_clusters)
df_refined = df_refined[['Text', 'Topic', 'Cluster']]
df_refined.reset_index(drop=True, inplace=True)

# Print clusters cohesively
print_clusters(num_clusters, df_refined, 5)

# Save data as CSV
df_refined.to_csv(data_path[:-4]+'.csv')