from sentence_transformers import SentenceTransformer, util
import numpy as np
import pylcs
import pandas as pd

# Global variables for the pipeline

red1 = None
red2 = None

def topic(s1:str):
    
    # Returns the topic extracted from the entire output.
    
    return s1[:s1.find('.')]

def preprocess(data:list, refer=None):
    
    # Applies three proprocessing steps: turns every item to lower case, selects only short strings, and eliminates duplicates
    
    data = [datum.lower() for datum in data]
    processed = {}
    
    if refer:
                
        for i in range(len(data)):
            
            if len(data[i]) < 25:
                processed[data[i]] = refer[i]
        
    else:
        data_processed = [datum for datum in data if len(datum) < 25]
        processed = dict(zip(data_processed, data_processed))
        
    global red1
    red1 = (len(data) - len(processed.keys()))/len(data) * 100
    
    return processed

def overlap(s1:str, s2:str):
    
    # Returns the lexical overlap using longest common subsequence between two strings
    
    if len(s1) > len(s2):
        return pylcs.lcs(s1,s2)/float(len(s1))
    else:
        return pylcs.lcs(s1,s2)/float(len(s2))
    
def kmeans(num_clusters:int, data:list):
    
    # Performs K-Means Clustering on data and returns cluster assignments
    
    from sklearn.cluster import KMeans
        
    clustering_model = KMeans(n_clusters = num_clusters)
    clustering_model.fit(data)
    cluster_assignment = clustering_model.labels_

    return cluster_assignment

def optimize_k(data:list):
    
    # This function is used to automatically determine the optimal value of K for K-Means Clustering
    
    from sklearn.cluster import KMeans
    import math
    
    dists = []
    grads = []
    K = range(1,70)
    iner_drop = 0
    iner_last = 0
    
    for n in K:
        k_model = KMeans(n_clusters = n)
        k_model.fit(data)
        dists.append(k_model.inertia_)

    def calc_dist(x1,y1,a,b,c):
        return abs((a*x1 + b*y1 + c))/(math.sqrt(a**2 + b**2))
        
    a = dists[0] - dists[-1]
    b = K[-1] - K[0]
    c1 = K[0] * dists[-1]
    c2 = K[-1] * dists[0]
    c = c1 - c2
        
    dists_line = []

    for k in range(K[-1]):
        dists_line.append(calc_dist(K[k], dists[k], a, b, c))
                
    num_clusters = dists_line.index(max(dists_line))+1
        
    return num_clusters

def reduce_dims(data, alg='tsne', num_components=2):
    
    # Used to reduce dimensions using TSNE or PCA algorithms
    
    topics_red = None
    
    if alg == 'tsne':
        
        from sklearn.manifold import TSNE

        topics_red = TSNE(n_components=num_components).fit_transform(data)
        
    elif alg == 'pca':
        
        from sklearn.decomposition import PCA
        topics_red = PCA(n_components=num_components,svd_solver='full').fit_transform(data)
            
    return topics_red

def get_clusters(num_clusters:int, df):
    
    # Prepares and returns clusters as list of lists
    
    clusters = []
    
    for i in range(num_clusters):
        
        clust_sent = np.where(df['Cluster'] == i)
        clust_points = []
        
        for k in clust_sent[0]:
            
            clust_points.append(df['Topic'][k])
            
        clusters.append(clust_points)
    
    return clusters

def refine_clusters(df, num_clusters:int, threshold=0.7):
    
    # Eliminates lexically similar items using Longest Common Subsequence
    
    refined_clusters = pd.DataFrame()
    reductions = []

    for i in range(num_clusters):

        df_cluster = df.where(df['Cluster'] == i+1)

        df_cluster.dropna(subset=['Text'], inplace=True)

        refined_cluster = pd.DataFrame()

        for j in range(df_cluster.shape[0]-1):
            flag = True

            for k in range(j+1,df_cluster.shape[0]):

                overlap_ = overlap(df_cluster.iloc[j]['Topic'],df_cluster.iloc[k]['Topic'])

                if overlap_ > 0.7:
                    #print('Hit')
                    flag = False
                    break

            if flag:
                    refined_cluster = refined_cluster.append(df_cluster.iloc[j])

        if df_cluster.shape[0]:
                reductions.append((df_cluster.shape[0] - refined_cluster.shape[0]) / df_cluster.shape[0])
                refined_clusters = refined_clusters.append(refined_cluster)

    global red2
    red2 = np.average(np.array(reductions))*100
    
    return refined_clusters

def print_clusters(num_clusters:int, df, n=10):
    
    # Prints clusters in a cohesive manner and also displays how much redundant data has been eliminated

    for i in range(num_clusters):
        
        df_cluster = df.where(df['Cluster'] == i+1)
        df_cluster.dropna(subset=['Text'], inplace=True)
        print(df_cluster.head(n))
        
    global red1, red2
    print(f'\nData trimmed by {red1:.2f}% in preprocessing step, and by {red2:.2f}% in cluster refinement step.\n')

