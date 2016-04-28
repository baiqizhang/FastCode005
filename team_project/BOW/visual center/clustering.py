# Load all the sift features
import pickle
with open("/home/hadoop/features.pickle", "rb") as f:
    features = pickle.load(f)

# Clustering
import time
import sklearn
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(max_iter=100, n_clusters=10000)
start = time.time()
print "Start clustering"
kmeans.fit(features)

end = time.time()
print end - start
with open("cluster_centers.pickle", "wb") as f:
    pickle.dump(kmeans.cluster_centers_, f)





