import pandas as pd
from sklearn.cluster import KMeans


class ClusterModule:
    def __init__(self, all_items):
        self.all_items = all_items

    def cluster_categories(self, affinity, k):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(affinity.values)
        df = pd.DataFrame({"Item": self.all_items, "Cluster": labels})
        df = df.sort_values("Cluster").reset_index(drop=True)
        clustered_items = df["Item"].tolist()
        return clustered_items
