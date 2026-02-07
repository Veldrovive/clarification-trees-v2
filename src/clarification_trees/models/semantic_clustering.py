"""
Uses a sentence transformer to cluster LLM outputs by semantic similarity
"""

from omegaconf import DictConfig
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

class SemanticClusterer:
    def __init__(self, semantic_cluster_cfg: DictConfig, device: str):
        self.sentence_transformers_key = semantic_cluster_cfg.sentence_transformers_key
        self.similarity_threshold = semantic_cluster_cfg.similarity_threshold
        self.clustering_method = semantic_cluster_cfg.clustering_method
        self.exemplar_selection_method = semantic_cluster_cfg.exemplar_selection_method
        self.device = device

        if self.clustering_method == "agglomerative":
            self.setup_agglomerative_clustering()
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        self.model = SentenceTransformer(self.sentence_transformers_key, device=self.device)

    def setup_agglomerative_clustering(self):
        dist_threshold = 1.0 - self.similarity_threshold
        self.clusterer = AgglomerativeClustering(
            n_clusters=None,          # Find number of clusters automatically
            distance_threshold=dist_threshold, 
            metric='precomputed',     # We are providing the distance matrix
            linkage='complete'        # 'complete' linkage ensures ALL items in a cluster 
                                      # are close to each other (avoids chaining)
        )

    def run_agglomerative_clustering(self, embeddings: np.ndarray) -> list[list[int]]:
        """
        Runs agglomerative clustering on the embeddings.
        Embeddings are of shape (n_samples, n_features).
        Returns a list of lists of indices of the clusters.
        """
        n_samples = embeddings.shape[0]

        cosine_sim_matrix = np.inner(embeddings, embeddings)
        cosine_dist_matrix = 1.0 - cosine_sim_matrix
        cosine_dist_matrix = np.maximum(cosine_dist_matrix, 0.0)
        
        cluster_ids = self.clusterer.fit_predict(cosine_dist_matrix)
        n_clusters = len(set(cluster_ids))
        
        clusters = [[] for _ in range(n_clusters)]
        for i, cluster_id in enumerate(cluster_ids):
            clusters[cluster_id].append(i)
        
        return clusters

    def cluster(self, texts: list[str]) -> tuple[list[list[str]], list[str]]:
        """
        Clusters the texts by semantic similarity.
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

        if self.clustering_method == "agglomerative":
            cluster_indices = self.run_agglomerative_clustering(embeddings)
        else:
            raise ValueError(f"Unknown clustering method: {self.clustering_method}")

        clusters = [[] for _ in range(len(cluster_indices))]
        semantic_centers = []
        for cluster_index, clustered_indices in enumerate(cluster_indices):
            for index in clustered_indices:
                clusters[cluster_index].append(texts[index])

            if self.exemplar_selection_method == "closest_to_mean":
                # Compute the semantic center as the embeddings that is closest to the mean embedding for this cluster
                if len(clustered_indices) == 1:
                    semantic_center_idx = 0
                elif len(clustered_indices) == 2:
                    # Then there is no way to tell which is better so we take the longer one
                    if len(texts[clustered_indices[0]]) > len(texts[clustered_indices[1]]):
                        semantic_center_idx = 0
                    else:
                        semantic_center_idx = 1
                else:
                    # Now we can actually compute the mean embedding
                    mean_embedding = np.mean(embeddings[clustered_indices], axis=0)
                    distances = np.linalg.norm(embeddings[clustered_indices] - mean_embedding, axis=1)
                    semantic_center_idx = np.argmin(distances)
            elif self.exemplar_selection_method == "longest":
                semantic_center_idx = np.argmax([len(texts[i]) for i in clustered_indices])
            elif self.exemplar_selection_method == "shortest":
                semantic_center_idx = np.argmin([len(texts[i]) for i in clustered_indices])
            elif self.exemplar_selection_method == "random":
                semantic_center_idx = np.random.choice(clustered_indices)
            else:
                raise ValueError(f"Unknown exemplar selection method: {self.exemplar_selection_method}")
            
            semantic_centers.append(texts[clustered_indices[semantic_center_idx]])
        
        return clusters, semantic_centers

if __name__ == "__main__":
    import dotenv
    dotenv.load_dotenv()

    test_semantic_clustering_config = DictConfig({
        "sentence_transformers_key": "all-MiniLM-L6-v2",
        "similarity_threshold": 0.85,
        "clustering_method": "agglomerative"
    })
    semantic_clusterer = SemanticClusterer(test_semantic_clustering_config, "cuda:7")
    texts = ["What color is the image?", "What color is the image?", "What color is the picture?", "Is that a dog?", "Is that a cat?"]
    clusters, semantic_centers = semantic_clusterer.cluster(texts)
    print(clusters)
    print(semantic_centers)