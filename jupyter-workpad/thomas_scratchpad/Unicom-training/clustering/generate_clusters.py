import numpy as np
from scipy.io import savemat, loadmat
import glob
import faiss
import pandas as pd


def combine_and_cluster(n_clusters=3000):
    """
    Load all .mat files, combine embeddings, and perform clustering
    """
    # Get all mat files -- Performance??????
    mat_files = sorted(glob.glob("embeddings_batch_*.mat"))  # Sort to maintain order

    # Lists to store all embeddings and metadata
    all_image_embeddings = []
    all_text_embeddings = []
    all_image_paths = []
    all_scientific_names = []
    all_categories = []

    # Load each file
    for mat_file in mat_files:
        print(f"Loading {mat_file}")
        data = loadmat(mat_file)
        all_image_embeddings.extend(data["image_embeddings"])
        all_text_embeddings.extend(data["text_embeddings"])
        all_image_paths.extend(data["image_paths"])
        all_scientific_names.extend(data["scientific_names"])
        all_categories.extend(data["categories"])

    # Convert to numpy arrays
    image_embeddings = np.array(all_image_embeddings)
    text_embeddings = np.array(all_text_embeddings)

    # Create joint embeddings
    joint_embeddings = (image_embeddings + text_embeddings) / 2

    # Perform clustering
    n, d = joint_embeddings.shape
    n_clusters = min(n_clusters, n)

    print(f"Clustering {n} samples into {n_clusters} clusters...")
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True)
    kmeans.train(joint_embeddings)

    # Get cluster assignments
    D, I = kmeans.index.search(joint_embeddings, 1)

    # Create results DataFrame
    results_df = pd.DataFrame(
        {
            "image_path": all_image_paths,
            "scientific_name": all_scientific_names,
            "category": all_categories,
            "cluster_id": I.flatten(),
            "distance_to_centroid": D.flatten(),
        }
    )

    # Save results
    results_df.to_csv("clustering_results.csv", index=False)
    print("Saved clustering_results.csv")

    # Also save as .mat if needed
    save_dict = {
        "cluster_ids": I.flatten(),
        "distances": D.flatten(),
        "image_paths": all_image_paths,
        "scientific_names": all_scientific_names,
        "categories": all_categories,
    }
    savemat("clustering_results.mat", save_dict)
    print("Saved clustering_results.mat")

    return results_df


if __name__ == "__main__":
    results = combine_and_cluster(n_clusters=3000)
