import numpy as np
from scipy.io import savemat, loadmat
import glob
import faiss
import pandas as pd
from pathlib import Path


def combine_and_cluster(embeddings_dir, n_clusters=3000):
    """
    Load all .mat files from specified directory, combine embeddings, and perform clustering

    Parameters:
    -----------
    embeddings_dir : str
        Directory containing the embedding .mat files and where results will be saved
    n_clusters : int
        Number of clusters to create
    """
    np.random.seed(5)

    embeddings_dir = Path(embeddings_dir)
    if not embeddings_dir.exists():
        raise ValueError(f"Directory {embeddings_dir} does not exist")

    mat_files = sorted(list(embeddings_dir.glob("embeddings_batch_*.mat")))

    all_image_embeddings = []
    all_text_embeddings = []
    all_image_paths = []
    all_scientific_names = []
    all_categories = []

    # Load and clean data from each file
    for mat_file in mat_files:
        print(f"Loading {mat_file}")
        data = loadmat(mat_file)
        all_image_embeddings.extend(data["image_embeddings"])
        all_text_embeddings.extend(data["text_embeddings"])

        # Clean strings when loading
        all_image_paths.extend([str(p).strip() for p in data["image_paths"]])
        all_scientific_names.extend([str(s).strip() for s in data["scientific_names"]])
        all_categories.extend([str(c).strip() for c in data["categories"]])

    image_embeddings = np.array(all_image_embeddings)
    text_embeddings = np.array(all_text_embeddings)

    joint_embeddings = (image_embeddings + text_embeddings) / 2

    n, d = joint_embeddings.shape
    n_clusters = min(n_clusters, n)

    print(f"Clustering {n} samples into {n_clusters} clusters...")
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True)
    kmeans.train(joint_embeddings)

    D, I = kmeans.index.search(joint_embeddings, 1)

    # Create DataFrame with cleaned data
    results_df = pd.DataFrame(
        {
            "image_path": [p.strip() for p in all_image_paths],
            "scientific_name": [s.strip() for s in all_scientific_names],
            "category": [c.strip() for c in all_categories],
            "cluster_id": I.flatten(),
            "distance_to_centroid": D.flatten(),
        }
    )

    csv_filename = embeddings_dir / f"clustering_results_{n_clusters}_clusters.csv"
    mat_filename = embeddings_dir / f"clustering_results_{n_clusters}_clusters.mat"

    # Save CSV with clean formatting
    results_df.to_csv(csv_filename, index=False, na_rep="NA")
    print(f"Saved {csv_filename}")

    # Clean strings before saving to mat
    save_dict = {
        "cluster_ids": I.flatten(),
        "distances": D.flatten(),
        "image_paths": np.array([p.strip() for p in all_image_paths]),
        "scientific_names": np.array([s.strip() for s in all_scientific_names]),
        "categories": np.array([c.strip() for c in all_categories]),
    }
    savemat(mat_filename, save_dict)
    print(f"Saved {mat_filename}")

    return results_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster embeddings from .mat files")
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        required=True,
        help="Directory containing embedding .mat files",
    )
    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3000,
        help="Number of clusters (default: 3000)",
    )

    args = parser.parse_args()

    results = combine_and_cluster(
        embeddings_dir=args.embeddings_dir, n_clusters=args.n_clusters
    )
