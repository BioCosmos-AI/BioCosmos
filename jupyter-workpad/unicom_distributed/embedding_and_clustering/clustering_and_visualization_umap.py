#!/usr/bin/env python3
import os
import sqlite3
import argparse
import logging
import numpy as np
import pandas as pd
import time
import json
from datetime import datetime
from tqdm import tqdm
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import signal
import sys
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import tarfile
import io
from PIL import Image

# Import UMAP for dimension reduction
import umap

# Try to import GPU libraries, but fall back gracefully if not available
try:
    import cuml
    from cuml.manifold import UMAP as cuUMAP
    from cuml.cluster import KMeans as cuKMeans
    from cuml.metrics import pairwise_distances

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"clustering_and_viz_umap_{timestamp}.log")

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger


def load_data_for_species(db_path, table_name, target_species, logger):
    """Load data for specific species from SQLite database."""
    logger.info(
        f"Loading data for {len(target_species)} specific species from {db_path}"
    )

    try:
        # Format species list for SQL query
        species_str = ", ".join(f"'{species}'" for species in target_species)

        # Connect to database
        connection = sqlite3.connect(db_path, timeout=300.0)

        # Query for specified species
        query = f"""
        SELECT uuid, shard_id, species_name, embedding
        FROM {table_name}
        WHERE species_name IN ({species_str})
        """

        logger.info(f"Executing query: {query}")

        # Load data
        df = pd.read_sql_query(query, connection)
        logger.info(
            f"Retrieved {len(df)} rows for {len(df['species_name'].unique())} species"
        )

        # Create embeddings dictionary
        embeddings_dict = {}
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Processing embeddings"):
            try:
                if row["embedding"] is not None:
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                    embeddings_dict[row["uuid"]] = embedding
            except Exception as e:
                logger.warning(f"Error with embedding for UUID {row['uuid']}: {str(e)}")

        connection.close()
        logger.info(f"Processed {len(embeddings_dict)} valid embeddings")

        return df, embeddings_dict

    except Exception as e:
        logger.error(f"Error loading species data: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return pd.DataFrame(), {}


def find_optimal_k(embeddings, k_range, use_gpu, logger):
    """Find optimal number of clusters using silhouette scores."""
    best_score = -1
    best_k = 2  # Default to 2 clusters

    logger.info(f"Finding optimal k in range {k_range}")

    # Try GPU implementation if available
    if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
        try:
            logger.info("Using GPU-accelerated K-means for hyperparameter search")

            for k in range(k_range[0], k_range[1] + 1):
                try:
                    # Use cuML KMeans
                    kmeans = cuKMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)

                    # Convert to NumPy if needed
                    if hasattr(labels, "get"):
                        labels_cpu = labels.get()
                    elif hasattr(labels, "to_numpy"):
                        labels_cpu = labels.to_numpy()
                    elif isinstance(labels, torch.Tensor):
                        labels_cpu = labels.cpu().numpy()
                    else:
                        labels_cpu = np.array(labels)

                    # Convert embeddings to CPU if needed
                    if hasattr(embeddings, "get"):
                        embeddings_cpu = embeddings.get()
                    elif hasattr(embeddings, "to_numpy"):
                        embeddings_cpu = embeddings.to_numpy()
                    elif isinstance(embeddings, torch.Tensor):
                        embeddings_cpu = embeddings.cpu().numpy()
                    else:
                        embeddings_cpu = np.array(embeddings)

                    # Skip if only one cluster
                    if len(np.unique(labels_cpu)) < 2:
                        continue

                    score = silhouette_score(embeddings_cpu, labels_cpu)
                    logger.info(f"K={k}, Silhouette Score={score:.4f}")

                    if score > best_score:
                        best_score = score
                        best_k = k

                except Exception as e:
                    logger.error(f"Error evaluating k={k}: {str(e)}")
                    continue

            return best_k, best_score

        except Exception as e:
            logger.warning(f"GPU processing failed: {str(e)}, falling back to CPU")
            # Continue to CPU implementation

    # CPU implementation
    logger.info("Using CPU K-means for hyperparameter search")
    for k in range(k_range[0], k_range[1] + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            # Skip if only one cluster
            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels)
            logger.info(f"K={k}, Silhouette Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        except Exception as e:
            logger.error(f"Error evaluating k={k}: {str(e)}")
            continue

    return best_k, best_score


def generate_umap(embeddings, use_gpu, n_neighbors, min_dist, logger, n_components=2):
    """Generate UMAP representation using GPU if available."""
    logger.info(
        f"Generating {n_components}D UMAP projection for {len(embeddings)} samples"
    )

    # Adjust n_neighbors if needed (must be smaller than n_samples)
    adjusted_n_neighbors = min(n_neighbors, len(embeddings) - 1)
    if adjusted_n_neighbors < n_neighbors:
        logger.info(
            f"Adjusted n_neighbors from {n_neighbors} to {adjusted_n_neighbors} due to small dataset size"
        )

    # Try GPU implementation first if requested
    if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
        try:
            logger.info(
                f"Attempting GPU-accelerated UMAP with n_neighbors={adjusted_n_neighbors}, min_dist={min_dist}"
            )
            umap_reducer = cuUMAP(
                n_components=n_components,
                n_neighbors=adjusted_n_neighbors,
                min_dist=min_dist,
                random_state=42,
            )
            umap_result = umap_reducer.fit_transform(embeddings)

            # Convert to NumPy if needed
            if hasattr(umap_result, "get"):
                umap_result = umap_result.get()
            elif hasattr(umap_result, "to_numpy"):
                umap_result = umap_result.to_numpy()
            elif isinstance(umap_result, torch.Tensor):
                umap_result = umap_result.cpu().numpy()
            else:
                umap_result = np.array(umap_result)

            if umap_result is not None:
                # Min-max normalization to [0, 1] range
                umap_min = np.min(umap_result, axis=0)
                umap_max = np.max(umap_result, axis=0)
                umap_result = (umap_result - umap_min) / (umap_max - umap_min + 1e-8)

            logger.info("Successfully used GPU-accelerated UMAP")
            return umap_result
        except Exception as e:
            logger.warning(f"GPU UMAP failed, falling back to CPU: {str(e)}")

    # Fall back to CPU UMAP
    try:
        logger.info(
            f"Using CPU UMAP with n_neighbors={adjusted_n_neighbors}, min_dist={min_dist}"
        )
        umap_reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=adjusted_n_neighbors,
            min_dist=min_dist,
            random_state=42,
        )
        umap_result = umap_reducer.fit_transform(embeddings)
        if umap_result is not None:
            # Min-max normalization to [0, 1] range
            umap_min = np.min(umap_result, axis=0)
            umap_max = np.max(umap_result, axis=0)
            umap_result = (umap_result - umap_min) / (umap_max - umap_min + 1e-8)

        logger.info("Successfully used CPU UMAP")
        return umap_result
    except Exception as e:
        logger.error(f"CPU UMAP failed: {str(e)}")
        return None


def perform_umap_and_clustering(
    species_df, embeddings_dict, min_k, max_k, n_neighbors, min_dist, use_gpu, logger
):
    """Perform UMAP and clustering, keeping UMAP coordinates for visualization."""
    species_name = species_df["species_name"].iloc[0]
    logger.info(f"Processing {species_name} with {len(species_df)} samples")

    # Extract embeddings for this species
    uuids = species_df["uuid"].tolist()
    embeddings_list = []
    valid_indices = []

    for i, uuid in enumerate(uuids):
        if uuid in embeddings_dict:
            emb = embeddings_dict[uuid]
            # Check for NaN or zero embeddings
            if not np.isnan(emb).any() and not (emb == 0).all():
                embeddings_list.append(emb)
                valid_indices.append(i)

    if len(embeddings_list) < min_k:
        logger.warning(
            f"Not enough valid embeddings for {species_name} (found {len(embeddings_list)}, need {min_k})"
        )
        return None, None, None

    embeddings = np.array(embeddings_list)
    valid_df_indices = species_df.index[valid_indices]

    # Get valid dataframe with only rows that have valid embeddings
    valid_df = species_df.iloc[valid_indices].copy().reset_index(drop=True)

    # Compute UMAP
    logger.info(f"Computing UMAP for {species_name} ({len(embeddings)} samples)")
    umap_result = generate_umap(embeddings, use_gpu, n_neighbors, min_dist, logger)

    if umap_result is None or len(umap_result) == 0:
        logger.error(f"UMAP failed to produce output for {species_name}")
        return None, None, None

    # Add UMAP coordinates to dataframe
    valid_df["x"] = umap_result[:, 0]
    valid_df["y"] = umap_result[:, 1]

    # Find optimal number of clusters
    logger.info(f"Finding optimal k for {species_name}")
    optimal_k, score = find_optimal_k(umap_result, [min_k, max_k], use_gpu, logger)
    score_formatted = f"{score:.4f}" if score is not None else "0.0000"
    logger.info(f"Optimal k for {species_name}: {optimal_k} (score: {score_formatted})")

    # Perform final clustering
    logger.info(f"Performing final clustering with k={optimal_k}")
    clusters = None

    # Try GPU K-means
    if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
        try:
            kmeans = cuKMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(umap_result)

            # Convert to NumPy if needed
            if hasattr(clusters, "get"):
                clusters = clusters.get()
            elif hasattr(clusters, "to_numpy"):
                clusters = clusters.to_numpy()
            elif isinstance(clusters, torch.Tensor):
                clusters = clusters.cpu().numpy()
            else:
                clusters = np.array(clusters)

            logger.info("Used GPU-accelerated K-means for final clustering")
        except Exception as e:
            logger.warning(f"GPU K-means failed, falling back to CPU: {str(e)}")
            clusters = None

    # Fall back to CPU K-means
    if clusters is None:
        try:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(umap_result)
            logger.info("Used CPU K-means for final clustering")
        except Exception as e:
            logger.error(f"CPU K-means failed: {str(e)}")
            # Assign all to cluster 0 as fallback
            clusters = np.zeros(len(embeddings), dtype=int)

    # Add cluster assignments to dataframe
    valid_df["cluster"] = clusters

    # Create statistics
    stats = {
        "count": len(valid_df),
        "method": "umap_clustering",
        "optimal_k": optimal_k,
        "silhouette_score": score,
    }

    return valid_df, stats, embeddings


def extract_image_from_tar(uuid, shard_id, data_dir, logger):
    """Extract an image from a WebDataset tar file based on UUID and shard_id."""
    # Format shard filename with leading zeros
    try:
        shard_num = int(shard_id)
        shard_filename = f"shard-{shard_num:06d}.tar"
    except (ValueError, TypeError):
        logger.error(f"Invalid shard_id: {shard_id}")
        return None

    tar_path = os.path.join(data_dir, shard_filename)

    logger.info(f"Extracting image for UUID {uuid} from shard {shard_filename}")

    try:
        # Check if tar file exists
        if not os.path.exists(tar_path):
            logger.error(f"Tar file not found: {tar_path}")
            return None

        # Open tar file
        with tarfile.open(tar_path, "r") as tar:
            # Try different image extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                image_name = f"{uuid}{ext}"
                try:
                    # Extract the file if it exists
                    member = tar.getmember(image_name)
                    f = tar.extractfile(member)
                    if f:
                        img_data = f.read()
                        img = Image.open(io.BytesIO(img_data))
                        return img
                except (KeyError, tarfile.ReadError):
                    # File not found with this extension, try next
                    continue

            # If we got here, we didn't find any image file
            logger.warning(f"No image file found for UUID {uuid} in {tar_path}")
            return None

    except Exception as e:
        logger.error(f"Error extracting image: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def get_cluster_representatives(df, embeddings, umap_result, logger):
    """Find representative samples for each cluster (closest to centroid)."""
    logger.info("Finding representative samples for each cluster")

    # Get unique clusters
    clusters = sorted(df["cluster"].unique())
    representatives = {}

    for cluster_id in clusters:
        logger.info(f"Finding representative for cluster {cluster_id}")

        # Get indices for this cluster
        cluster_mask = df["cluster"] == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # If only one sample, it's the representative
        if len(cluster_indices) == 1:
            idx = cluster_indices[0]
            representatives[cluster_id] = {
                "index": idx,
                "uuid": df.iloc[idx]["uuid"],
                "shard_id": df.iloc[idx]["shard_id"],
                "position": (df.iloc[idx]["x"], df.iloc[idx]["y"]),
            }
            continue

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]

        # Calculate cluster centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find the sample closest to the centroid
        distances = np.sqrt(np.sum((cluster_embeddings - centroid) ** 2, axis=1))
        closest_idx = cluster_indices[np.argmin(distances)]

        # Get position in UMAP space
        position = (df.iloc[closest_idx]["x"], df.iloc[closest_idx]["y"])

        representatives[cluster_id] = {
            "index": closest_idx,
            "uuid": df.iloc[closest_idx]["uuid"],
            "shard_id": df.iloc[closest_idx]["shard_id"],
            "position": position,
        }

    logger.info(f"Found {len(representatives)} representative samples")
    return representatives


def generate_visualization(
    species_name,
    species_df,
    representatives,
    data_dir,
    output_dir,
    logger,
    jitter_amount=0.05,
):
    """Generate and save UMAP visualization for a species with representative images."""
    logger.info(f"Generating visualization for {species_name}")

    # Add jitter to spread out points
    x_range = species_df["x"].max() - species_df["x"].min()
    y_range = species_df["y"].max() - species_df["y"].min()

    species_df["x_jittered"] = species_df["x"] + np.random.normal(
        0, jitter_amount * x_range, len(species_df)
    )
    species_df["y_jittered"] = species_df["y"] + np.random.normal(
        0, jitter_amount * y_range, len(species_df)
    )

    # Set up plot
    plt.figure(figsize=(14, 12))
    sns.set_style("whitegrid")

    # Get unique clusters and create color palette
    unique_clusters = sorted(species_df["cluster"].unique())
    num_clusters = len(unique_clusters)

    if num_clusters > 10:
        colors = sns.color_palette("husl", num_clusters)
    else:
        colors = sns.color_palette("Set1", num_clusters)

    cmap = ListedColormap(colors)

    # Create scatter plot with cluster colors
    scatter = plt.scatter(
        # species_df["x_jittered"],
        # species_df["y_jittered"],
        species_df["x"],
        species_df["y"],
        c=species_df["cluster"],
        cmap=cmap,
        s=50,
        alpha=0.6,
        edgecolor="k",
        linewidth=0.5,
    )

    # Add legend
    legend_elements = []
    for i, cluster_id in enumerate(unique_clusters):
        label = f"Cluster {int(cluster_id)}"
        legend_elements.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=colors[i % len(colors)],
                markersize=10,
                label=f"{label} ({sum(species_df['cluster'] == cluster_id)} samples)",
            )
        )

    plt.legend(
        handles=legend_elements,
        title="Clusters",
        loc="best",
        fontsize="medium",
        title_fontsize="large",
    )

    # Calculate appropriate image size based on number of clusters
    img_zoom = max(0.1, min(0.7, 1.0 / (max(len(representatives), 1) ** 0.5)))
    logger.info(f"Using image zoom factor: {img_zoom:.2f}")

    # Add representative images
    if representatives:
        for cluster_id, rep in representatives.items():
            logger.info(f"Adding representative image for cluster {cluster_id}")

            # Extract image from tar file
            img = extract_image_from_tar(rep["uuid"], rep["shard_id"], data_dir, logger)

            if img is not None:
                # Create OffsetImage
                imagebox = OffsetImage(img, zoom=img_zoom)

                # Create annotation box - use original coordinates, not jittered ones
                edge_color = colors[
                    list(sorted(unique_clusters)).index(cluster_id) % len(colors)
                ]

                ab = AnnotationBbox(
                    imagebox,
                    rep["position"],
                    pad=0.0,
                    frameon=True,
                    bboxprops=dict(edgecolor=edge_color, linewidth=3),
                )

                # Add to plot
                plt.gca().add_artist(ab)
            else:
                logger.warning(
                    f"Could not add image for cluster {cluster_id} (UUID: {rep['uuid']})"
                )
    else:
        logger.warning("No representative images found to display")

    # Get cluster counts
    cluster_counts = species_df["cluster"].value_counts().to_dict()
    cluster_info = ", ".join(
        [f"Cluster {int(k)}: {v}" for k, v in cluster_counts.items()]
    )

    # Titles and labels
    plt.title(
        f"UMAP Visualization of {species_name}\n({len(species_df)} samples)",
        fontsize=16,
    )
    plt.xlabel("UMAP dimension 1", fontsize=14)
    plt.ylabel("UMAP dimension 2", fontsize=14)

    # Add a text box with cluster counts
    if cluster_info:
        plt.figtext(
            0.5,
            0.01,
            f"Cluster counts: {cluster_info}",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
        )

    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    safe_name = species_name.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"{safe_name}_umap_with_images.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")
    return output_path


def numpy_to_python_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: numpy_to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_python_types(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_python_types(i) for i in obj)
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description="UMAP-based clustering and visualization for specific species"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--table-name",
        type=str,
        default="image_embeddings",
        help="Name of table in SQLite",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory for log files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train",
        help="Directory containing WebDataset tar files",
    )
    parser.add_argument("--min-k", type=int, default=2, help="Minimum clusters to try")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum clusters to try")
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="UMAP n_neighbors parameter (local neighborhood size)",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="UMAP min_dist parameter (minimum distance between points)",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration if available",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=[
            "Abagrotis alternata",
            "Abaeis nicippe",
            "Hemicircus canente",
            "Hemigomphus comitatus",
            "Zyrphelis crenata",
            "Zygaena oxytropis",
        ],
        help="List of species to analyze",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting UMAP-based clustering and visualization script")

    # Check GPU availability
    if args.use_gpu:
        if torch.cuda.is_available():
            logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            if CUML_AVAILABLE:
                logger.info("cuML is available for GPU acceleration")
            else:
                logger.info(
                    "cuML is not available, will use GPU only for PyTorch operations"
                )
        else:
            logger.warning("CUDA not available, falling back to CPU processing")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data for target species
    df, embeddings_dict = load_data_for_species(
        args.db_path, args.table_name, args.species, logger
    )

    if len(df) == 0:
        logger.error("No data loaded. Exiting.")
        return

    # Process each species
    all_species_stats = {}

    for species_name in args.species:
        logger.info(f"Processing species: {species_name}")

        # Get data for this species
        species_df = df[df["species_name"] == species_name]

        if len(species_df) == 0:
            logger.warning(f"No data found for species {species_name}")
            continue

        # Perform UMAP and clustering
        processed_df, stats, embeddings = perform_umap_and_clustering(
            species_df,
            embeddings_dict,
            args.min_k,
            args.max_k,
            args.n_neighbors,
            args.min_dist,
            args.use_gpu,
            logger,
        )

        if processed_df is None or len(processed_df) == 0:
            logger.warning(f"Processing failed for {species_name}")
            continue

        # Store statistics
        all_species_stats[species_name] = stats

        # Find representative samples
        umap_result = np.column_stack((processed_df["x"], processed_df["y"]))
        representatives = get_cluster_representatives(
            processed_df, embeddings, umap_result, logger
        )

        # Generate visualization
        vis_path = generate_visualization(
            species_name,
            processed_df,
            representatives,
            args.data_dir,
            args.output_dir,
            logger,
        )

    # Save statistics
    stats_path = os.path.join(args.output_dir, "clustering_stats_umap.json")

    # Convert numpy types to standard Python types for JSON serialization
    converted_stats = numpy_to_python_types(all_species_stats)

    with open(stats_path, "w") as f:
        json.dump(converted_stats, f, indent=2)

    logger.info(f"Saved statistics to {stats_path}")
    logger.info("All processing completed successfully")


if __name__ == "__main__":
    main()
