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
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
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
import traceback

# Try to import HDBSCAN, handle cases where it's not installed
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available. HDBSCAN clustering will be skipped.")


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sundry_tsne_{timestamp}.log")

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
        logger.info(f"Connecting to database at {db_path}")
        connection = sqlite3.connect(db_path, timeout=300.0)
        logger.info("Connected to database successfully")

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
        logger.error(traceback.format_exc())
        return pd.DataFrame(), {}


def normalize_embeddings(embeddings):
    """Normalize embeddings to unit length for cosine similarity."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-10)  # Avoid division by zero


def generate_tsne(embeddings, logger, n_components=2):
    """Generate t-SNE representation."""
    logger.info(
        f"Generating {n_components}D t-SNE projection for {len(embeddings)} samples"
    )

    try:
        tsne = TSNE(n_components=n_components, random_state=42)
        logger.info("Starting t-SNE fit_transform - this may take a while...")
        tsne_result = tsne.fit_transform(embeddings)

        logger.info(
            f"Successfully generated t-SNE projection with shape {tsne_result.shape}"
        )
        # Verify no NaN values in result
        if np.isnan(tsne_result).any():
            logger.warning("t-SNE result contains NaN values!")
            # Replace NaNs with zeros as a fallback
            tsne_result = np.nan_to_num(tsne_result)

        return tsne_result
    except Exception as e:
        logger.error(f"t-SNE projection failed: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def kmeans_clustering(data, min_k, max_k, logger):
    """Perform k-means clustering with silhouette score optimization."""
    logger.info(f"Performing K-means clustering with k range [{min_k}-{max_k}]")

    best_score = -1
    best_k = min_k
    best_labels = None

    for k in range(min_k, max_k + 1):
        try:
            logger.info(f"Trying K-means with k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)
            logger.info(f"K-means with k={k} completed successfully")

            # Skip if only one cluster
            if len(np.unique(labels)) < 2:
                logger.warning(
                    f"K-means with k={k} produced only one cluster, skipping"
                )
                continue

            score = silhouette_score(data, labels)
            logger.info(f"K={k}, Silhouette Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels

        except Exception as e:
            logger.error(f"Error during K-means with k={k}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    logger.info(f"Optimal k: {best_k} with silhouette score: {best_score:.4f}")

    # If optimization failed, use minimum k
    if best_labels is None:
        logger.warning(f"K-means optimization failed, defaulting to k={min_k}")
        try:
            kmeans = KMeans(n_clusters=min_k, random_state=42, n_init=10)
            best_labels = kmeans.fit_predict(data)
            best_k = min_k
        except Exception as e:
            logger.error(f"Error during fallback K-means with k={min_k}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return dummy labels as last resort
            best_labels = np.zeros(len(data), dtype=int)

    return best_labels, best_k, best_score


def hierarchical_clustering(data, strategy, logger):
    """Perform hierarchical clustering with specified strategy."""
    logger.info(f"Performing hierarchical clustering with {strategy} strategy")

    try:
        # Compute linkage matrix
        logger.info("Computing linkage matrix...")
        Z = linkage(data, method="ward")
        logger.info("Linkage matrix computed successfully")

        if strategy == "silhouette":
            # Find optimal threshold using silhouette scores
            dists = np.sort(np.unique(Z[:, 2]))
            logger.info(f"Testing {len(dists)} potential distance thresholds")
            best_score, best_threshold = -1, None

            for i, t in enumerate(dists):
                if i % 10 == 0:  # Log progress every 10th iteration
                    logger.info(f"Testing threshold {i+1}/{len(dists)}: {t:.2f}")

                clusters_temp = fcluster(Z, t=t, criterion="distance")
                n_clusters = len(np.unique(clusters_temp))

                if n_clusters < 2:
                    continue

                score = silhouette_score(data, clusters_temp)
                if i % 10 == 0:  # Log progress
                    logger.info(
                        f"Threshold={t:.2f}, Clusters={n_clusters}, Score={score:.4f}"
                    )

                if score > best_score:
                    best_score, best_threshold = score, t

            if best_threshold is None:
                # Fallback to a reasonable threshold
                best_threshold = np.median(Z[:, 2])
                logger.warning(
                    f"Silhouette analysis failed; using median threshold: {best_threshold:.2f}"
                )
            else:
                logger.info(
                    f"Optimal threshold: {best_threshold:.2f} with silhouette score: {best_score:.4f}"
                )

        elif strategy == "gap":
            # Use gap statistic approach (find biggest gap in distances)
            sorted_dists = np.sort(Z[:, 2])
            gaps = np.diff(sorted_dists)
            max_gap_index = np.argmax(gaps)
            best_threshold = sorted_dists[max_gap_index + 1]
            logger.info(
                f"Gap-based threshold: {best_threshold:.2f} (max gap: {gaps[max_gap_index]:.2f})"
            )
            best_score = None
        else:
            logger.error(f"Unknown hierarchical strategy: {strategy}")
            return None, None, None

        # Apply clustering with the determined threshold
        logger.info(f"Applying final clustering with threshold={best_threshold:.2f}")
        labels = fcluster(Z, t=best_threshold, criterion="distance")
        n_clusters = len(np.unique(labels))
        logger.info(f"Final clustering produced {n_clusters} clusters")

        # Compute silhouette score if not already done
        if best_score is None and len(np.unique(labels)) >= 2:
            best_score = silhouette_score(data, labels)
            logger.info(f"Final silhouette score: {best_score:.4f}")

        return labels, best_threshold, best_score

    except Exception as e:
        logger.error(
            f"Error in hierarchical clustering with {strategy} strategy: {str(e)}"
        )
        logger.error(traceback.format_exc())
        # Return dummy labels as fallback
        return np.zeros(len(data), dtype=int), 0.0, -1.0


def dbscan_clustering(data, min_samples, eps_range, logger):
    """Perform DBSCAN clustering with eps optimization."""
    logger.info(
        f"Performing DBSCAN clustering with eps range [{min(eps_range):.2f}-{max(eps_range):.2f}]"
    )

    best_score = -1
    best_eps = None
    best_labels = None

    for eps in eps_range:
        try:
            logger.info(f"Trying DBSCAN with eps={eps:.2f}, min_samples={min_samples}")
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(data)

            # Count clusters and noise points
            n_clusters = len(np.unique(labels[labels != -1]))
            n_noise = np.sum(labels == -1)
            logger.info(
                f"DBSCAN with eps={eps:.2f} found {n_clusters} clusters and {n_noise} noise points"
            )

            # Skip if all noise or only one cluster (excluding noise)
            if n_clusters < 2:
                logger.info(
                    f"Eps={eps:.2f} - Found only {n_clusters} clusters (excluding noise), skipping"
                )
                continue

            # Calculate silhouette score (ignore noise points with label -1)
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) < 2:
                logger.info(
                    f"Eps={eps:.2f} - Not enough non-noise points for silhouette score, skipping"
                )
                continue

            score = silhouette_score(data[non_noise_mask], labels[non_noise_mask])
            logger.info(
                f"Eps={eps:.2f}, Clusters={n_clusters}, Noise Points={n_noise}, Silhouette Score={score:.4f}"
            )

            if score > best_score:
                best_score = score
                best_eps = eps
                best_labels = labels

        except Exception as e:
            logger.error(f"Error during DBSCAN with eps={eps}: {str(e)}")
            logger.error(traceback.format_exc())
            continue

    if best_labels is None:
        logger.warning("DBSCAN optimization failed, using median eps value")
        eps = np.median(eps_range)
        try:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            best_labels = dbscan.fit_predict(data)
            best_eps = eps

            # Try to compute score for the fallback
            non_noise_mask = best_labels != -1
            if np.sum(non_noise_mask) >= 2:
                best_score = silhouette_score(
                    data[non_noise_mask], best_labels[non_noise_mask]
                )
            else:
                best_score = None
        except Exception as e:
            logger.error(f"Error during fallback DBSCAN with eps={eps}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return dummy labels as last resort
            best_labels = np.zeros(len(data), dtype=int) - 1  # All noise

    logger.info(
        f"Optimal DBSCAN eps: {best_eps:.2f}, Total clusters: {len(np.unique(best_labels[best_labels != -1]))}"
    )
    return best_labels, best_eps, best_score


def hdbscan_clustering(data, min_cluster_size, logger):
    """Perform HDBSCAN clustering."""
    if not HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN not available, skipping")
        return None, None, None

    logger.info(
        f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}"
    )

    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(data)

        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = np.sum(labels == -1)
        logger.info(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points")

        # Calculate silhouette score if possible
        score = None
        if n_clusters >= 2:
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) >= 2:
                score = silhouette_score(data[non_noise_mask], labels[non_noise_mask])
                logger.info(f"HDBSCAN Silhouette Score: {score:.4f}")

        return labels, min_cluster_size, score
    except Exception as e:
        logger.error(f"Error during HDBSCAN clustering: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None


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
        logger.error(traceback.format_exc())
        return None


def get_cluster_representatives(df, clusters, tsne_result, logger):
    """Find representative samples for each cluster (closest to centroid)."""
    logger.info("Finding representative samples for each cluster")

    unique_clusters = np.unique(clusters)
    # Skip noise cluster (-1) for DBSCAN/HDBSCAN
    if -1 in unique_clusters:
        unique_clusters = unique_clusters[unique_clusters != -1]

    representatives = {}

    for cluster_id in unique_clusters:
        logger.info(f"Finding representative for cluster {cluster_id}")

        # Get indices for this cluster
        cluster_mask = clusters == cluster_id
        cluster_indices = np.where(cluster_mask)[0]

        # If only one sample, it's the representative
        if len(cluster_indices) == 1:
            idx = cluster_indices[0]
            representatives[cluster_id] = {
                "index": idx,
                "uuid": df.iloc[idx]["uuid"],
                "shard_id": df.iloc[idx]["shard_id"],
                "position": (tsne_result[idx, 0], tsne_result[idx, 1]),
            }
            continue

        # Calculate centroid in t-SNE space
        cluster_coords = tsne_result[cluster_indices]
        centroid = np.mean(cluster_coords, axis=0)

        # Find closest point to centroid
        distances = np.sqrt(np.sum((cluster_coords - centroid) ** 2, axis=1))
        closest_idx = cluster_indices[np.argmin(distances)]

        # Get position in t-SNE space
        position = (tsne_result[closest_idx, 0], tsne_result[closest_idx, 1])

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
    tsne_result,
    clusters,
    method_name,
    parameter_info,
    score,
    representatives,
    data_dir,
    output_dir,
    logger,
    zoom=0.15,
):
    """Generate and save t-SNE visualization for a species with cluster coloring."""
    logger.info(f"Generating visualization for {species_name} using {method_name}")

    try:
        # Set up plot
        plt.figure(figsize=(14, 12))
        sns.set_style("whitegrid")

        # Get unique clusters and create color palette
        unique_clusters = np.unique(clusters)

        # Handle noise cluster for DBSCAN/HDBSCAN
        has_noise = -1 in unique_clusters
        if has_noise:
            # Get clusters excluding noise
            regular_clusters = unique_clusters[unique_clusters != -1]
            num_regular_clusters = len(regular_clusters)

            # Create colors: black for noise, colormap for others
            if num_regular_clusters > 10:
                colors = sns.color_palette("husl", num_regular_clusters)
            else:
                colors = sns.color_palette("Set1", max(num_regular_clusters, 1))

            # Create a mapping from cluster ID to color index
            cluster_to_color = {
                cluster_id: i for i, cluster_id in enumerate(regular_clusters)
            }
        else:
            # No noise cluster
            num_clusters = len(unique_clusters)
            if num_clusters > 10:
                colors = sns.color_palette("husl", num_clusters)
            else:
                colors = sns.color_palette("Set1", max(num_clusters, 1))

            # Create a mapping from cluster ID to color index
            cluster_to_color = {
                cluster_id: i for i, cluster_id in enumerate(unique_clusters)
            }

        # Create scatter plot with cluster colors
        for i, c in enumerate(unique_clusters):
            mask = clusters == c
            if not np.any(mask):
                logger.warning(f"No points in cluster {c}, skipping")
                continue

            if c == -1:  # Noise points
                plt.scatter(
                    tsne_result[mask, 0],
                    tsne_result[mask, 1],
                    c="black",
                    s=50,
                    alpha=0.6,
                    edgecolor="k",
                    linewidth=0.5,
                    label=f"Noise ({np.sum(mask)} points)",
                )
            else:
                color_idx = cluster_to_color[c]
                plt.scatter(
                    tsne_result[mask, 0],
                    tsne_result[mask, 1],
                    c=[colors[color_idx]],
                    s=50,
                    alpha=0.6,
                    edgecolor="k",
                    linewidth=0.5,
                    label=f"Cluster {int(c)} ({np.sum(mask)} points)",
                )

        # Add representative images if available
        if representatives:
            for cluster_id, rep in representatives.items():
                logger.info(f"Adding representative image for cluster {cluster_id}")

                # Extract image from tar file
                img = extract_image_from_tar(
                    rep["uuid"], rep["shard_id"], data_dir, logger
                )

                if img is not None:
                    # Create OffsetImage
                    imagebox = OffsetImage(img, zoom=zoom)

                    # Create annotation box
                    if cluster_id == -1:  # Noise cluster
                        edge_color = "black"
                    else:
                        edge_color = colors[cluster_to_color[cluster_id]]

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

        # Add a score value if available
        score_text = f"Score: {score:.4f}" if score is not None else "Score: N/A"

        # Titles and labels
        plt.title(
            f"t-SNE Visualization of {species_name}\n{method_name} Clustering: {parameter_info}, {score_text}",
            fontsize=16,
        )
        plt.xlabel("t-SNE dimension 1", fontsize=14)
        plt.ylabel("t-SNE dimension 2", fontsize=14)

        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        if handles:
            plt.legend(
                handles=handles,
                labels=labels,
                loc="best",
                fontsize="medium",
                title="Clusters",
            )
        else:
            logger.warning(
                "No legend handles found. This may indicate an issue with clustering."
            )

        plt.tight_layout()

        # Save the figure
        os.makedirs(output_dir, exist_ok=True)
        safe_name = species_name.replace(" ", "_").lower()
        output_path = os.path.join(
            output_dir, f"{safe_name}_{method_name.lower()}_tsne.png"
        )
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Saved visualization to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        logger.error(traceback.format_exc())
        return None


def process_species_data(
    species_name,
    species_df,
    embeddings_dict,
    min_k,
    max_k,
    dbscan_min_samples,
    dbscan_eps_range,
    hdbscan_min_cluster_size,
    data_dir,
    output_dir,
    logger,
):
    """Process a single species with multiple clustering methods."""
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

    if len(embeddings_list) < 2:
        logger.warning(
            f"Not enough valid embeddings for {species_name} (found {len(embeddings_list)}, need at least 2)"
        )
        return {}

    embeddings = np.array(embeddings_list)
    valid_df = species_df.iloc[valid_indices].copy().reset_index(drop=True)

    # Generate t-SNE projection
    logger.info(f"Computing t-SNE for {species_name} ({len(embeddings)} samples)")
    tsne_result = generate_tsne(embeddings, logger)

    if tsne_result is None or len(tsne_result) == 0:
        logger.error(f"t-SNE failed to produce output for {species_name}")
        return {}

    stats = {}
    method_results = {}

    # 1. K-means clustering
    logger.info(f"Starting K-means clustering for {species_name}")
    kmeans_labels, kmeans_k, kmeans_score = kmeans_clustering(
        tsne_result, min_k, max_k, logger
    )

    if kmeans_labels is not None:
        kmeans_reps = get_cluster_representatives(
            valid_df, kmeans_labels, tsne_result, logger
        )
        kmeans_vis = generate_visualization(
            species_name,
            valid_df,
            tsne_result,
            kmeans_labels,
            "KMeans",
            f"k={kmeans_k}",
            kmeans_score,
            kmeans_reps,
            data_dir,
            output_dir,
            logger,
        )
        method_results["kmeans"] = {
            "labels": kmeans_labels.tolist(),
            "k": kmeans_k,
            "score": kmeans_score,
            "vis_path": kmeans_vis,
        }

    # 2. Hierarchical clustering - silhouette
    logger.info(f"Starting hierarchical clustering (silhouette) for {species_name}")
    hier_sil_labels, hier_sil_threshold, hier_sil_score = hierarchical_clustering(
        tsne_result, "silhouette", logger
    )

    if hier_sil_labels is not None:
        hier_sil_reps = get_cluster_representatives(
            valid_df, hier_sil_labels, tsne_result, logger
        )
        hier_sil_vis = generate_visualization(
            species_name,
            valid_df,
            tsne_result,
            hier_sil_labels,
            "Hierarchical-Silhouette",
            f"threshold={hier_sil_threshold:.2f}",
            hier_sil_score,
            hier_sil_reps,
            data_dir,
            output_dir,
            logger,
        )
        method_results["hierarchical_silhouette"] = {
            "labels": hier_sil_labels.tolist(),
            "threshold": hier_sil_threshold,
            "score": hier_sil_score,
            "vis_path": hier_sil_vis,
        }

    # 3. Hierarchical clustering - gap
    logger.info(f"Starting hierarchical clustering (gap) for {species_name}")
    hier_gap_labels, hier_gap_threshold, hier_gap_score = hierarchical_clustering(
        tsne_result, "gap", logger
    )

    if hier_gap_labels is not None:
        hier_gap_reps = get_cluster_representatives(
            valid_df, hier_gap_labels, tsne_result, logger
        )
        hier_gap_vis = generate_visualization(
            species_name,
            valid_df,
            tsne_result,
            hier_gap_labels,
            "Hierarchical-Gap",
            f"threshold={hier_gap_threshold:.2f}",
            hier_gap_score,
            hier_gap_reps,
            data_dir,
            output_dir,
            logger,
        )
        method_results["hierarchical_gap"] = {
            "labels": hier_gap_labels.tolist(),
            "threshold": hier_gap_threshold,
            "score": hier_gap_score,
            "vis_path": hier_gap_vis,
        }

    # 4. DBSCAN clustering
    logger.info(f"Starting DBSCAN clustering for {species_name}")
    dbscan_labels, dbscan_eps, dbscan_score = dbscan_clustering(
        tsne_result, dbscan_min_samples, dbscan_eps_range, logger
    )

    if dbscan_labels is not None:
        dbscan_reps = get_cluster_representatives(
            valid_df, dbscan_labels, tsne_result, logger
        )
        dbscan_vis = generate_visualization(
            species_name,
            valid_df,
            tsne_result,
            dbscan_labels,
            "DBSCAN",
            f"eps={dbscan_eps:.2f}, min_samples={dbscan_min_samples}",
            dbscan_score,
            dbscan_reps,
            data_dir,
            output_dir,
            logger,
        )
        method_results["dbscan"] = {
            "labels": dbscan_labels.tolist(),
            "eps": dbscan_eps,
            "min_samples": dbscan_min_samples,
            "score": dbscan_score,
            "vis_path": dbscan_vis,
        }

    # 5. HDBSCAN clustering (if available)
    if HDBSCAN_AVAILABLE:
        logger.info(f"Starting HDBSCAN clustering for {species_name}")
        hdbscan_labels, hdbscan_min_size, hdbscan_score = hdbscan_clustering(
            tsne_result, hdbscan_min_cluster_size, logger
        )

        if hdbscan_labels is not None:
            hdbscan_reps = get_cluster_representatives(
                valid_df, hdbscan_labels, tsne_result, logger
            )
            hdbscan_vis = generate_visualization(
                species_name,
                valid_df,
                tsne_result,
                hdbscan_labels,
                "HDBSCAN",
                f"min_cluster_size={hdbscan_min_cluster_size}",
                hdbscan_score,
                hdbscan_reps,
                data_dir,
                output_dir,
                logger,
            )
            method_results["hdbscan"] = {
                "labels": hdbscan_labels.tolist(),
                "min_cluster_size": hdbscan_min_cluster_size,
                "score": hdbscan_score,
                "vis_path": hdbscan_vis,
            }

    # Store summary statistics
    stats = {
        "total_samples": len(valid_df),
        "species_name": species_name,
        "method_results": method_results,
    }

    return stats


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
        description="t-SNE-based clustering and visualization with multiple methods for specific species"
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
        "--dbscan-min-samples",
        type=int,
        default=25,
        help="DBSCAN min_samples parameter (minimum points to form a core point)",
    )
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=25,
        help="HDBSCAN min_cluster_size parameter (minimum points to form a cluster)",
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

    # Add DBSCAN eps range parameters
    parser.add_argument(
        "--dbscan-min-eps",
        type=float,
        default=0.5,
        help="Minimum DBSCAN eps value to try",
    )
    parser.add_argument(
        "--dbscan-max-eps",
        type=float,
        default=2.0,
        help="Maximum DBSCAN eps value to try",
    )
    parser.add_argument(
        "--dbscan-eps-steps",
        type=int,
        default=4,
        help="Number of eps values to try between min and max",
    )

    # Parse arguments
    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_dir)
    logger.info("Starting t-SNE-based clustering and visualization")
    logger.info(f"Arguments: {args}")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Generate DBSCAN eps range
    dbscan_eps_range = np.linspace(
        args.dbscan_min_eps, args.dbscan_max_eps, args.dbscan_eps_steps
    )

    # Load data for all target species
    try:
        df, embeddings_dict = load_data_for_species(
            args.db_path, args.table_name, args.species, logger
        )

        if df.empty or not embeddings_dict:
            logger.error("Failed to load data for target species")
            return 1

        # Process each species
        all_stats = {}
        for species_name in args.species:
            logger.info(f"===== Processing species: {species_name} =====")

            # Filter dataframe for this species
            species_df = (
                df[df["species_name"] == species_name].copy().reset_index(drop=True)
            )

            if len(species_df) < 10:
                logger.warning(
                    f"Not enough samples for {species_name} (found {len(species_df)}, need at least 10)"
                )
                continue

            # Process this species
            species_stats = process_species_data(
                species_name,
                species_df,
                embeddings_dict,
                args.min_k,
                args.max_k,
                args.dbscan_min_samples,
                dbscan_eps_range,
                args.hdbscan_min_cluster_size,
                args.data_dir,
                args.output_dir,
                logger,
            )

            if species_stats:
                all_stats[species_name] = species_stats

        # Save overall results
        if all_stats:
            stats_file = os.path.join(args.output_dir, "clustering_stats.json")
            with open(stats_file, "w") as f:
                json.dump(numpy_to_python_types(all_stats), f, indent=2)
            logger.info(f"Saved overall statistics to {stats_file}")
        else:
            logger.warning("No statistics generated for any species")

        logger.info("t-SNE-based clustering and visualization completed successfully")
        return 0

    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
