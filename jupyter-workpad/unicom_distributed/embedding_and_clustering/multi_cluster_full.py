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
import pickle
import signal
import sys
import traceback

# Import GPU libraries (cuML), assuming they are available
import cuml
from cuml.manifold import TSNE
from cuml.cluster import KMeans
from cuml.metrics import pairwise_distances

# Try to import HDBSCAN
try:
    import hdbscan

    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    print("HDBSCAN not available. HDBSCAN clustering will be skipped.")

from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.metrics import silhouette_score


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"clustering_tsne_multiple_{timestamp}.log")

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


def save_checkpoint(species_processed, results_dict, stats_dict, checkpoint_dir):
    """Save checkpoint data to allow resuming after interruption.

    Args:
        species_processed: List of species names that have been processed
        results_dict: Dictionary containing clustering results for each method
        stats_dict: Dictionary of statistics for each method
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")

    # First save to a temporary file to avoid corrupting existing checkpoint if interrupted
    temp_file = checkpoint_file + ".tmp"
    try:
        with open(temp_file, "wb") as f:
            pickle.dump(
                {
                    "species_processed": species_processed,
                    "results": results_dict,
                    "stats": stats_dict,
                    "timestamp": time.time(),
                },
                f,
            )

        # Atomic rename to avoid corruption
        if os.path.exists(checkpoint_file):
            os.replace(temp_file, checkpoint_file)
        else:
            os.rename(temp_file, checkpoint_file)

        return True
    except Exception as e:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return False


def load_checkpoint(checkpoint_dir, logger):
    """Load checkpoint data if available.

    Args:
        checkpoint_dir: Directory containing checkpoints
        logger: Logger for output messages

    Returns:
        Dictionary with checkpoint data if found, None otherwise
    """
    checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.pkl")
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "rb") as f:
                checkpoint = pickle.load(f)

            # Verify checkpoint has expected data
            required_keys = ["species_processed", "results", "stats", "timestamp"]
            if all(key in checkpoint for key in required_keys):
                age = time.time() - checkpoint["timestamp"]
                age_hours = age / 3600
                logger.info(
                    f"Loaded checkpoint with {len(checkpoint['species_processed'])} "
                    f"processed species (age: {age_hours:.2f} hours)"
                )
                return checkpoint
            else:
                logger.warning(f"Checkpoint file missing required data, ignoring")
                return None
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    return None


def load_data_from_sqlite(db_path, table_name, logger):
    """Load data from SQLite database into a DataFrame."""
    logger.info(f"Loading data from {db_path} into memory")

    # Set initial retry parameters
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            # Use a simple query to get all data
            connection = sqlite3.connect(db_path, timeout=300.0)

            query = f"SELECT uuid, species_name, embedding FROM {table_name}"
            logger.info("Executing SELECT query to retrieve data")

            # Read in chunks to manage memory
            chunk_size = 100000
            chunks = []

            for chunk_df in pd.read_sql_query(query, connection, chunksize=chunk_size):
                # Filter for non-null species names in pandas
                chunk_df = chunk_df[
                    chunk_df["species_name"].notna() & (chunk_df["species_name"] != "")
                ]
                chunks.append(chunk_df)
                logger.info(
                    f"Loaded chunk with {len(chunk_df)} rows, total so far: {sum(len(df) for df in chunks)}"
                )

            # Combine all chunks
            raw_df = pd.concat(chunks, ignore_index=True)
            connection.close()

            logger.info(f"Successfully loaded {len(raw_df)} total rows")

            # Get unique species and sort them
            all_species = sorted(raw_df["species_name"].unique())
            logger.info(f"Found {len(all_species)} unique species")

            # Create embeddings dictionary
            embeddings_dict = {}
            for i, row in tqdm(
                raw_df.iterrows(), total=len(raw_df), desc="Processing embeddings"
            ):
                try:
                    embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                    embeddings_dict[row["uuid"]] = embedding
                except Exception as e:
                    logger.warning(
                        f"Error with embedding for UUID {row['uuid']}: {str(e)}"
                    )
                    embeddings_dict[row["uuid"]] = np.zeros(1024, dtype=np.float32)

            return raw_df, embeddings_dict, all_species

        except Exception as e:
            logger.error(f"Error loading data on attempt {attempt+1}: {str(e)}")
            logger.error(traceback.format_exc())

            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Maximum retries reached, giving up")
                return pd.DataFrame(), {}, []


def kmeans_clustering(data, min_k, max_k, logger):
    """Perform k-means clustering with silhouette score optimization using GPU."""
    logger.info(f"Performing K-means clustering with k range [{min_k}-{max_k}]")

    best_score = -1
    best_k = min_k
    best_labels = None

    for k in range(min_k, max_k + 1):
        try:
            logger.info(f"Trying K-means with k={k}")
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(data)

            # Convert cuML array to NumPy if needed
            if hasattr(labels, "get"):
                labels_cpu = labels.get()
            else:
                labels_cpu = np.array(labels)

            # Convert data to NumPy if needed
            if hasattr(data, "get"):
                data_cpu = data.get()
            else:
                data_cpu = np.array(data)

            # Skip if only one cluster
            if len(np.unique(labels_cpu)) < 2:
                logger.warning(
                    f"K-means with k={k} produced only one cluster, skipping"
                )
                continue

            score = silhouette_score(data_cpu, labels_cpu)
            logger.info(f"K={k}, Silhouette Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k
                best_labels = labels_cpu

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
            labels = kmeans.fit_predict(data)

            # Convert cuML array to NumPy if needed
            if hasattr(labels, "get"):
                best_labels = labels.get()
            else:
                best_labels = np.array(labels)

            best_k = min_k

            # Calculate score for the fallback
            if hasattr(data, "get"):
                data_cpu = data.get()
            else:
                data_cpu = np.array(data)

            best_score = silhouette_score(data_cpu, best_labels)

        except Exception as e:
            logger.error(f"Error during fallback K-means with k={min_k}: {str(e)}")
            logger.error(traceback.format_exc())
            # Return dummy labels as last resort
            best_labels = np.zeros(len(data), dtype=int)
            best_score = -1.0

    return best_labels, best_k, best_score


def hierarchical_clustering(data, strategy, logger):
    """Perform hierarchical clustering with specified strategy."""
    logger.info(f"Performing hierarchical clustering with {strategy} strategy")

    try:
        # Convert to CPU if needed for scipy
        if hasattr(data, "get"):
            data_cpu = data.get()
        else:
            data_cpu = np.array(data)

        # Compute linkage matrix
        logger.info("Computing linkage matrix...")
        Z = linkage(data_cpu, method="ward")
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

                score = silhouette_score(data_cpu, clusters_temp)
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
                best_score = -1.0
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

            # Apply clustering with the determined threshold
            labels = fcluster(Z, t=best_threshold, criterion="distance")
            n_clusters = len(np.unique(labels))

            # Calculate silhouette score if there are at least 2 clusters
            if n_clusters >= 2:
                best_score = silhouette_score(data_cpu, labels)
                logger.info(f"Gap strategy silhouette score: {best_score:.4f}")
            else:
                best_score = -1.0
                logger.warning(
                    f"Gap strategy produced only {n_clusters} clusters, invalid for silhouette score"
                )
        else:
            logger.error(f"Unknown hierarchical strategy: {strategy}")
            return None, None, None

        # Apply clustering with the determined threshold
        logger.info(f"Applying final clustering with threshold={best_threshold:.2f}")
        labels = fcluster(Z, t=best_threshold, criterion="distance")
        n_clusters = len(np.unique(labels))
        logger.info(f"Final clustering produced {n_clusters} clusters")

        # Compute silhouette score if not already done
        if best_score is None and n_clusters >= 2:
            best_score = silhouette_score(data_cpu, labels)
            logger.info(f"Final silhouette score: {best_score:.4f}")
        elif n_clusters < 2:
            best_score = -1.0
            logger.warning(
                "Final clustering produced less than 2 clusters, invalid for silhouette score"
            )

        return labels, best_threshold, best_score

    except Exception as e:
        logger.error(
            f"Error in hierarchical clustering with {strategy} strategy: {str(e)}"
        )
        logger.error(traceback.format_exc())
        # Return dummy labels as fallback
        return np.zeros(len(data), dtype=int), 0.0, -1.0


def hdbscan_clustering(data, min_cluster_size, logger):
    """Perform HDBSCAN clustering."""
    if not HDBSCAN_AVAILABLE:
        logger.warning("HDBSCAN not available, skipping")
        return None, None, None

    logger.info(
        f"Performing HDBSCAN clustering with min_cluster_size={min_cluster_size}"
    )

    try:
        # Convert to CPU if needed for HDBSCAN
        if hasattr(data, "get"):
            data_cpu = data.get()
        else:
            data_cpu = np.array(data)

        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        labels = clusterer.fit_predict(data_cpu)

        n_clusters = len(np.unique(labels[labels != -1]))
        n_noise = np.sum(labels == -1)
        logger.info(f"HDBSCAN found {n_clusters} clusters and {n_noise} noise points")

        # Calculate silhouette score if possible
        score = None
        if n_clusters >= 2:
            non_noise_mask = labels != -1
            if np.sum(non_noise_mask) >= 2:
                score = silhouette_score(
                    data_cpu[non_noise_mask], labels[non_noise_mask]
                )
                logger.info(f"HDBSCAN Silhouette Score: {score:.4f}")
            else:
                logger.warning("Not enough non-noise points for silhouette score")
                score = -1.0
        else:
            logger.warning(
                "HDBSCAN found less than 2 clusters, invalid for silhouette score"
            )
            score = -1.0

        return labels, min_cluster_size, score
    except Exception as e:
        logger.error(f"Error during HDBSCAN clustering: {str(e)}")
        logger.error(traceback.format_exc())
        return None, None, None


def process_under_sampled_species(
    species_df, embeddings_dict, outlier_threshold, logger
):
    """Process under-sampled species by finding outliers based on cosine similarity."""
    species_name = species_df["species_name"].iloc[0]

    try:
        # Extract embeddings for this species
        uuids = species_df["uuid"].tolist()

        # Safely collect embeddings
        valid_embeddings = []
        valid_indices = []
        for i, uuid in enumerate(uuids):
            if uuid in embeddings_dict:
                emb = embeddings_dict[uuid]
                if not np.isnan(emb).any() and not (emb == 0).all():
                    valid_embeddings.append(emb)
                    valid_indices.append(i)

        if len(valid_embeddings) == 0:
            logger.warning(f"No valid embeddings for {species_name}")
            return pd.Series(0, index=species_df.index), 0

        embeddings = np.array(valid_embeddings)
        valid_df_indices = species_df.index[valid_indices]

        if len(embeddings) <= 1:
            # Can't compute similarities with just one sample
            logger.info(
                f"Species {species_name} has only {len(embeddings)} samples - marking as valid"
            )
            return pd.Series(0, index=species_df.index), 0

        # Normalize embeddings for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms

        # Try GPU-accelerated distance calculation
        try:
            # Use GPU for distance calculation
            sim_matrix = 1 - pairwise_distances(
                normalized_embeddings, normalized_embeddings, metric="euclidean"
            )

            # Convert to NumPy if needed
            if hasattr(sim_matrix, "get"):
                sim_matrix = sim_matrix.get()
            else:
                sim_matrix = np.array(sim_matrix)

            logger.info("Used GPU-accelerated similarity calculation")
        except Exception as e:
            logger.error(f"GPU similarity calculation failed: {str(e)}")
            logger.error(traceback.format_exc())
            return pd.Series(0, index=species_df.index), 0

        # For each sample, compute average similarity to all other samples
        np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarity
        avg_similarities = sim_matrix.sum(axis=1) / max(
            len(embeddings) - 1, 1
        )  # Avoid division by zero

        # Compute mean and std of similarities
        mean_sim = np.mean(avg_similarities)
        std_sim = np.std(avg_similarities)

        # Handle edge case of zero standard deviation
        if std_sim == 0:
            logger.warning(
                f"Species {species_name}: zero standard deviation in similarities, no outliers identified"
            )
            return pd.Series(0, index=species_df.index), 0

        # Identify outliers as embeddings with avg similarity < mean - threshold*std
        threshold_value = mean_sim - outlier_threshold * std_sim
        is_outlier = avg_similarities < threshold_value

        logger.info(
            f"Species {species_name}: mean sim={mean_sim:.4f}, std={std_sim:.4f}, threshold={threshold_value:.4f}"
        )
        logger.info(f"Species {species_name}: {np.sum(is_outlier)} outliers identified")

        # Create a series with 0 for valid samples, -1 for outliers
        cluster_series = pd.Series(0, index=species_df.index)
        outlier_indices = [
            valid_df_indices[i] for i, is_out in enumerate(is_outlier) if is_out
        ]
        if outlier_indices:
            cluster_series.loc[outlier_indices] = -1

        return cluster_series, np.sum(is_outlier)

    except Exception as e:
        logger.error(f"Error processing under-sampled species {species_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return pd.Series(0, index=species_df.index), 0


def cluster_species(
    species_df,
    embeddings_dict,
    min_k,
    max_k,
    hdbscan_min_cluster_size,
    tsne_dims,
    logger,
):
    """Perform multiple clustering methods on a well-sampled species."""
    species_name = species_df["species_name"].iloc[0]

    clustering_results = {
        "kmeans": {"labels": None, "param": None, "score": None},
        "hierarchical_silhouette": {"labels": None, "param": None, "score": None},
        "hierarchical_gap": {"labels": None, "param": None, "score": None},
        "hdbscan": {"labels": None, "param": None, "score": None},
    }

    try:
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
            else:
                logger.warning(f"Missing embedding for UUID {uuid}")

        if len(embeddings_list) < min_k:
            logger.warning(
                f"Not enough valid embeddings for {species_name} (found {len(embeddings_list)}, need {min_k})"
            )
            return clustering_results, None

        embeddings = np.array(embeddings_list)
        valid_df_indices = species_df.index[valid_indices]

        # Generate t-SNE projection
        logger.info(f"Computing t-SNE for {species_name} ({len(embeddings)} samples)")
        try:
            tsne = TSNE(
                n_components=tsne_dims,
                method="exact",
                perplexity=15,
                random_state=42,
            )
            logger.info("Starting t-SNE fit_transform - this may take a while...")
            tsne_result = tsne.fit_transform(embeddings)

            # Convert cuML array to NumPy if needed
            if hasattr(tsne_result, "get"):
                tsne_result = tsne_result.get()
            else:
                tsne_result = np.array(tsne_result)

            logger.info(
                f"Successfully generated t-SNE projection with shape {tsne_result.shape}"
            )
        except Exception as e:
            logger.error(f"t-SNE projection failed: {str(e)}")
            logger.error(traceback.format_exc())
            return clustering_results, valid_df_indices

        # Verify t-SNE output
        if np.isnan(tsne_result).any():
            logger.warning("t-SNE result contains NaN values! Replacing with zeros.")
            tsne_result = np.nan_to_num(tsne_result)

        # Perform K-means clustering
        logger.info(f"Starting K-means clustering for {species_name}")
        kmeans_labels, kmeans_k, kmeans_score = kmeans_clustering(
            tsne_result, min_k, max_k, logger
        )

        if kmeans_labels is not None:
            clustering_results["kmeans"] = {
                "labels": kmeans_labels,
                "param": kmeans_k,
                "score": kmeans_score,
            }

        # Perform Hierarchical clustering - silhouette
        logger.info(f"Starting hierarchical clustering (silhouette) for {species_name}")
        hier_sil_labels, hier_sil_threshold, hier_sil_score = hierarchical_clustering(
            tsne_result, "silhouette", logger
        )

        if hier_sil_labels is not None:
            clustering_results["hierarchical_silhouette"] = {
                "labels": hier_sil_labels,
                "param": hier_sil_threshold,
                "score": hier_sil_score,
            }

        # Perform Hierarchical clustering - gap
        logger.info(f"Starting hierarchical clustering (gap) for {species_name}")
        hier_gap_labels, hier_gap_threshold, hier_gap_score = hierarchical_clustering(
            tsne_result, "gap", logger
        )

        if hier_gap_labels is not None:
            clustering_results["hierarchical_gap"] = {
                "labels": hier_gap_labels,
                "param": hier_gap_threshold,
                "score": hier_gap_score,
            }

        # Perform HDBSCAN clustering
        if HDBSCAN_AVAILABLE:
            logger.info(f"Starting HDBSCAN clustering for {species_name}")
            hdbscan_labels, hdbscan_min_size, hdbscan_score = hdbscan_clustering(
                tsne_result, hdbscan_min_cluster_size, logger
            )

            if hdbscan_labels is not None:
                clustering_results["hdbscan"] = {
                    "labels": hdbscan_labels,
                    "param": hdbscan_min_size,
                    "score": hdbscan_score,
                }

        return clustering_results, valid_df_indices

    except Exception as e:
        logger.error(f"Error clustering {species_name}: {str(e)}")
        logger.error(traceback.format_exc())
        return clustering_results, None


def process_all_species(df, embeddings_dict, species_list, args, logger):
    """Process all species with checkpointing."""
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    columns_config = {
        "kmeans": {
            "cluster_col": args.kmeans_column,
            "score_col": args.kmeans_score_column,
        },
        "hierarchical_silhouette": {
            "cluster_col": args.hier_sil_column,
            "score_col": args.hier_sil_score_column,
        },
        "hierarchical_gap": {
            "cluster_col": args.hier_gap_column,
            "score_col": args.hier_gap_score_column,
        },
        "hdbscan": {
            "cluster_col": args.hdbscan_column,
            "score_col": args.hdbscan_score_column,
        },
    }

    # Try to load checkpoint
    checkpoint = load_checkpoint(checkpoint_dir, logger)
    if checkpoint:
        results_dict = checkpoint["results"]
        stats_dict = checkpoint["stats"]
        processed_species = set(checkpoint["species_processed"])
        logger.info(
            f"Resuming from checkpoint with {len(processed_species)} species already processed"
        )
    else:
        # Initialize results dictionary with Series for each clustering method
        results_dict = {}
        for method, cols in columns_config.items():
            results_dict[method] = pd.Series(index=df.index, dtype=int)

        stats_dict = {}
        processed_species = set()

    total_species = len(species_list)
    remaining_species = [s for s in species_list if s not in processed_species]
    logger.info(
        f"Processing {len(remaining_species)} remaining species out of {total_species} total"
    )

    # Track processed species for checkpointing
    newly_processed = []
    checkpoint_interval = 5  # Save checkpoint every 5 species
    last_checkpoint_time = time.time()
    time_based_checkpoint_interval = 10 * 60  # 10 minutes

    try:
        for i, species_name in enumerate(
            tqdm(remaining_species, desc="Processing species")
        ):
            # Get data for this species
            species_df = df[df["species_name"] == species_name]
            if len(species_df) == 0:
                logger.warning(f"No data found for species {species_name}, skipping")
                continue

            current_progress = len(processed_species) + i + 1
            logger.info(
                f"Processing {species_name} ({current_progress}/{total_species}) with {len(species_df)} samples"
            )

            # Determine processing method based on sample count
            is_well_sampled = len(species_df) >= args.min_samples_for_clustering

            if is_well_sampled:
                # Cluster well-sampled species with multiple methods
                logger.info(
                    f"{species_name} is well-sampled ({len(species_df)} samples)"
                )
                clustering_results, valid_indices = cluster_species(
                    species_df,
                    embeddings_dict,
                    args.min_k,
                    args.max_k,
                    args.hdbscan_min_cluster_size,
                    args.tsne_dims,
                    logger,
                )

                if valid_indices is None:
                    logger.warning(f"Failed to process {species_name}, skipping")
                    continue

                # Store cluster and score information for each method
                species_stats = {
                    "count": len(species_df),
                    "valid_samples": len(valid_indices),
                    "method": "multi_clustering",
                    "clustering_params": {},
                }

                # Process each clustering method
                for method, result in clustering_results.items():
                    if result["labels"] is not None:
                        # Create a full series for all records in this species
                        cluster_series = pd.Series(
                            -1, index=species_df.index
                        )  # Default to -1 (noise/outlier)

                        # Special handling for HDBSCAN - keep noise points as -1
                        if method == "hdbscan":
                            # For records that aren't noise, apply cluster labels
                            non_noise_indices = [
                                valid_indices[i]
                                for i, label in enumerate(result["labels"])
                                if label != -1
                            ]
                            non_noise_labels = [
                                label for label in result["labels"] if label != -1
                            ]
                            if non_noise_indices:
                                cluster_series.loc[non_noise_indices] = non_noise_labels
                        else:
                            # For other methods, apply all cluster labels
                            cluster_series.loc[valid_indices] = result["labels"]

                        # Store the clustering results
                        results_dict[method].loc[species_df.index] = cluster_series

                        # Store stats for this clustering method
                        method_stats = {
                            "param_value": result["param"],
                            "silhouette_score": result["score"],
                        }

                        # Add method-specific details
                        if method == "kmeans":
                            method_stats["k"] = result["param"]
                        elif (
                            method == "hierarchical_silhouette"
                            or method == "hierarchical_gap"
                        ):
                            method_stats["threshold"] = result["param"]
                        elif method == "hdbscan":
                            method_stats["min_cluster_size"] = result["param"]
                            method_stats["noise_points"] = np.sum(
                                result["labels"] == -1
                            )

                        species_stats["clustering_params"][method] = method_stats

            else:
                # Process under-sampled species with outlier detection
                logger.info(
                    f"{species_name} is under-sampled ({len(species_df)} samples)"
                )
                outlier_series, outlier_count = process_under_sampled_species(
                    species_df, embeddings_dict, args.outlier_threshold, logger
                )

                # Use the same outlier results for all methods
                for method in columns_config.keys():
                    results_dict[method].loc[species_df.index] = outlier_series

                # Store stats for under-sampled species
                species_stats = {
                    "count": len(species_df),
                    "method": "outlier_detection",
                    "outlier_count": outlier_count,
                }

            # Store results for this species
            stats_dict[species_name] = species_stats
            newly_processed.append(species_name)

            # Save checkpoint based on count or time
            current_time = time.time()
            time_to_checkpoint = (len(newly_processed) % checkpoint_interval == 0) or (
                (current_time - last_checkpoint_time) > time_based_checkpoint_interval
            )

            if time_to_checkpoint:
                all_processed = list(processed_species) + newly_processed
                if save_checkpoint(
                    all_processed, results_dict, stats_dict, checkpoint_dir
                ):
                    logger.info(
                        f"Saved checkpoint: {len(all_processed)}/{total_species} species processed"
                    )
                    last_checkpoint_time = current_time
                else:
                    logger.warning("Failed to save checkpoint")

            # Log progress periodically
            if (i + 1) % 10 == 0 or (i + 1) == len(remaining_species):
                progress_pct = (current_progress / total_species) * 100
                logger.info(
                    f"Progress: {current_progress}/{total_species} species processed ({progress_pct:.1f}%)"
                )

        # Save final checkpoint
        all_processed = list(processed_species) + newly_processed
        if save_checkpoint(all_processed, results_dict, stats_dict, checkpoint_dir):
            logger.info(
                f"Saved final checkpoint: {len(all_processed)}/{total_species} species processed"
            )

    except Exception as e:
        logger.error(f"Error in process_all_species: {str(e)}")
        logger.error(traceback.format_exc())

        # Save checkpoint on error to preserve progress
        if newly_processed:
            all_processed = list(processed_species) + newly_processed
            if save_checkpoint(all_processed, results_dict, stats_dict, checkpoint_dir):
                logger.info(
                    f"Saved emergency checkpoint after error: {len(all_processed)}/{total_species} species processed"
                )

        # Re-raise to allow main error handling to take over
        raise

    return results_dict, stats_dict


def write_results_to_sqlite(
    db_path, table_name, columns_config, df, results_dict, logger
):
    """Write clustering results back to SQLite database with robust error handling."""
    # Set initial retry parameters
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(f"Writing results to {db_path} (attempt {attempt+1})")

            # Connect to database with longer timeout
            conn = sqlite3.connect(db_path, timeout=120.0, isolation_level="EXCLUSIVE")
            logger.info(f"Connected to database successfully")

            # Set pragmas for better write performance
            try:
                conn.execute("PRAGMA synchronous = NORMAL")
                conn.execute("PRAGMA busy_timeout = 60000")  # 60 second busy timeout
                conn.commit()
                logger.info("Database PRAGMA settings applied successfully")
            except sqlite3.OperationalError as pragma_err:
                logger.warning(f"Could not set all PRAGMA settings: {pragma_err}")

            # Ensure all columns exist
            for method, cols in columns_config.items():
                cluster_col = cols["cluster_col"]
                score_col = cols["score_col"]

                try:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {cluster_col} INTEGER"
                    )
                    logger.info(f"Column {cluster_col} added")
                except sqlite3.OperationalError:
                    logger.info(f"Column {cluster_col} already exists")

                try:
                    conn.execute(
                        f"ALTER TABLE {table_name} ADD COLUMN {score_col} REAL"
                    )
                    logger.info(f"Column {score_col} added")
                except sqlite3.OperationalError:
                    logger.info(f"Column {score_col} already exists")

            conn.commit()

            # Start transaction
            conn.execute("BEGIN TRANSACTION")

            # Prepare and execute update statement
            cursor = conn.cursor()

            # Process each clustering method
            total_updated_rows = 0

            for method, cols in columns_config.items():
                cluster_col = cols["cluster_col"]
                score_col = cols["score_col"]

                # Get clustering results for this method
                cluster_results = results_dict[method]

                # Prepare data for update - ensure cluster results are standard Python ints
                update_df = pd.DataFrame(
                    {
                        "uuid": df["uuid"],
                        "species_name": df["species_name"],
                        cluster_col: cluster_results.astype(int),
                    }
                )

                # Update in batches to avoid memory issues
                batch_size = 5000  # Smaller batch size to reduce lock time
                total_rows = len(update_df)
                method_updated_rows = 0

                for i in tqdm(
                    range(0, total_rows, batch_size), desc=f"Writing {method} clusters"
                ):
                    batch = update_df.iloc[i : i + batch_size]
                    batch_updates = []

                    for _, row in batch.iterrows():
                        # Make sure we're passing the column value first, then the UUID
                        cluster_val = int(row[cluster_col])  # Force Python int type
                        uuid_val = row["uuid"]
                        batch_updates.append((cluster_val, uuid_val))

                    try:
                        cursor.executemany(
                            f"UPDATE {table_name} SET {cluster_col} = ? WHERE uuid = ?",
                            batch_updates,
                        )

                        # Commit every few batches to reduce lock time
                        if i % (batch_size * 10) == 0 and i > 0:
                            conn.commit()
                            conn.execute("BEGIN TRANSACTION")
                            logger.info(f"Intermediate commit at {i} rows for {method}")

                        method_updated_rows += len(batch)
                    except sqlite3.OperationalError as update_err:
                        logger.error(
                            f"Error updating batch {i//batch_size} for {method}: {update_err}"
                        )
                        # Try to continue with next batch
                        continue

                logger.info(f"Updated {method_updated_rows} rows for {method} clusters")
                total_updated_rows += method_updated_rows

                # Now update silhouette scores for each species
                species_score_updates = []

                # Get unique species
                unique_species = df["species_name"].unique()

                for species_name in unique_species:
                    species_indices = df[df["species_name"] == species_name].index
                    # Use the first value that isn't -1 as the representative score
                    # If all are -1, use -1 as the score
                    species_scores = cluster_results.loc[species_indices]

                    # Get score from stats dictionary if available, otherwise use default value
                    if species_name in stats_dict:
                        if (
                            "clustering_params" in stats_dict[species_name]
                            and method in stats_dict[species_name]["clustering_params"]
                        ):
                            score = stats_dict[species_name]["clustering_params"][
                                method
                            ]["silhouette_score"]
                        else:
                            score = -1.0
                    else:
                        score = -1.0

                    # Add an update for each record in this species
                    for idx in species_indices:
                        uuid_val = df.loc[idx, "uuid"]
                        species_score_updates.append((score, uuid_val))

                # Update scores in batches
                batch_size = 5000
                for i in tqdm(
                    range(0, len(species_score_updates), batch_size),
                    desc=f"Writing {method} scores",
                ):
                    batch = species_score_updates[i : i + batch_size]

                    try:
                        cursor.executemany(
                            f"UPDATE {table_name} SET {score_col} = ? WHERE uuid = ?",
                            batch,
                        )

                        # Commit every few batches to reduce lock time
                        if i % (batch_size * 10) == 0 and i > 0:
                            conn.commit()
                            conn.execute("BEGIN TRANSACTION")
                            logger.info(
                                f"Intermediate commit at {i} rows for {method} scores"
                            )

                    except sqlite3.OperationalError as update_err:
                        logger.error(
                            f"Error updating batch {i//batch_size} for {method} scores: {update_err}"
                        )
                        # Try to continue with next batch
                        continue

                logger.info(
                    f"Updated {len(species_score_updates)} rows for {method} scores"
                )

            # Commit final transaction
            try:
                conn.commit()
                logger.info(f"Successfully updated {total_updated_rows} total rows")
            except sqlite3.OperationalError as commit_err:
                logger.error(f"Error during final commit: {commit_err}")
                if attempt < max_retries - 1:
                    conn.close()
                    continue
                else:
                    return total_updated_rows

            # Close connection
            conn.close()
            logger.info("Database connection closed successfully")

            return total_updated_rows

        except sqlite3.Error as e:
            logger.error(f"SQLite error on attempt {attempt+1}: {str(e)}")
            if "conn" in locals():
                try:
                    conn.rollback()
                    logger.info("Rolled back transaction after error")
                except:
                    pass
                try:
                    conn.close()
                    logger.info("Closed database connection after error")
                except:
                    pass

            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Maximum retries reached, giving up")
                return 0

        except Exception as e:
            logger.error(f"Unexpected error on attempt {attempt+1}: {str(e)}")
            logger.error(traceback.format_exc())

            if "conn" in locals():
                try:
                    conn.rollback()
                    logger.info("Rolled back transaction after error")
                except:
                    pass
                try:
                    conn.close()
                    logger.info("Closed database connection after error")
                except:
                    pass

            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Maximum retries reached, giving up")
                return 0


def save_stats(stats, output_file, logger):
    """Save statistics to a JSON file."""
    try:
        # Convert numpy types to Python types
        clean_stats = numpy_to_python_types(stats)

        with open(output_file, "w") as f:
            json.dump(clean_stats, f, indent=2)

        logger.info(f"Saved clustering statistics to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving stats: {str(e)}")
        logger.error(traceback.format_exc())
        return False


def numpy_to_python_types(obj):
    """Convert numpy types to standard Python types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
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
        description="Multi-method clustering of image embeddings by species"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--table-name", default="image_embeddings", help="Name of table in SQLite"
    )

    # Column names for each clustering method
    parser.add_argument(
        "--kmeans-column",
        default="kmeans_cluster",
        help="Column name to store K-means clustering results",
    )
    parser.add_argument(
        "--kmeans-score-column",
        default="kmeans_silhouette_score",
        help="Column name to store K-means silhouette scores",
    )
    parser.add_argument(
        "--hier-sil-column",
        default="hier_silhouette_cluster",
        help="Column name to store hierarchical silhouette clustering results",
    )
    parser.add_argument(
        "--hier-sil-score-column",
        default="hier_silhouette_score",
        help="Column name to store hierarchical silhouette scores",
    )
    parser.add_argument(
        "--hier-gap-column",
        default="hier_gap_cluster",
        help="Column name to store hierarchical gap clustering results",
    )
    parser.add_argument(
        "--hier-gap-score-column",
        default="hier_gap_silhouette_score",
        help="Column name to store hierarchical gap silhouette scores",
    )
    parser.add_argument(
        "--hdbscan-column",
        default="hdbscan_cluster",
        help="Column name to store HDBSCAN clustering results",
    )
    parser.add_argument(
        "--hdbscan-score-column",
        default="hdbscan_silhouette_score",
        help="Column name to store HDBSCAN silhouette scores",
    )

    parser.add_argument("--log-dir", required=True, help="Directory for log files")
    parser.add_argument(
        "--min-samples-for-clustering",
        type=int,
        default=25,
        help="Minimum samples required for clustering",
    )
    parser.add_argument("--min-k", type=int, default=2, help="Minimum clusters to try")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum clusters to try")
    parser.add_argument(
        "--hdbscan-min-cluster-size",
        type=int,
        default=25,
        help="HDBSCAN min_cluster_size parameter",
    )
    parser.add_argument(
        "--tsne-dims",
        type=int,
        default=2,
        choices=[2, 3],
        help="t-SNE dimensions (2D or 3D)",
    )
    parser.add_argument(
        "--outlier-threshold",
        type=float,
        default=2.0,
        help="Threshold in std deviations for outlier detection",
    )
    parser.add_argument(
        "--stats-output",
        default="clustering_stats_multiple.json",
        help="JSON file to save clustering statistics",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--skip-db-write",
        action="store_true",
        help="Skip writing results back to the database (for testing)",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_dir)
    start_time = time.time()

    # Set up signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Log start information
    logger.info("Starting multi-method species clustering")
    logger.info(
        f"Parameters: db={args.db_path}, table={args.table_name}, "
        f"min_samples={args.min_samples_for_clustering}, "
        f"k_range=[{args.min_k}, {args.max_k}], tsne_dims={args.tsne_dims}, "
        f"hdbscan_min_cluster_size={args.hdbscan_min_cluster_size}, "
        f"resume={args.resume}"
    )

    # Check GPU availability
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        logger.info("Using cuML for GPU acceleration")
    else:
        logger.error("CUDA not available, but required for this script")
        return 1

    # Create checkpoints directory
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Check if we can resume from a checkpoint
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_dir, logger)

    # If resuming and we have a valid checkpoint with complete results, we can skip loading from DB
    df = pd.DataFrame()
    embeddings_dict = {}
    species_list = []

    # Set up columns configuration
    columns_config = {
        "kmeans": {
            "cluster_col": args.kmeans_column,
            "score_col": args.kmeans_score_column,
        },
        "hierarchical_silhouette": {
            "cluster_col": args.hier_sil_column,
            "score_col": args.hier_sil_score_column,
        },
        "hierarchical_gap": {
            "cluster_col": args.hier_gap_column,
            "score_col": args.hier_gap_score_column,
        },
        "hdbscan": {
            "cluster_col": args.hdbscan_column,
            "score_col": args.hdbscan_score_column,
        },
    }

    if not checkpoint or "complete" not in checkpoint or not checkpoint["complete"]:
        # Load data from SQLite
        load_start = time.time()
        df, embeddings_dict, species_list = load_data_from_sqlite(
            args.db_path, args.table_name, logger
        )

        if len(df) == 0:
            logger.error("No data loaded from database")
            return 1

        load_time = time.time() - load_start
        logger.info(f"Data loading completed in {load_time:.2f}s")
    else:
        logger.info(
            "Found complete results in checkpoint, skipping data loading and processing"
        )
        df = checkpoint.get("df", pd.DataFrame())
        results_dict = checkpoint["results"]
        stats_dict = checkpoint["stats"]

        # Verify we have all methods in the results
        for method in columns_config.keys():
            if method not in results_dict:
                logger.error(f"Checkpoint missing results for {method}")
                return 1

    # Process species if we don't have complete results yet
    if not checkpoint or "complete" not in checkpoint or not checkpoint["complete"]:
        process_start = time.time()
        results_dict, stats_dict = process_all_species(
            df, embeddings_dict, species_list, args, logger
        )
        process_time = time.time() - process_start
        logger.info(f"Processing completed in {process_time:.2f}s")

        # Mark processing as complete in the checkpoint
        all_processed = list(species_list)
        save_checkpoint(all_processed, results_dict, stats_dict, checkpoint_dir)

        # Save a final "complete" flag
        with open(os.path.join(checkpoint_dir, "checkpoint.pkl"), "rb") as f:
            checkpoint = pickle.load(f)
        checkpoint["complete"] = True
        checkpoint["df"] = df  # Save DataFrame for reference
        with open(os.path.join(checkpoint_dir, "checkpoint.pkl"), "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info("Marked checkpoint as complete")

    # Write results back to SQLite unless skipped
    write_time = 0
    if not args.skip_db_write:
        write_start = time.time()
        total_updated_rows = write_results_to_sqlite(
            args.db_path, args.table_name, columns_config, df, results_dict, logger
        )
        write_time = time.time() - write_start
        logger.info(f"Database update completed in {write_time:.2f}s")

    # Save statistics
    stats_file = os.path.join(args.log_dir, args.stats_output)
    save_stats(stats_dict, stats_file, logger)

    # Log final statistics
    total_time = time.time() - start_time
    logger.info("\nFinal Statistics:")
    logger.info(f"Total species processed: {len(stats_dict)}")
    logger.info(f"Total records processed: {len(df)}")

    if not args.skip_db_write:
        logger.info(f"Total database updates: {total_updated_rows}")

    logger.info(f"Total processing time: {total_time:.2f} seconds")

    # Only show detailed time breakdown if we did actual work
    if "process_time" in locals() and process_time > 0 or write_time > 0:
        load_time = load_time if "load_time" in locals() else 0
        process_pct = (
            (process_time / total_time * 100)
            if "process_time" in locals() and process_time > 0
            else 0
        )
        write_pct = (write_time / total_time * 100) if write_time > 0 else 0
        load_pct = (load_time / total_time * 100) if load_time > 0 else 0

        logger.info(f"  - Data loading: {load_time:.2f}s ({load_pct:.1f}%)")
        if "process_time" in locals():
            logger.info(f"  - Processing: {process_time:.2f}s ({process_pct:.1f}%)")
        if write_time > 0:
            logger.info(f"  - Database update: {write_time:.2f}s ({write_pct:.1f}%)")

    logger.info("Multi-method clustering job completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
