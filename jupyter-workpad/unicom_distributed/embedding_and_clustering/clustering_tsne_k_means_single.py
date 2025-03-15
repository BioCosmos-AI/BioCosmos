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
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pickle
import signal
import sys

# Try to import GPU libraries, but fall back gracefully if not available
try:
    import cuml
    from cuml.manifold import TSNE as cuTSNE
    from cuml.cluster import KMeans as cuKMeans
    from cuml.metrics import pairwise_distances

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"clustering_tsne_k_means_single_{timestamp}.log")

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


def save_checkpoint(species_processed, results_so_far, stats_so_far, checkpoint_dir):
    """Save checkpoint data to allow resuming after interruption.

    Args:
        species_processed: List of species names that have been processed
        results_so_far: Series containing clustering results
        stats_so_far: Dictionary of statistics
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
                    "results": results_so_far,
                    "stats": stats_so_far,
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
            import traceback

            logger.error(traceback.format_exc())

            if attempt < max_retries - 1:
                wait_time = retry_delay * (attempt + 1)
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                logger.error("Maximum retries reached, giving up")
                return pd.DataFrame(), {}, []


def find_optimal_k_gpu(embeddings, k_range, logger):
    """Find optimal number of clusters using GPU-accelerated KMeans and silhouette scores."""
    best_score = -1
    best_k = 2  # Default to 2 clusters

    if not CUML_AVAILABLE:
        logger.warning("cuML is not available, falling back to CPU implementation")
        return find_optimal_k_cpu(embeddings, k_range, logger)

    try:
        # For cuML input, we don't need to convert to tensor
        # cuML will handle the input directly
        for k in range(k_range[0], k_range[1] + 1):
            try:
                # Use cuML KMeans implementation
                kmeans = cuKMeans(n_clusters=k, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                # Explicitly convert cuML array to NumPy array
                if hasattr(labels, "get"):
                    labels_cpu = labels.get()
                elif hasattr(labels, "to_numpy"):
                    labels_cpu = labels.to_numpy()
                elif isinstance(labels, torch.Tensor):
                    labels_cpu = labels.cpu().numpy()
                else:
                    labels_cpu = np.array(labels)

                # Ensure embeddings are also in CPU NumPy format
                if hasattr(embeddings, "get"):
                    embeddings_cpu = embeddings.get()
                elif hasattr(embeddings, "to_numpy"):
                    embeddings_cpu = embeddings.to_numpy()
                elif isinstance(embeddings, torch.Tensor):
                    embeddings_cpu = embeddings.cpu().numpy()
                else:
                    embeddings_cpu = np.array(embeddings)

                # If we have only one cluster, silhouette score will fail
                if len(np.unique(labels_cpu)) < 2:
                    continue

                score = silhouette_score(embeddings_cpu, labels_cpu)
                logger.info(f"K={k}, Silhouette Score={score:.4f}")

                if score > best_score:
                    best_score = score
                    best_k = k

            except Exception as e:
                logger.error(f"Error evaluating k={k}: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())
                continue

        return best_k, best_score
    except Exception as e:
        logger.warning(f"GPU processing failed: {str(e)}, falling back to CPU")
        import traceback

        logger.error(traceback.format_exc())
        return find_optimal_k_cpu(embeddings, k_range, logger)


def find_optimal_k_cpu(embeddings, k_range, logger):
    """Find optimal number of clusters using CPU KMeans and silhouette scores."""
    best_score = -1
    best_k = 2  # Default to 2 clusters

    for k in range(k_range[0], k_range[1] + 1):
        try:
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(embeddings)

            # If we have only one cluster, silhouette score will fail
            if len(np.unique(labels)) < 2:
                continue

            score = silhouette_score(embeddings, labels)
            logger.debug(f"K={k}, Silhouette Score={score:.4f}")

            if score > best_score:
                best_score = score
                best_k = k

        except Exception as e:
            logger.error(f"Error evaluating k={k}: {str(e)}")
            continue

    return best_k, best_score


def cluster_well_sampled_species(
    species_df, embeddings_dict, min_k, max_k, tsne_dims, use_gpu, logger
):
    """Cluster a well-sampled species using t-SNE and KMeans with optional GPU acceleration."""
    species_name = species_df["species_name"].iloc[0]

    try:
        # Extract embeddings for this species
        uuids = species_df["uuid"].tolist()

        # Safely extract embeddings, handling any missing keys
        embeddings_list = []
        valid_indices = []
        for i, uuid in enumerate(uuids):
            if uuid in embeddings_dict:
                emb = embeddings_dict[uuid]
                # Check for NaN or zero embeddings
                if np.isnan(emb).any() or (emb == 0).all():
                    logger.warning(
                        f"Invalid embedding for UUID {uuid} (NaN or all zeros)"
                    )
                    continue
                embeddings_list.append(emb)
                valid_indices.append(i)
            else:
                logger.warning(f"Missing embedding for UUID {uuid}")

        if len(embeddings_list) < min_k:
            logger.warning(
                f"Not enough valid embeddings for {species_name} (found {len(embeddings_list)}, need {min_k})"
            )
            return pd.Series(0, index=species_df.index), None, None

        embeddings = np.array(embeddings_list)

        # Get valid dataframe indices
        valid_df_indices = species_df.index[valid_indices]

        # Verify embedding dimensions are consistent
        embedding_dims = embeddings.shape[1]
        logger.info(
            f"Computing {tsne_dims}D t-SNE for {species_name} ({len(embeddings)} samples, {embedding_dims} dimensions)"
        )

        # t-SNE computation with proper error handling
        tsne_result = None
        if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
            try:
                # Use GPU-accelerated t-SNE
                tsne = cuTSNE(n_components=tsne_dims, random_state=42)
                tsne_result = tsne.fit_transform(embeddings)
                logger.info("Used GPU-accelerated t-SNE")
            except Exception as e:
                logger.warning(f"GPU t-SNE failed, falling back to CPU: {str(e)}")
                tsne_result = None

        # Fall back to CPU t-SNE if needed
        if tsne_result is None:
            try:
                tsne = TSNE(n_components=tsne_dims, random_state=42)
                tsne_result = tsne.fit_transform(embeddings)
                logger.info("Used CPU t-SNE")
            except Exception as e:
                logger.error(f"CPU t-SNE failed for {species_name}: {str(e)}")
                return pd.Series(0, index=species_df.index), None, None

        # Ensure tsne_result is in the proper format for further processing
        if hasattr(tsne_result, "get"):
            tsne_result_numpy = tsne_result.get()
        elif hasattr(tsne_result, "to_numpy"):
            tsne_result_numpy = tsne_result.to_numpy()
        elif isinstance(tsne_result, torch.Tensor):
            tsne_result_numpy = tsne_result.cpu().numpy()
        else:
            tsne_result_numpy = np.array(tsne_result)

        # Verify t-SNE output
        if tsne_result_numpy is None or len(tsne_result_numpy) == 0:
            logger.error(f"t-SNE failed to produce output for {species_name}")
            return pd.Series(0, index=species_df.index), None, None

        # Find optimal k using silhouette score
        logger.info(f"Finding optimal k for {species_name}")

        optimal_k = None
        score = None

        if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
            try:
                # Pass the NumPy version of tsne_result to find_optimal_k_gpu
                optimal_k, score = find_optimal_k_gpu(
                    tsne_result_numpy, [min_k, max_k], logger
                )
                logger.info("Used GPU-accelerated KMeans for hyperparameter search")
            except Exception as e:
                logger.warning(
                    f"GPU KMeans optimization failed, falling back to CPU: {str(e)}"
                )
                optimal_k = None

        # Fall back to CPU K-means optimization if needed
        if optimal_k is None:
            try:
                optimal_k, score = find_optimal_k_cpu(
                    tsne_result_numpy, [min_k, max_k], logger
                )
                logger.info("Used CPU KMeans for hyperparameter search")
            except Exception as e:
                logger.error(
                    f"CPU KMeans optimization failed for {species_name}: {str(e)}"
                )
                # Use default K as fallback
                optimal_k = min_k
                score = 0.0

        # logger.info(
        #     f"Optimal k for {species_name}: {optimal_k} (score: {score:.4f if score is not None else 0.0})"
        # )
        score_formatted = f"{score:.4f}" if score is not None else "0.0000"
        logger.info(
            f"Optimal k for {species_name}: {optimal_k} (score: {score_formatted})"
        )

        # Perform final clustering with optimal k
        clusters = None
        if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
            try:
                kmeans = cuKMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(tsne_result_numpy)

                # Explicitly convert to NumPy array
                if hasattr(clusters, "get"):
                    clusters = clusters.get()
                elif hasattr(clusters, "to_numpy"):
                    clusters = clusters.to_numpy()
                elif isinstance(clusters, torch.Tensor):
                    clusters = clusters.cpu().numpy()
                else:
                    clusters = np.array(clusters)

                logger.info("Used GPU-accelerated KMeans for final clustering")
            except Exception as e:
                logger.warning(
                    f"GPU final KMeans failed, falling back to CPU: {str(e)}"
                )
                clusters = None

        # Fall back to CPU KMeans if needed
        if clusters is None:
            try:
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(tsne_result_numpy)
                logger.info("Used CPU KMeans for final clustering")
            except Exception as e:
                logger.error(f"CPU final KMeans failed for {species_name}: {str(e)}")
                # Assign all to cluster 0 as fallback
                clusters = np.zeros(len(embeddings), dtype=int)

        # Create a series with the clustering results for valid samples
        result_series = pd.Series(0, index=species_df.index)
        result_series.loc[valid_df_indices] = clusters

        return result_series, optimal_k, score

    except Exception as e:
        logger.error(f"Error clustering {species_name}: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return pd.Series(0, index=species_df.index), None, None


def process_under_sampled_species(
    species_df, embeddings_dict, outlier_threshold, use_gpu, logger
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

        # Compute cosine similarities
        sim_matrix = None
        if use_gpu and CUML_AVAILABLE and torch.cuda.is_available():
            try:
                # Try GPU-accelerated distance calculation
                sim_matrix = 1 - pairwise_distances(
                    normalized_embeddings, normalized_embeddings, metric="euclidean"
                )

                # Explicitly convert to NumPy
                if hasattr(sim_matrix, "get"):
                    sim_matrix = sim_matrix.get()
                elif hasattr(sim_matrix, "to_numpy"):
                    sim_matrix = sim_matrix.to_numpy()
                elif isinstance(sim_matrix, torch.Tensor):
                    sim_matrix = sim_matrix.cpu().numpy()
                else:
                    sim_matrix = np.array(sim_matrix)

                logger.info("Used GPU-accelerated similarity calculation")
            except Exception as e:
                logger.warning(
                    f"GPU similarity calculation failed, falling back to CPU: {str(e)}"
                )
                sim_matrix = None

        # Fall back to CPU similarity calculation if needed
        if sim_matrix is None:
            # Use CPU for cosine similarity
            sim_matrix = np.matmul(normalized_embeddings, normalized_embeddings.T)
            logger.info("Used CPU similarity calculation")

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
        import traceback

        logger.error(traceback.format_exc())
        return pd.Series(0, index=species_df.index), 0


def process_all_species(df, embeddings_dict, species_list, args, logger):
    """Process all species with checkpointing."""
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")

    # Try to load checkpoint
    checkpoint = load_checkpoint(checkpoint_dir, logger)
    if checkpoint:
        results = checkpoint["results"]
        stats = checkpoint["stats"]
        processed_species = set(checkpoint["species_processed"])
        logger.info(
            f"Resuming from checkpoint with {len(processed_species)} species already processed"
        )
    else:
        results = pd.Series(index=df.index, dtype=int)
        stats = {}
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
                # Cluster well-sampled species
                logger.info(
                    f"{species_name} is well-sampled ({len(species_df)} samples)"
                )
                clusters, optimal_k, score = cluster_well_sampled_species(
                    species_df,
                    embeddings_dict,
                    args.min_k,
                    args.max_k,
                    args.tsne_dims,
                    args.use_gpu,
                    logger,
                )
                method = "clustering"
                cluster_stats = {"optimal_k": optimal_k, "silhouette_score": score}

            else:
                # Process under-sampled species
                logger.info(
                    f"{species_name} is under-sampled ({len(species_df)} samples)"
                )
                clusters, outlier_count = process_under_sampled_species(
                    species_df,
                    embeddings_dict,
                    args.outlier_threshold,
                    args.use_gpu,
                    logger,
                )
                method = "outlier_detection"
                cluster_stats = {"outlier_count": outlier_count}

            # Store results
            results[species_df.index] = clusters
            stats[species_name] = {
                "count": len(species_df),
                "method": method,
                **cluster_stats,
            }
            newly_processed.append(species_name)

            # Save checkpoint based on count or time
            current_time = time.time()
            time_to_checkpoint = (len(newly_processed) % checkpoint_interval == 0) or (
                (current_time - last_checkpoint_time) > time_based_checkpoint_interval
            )

            if time_to_checkpoint:
                all_processed = list(processed_species) + newly_processed
                if save_checkpoint(all_processed, results, stats, checkpoint_dir):
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
        if save_checkpoint(all_processed, results, stats, checkpoint_dir):
            logger.info(
                f"Saved final checkpoint: {len(all_processed)}/{total_species} species processed"
            )

    except Exception as e:
        logger.error(f"Error in process_all_species: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())

        # Save checkpoint on error to preserve progress
        if newly_processed:
            all_processed = list(processed_species) + newly_processed
            if save_checkpoint(all_processed, results, stats, checkpoint_dir):
                logger.info(
                    f"Saved emergency checkpoint after error: {len(all_processed)}/{total_species} species processed"
                )

        # Re-raise to allow main error handling to take over
        raise

    return results, stats


def write_results_to_sqlite(
    db_path, table_name, column_name, df, cluster_results, logger
):
    """Write clustering results back to SQLite database with robust error handling."""
    # Set initial retry parameters
    max_retries = 5
    retry_delay = 5  # seconds

    for attempt in range(max_retries):
        try:
            logger.info(
                f"Writing results to {db_path}, column {column_name} (attempt {attempt+1})"
            )

            # Prepare data for update - ensure cluster results are standard Python ints
            update_df = pd.DataFrame(
                {"uuid": df["uuid"], column_name: cluster_results.astype(int)}
            )

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

            # Ensure column exists
            try:
                conn.execute(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} INTEGER"
                )
                conn.commit()
                logger.info(f"Column {column_name} added")
            except sqlite3.OperationalError:
                logger.info(f"Column {column_name} already exists")

            # Start transaction
            conn.execute("BEGIN TRANSACTION")

            # Prepare and execute update statement
            cursor = conn.cursor()

            # Update in batches to avoid memory issues
            batch_size = 5000  # Smaller batch size to reduce lock time
            total_rows = len(update_df)
            updated_rows = 0

            for i in tqdm(range(0, total_rows, batch_size), desc="Writing to database"):
                batch = update_df.iloc[i : i + batch_size]
                batch_updates = []
                for _, row in batch.iterrows():
                    # Make sure we're passing the column value first, then the UUID
                    cluster_val = int(row[column_name])  # Force Python int type
                    uuid_val = row["uuid"]
                    batch_updates.append((cluster_val, uuid_val))

                try:
                    cursor.executemany(
                        f"UPDATE {table_name} SET {column_name} = ? WHERE uuid = ?",
                        batch_updates,
                    )
                    # Commit every few batches to reduce lock time
                    if i % (batch_size * 10) == 0 and i > 0:
                        conn.commit()
                        conn.execute("BEGIN TRANSACTION")
                        logger.info(f"Intermediate commit at {i} rows")

                    updated_rows += len(batch)
                except sqlite3.OperationalError as update_err:
                    logger.error(f"Error updating batch {i//batch_size}: {update_err}")
                    # Try to continue with next batch
                    continue

            # Commit final transaction
            try:
                conn.commit()
                logger.info(f"Successfully updated {updated_rows} rows")
            except sqlite3.OperationalError as commit_err:
                logger.error(f"Error during final commit: {commit_err}")
                if attempt < max_retries - 1:
                    conn.close()
                    continue
                else:
                    return updated_rows

            # Close connection
            conn.close()
            logger.info("Database connection closed successfully")

            return updated_rows

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
            import traceback

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
        with open(output_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved clustering statistics to {output_file}")
        return True
    except Exception as e:
        logger.error(f"Error saving stats: {str(e)}")
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
        description="Single-node clustering of image embeddings by species"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--table-name", default="image_embeddings", help="Name of table in SQLite"
    )
    parser.add_argument(
        "--column-name",
        default="proto_cluster",
        help="Column name to store clustering results",
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")
    parser.add_argument(
        "--min-samples-for-clustering",
        type=int,
        default=50,
        help="Minimum samples required for clustering",
    )
    parser.add_argument("--min-k", type=int, default=2, help="Minimum clusters to try")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum clusters to try")
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
        default="clustering_stats.json",
        help="JSON file to save clustering statistics",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration for t-SNE and KMeans if available",
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
    logger.info("Starting single-node species clustering")
    logger.info(
        f"Parameters: db={args.db_path}, table={args.table_name}, "
        f"column={args.column_name}, min_samples={args.min_samples_for_clustering}, "
        f"k_range=[{args.min_k}, {args.max_k}], tsne_dims={args.tsne_dims}, "
        f"use_gpu={args.use_gpu}, resume={args.resume}"
    )

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

    if not checkpoint or "complete" not in checkpoint or not checkpoint["complete"]:
        # Load data from SQLite
        load_start = time.time()
        df, embeddings_dict, species_list = load_data_from_sqlite(
            args.db_path, args.table_name, logger
        )

        if len(df) == 0:
            logger.error("No data loaded from database")
            return

        load_time = time.time() - load_start
        logger.info(f"Data loading completed in {load_time:.2f}s")
    else:
        logger.info(
            "Found complete results in checkpoint, skipping data loading and processing"
        )
        df = checkpoint["df"] if "df" in checkpoint else pd.DataFrame()
        results = checkpoint["results"]
        stats = checkpoint["stats"]
        # Skip to writing results

    # Process species if we don't have complete results yet
    if not checkpoint or "complete" not in checkpoint or not checkpoint["complete"]:
        process_start = time.time()
        cluster_results, stats = process_all_species(
            df, embeddings_dict, species_list, args, logger
        )
        process_time = time.time() - process_start
        logger.info(f"Processing completed in {process_time:.2f}s")

        # Mark processing as complete in the checkpoint
        all_processed = list(species_list)
        save_checkpoint(all_processed, cluster_results, stats, checkpoint_dir)
        # Save a final "complete" flag
        with open(os.path.join(checkpoint_dir, "checkpoint.pkl"), "rb") as f:
            checkpoint = pickle.load(f)
        checkpoint["complete"] = True
        checkpoint["df"] = df  # Save DataFrame for reference
        with open(os.path.join(checkpoint_dir, "checkpoint.pkl"), "wb") as f:
            pickle.dump(checkpoint, f)
        logger.info("Marked checkpoint as complete")
    else:
        # Use the results from the checkpoint
        cluster_results = checkpoint["results"]
        stats = checkpoint["stats"]
        process_time = 0  # No processing time in this case

    # Convert numpy types to Python types for JSON serialization
    stats = numpy_to_python_types(stats)

    # Write results back to SQLite unless skipped
    write_time = 0
    updated_rows = 0
    if not args.skip_db_write:
        write_start = time.time()
        updated_rows = write_results_to_sqlite(
            args.db_path,
            args.table_name,
            args.column_name,
            df,
            cluster_results,
            logger,
        )
        write_time = time.time() - write_start
        logger.info(f"Database update completed in {write_time:.2f}s")

    # Save statistics
    stats_file = os.path.join(args.log_dir, args.stats_output)
    save_stats(stats, stats_file, logger)

    # Log final statistics
    total_time = time.time() - start_time
    logger.info("\nFinal Statistics:")
    logger.info(f"Total species processed: {len(stats)}")
    logger.info(f"Total records processed: {len(df)}")
    logger.info(f"Total database updates: {updated_rows}")
    logger.info(f"Total processing time: {total_time:.2f} seconds")

    # Only show detailed time breakdown if we did actual work
    if process_time > 0 or write_time > 0:
        load_time = load_time if "load_time" in locals() else 0
        process_pct = (process_time / total_time * 100) if process_time > 0 else 0
        write_pct = (write_time / total_time * 100) if write_time > 0 else 0
        load_pct = (load_time / total_time * 100) if load_time > 0 else 0

        logger.info(f"  - Data loading: {load_time:.2f}s ({load_pct:.1f}%)")
        logger.info(f"  - Processing: {process_time:.2f}s ({process_pct:.1f}%)")
        logger.info(f"  - Database update: {write_time:.2f}s ({write_pct:.1f}%)")

    logger.info("Clustering job completed successfully")


if __name__ == "__main__":
    main()
