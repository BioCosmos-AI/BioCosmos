import os
import sqlite3
import argparse
import logging
import numpy as np
import pandas as pd
import time
import json
import torch
import torch.distributed as dist
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import cudf
import cuml
from cuml.manifold import TSNE as cuTSNE
from cuml.cluster import KMeans as cuKMeans
from cuml.metrics import pairwise_distances
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def setup_logging(rank, log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"clustering_tsne_k_means_1_{timestamp}_rank{rank}.log")

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def setup_distributed(log_dir):
    """Set up distributed processing environment."""
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    world_size = int(os.environ.get("SLURM_NTASKS", "1"))
    local_rank = int(os.environ.get("SLURM_LOCALID", "0"))
    
    # Get the hostname of the first node
    nodes = os.environ.get("SLURM_NODELIST", "localhost")
    if "[" in nodes:
        # Handle node ranges like "c0800a-s[11,17,23]"
        prefix = nodes.split("[")[0]
        node_nums = nodes.split("[")[1].split("]")[0].split(",")
        master_addr = f"{prefix}{node_nums[0]}"
    else:
        # Handle single node or comma-separated list
        master_addr = nodes.split(",")[0]

    master_port = int(os.environ.get("MASTER_PORT", "12355"))

    # Set environment variables for distributed setup
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Initialize process group if running in distributed mode
    if world_size > 1:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
    
    # Set up logging
    logger = setup_logging(rank, log_dir)
    
    logger.info(f"Initialized process: rank {rank}/{world_size}")
    if world_size > 1:
        logger.info(f"Master: {master_addr}:{master_port}, Local rank: {local_rank}")
    
    return rank, world_size, local_rank, logger


def load_data_from_sqlite(db_path, table_name, rank, world_size, logger):
    """Load data from SQLite into a DataFrame, distributed by species alphabet ranges."""
    logger.info(f"Loading data from {db_path}")
    
    try:
        conn = sqlite3.connect(db_path, timeout=60.0)
        
        # First, get distinct species and their counts
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT species_name, COUNT(*) as count 
            FROM image_embeddings 
            WHERE species_name IS NOT NULL AND species_name != '' 
            GROUP BY species_name
            ORDER BY species_name
            """
        )
        all_species = cursor.fetchall()
        
        # Distribute species across ranks alphabetically
        # This helps balance load while keeping related species together
        species_per_rank = len(all_species) // world_size
        start_idx = rank * species_per_rank
        end_idx = (rank + 1) * species_per_rank if rank < world_size - 1 else len(all_species)
        
        my_species = all_species[start_idx:end_idx]
        species_names = [s[0] for s in my_species]
        
        logger.info(f"This rank will process {len(species_names)} species")
        
        # Build query to get data for assigned species
        placeholders = ','.join(['?'] * len(species_names))
        query = f"""
        SELECT uuid, species_name, embedding 
        FROM {table_name}
        WHERE species_name IN ({placeholders})
        """
        
        logger.info("Executing query...")
        cursor.execute(query, species_names)
        
        # Initialize lists to store data
        uuids = []
        species_names_list = []
        embeddings = []
        
        # Fetch data in chunks to manage memory
        chunk_size = 10000
        row_count = 0
        
        logger.info("Fetching data in chunks...")
        while True:
            rows = cursor.fetchmany(chunk_size)
            if not rows:
                break
                
            for uuid, species_name, embedding_blob in rows:
                uuids.append(uuid)
                species_names_list.append(species_name)
                
                # Convert BLOB to numpy array
                embedding = np.frombuffer(embedding_blob, dtype=np.float32)
                embeddings.append(embedding)
                
            row_count += len(rows)
            if row_count % 100000 == 0:
                logger.info(f"Loaded {row_count} rows so far...")
        
        conn.close()
        
        # Create DataFrame
        df = pd.DataFrame({
            'uuid': uuids,
            'species_name': species_names_list
        })
        
        # Store embeddings separately as they're numpy arrays
        embeddings_dict = {uuid: emb for uuid, emb in zip(uuids, embeddings)}
        
        logger.info(f"Loaded {len(df)} rows into DataFrame")
        return df, embeddings_dict, species_names
        
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        if 'conn' in locals():
            conn.close()
        return pd.DataFrame(), {}, []


def find_optimal_k_gpu(embeddings, k_range, logger):
    """Find optimal number of clusters using GPU-accelerated KMeans and silhouette scores."""
    best_score = -1
    best_k = 2  # Default to 2 clusters
    
    # Move data to GPU
    device = torch.device("cuda")
    emb_tensor = torch.tensor(embeddings, device=device, dtype=torch.float32)
    
    for k in range(k_range[0], k_range[1] + 1):
        try:
            # Use cuML KMeans implementation
            kmeans = cuKMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(emb_tensor)
            
            # Move results back to CPU for silhouette calculation
            labels_cpu = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            embeddings_cpu = embeddings
            
            # If we have only one cluster, silhouette score will fail
            if len(np.unique(labels_cpu)) < 2:
                continue
                
            score = silhouette_score(embeddings_cpu, labels_cpu)
            logger.debug(f"K={k}, Silhouette Score={score:.4f}")
            
            if score > best_score:
                best_score = score
                best_k = k
                
        except Exception as e:
            logger.error(f"Error evaluating k={k}: {str(e)}")
            continue
    
    return best_k, best_score


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


def cluster_well_sampled_species(species_df, embeddings_dict, min_k, max_k, tsne_dims, use_gpu, logger):
    """Cluster a well-sampled species using t-SNE and KMeans with optional GPU acceleration."""
    species_name = species_df['species_name'].iloc[0]
    
    try:
        # Extract embeddings for this species
        uuids = species_df['uuid'].tolist()
        embeddings = np.array([embeddings_dict[uuid] for uuid in uuids])
        
        # Compute t-SNE embedding
        logger.info(f"Computing {tsne_dims}D t-SNE for {species_name} ({len(embeddings)} samples)")
        
        if use_gpu and torch.cuda.is_available():
            try:
                # Use GPU-accelerated t-SNE
                tsne = cuTSNE(n_components=tsne_dims, random_state=42)
                tsne_result = tsne.fit_transform(embeddings)
                
                # Convert back to numpy if needed
                if not isinstance(tsne_result, np.ndarray):
                    tsne_result = tsne_result.get() if hasattr(tsne_result, 'get') else tsne_result.cpu().numpy()
                
                logger.info("Used GPU-accelerated t-SNE")
            except Exception as e:
                logger.warning(f"GPU t-SNE failed, falling back to CPU: {str(e)}")
                tsne = TSNE(n_components=tsne_dims, random_state=42)
                tsne_result = tsne.fit_transform(embeddings)
        else:
            # Use CPU t-SNE
            tsne = TSNE(n_components=tsne_dims, random_state=42)
            tsne_result = tsne.fit_transform(embeddings)
        
        # Find optimal k using silhouette score
        logger.info(f"Finding optimal k for {species_name}")
        
        if use_gpu and torch.cuda.is_available():
            try:
                optimal_k, score = find_optimal_k_gpu(tsne_result, [min_k, max_k], logger)
                logger.info("Used GPU-accelerated KMeans")
            except Exception as e:
                logger.warning(f"GPU KMeans failed, falling back to CPU: {str(e)}")
                optimal_k, score = find_optimal_k_cpu(tsne_result, [min_k, max_k], logger)
        else:
            optimal_k, score = find_optimal_k_cpu(tsne_result, [min_k, max_k], logger)
            
        logger.info(f"Optimal k for {species_name}: {optimal_k} (score: {score:.4f})")
        
        # Perform final clustering with optimal k
        if use_gpu and torch.cuda.is_available():
            try:
                kmeans = cuKMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(tsne_result)
                if not isinstance(clusters, np.ndarray):
                    clusters = clusters.get() if hasattr(clusters, 'get') else clusters.cpu().numpy()
            except Exception as e:
                logger.warning(f"GPU final KMeans failed, falling back to CPU: {str(e)}")
                kmeans = KMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(tsne_result)
        else:
            kmeans = KMeans(n_clusters=optimal_k, random_state=42)
            clusters = kmeans.fit_predict(tsne_result)
        
        # Return clusters and stats
        return pd.Series(clusters, index=species_df.index), optimal_k, score
        
    except Exception as e:
        logger.error(f"Error clustering {species_name}: {str(e)}")
        return pd.Series(0, index=species_df.index), None, None


def process_under_sampled_species(species_df, embeddings_dict, outlier_threshold, use_gpu, logger):
    """Process under-sampled species by finding outliers based on cosine similarity."""
    species_name = species_df['species_name'].iloc[0]
    
    try:
        # Extract embeddings for this species
        uuids = species_df['uuid'].tolist()
        embeddings = np.array([embeddings_dict[uuid] for uuid in uuids])
        
        if len(embeddings) <= 1:
            # Can't compute similarities with just one sample
            logger.info(f"Species {species_name} has only {len(embeddings)} samples - marking as valid")
            return pd.Series(0, index=species_df.index), 0
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Compute cosine similarities
        if use_gpu and torch.cuda.is_available():
            try:
                # Try GPU-accelerated distance calculation
                sim_matrix = 1 - pairwise_distances(
                    normalized_embeddings, 
                    normalized_embeddings,
                    metric='euclidean'
                )
                if not isinstance(sim_matrix, np.ndarray):
                    sim_matrix = sim_matrix.get() if hasattr(sim_matrix, 'get') else sim_matrix.cpu().numpy()
                logger.info("Used GPU-accelerated similarity calculation")
            except Exception as e:
                logger.warning(f"GPU similarity calculation failed, falling back to CPU: {str(e)}")
                sim_matrix = 1 - np.matmul(normalized_embeddings, normalized_embeddings.T)
        else:
            # Use CPU for cosine similarity
            sim_matrix = 1 - np.matmul(normalized_embeddings, normalized_embeddings.T)
        
        # For each sample, compute average similarity to all other samples
        np.fill_diagonal(sim_matrix, 0)  # Exclude self-similarity
        avg_similarities = sim_matrix.sum(axis=1) / (len(embeddings) - 1)
        
        # Compute mean and std of similarities
        mean_sim = np.mean(avg_similarities)
        std_sim = np.std(avg_similarities)
        
        # Identify outliers as embeddings with avg similarity < mean - threshold*std
        threshold_value = mean_sim - outlier_threshold * std_sim
        is_outlier = avg_similarities < threshold_value
        
        logger.info(f"Species {species_name}: mean sim={mean_sim:.4f}, std={std_sim:.4f}")
        logger.info(f"Species {species_name}: {np.sum(is_outlier)} outliers identified")
        
        # Create a series with 0 for valid samples, -1 for outliers
        cluster_series = pd.Series(0, index=species_df.index)
        outlier_indices = species_df.index[is_outlier]
        cluster_series.loc[outlier_indices] = -1
        
        return cluster_series, np.sum(is_outlier)
        
    except Exception as e:
        logger.error(f"Error processing under-sampled species {species_name}: {str(e)}")
        return pd.Series(0, index=species_df.index), 0


def process_all_species(df, embeddings_dict, species_list, args, rank, logger):
    """Process all species assigned to this rank."""
    results = pd.Series(index=df.index, dtype=int)
    stats = {}
    
    total_species = len(species_list)
    logger.info(f"Rank {rank} processing {total_species} unique species")
    
    for i, species_name in enumerate(tqdm(species_list, desc=f"Rank {rank} processing species")):
        # Get data for this species
        species_df = df[df['species_name'] == species_name]
        if len(species_df) == 0:
            continue
            
        logger.info(f"Processing {species_name} ({i+1}/{total_species}) with {len(species_df)} samples")
        
        is_well_sampled = len(species_df) >= args.min_samples_for_clustering
        
        if is_well_sampled:
            # Cluster well-sampled species
            logger.info(f"{species_name} is well-sampled ({len(species_df)} samples)")
            clusters, optimal_k, score = cluster_well_sampled_species(
                species_df, embeddings_dict,
                args.min_k, args.max_k, args.tsne_dims,
                args.use_gpu, logger
            )
            method = "clustering"
            cluster_stats = {"optimal_k": optimal_k, "silhouette_score": score}
            
        else:
            # Process under-sampled species
            logger.info(f"{species_name} is under-sampled ({len(species_df)} samples)")
            clusters, outlier_count = process_under_sampled_species(
                species_df, embeddings_dict,
                args.outlier_threshold, args.use_gpu, logger
            )
            method = "outlier_detection"
            cluster_stats = {"outlier_count": outlier_count}
        
        # Store results
        results[species_df.index] = clusters
        stats[species_name] = {
            "count": len(species_df),
            "method": method,
            **cluster_stats
        }
        
        # Log progress periodically
        if (i+1) % 10 == 0 or (i+1) == total_species:
            logger.info(f"Progress: {i+1}/{total_species} species processed")
    
    return results, stats


def write_results_to_sqlite(db_path, table_name, column_name, df, cluster_results, logger):
    """Write clustering results back to SQLite database."""
    try:
        logger.info(f"Writing results to {db_path}, column {column_name}")
        
        # Prepare data for update
        update_df = pd.DataFrame({
            'uuid': df['uuid'],
            column_name: cluster_results
        })
        
        # Connect to database
        conn = sqlite3.connect(db_path, timeout=60.0)
        
        # Ensure column exists
        try:
            conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} INTEGER")
            conn.commit()
            logger.info(f"Column {column_name} added")
        except sqlite3.OperationalError:
            logger.info(f"Column {column_name} already exists")
        
        # Start transaction
        conn.execute("BEGIN TRANSACTION")
        
        # Prepare and execute update statement
        cursor = conn.cursor()
        
        # Update in batches to avoid memory issues
        batch_size = 10000
        total_rows = len(update_df)
        
        for i in tqdm(range(0, total_rows, batch_size), desc="Writing to database"):
            batch = update_df.iloc[i:i+batch_size]
            batch_updates = list(batch.itertuples(index=False, name=None))
            cursor.executemany(
                f"UPDATE {table_name} SET {column_name} = ? WHERE uuid = ?",
                [(int(cluster), uuid) for cluster, uuid in batch_updates]
            )
        
        # Commit transaction
        conn.commit()
        logger.info(f"Successfully updated {total_rows} rows")
        
        # Close connection
        conn.close()
        
        return total_rows
        
    except Exception as e:
        logger.error(f"Error writing results to database: {str(e)}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return 0


def gather_and_save_stats(stats, args, rank, world_size, logger):
    """Gather statistics from all ranks and save to a JSON file."""
    # If single process, just save directly
    if world_size == 1:
        stats_file = os.path.join(args.log_dir, args.stats_output)
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved clustering statistics to {stats_file}")
        return stats
    
    # In distributed mode, gather stats from all processes
    all_stats = [None] * world_size
    
    # Convert stats to JSON string for gathering
    stats_json = json.dumps(stats)
    
    # Gather stats from all processes
    if rank == 0:
        stats_list = [None] * world_size
        dist.gather_object(stats_json, stats_list, dst=0)
        
        # Combine stats
        combined_stats = {}
        for s in stats_list:
            if s is not None:
                rank_stats = json.loads(s)
                combined_stats.update(rank_stats)
        
        # Save combined stats
        stats_file = os.path.join(args.log_dir, args.stats_output)
        with open(stats_file, 'w') as f:
            json.dump(combined_stats, f, indent=2)
        logger.info(f"Saved combined clustering statistics to {stats_file}")
        return combined_stats
    else:
        dist.gather_object(stats_json, None, dst=0)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Distributed clustering of image embeddings by species"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--table-name", default="image_embeddings", help="Name of table in SQLite")
    parser.add_argument("--column-name", default="proto_cluster", help="Column name to store clustering results")
    parser.add_argument("--log-dir", required=True, help="Directory for log files")
    parser.add_argument("--min-samples-for-clustering", type=int, default=50, 
                        help="Minimum samples required for clustering")
    parser.add_argument("--min-k", type=int, default=2, help="Minimum clusters to try")
    parser.add_argument("--max-k", type=int, default=10, help="Maximum clusters to try")
    parser.add_argument("--tsne-dims", type=int, default=2, choices=[2, 3], 
                        help="t-SNE dimensions (2D or 3D)")
    parser.add_argument("--outlier-threshold", type=float, default=2.0,
                        help="Threshold in std deviations for outlier detection")
    parser.add_argument("--stats-output", default="clustering_stats.json",
                        help="JSON file to save clustering statistics")
    parser.add_argument("--use-gpu", action="store_true", default=True,
                        help="Use GPU acceleration for t-SNE and KMeans")

    args = parser.parse_args()

    # Set up distributed environment
    rank, world_size, local_rank, logger = setup_distributed(args.log_dir)
    start_time = time.time()

    # Log start information
    logger.info("Starting distributed species clustering")
    logger.info(f"Parameters: db={args.db_path}, table={args.table_name}, "
                f"column={args.column_name}, min_samples={args.min_samples_for_clustering}, "
                f"k_range=[{args.min_k}, {args.max_k}], tsne_dims={args.tsne_dims}, "
                f"use_gpu={args.use_gpu}")

    try:
        # Load data from SQLite
        load_start = time.time()
        df, embeddings_dict, species_list = load_data_from_sqlite(
            args.db_path, args.table_name, rank, world_size, logger
        )
        if len(df) == 0:
            logger.error("No data loaded from database")
            return
            
        load_time = time.time() - load_start
        logger.info(f"Data loading completed in {load_time:.2f}s")
        
        # Process assigned species
        process_start = time.time()
        cluster_results, stats = process_all_species(
            df, embeddings_dict, species_list, args, rank, logger
        )
        process_time = time.time() - process_start
        logger.info(f"Processing completed in {process_time:.2f}s")
        
        # Write results back to SQLite
        write_start = time.time()
        updated_rows = write_results_to_sqlite(
            args.db_path, args.table_name, args.column_name, 
            df, cluster_results, logger
        )
        write_time = time.time() - write_start
        logger.info(f"Database update completed in {write_time:.2f}s")
        
        # Gather and save statistics from all ranks
        gather_and_save_stats(stats, args, rank, world_size, logger)
        
        # Log final statistics
        total_time = time.time() - start_time
        logger.info("\nFinal Statistics:")
        logger.info(f"Total species processed by this rank: {len(stats)}")
        logger.info(f"Total records processed by this rank: {len(df)}")
        logger.info(f"Total updates by this rank: {updated_rows}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")
        logger.info(f"  - Data loading: {load_time:.2f}s ({load_time/total_time*100:.1f}%)")
        logger.info(f"  - Processing: {process_time:.2f}s ({process_time/total_time*100:.1f}%)")
        logger.info(f"  - Database update: {write_time:.2f}s ({write_time/total_time*100:.1f}%)")
        
        # Clean up distributed environment
        if world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        if 'dist' in globals() and world_size > 1:
            dist.destroy_process_group()
        raise


if __name__ == "__main__":
    main()