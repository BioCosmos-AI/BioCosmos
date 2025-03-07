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
import pickle
import signal
import sys


def setup_logging(rank, log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(
        log_dir, f"clustering_tsne_k_means_1_{timestamp}_rank{rank}.log"
    )

    # Configure logging
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter(
        f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def save_checkpoint(
    rank, species_processed, results_so_far, stats_so_far, checkpoint_dir
):
    """Save checkpoint data to allow resuming after interruption.

    Args:
        rank: The process rank
        species_processed: List of species names that have been processed
        results_so_far: Series containing clustering results
        stats_so_far: Dictionary of statistics
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_rank_{rank}.pkl")

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


def load_checkpoint(rank, checkpoint_dir, logger):
    """Load checkpoint data if available.

    Args:
        rank: The process rank
        checkpoint_dir: Directory containing checkpoints
        logger: Logger for output messages

    Returns:
        Dictionary with checkpoint data if found, None otherwise
    """
    checkpoint_file = os.path.join(checkpoint_dir, f"checkpoint_rank_{rank}.pkl")
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


def setup_distributed(output_dir):
    """Set up distributed training environment using GLOO backend for better compatibility."""
    # Set up basic logging first
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    # Get distributed parameters
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Get master address from environment variable (set in the SLURM script)
    master_addr = os.environ.get("MASTER_ADDR")
    if not master_addr:
        # Fallback if not set in environment
        import socket

        master_addr = socket.gethostname()

    master_port = int(os.environ.get("MASTER_PORT", "12355"))

    logging.info(f"Using master node: {master_addr}")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Add some debugging info
    logging.info(f"SLURM_NODELIST: {os.environ['SLURM_NODELIST']}")
    logging.info(f"SLURM_PROCID: {rank}")
    logging.info(f"MASTER_ADDR: {master_addr}")
    logging.info(f"MASTER_PORT: {master_port}")

    # Initialize process group - explicitly use GLOO backend for better compatibility
    try:
        # Use a longer timeout for initialization
        timeout = datetime.timedelta(minutes=30)
        dist.init_process_group(
            "gloo", rank=rank, world_size=world_size, timeout=timeout
        )
        logging.info("Successfully initialized process group with GLOO backend")
    except Exception as e:
        logging.error(f"Failed to initialize with GLOO: {e}")
        raise

    # Set device for operations that need a GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logging.info(f"Set CUDA device to local_rank {local_rank}")
    else:
        logging.warning("CUDA not available, using CPU only")

    # Now set up proper logging to file
    logger = setup_logging(rank, output_dir)

    logger.info(
        f"Initialized distributed process: rank {rank}/{world_size} on {master_addr}"
    )

    return rank, world_size, local_rank, logger


# def setup_distributed(output_dir):
#     """Set up distributed training environment."""
#     # Set up basic logging first
#     logging.basicConfig(
#         level=logging.INFO,
#         format="%(asctime)s - %(levelname)s - %(message)s",
#         handlers=[logging.StreamHandler()],
#     )

#     # Get distributed parameters
#     rank = int(os.environ["SLURM_PROCID"])
#     world_size = int(os.environ["SLURM_NTASKS"])
#     local_rank = int(os.environ["SLURM_LOCALID"])

#     # Get master address from environment variable (set in the SLURM script)
#     master_addr = os.environ.get("MASTER_ADDR")
#     if not master_addr:
#         # Fallback if not set in environment
#         import socket

#         master_addr = socket.gethostname()

#     master_port = int(os.environ.get("MASTER_PORT", "12355"))

#     logging.info(f"Using master node: {master_addr}")
#     os.environ["MASTER_ADDR"] = master_addr
#     os.environ["MASTER_PORT"] = str(master_port)

#     # Add some debugging info
#     logging.info(f"SLURM_NODELIST: {os.environ['SLURM_NODELIST']}")
#     logging.info(f"SLURM_PROCID: {rank}")
#     logging.info(f"MASTER_ADDR: {master_addr}")
#     logging.info(f"MASTER_PORT: {master_port}")

#     # Initialize process group
#     try:
#         dist.init_process_group("nccl", rank=rank, world_size=world_size)
#         logging.info("Successfully initialized process group with NCCL backend")
#     except Exception as e:
#         logging.error(f"Failed to initialize with NCCL: {e}")
#         logging.info("Trying with GLOO backend instead...")
#         try:
#             # If NCCL fails, try GLOO
#             dist.init_process_group("gloo", rank=rank, world_size=world_size)
#             logging.info("Successfully initialized process group with GLOO backend")
#         except Exception as e2:
#             logging.error(f"Failed to initialize with GLOO: {e2}")
#             raise

#     torch.cuda.set_device(local_rank)

#     # Now set up proper logging to file
#     logger = setup_logging(rank, output_dir)

#     logger.info(
#         f"Initialized distributed process: rank {rank}/{world_size} on {master_addr}"
#     )

#     return rank, world_size, local_rank, logger


def load_and_distribute_data(db_path, table_name, rank, world_size, logger):
    """Rank 0 loads entire data and distributes to other ranks using GLOO backend."""

    # Create a GLOO process group for reliable data sharing
    try:
        # Initialize a separate GLOO process group for data distribution
        gloo_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        logger.info(f"Created GLOO process group for data distribution")
    except Exception as e:
        logger.error(f"Error creating GLOO group: {str(e)}")
        # Continue with default process group
        gloo_group = None
        logger.info("Will use default process group for communication")

    if rank == 0:
        logger.info(f"Rank 0: Loading entire table from {db_path} into memory")

        try:
            # Only rank 0 accesses the database
            connection = sqlite3.connect(db_path, timeout=300.0)

            # Use a simple query to get all data
            query = f"SELECT uuid, species_name, embedding FROM {table_name}"
            logger.info("Executing simple SELECT query")

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

            # Distribute species among ranks
            species_chunks = np.array_split(all_species, world_size)

            # Prepare data for each rank
            rank_data = []
            for r in range(world_size):
                r_species = species_chunks[r].tolist()
                r_df = raw_df[raw_df["species_name"].isin(r_species)].copy()
                r_uuids = r_df["uuid"].tolist()
                r_embeddings = {
                    uuid: embeddings_dict[uuid]
                    for uuid in r_uuids
                    if uuid in embeddings_dict
                }

                rank_data.append(
                    {
                        "df": r_df,
                        "embeddings_dict": r_embeddings,
                        "species_list": r_species,
                    }
                )

                logger.info(
                    f"Prepared data for rank {r}: {len(r_species)} species, {len(r_df)} samples"
                )

            # Free up memory
            del raw_df
            del embeddings_dict

            # Set success flag
            success_flag = 1

        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            # Failed to load data
            rank_data = [
                {"df": pd.DataFrame(), "embeddings_dict": {}, "species_list": []}
                for _ in range(world_size)
            ]
            success_flag = 0
    else:
        # Other ranks don't need these initialized
        success_flag = 0
        rank_data = None

    # Broadcast success flag from rank 0 to all ranks
    # Create tensor on the appropriate device based on available process group
    if gloo_group:
        # GLOO can handle CPU tensors
        success_tensor = torch.tensor([success_flag], dtype=torch.int32)
        dist.broadcast(success_tensor, 0, group=gloo_group)
    else:
        # Default group might be NCCL which requires CUDA tensors
        if torch.cuda.is_available():
            success_tensor = torch.tensor(
                [success_flag], dtype=torch.int32, device="cuda"
            )
            dist.broadcast(success_tensor, 0)
        else:
            # If no CUDA available, try with CPU tensor but this might fail with NCCL
            logger.warning(
                "No CUDA available but might be using NCCL backend - this could fail"
            )
            success_tensor = torch.tensor([success_flag], dtype=torch.int32)
            try:
                dist.broadcast(success_tensor, 0)
            except RuntimeError as e:
                logger.error(f"Failed to broadcast with default group: {e}")
                logger.error(
                    "This likely means NCCL is being used with CPU tensors, which is not supported"
                )
                if rank == 0:
                    # If rank 0, create a dummy success value for other ranks
                    success_flag = 0
                else:
                    # If not rank 0, assume failure
                    success_flag = 0

    success_flag = success_tensor.item()

    if success_flag == 0:
        logger.error("Rank 0 failed to load data, aborting")
        return pd.DataFrame(), {}, []

    # Now distribute data to all ranks
    if rank == 0:
        # For rank 0, just use its own data directly
        my_data = rank_data[0]

        # For each rank > 0, send their data
        for r in range(1, world_size):
            try:
                # Pickle the data for this rank
                pickled_data = pickle.dumps(rank_data[r])

                # Send the size first as an integer tensor
                # Make sure to use the right device for the tensor
                if gloo_group:
                    size_tensor = torch.tensor([len(pickled_data)], dtype=torch.int64)
                    dist.send(size_tensor, dst=r, group=gloo_group)
                else:
                    if torch.cuda.is_available():
                        size_tensor = torch.tensor(
                            [len(pickled_data)], dtype=torch.int64, device="cuda"
                        )
                    else:
                        size_tensor = torch.tensor(
                            [len(pickled_data)], dtype=torch.int64
                        )
                    dist.send(size_tensor, dst=r)

                logger.info(f"Sent size {size_tensor.item()} to rank {r}")

                # Break data into chunks of 10MB to avoid memory issues
                chunk_size = 10 * 1024 * 1024
                for i in range(0, len(pickled_data), chunk_size):
                    end = min(i + chunk_size, len(pickled_data))
                    chunk = pickled_data[i:end]

                    # Make sure to use the right device for the tensor
                    if gloo_group:
                        chunk_tensor = torch.ByteTensor(list(chunk))
                        dist.send(chunk_tensor, dst=r, group=gloo_group)
                    else:
                        if torch.cuda.is_available():
                            chunk_tensor = torch.tensor(
                                list(chunk), dtype=torch.uint8, device="cuda"
                            )
                        else:
                            chunk_tensor = torch.ByteTensor(list(chunk))
                        dist.send(chunk_tensor, dst=r)

                    logger.info(
                        f"Sent chunk {i//chunk_size + 1} of {(len(pickled_data) + chunk_size - 1)//chunk_size} to rank {r}"
                    )
            except Exception as e:
                logger.error(f"Error sending data to rank {r}: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

        return my_data["df"], my_data["embeddings_dict"], my_data["species_list"]
    else:
        # For other ranks, receive data from rank 0
        try:
            # First receive the size
            if gloo_group:
                size_tensor = torch.tensor([0], dtype=torch.int64)
                dist.recv(size_tensor, src=0, group=gloo_group)
            else:
                if torch.cuda.is_available():
                    size_tensor = torch.tensor([0], dtype=torch.int64, device="cuda")
                else:
                    size_tensor = torch.tensor([0], dtype=torch.int64)
                dist.recv(size_tensor, src=0)

            total_size = size_tensor.item()
            logger.info(f"Will receive {total_size} bytes from rank 0")

            # Receive the data in chunks
            data_chunks = []
            chunk_size = 10 * 1024 * 1024
            num_chunks = (total_size + chunk_size - 1) // chunk_size

            for i in range(num_chunks):
                # For the last chunk, use the remaining size
                if i == num_chunks - 1:
                    remaining = total_size - i * chunk_size
                    if gloo_group:
                        chunk_tensor = torch.empty(remaining, dtype=torch.uint8)
                    else:
                        if torch.cuda.is_available():
                            chunk_tensor = torch.empty(
                                remaining, dtype=torch.uint8, device="cuda"
                            )
                        else:
                            chunk_tensor = torch.empty(remaining, dtype=torch.uint8)
                else:
                    if gloo_group:
                        chunk_tensor = torch.empty(chunk_size, dtype=torch.uint8)
                    else:
                        if torch.cuda.is_available():
                            chunk_tensor = torch.empty(
                                chunk_size, dtype=torch.uint8, device="cuda"
                            )
                        else:
                            chunk_tensor = torch.empty(chunk_size, dtype=torch.uint8)

                if gloo_group:
                    dist.recv(chunk_tensor, src=0, group=gloo_group)
                else:
                    dist.recv(chunk_tensor, src=0)

                # If tensor is on GPU, move to CPU for conversion to bytes
                if chunk_tensor.device.type == "cuda":
                    chunk_tensor = chunk_tensor.cpu()

                data_chunks.append(chunk_tensor.numpy().tobytes())
                logger.info(f"Received chunk {i+1}/{num_chunks} from rank 0")

            # Combine all chunks
            all_data = b"".join(data_chunks)

            # Deserialize the data
            my_data = pickle.loads(all_data)
            logger.info(
                f"Successfully deserialized data: {len(my_data['species_list'])} species, {len(my_data['df'])} samples"
            )

            return my_data["df"], my_data["embeddings_dict"], my_data["species_list"]
        except Exception as e:
            logger.error(f"Error receiving data from rank 0: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return pd.DataFrame(), {}, []


# def load_data_from_sqlite(db_path, table_name, rank, world_size, logger):
#     """Load entire SQLite table into DataFrame, then do all filtering in pandas."""
#     logger.info(f"Loading entire table from {db_path} into memory")

#     # Set initial retry parameters
#     max_retries = 5
#     retry_delay = 5  # seconds

#     for attempt in range(max_retries):
#         try:
#             # Add delay between ranks to reduce contention
#             time.sleep(rank * 3)  # Stagger access based on rank

#             # Use simplest possible query - just read the entire table
#             logger.info(f"Attempt {attempt+1}: Reading entire table into DataFrame")
#             query = f"SELECT uuid, species_name, embedding FROM {table_name}"

#             # Read the data without processing the BLOB column yet
#             raw_df = pd.read_sql_query(query, f"sqlite:///{db_path}")
#             logger.info(f"Successfully loaded {len(raw_df)} rows into DataFrame")

#             # Now do all the filtering in pandas
#             logger.info("Filtering in pandas and distributing species")

#             # Filter for non-null species names
#             raw_df = raw_df[
#                 raw_df["species_name"].notna() & (raw_df["species_name"] != "")
#             ]
#             logger.info(f"After filtering: {len(raw_df)} rows remain")

#             if len(raw_df) == 0:
#                 logger.warning("No valid data found after filtering")
#                 return pd.DataFrame(), {}, []

#             # Extract and convert embeddings
#             embeddings_dict = {}
#             for i, row in tqdm(
#                 raw_df.iterrows(), total=len(raw_df), desc="Processing embeddings"
#             ):
#                 try:
#                     # Convert BLOB to numpy array
#                     embedding = np.frombuffer(row["embedding"], dtype=np.float32)
#                     embeddings_dict[row["uuid"]] = embedding
#                 except Exception as embed_err:
#                     logger.warning(
#                         f"Error with embedding for UUID {row['uuid']}: {str(embed_err)}"
#                     )
#                     # Add empty embedding as placeholder
#                     embeddings_dict[row["uuid"]] = np.zeros(1024, dtype=np.float32)

#             # Get unique species and sort them for consistent distribution
#             all_species = sorted(raw_df["species_name"].unique())
#             logger.info(f"Found {len(all_species)} unique species")

#             # Distribute species alphabetically, ensuring all samples of a species
#             # go to the same rank
#             species_chunks = np.array_split(all_species, world_size)
#             my_species = species_chunks[rank].tolist()

#             logger.info(f"Rank {rank} will process {len(my_species)} species")
#             if len(my_species) > 0:
#                 logger.info(f"Species range: {my_species[0]} to {my_species[-1]}")

#             # Filter DataFrame to only include species for this rank
#             filtered_df = raw_df[raw_df["species_name"].isin(my_species)].copy()
#             logger.info(f"Filtered to {len(filtered_df)} rows for assigned species")

#             # Clean up memory
#             del raw_df

#             # Return results
#             return filtered_df, embeddings_dict, my_species

#         except Exception as e:
#             logger.error(f"Error loading data on attempt {attempt+1}: {str(e)}")
#             import traceback

#             logger.error(traceback.format_exc())

#             if attempt < max_retries - 1:
#                 wait_time = retry_delay * (attempt + 1)
#                 logger.info(f"Retrying in {wait_time} seconds...")
#                 time.sleep(wait_time)
#             else:
#                 logger.error("Maximum retries reached, giving up")
#                 return pd.DataFrame(), {}, []


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
            labels_cpu = (
                labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels
            )
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

        # Compute t-SNE embedding - check for embedding size issues
        if embeddings.shape[1] != embeddings.shape[1]:
            logger.error(f"Inconsistent embedding dimensions for {species_name}")
            return pd.Series(0, index=species_df.index), None, None

        logger.info(
            f"Computing {tsne_dims}D t-SNE for {species_name} ({len(embeddings)} samples)"
        )

        # t-SNE computation with proper error handling
        tsne_result = None
        if use_gpu and torch.cuda.is_available():
            try:
                # Use GPU-accelerated t-SNE
                tsne = cuTSNE(n_components=tsne_dims, random_state=42)
                tsne_result = tsne.fit_transform(embeddings)

                # Convert back to numpy if needed
                if not isinstance(tsne_result, np.ndarray):
                    tsne_result = (
                        tsne_result.get()
                        if hasattr(tsne_result, "get")
                        else tsne_result.cpu().numpy()
                    )

                logger.info("Used GPU-accelerated t-SNE")
            except Exception as e:
                logger.warning(f"GPU t-SNE failed, falling back to CPU: {str(e)}")
                tsne_result = None

        # Fall back to CPU t-SNE if needed
        if tsne_result is None:
            try:
                tsne = TSNE(n_components=tsne_dims, random_state=42)
                tsne_result = tsne.fit_transform(embeddings)
            except Exception as e:
                logger.error(f"CPU t-SNE failed for {species_name}: {str(e)}")
                return pd.Series(0, index=species_df.index), None, None

        # Verify t-SNE output
        if tsne_result is None or len(tsne_result) == 0:
            logger.error(f"t-SNE failed to produce output for {species_name}")
            return pd.Series(0, index=species_df.index), None, None

        # Find optimal k using silhouette score
        logger.info(f"Finding optimal k for {species_name}")

        optimal_k = None
        score = None

        if use_gpu and torch.cuda.is_available():
            try:
                optimal_k, score = find_optimal_k_gpu(
                    tsne_result, [min_k, max_k], logger
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
                    tsne_result, [min_k, max_k], logger
                )
            except Exception as e:
                logger.error(
                    f"CPU KMeans optimization failed for {species_name}: {str(e)}"
                )
                # Use default K as fallback
                optimal_k = min_k
                score = 0.0

        logger.info(f"Optimal k for {species_name}: {optimal_k} (score: {score:.4f})")

        # Perform final clustering with optimal k
        clusters = None
        if use_gpu and torch.cuda.is_available():
            try:
                kmeans = cuKMeans(n_clusters=optimal_k, random_state=42)
                clusters = kmeans.fit_predict(tsne_result)
                if not isinstance(clusters, np.ndarray):
                    clusters = (
                        clusters.get()
                        if hasattr(clusters, "get")
                        else clusters.cpu().numpy()
                    )
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
                clusters = kmeans.fit_predict(tsne_result)
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
        embeddings = np.array([embeddings_dict[uuid] for uuid in uuids])

        if len(embeddings) <= 1:
            # Can't compute similarities with just one sample
            logger.info(
                f"Species {species_name} has only {len(embeddings)} samples - marking as valid"
            )
            return pd.Series(0, index=species_df.index), 0

        # Normalize embeddings for cosine similarity
        normalized_embeddings = embeddings / np.linalg.norm(
            embeddings, axis=1, keepdims=True
        )

        # Compute cosine similarities
        if use_gpu and torch.cuda.is_available():
            try:
                # Try GPU-accelerated distance calculation
                sim_matrix = 1 - pairwise_distances(
                    normalized_embeddings, normalized_embeddings, metric="euclidean"
                )
                if not isinstance(sim_matrix, np.ndarray):
                    sim_matrix = (
                        sim_matrix.get()
                        if hasattr(sim_matrix, "get")
                        else sim_matrix.cpu().numpy()
                    )
                logger.info("Used GPU-accelerated similarity calculation")
            except Exception as e:
                logger.warning(
                    f"GPU similarity calculation failed, falling back to CPU: {str(e)}"
                )
                sim_matrix = 1 - np.matmul(
                    normalized_embeddings, normalized_embeddings.T
                )
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

        logger.info(
            f"Species {species_name}: mean sim={mean_sim:.4f}, std={std_sim:.4f}"
        )
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
    """Process all species assigned to this rank with checkpointing."""
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")

    # Try to load checkpoint
    checkpoint = load_checkpoint(rank, checkpoint_dir, logger)
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
        f"Rank {rank} processing {len(remaining_species)} remaining species out of {total_species} total"
    )

    # Track processed species for checkpointing
    newly_processed = []
    checkpoint_interval = 5  # Save checkpoint every 5 species
    last_checkpoint_time = time.time()
    time_based_checkpoint_interval = 10 * 60  # 10 minutes

    try:
        for i, species_name in enumerate(
            tqdm(remaining_species, desc=f"Rank {rank} processing species")
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
                if save_checkpoint(rank, all_processed, results, stats, checkpoint_dir):
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
        if save_checkpoint(rank, all_processed, results, stats, checkpoint_dir):
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
            if save_checkpoint(rank, all_processed, results, stats, checkpoint_dir):
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


def gather_and_save_stats(stats, args, rank, world_size, logger):
    """Gather statistics from all ranks and save to a JSON file."""
    # Convert numpy types to Python native types
    stats = numpy_to_python_types(stats)

    # If single process, just save directly
    if world_size == 1:
        stats_file = os.path.join(args.log_dir, args.stats_output)
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved clustering statistics to {stats_file}")
        return stats

    # In distributed mode, gather stats from all processes
    try:
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
            with open(stats_file, "w") as f:
                json.dump(combined_stats, f, indent=2)
            logger.info(f"Saved combined clustering statistics to {stats_file}")
            return combined_stats
        else:
            dist.gather_object(stats_json, None, dst=0)
            return None
    except Exception as e:
        logger.error(f"Error gathering/saving stats: {str(e)}")
        # Fallback: save just this rank's stats
        stats_file = os.path.join(
            args.log_dir, f"{args.stats_output.split('.')[0]}_rank{rank}.json"
        )
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved rank-specific stats to {stats_file}")
        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Distributed clustering of image embeddings by species"
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
        help="Use GPU acceleration for t-SNE and KMeans",
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

    # Set up distributed environment
    try:
        rank, world_size, local_rank, logger = setup_distributed(args.log_dir)
        start_time = time.time()

        # Set up signal handlers for graceful shutdown
        def signal_handler(sig, frame):
            logger.info(f"Received signal {sig}, shutting down gracefully...")
            if "dist" in globals() and world_size > 1:
                try:
                    dist.destroy_process_group()
                except:
                    pass
            sys.exit(0)

        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)

        # Log start information
        logger.info("Starting distributed species clustering")
        logger.info(
            f"Parameters: db={args.db_path}, table={args.table_name}, "
            f"column={args.column_name}, min_samples={args.min_samples_for_clustering}, "
            f"k_range=[{args.min_k}, {args.max_k}], tsne_dims={args.tsne_dims}, "
            f"use_gpu={args.use_gpu}, resume={args.resume}"
        )

        # Create checkpoints directory
        checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Check if we can resume from a checkpoint
        checkpoint = None
        if args.resume:
            checkpoint = load_checkpoint(rank, checkpoint_dir, logger)

        # If resuming and we have a valid checkpoint with complete results, we can skip loading from DB
        df = pd.DataFrame()
        embeddings_dict = {}
        species_list = []

        if not checkpoint or "complete" not in checkpoint or not checkpoint["complete"]:
            # Load data from SQLite
            load_start = time.time()
            # df, embeddings_dict, species_list = load_data_from_sqlite(
            #     args.db_path, args.table_name, rank, world_size, logger
            # )
            df, embeddings_dict, species_list = load_and_distribute_data(
                args.db_path, args.table_name, rank, world_size, logger
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

        # Process assigned species if we don't have complete results yet
        if not checkpoint or "complete" not in checkpoint or not checkpoint["complete"]:
            process_start = time.time()
            cluster_results, stats = process_all_species(
                df, embeddings_dict, species_list, args, rank, logger
            )
            process_time = time.time() - process_start
            logger.info(f"Processing completed in {process_time:.2f}s")

            # Mark processing as complete in the checkpoint
            all_processed = list(species_list)
            save_checkpoint(rank, all_processed, cluster_results, stats, checkpoint_dir)
            # Save a final "complete" flag
            with open(
                os.path.join(checkpoint_dir, f"checkpoint_rank_{rank}.pkl"), "rb"
            ) as f:
                checkpoint = pickle.load(f)
            checkpoint["complete"] = True
            checkpoint["df"] = df  # Save DataFrame for reference
            with open(
                os.path.join(checkpoint_dir, f"checkpoint_rank_{rank}.pkl"), "wb"
            ) as f:
                pickle.dump(checkpoint, f)
            logger.info("Marked checkpoint as complete")
        else:
            # Use the results from the checkpoint
            cluster_results = checkpoint["results"]
            stats = checkpoint["stats"]
            process_time = 0  # No processing time in this case

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

        # Gather and save statistics from all ranks
        gather_and_save_stats(stats, args, rank, world_size, logger)

        # Log final statistics
        total_time = time.time() - start_time
        logger.info("\nFinal Statistics:")
        logger.info(f"Total species processed by this rank: {len(stats)}")
        logger.info(f"Total records processed by this rank: {len(df)}")
        logger.info(f"Total updates by this rank: {updated_rows}")
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

        # Clean up distributed environment
        if world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        if "dist" in globals() and world_size > 1:
            try:
                dist.destroy_process_group()
            except:
                pass
        raise


if __name__ == "__main__":
    main()
