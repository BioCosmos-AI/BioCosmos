#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Migrate data from SQLite to HDF5 format for faster loading.
"""

import os
import sys
import time
import argparse
import logging
import sqlite3
import pickle
import json
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
from datetime import datetime
import signal


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"sqlite_to_hdf5_{timestamp}.log")

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


def save_checkpoint(last_processed_id, total_processed, checkpoint_dir):
    """Save checkpoint data to allow resuming after interruption.

    Args:
        last_processed_id: The last processed row ID
        total_processed: Total number of rows processed
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
                    "last_processed_id": last_processed_id,
                    "total_processed": total_processed,
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
        print(f"Error saving checkpoint: {str(e)}")
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
            required_keys = ["last_processed_id", "total_processed", "timestamp"]
            if all(key in checkpoint for key in required_keys):
                age = time.time() - checkpoint["timestamp"]
                age_hours = age / 3600
                logger.info(
                    f"Loaded checkpoint with {checkpoint['total_processed']} "
                    f"processed rows (age: {age_hours:.2f} hours)"
                )
                return checkpoint
            else:
                logger.warning(f"Checkpoint file missing required data, ignoring")
                return None
        except Exception as e:
            logger.error(f"Error loading checkpoint: {str(e)}")
            return None
    return None


def get_table_schema(conn, table_name, logger):
    """Get the schema of the SQLite table.

    Args:
        conn: SQLite connection
        table_name: Name of the table
        logger: Logger for output messages

    Returns:
        List of column names and their types
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table_name})")
        schema = cursor.fetchall()

        # Schema format: (cid, name, type, notnull, dflt_value, pk)
        columns = [(col[1], col[2]) for col in schema]
        logger.info(f"Table schema: {columns}")
        return columns
    except Exception as e:
        logger.error(f"Error getting table schema: {str(e)}")
        return []


def get_row_count(conn, table_name, logger):
    """Get the total number of rows in the SQLite table.

    Args:
        conn: SQLite connection
        table_name: Name of the table
        logger: Logger for output messages

    Returns:
        Total number of rows
    """
    try:
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        logger.info(f"Total rows in SQLite table: {count}")
        return count
    except Exception as e:
        logger.error(f"Error getting row count: {str(e)}")
        return 0


def process_chunk(chunk_df, logger):
    """Process a chunk of data to prepare it for HDF5.

    Args:
        chunk_df: Pandas DataFrame with a chunk of data
        logger: Logger for output messages

    Returns:
        Processed DataFrame ready for HDF5
    """
    try:
        # Convert blob embeddings to numpy arrays
        embedding_arrays = []

        for i, row in enumerate(chunk_df["embedding"]):
            try:
                # Try to convert the blob to a numpy array
                if row is not None:
                    embedding = np.frombuffer(row, dtype=np.float32)
                    embedding_arrays.append(embedding)
                else:
                    # Handle NULL embeddings
                    embedding_arrays.append(np.zeros(1024, dtype=np.float32))
            except Exception as e:
                logger.warning(f"Error processing embedding at index {i}: {str(e)}")
                embedding_arrays.append(np.zeros(1024, dtype=np.float32))

        # Create a new DataFrame without the blob column
        processed_df = chunk_df.drop(columns=["embedding"])

        # Convert all object columns to string
        for col in processed_df.select_dtypes(include=["object"]).columns:
            processed_df[col] = processed_df[col].astype(str)

        # Handle any None/NaN values in string columns
        for col in ["uuid", "shard_id", "species_name", "common_name"]:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna("")

        # Add the processed embeddings as a separate item
        return processed_df, np.array(embedding_arrays)

    except Exception as e:
        logger.error(f"Error processing chunk: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return chunk_df.drop(columns=["embedding"]), np.array([])


def write_to_hdf5(h5file, processed_df, embeddings, group_name, logger):
    """Write processed data to HDF5 file.

    Args:
        h5file: Open HDF5 file
        processed_df: DataFrame with non-embedding columns
        embeddings: NumPy array of embeddings
        group_name: Name of the HDF5 group
        logger: Logger for output messages

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create or get the group
        if group_name in h5file:
            group = h5file[group_name]
        else:
            group = h5file.create_group(group_name)

        # Get the current size of datasets if they exist
        start_idx = 0
        if "uuid" in group:
            start_idx = len(group["uuid"])

        # Create or resize datasets for each column
        for col in processed_df.columns:
            data = processed_df[col].values

            # Handle different data types
            if col in ["uuid", "shard_id", "species_name", "common_name"]:
                # For string columns, ensure all values are strings
                str_data = [str(x) if x is not None else "" for x in data]

                # Use variable length string datatype
                string_dt = h5py.special_dtype(vlen=str)

                if col in group:
                    dataset = group[col]
                    curr_size = len(dataset)
                    dataset.resize((curr_size + len(str_data),))
                else:
                    dataset = group.create_dataset(
                        col,
                        shape=(len(str_data),),
                        maxshape=(None,),
                        dtype=string_dt,
                        chunks=True,
                        compression=None,
                    )

                dataset[start_idx : start_idx + len(str_data)] = str_data

            elif col in ["porto_suggested_cluster_exp_1", "single_node_cluster"]:
                # For integer columns, convert to numpy int32 array
                # Handle NaNs and None values too
                int_data = np.zeros(len(data), dtype=np.int32)
                for i, val in enumerate(data):
                    try:
                        if val is not None and not (
                            isinstance(val, float) and np.isnan(val)
                        ):
                            int_data[i] = int(val)
                    except (ValueError, TypeError):
                        logger.warning(f"Invalid value for {col}: {val}, using 0")

                if col in group:
                    dataset = group[col]
                    curr_size = len(dataset)
                    dataset.resize((curr_size + len(int_data),))
                else:
                    dataset = group.create_dataset(
                        col,
                        shape=(len(int_data),),
                        maxshape=(None,),
                        dtype=np.int32,
                        chunks=True,
                        compression=None,
                    )

                dataset[start_idx : start_idx + len(int_data)] = int_data

        # Add embeddings dataset
        if len(embeddings) > 0:
            embedding_dim = embeddings.shape[1] if embeddings.ndim > 1 else 0

            if "embeddings" in group:
                emb_dataset = group["embeddings"]
                curr_size = len(emb_dataset)
                emb_dataset.resize((curr_size + len(embeddings), embedding_dim))
            else:
                emb_dataset = group.create_dataset(
                    "embeddings",
                    shape=(len(embeddings), embedding_dim),
                    maxshape=(None, embedding_dim),
                    dtype=np.float32,
                    chunks=True,
                    compression=None,
                )

            emb_dataset[start_idx : start_idx + len(embeddings), :] = embeddings

        return True

    except Exception as e:
        logger.error(f"Error writing to HDF5: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


def migrate_data(args, checkpoint, logger):
    """Migrate data from SQLite to HDF5.

    Args:
        args: Command line arguments
        checkpoint: Checkpoint data if resuming
        logger: Logger for output messages

    Returns:
        True if successful, False otherwise
    """
    total_processed = 0
    last_id = 0

    if checkpoint:
        total_processed = checkpoint["total_processed"]
        last_id = checkpoint["last_processed_id"]
        logger.info(
            f"Resuming from last processed ID {last_id} ({total_processed} rows processed)"
        )

    try:
        # Connect to SQLite database
        logger.info(f"Connecting to SQLite database at {args.sqlite_path}")
        sqlite_conn = sqlite3.connect(args.sqlite_path, timeout=300.0)

        # Get table schema and row count
        schema = get_table_schema(sqlite_conn, args.table_name, logger)
        if not schema:
            logger.error("Failed to get table schema")
            return False

        total_rows = get_row_count(sqlite_conn, args.table_name, logger)
        if total_rows == 0:
            logger.error("No rows found in SQLite table")
            return False

        # Create or open HDF5 file
        # Use 'a' mode to append if file exists
        logger.info(f"Opening HDF5 file at {args.hdf5_path}")
        with h5py.File(args.hdf5_path, "a") as h5file:
            # Store metadata
            if "metadata" not in h5file:
                metadata_group = h5file.create_group("metadata")
                metadata_group.attrs["creation_date"] = datetime.now().isoformat()
                metadata_group.attrs["source_database"] = args.sqlite_path
                metadata_group.attrs["total_rows"] = total_rows

                # Store schema information
                schema_data = json.dumps(schema)
                metadata_group.attrs["schema"] = schema_data

            # Process data in chunks
            chunk_size = args.chunk_size

            # Check if we should truncate HDF5 file and start fresh when not resuming
            if not checkpoint and args.restart and args.table_name in h5file:
                logger.warning(
                    f"Restart requested - deleting existing '{args.table_name}' group"
                )
                del h5file[args.table_name]
                if "metadata" in h5file:
                    metadata_group = h5file["metadata"]
                    if "rows_migrated" in metadata_group.attrs:
                        del metadata_group.attrs["rows_migrated"]
                    if "migration_status" in metadata_group.attrs:
                        del metadata_group.attrs["migration_status"]

            # Prepare query with ID-based pagination for better performance
            query = f"""
                SELECT * FROM {args.table_name}
                WHERE rowid > ?
                ORDER BY rowid
                LIMIT ?
            """

            rows_left = total_rows - total_processed
            logger.info(f"Starting migration: {rows_left} rows left to process")

            while rows_left > 0:
                logger.info(f"Processing chunk starting from ID {last_id}...")

                # Read chunk from SQLite
                chunk_df = pd.read_sql_query(
                    query, sqlite_conn, params=(last_id, chunk_size)
                )

                if len(chunk_df) == 0:
                    logger.warning("No more rows to read from SQLite")
                    break

                # Get max rowid from this chunk for next iteration
                # Need to query the database again to get the correct rowid
                max_id_query = f"""
                    SELECT MAX(rowid) FROM {args.table_name}
                    WHERE uuid IN ({','.join(['?'] * len(chunk_df))})
                """
                try:
                    cursor = sqlite_conn.cursor()
                    cursor.execute(max_id_query, chunk_df["uuid"].tolist())
                    max_id_result = cursor.fetchone()
                    if max_id_result and max_id_result[0]:
                        last_id = max_id_result[0]
                        logger.info(f"Next chunk will start after rowid {last_id}")
                    else:
                        # Fallback: increment by chunk size
                        last_id += chunk_size
                        logger.warning(
                            f"Could not determine max rowid, using {last_id} for next chunk"
                        )
                except Exception as e:
                    logger.warning(
                        f"Error getting max rowid: {str(e)}, using increment method"
                    )
                    # Increment by chunk size as a fallback
                    last_id += chunk_size

                # Process the chunk
                processed_df, embeddings = process_chunk(chunk_df, logger)

                # Write to HDF5
                success = write_to_hdf5(
                    h5file, processed_df, embeddings, args.table_name, logger
                )
                if not success:
                    logger.error(f"Failed to write chunk to HDF5, stopping")
                    return False

                # Update progress
                total_processed += len(chunk_df)
                rows_left = total_rows - total_processed
                progress = (total_processed / total_rows) * 100

                logger.info(
                    f"Progress: {progress:.2f}% ({total_processed}/{total_rows} rows)"
                )

                # Save checkpoint
                checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
                save_checkpoint(last_id, total_processed, checkpoint_dir)

                # Commit HDF5 changes
                h5file.flush()

            # Verify the actual number of rows migrated
            if args.table_name in h5file and "uuid" in h5file[args.table_name]:
                actual_rows = len(h5file[args.table_name]["uuid"])
                logger.info(f"Verification: actual rows in HDF5: {actual_rows}")

                if actual_rows != total_rows:
                    logger.warning(
                        f"Row count mismatch: expected {total_rows}, got {actual_rows}"
                    )

                    # Update metadata with actual count
                    total_processed = actual_rows

            # Update metadata with completion status
            metadata_group = h5file["metadata"]
            metadata_group.attrs["completion_date"] = datetime.now().isoformat()
            metadata_group.attrs["rows_migrated"] = total_processed
            metadata_group.attrs["migration_status"] = "complete"

            logger.info(f"Migration completed: {total_processed} rows migrated to HDF5")
            return True

    except Exception as e:
        logger.error(f"Error during migration: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False
    finally:
        if "sqlite_conn" in locals():
            sqlite_conn.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate data from SQLite to HDF5")
    parser.add_argument("--sqlite-path", required=True, help="Path to SQLite database")
    parser.add_argument("--hdf5-path", required=True, help="Path to output HDF5 file")
    parser.add_argument(
        "--table-name", default="image_embeddings", help="SQLite table name"
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")
    parser.add_argument(
        "--chunk-size", type=int, default=100000, help="Chunk size for processing"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Restart migration from scratch (delete existing data)",
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
    logger.info("Starting SQLite to HDF5 migration")
    logger.info(
        f"Parameters: sqlite={args.sqlite_path}, hdf5={args.hdf5_path}, "
        f"table={args.table_name}, chunk_size={args.chunk_size}, resume={args.resume}"
    )

    # Create checkpoints directory
    checkpoint_dir = os.path.join(args.log_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load checkpoint if resuming
    checkpoint = None
    if args.resume:
        checkpoint = load_checkpoint(checkpoint_dir, logger)

    # Migrate data
    success = migrate_data(args, checkpoint, logger)

    # Log completion information
    total_time = time.time() - start_time
    total_minutes = total_time / 60

    if success:
        logger.info(f"Migration completed successfully in {total_minutes:.2f} minutes")
    else:
        logger.error(f"Migration failed after {total_minutes:.2f} minutes")
        sys.exit(1)


if __name__ == "__main__":
    main()
