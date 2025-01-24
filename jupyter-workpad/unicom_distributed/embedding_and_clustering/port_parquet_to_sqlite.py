import os
import sqlite3
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import argparse


def create_table_if_not_exists(conn, table_name):
    """Create the table if it doesn't exist with UUID, shard_id, and embedding columns."""
    schema = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        uuid TEXT,
        shard_id TEXT,
        embedding BLOB
    )
    """
    conn.execute(schema)

    # Create unique index on UUID if it doesn't exist
    index_sql = f"""
    CREATE UNIQUE INDEX IF NOT EXISTS idx_{table_name}_uuid 
    ON {table_name}(uuid)
    """
    conn.execute(index_sql)


def extract_shard_id(filename):
    """Extract the shard ID from the filename."""
    match = re.search(r"_(\d{6})\.parquet$", filename)
    return match.group(1) if match else None


def convert_to_blob(row):
    """Convert the dimensional columns to a binary blob."""
    vector = np.array([row[f"dim_{i}"] for i in range(1024)], dtype=np.float32)
    return sqlite3.Binary(vector.tobytes())


def process_parquet_file(conn, file_path, table_name):
    """Process a single parquet file and upsert its contents to SQLite."""
    # Read parquet file
    table = pq.read_table(file_path)
    print(f"\nSchema for {file_path}:")
    print(table.schema)

    df = table.to_pandas()
    print("\nDataFrame columns:")
    print(df.columns.tolist())
    print("\nDataFrame info:")
    print(df.info())

    # Add shard_id column
    shard_id = extract_shard_id(os.path.basename(file_path))

    # Convert dimensional data to blobs
    records = []
    for uuid, row in df.iterrows():  # uuid comes from the index
        records.append(
            (uuid, shard_id, convert_to_blob(row))  # Using the index value as uuid
        )

    # Prepare the upsert query
    upsert_query = f"""
    INSERT INTO {table_name} (uuid, shard_id, embedding)
    VALUES (?, ?, ?)
    ON CONFLICT(uuid) DO UPDATE SET
    shard_id=excluded.shard_id,
    embedding=excluded.embedding
    """

    # Execute the upsert
    conn.executemany(upsert_query, records)


def main():
    parser = argparse.ArgumentParser(
        description="Convert parquet files to SQLite database"
    )
    parser.add_argument(
        "--base-dir", required=True, help="Directory containing parquet files"
    )
    parser.add_argument("--db-path", required=True, help="Path for SQLite database")
    parser.add_argument(
        "--table-name", required=True, help="Name of the table in SQLite"
    )
    args = parser.parse_args()

    # Connect to SQLite database with optimized settings
    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -2000000")  # Use 2GB of cache

    try:
        # Create table if it doesn't exist
        create_table_if_not_exists(conn, args.table_name)

        # Get list of parquet files
        parquet_files = [f for f in os.listdir(args.base_dir) if f.endswith(".parquet")]

        # Process each parquet file with progress bar
        for file in tqdm(parquet_files, desc="Processing parquet files"):
            file_path = os.path.join(args.base_dir, file)

            # Start transaction for this file
            try:
                process_parquet_file(conn, file_path, args.table_name)
                conn.commit()
            except Exception as e:
                conn.rollback()
                print(f"Error processing {file}: {str(e)}")
                raise

        print(f"\nSuccessfully processed {len(parquet_files)} files")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
