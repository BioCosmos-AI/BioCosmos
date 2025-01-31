import os
import sqlite3
import tarfile
from tqdm import tqdm
import argparse


def alter_table_add_species_column(conn, table_name):
    """Add species_name column if it doesn't exist."""
    try:
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN species_name TEXT")
        print("Added species_name column successfully")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            print("species_name column already exists")
        else:
            raise


def get_scientific_name_from_tar(tar_path, uuid):
    """Extract scientific name from the tar file for given UUID."""
    try:
        with tarfile.open(tar_path, "r") as tar:
            scientific_name_file = f"{uuid}.scientific_name.txt"
            try:
                member = tar.getmember(scientific_name_file)
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode("utf-8").strip()
                    return content
            except KeyError:
                return None
    except Exception as e:
        print(f"Error processing {tar_path} for UUID {uuid}: {str(e)}")
        return None


def update_species_names(conn, table_name, tar_dir):
    """Update species names for all records in the database."""
    # Get all UUIDs and shard_ids
    cursor = conn.execute(
        f"SELECT uuid, shard_id FROM {table_name} WHERE species_name IS NULL"
    )
    rows = cursor.fetchall()

    print(f"Processing {len(rows)} records...")

    # Process in batches for efficiency
    batch_size = 1000
    updates = []

    for i, (uuid, shard_id) in enumerate(tqdm(rows)):
        tar_path = os.path.join(tar_dir, f"shard-{shard_id}.tar")
        species_name = get_scientific_name_from_tar(tar_path, uuid)

        updates.append((species_name, uuid))

        # Execute batch update
        if len(updates) >= batch_size or i == len(rows) - 1:
            conn.executemany(
                f"UPDATE {table_name} SET species_name = ? WHERE uuid = ?", updates
            )
            conn.commit()
            updates = []


def main():
    parser = argparse.ArgumentParser(
        description="Add and populate species_name column in SQLite database"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--table-name", required=True, help="Name of the table in SQLite"
    )
    parser.add_argument(
        "--tar-dir", required=True, help="Directory containing tar files"
    )

    args = parser.parse_args()

    # Connect to SQLite database with optimized settings
    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -2000000")  # Use 2GB of cache

    try:
        # Add the species_name column
        alter_table_add_species_column(conn, args.table_name)

        # Update species names
        update_species_names(conn, args.table_name, args.tar_dir)

        print("Species name update completed successfully")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
