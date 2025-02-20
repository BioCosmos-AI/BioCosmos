import os
import sqlite3
import tarfile
from tqdm import tqdm
import argparse
import logging
from datetime import datetime
from collections import defaultdict


def setup_logging(log_dir):
    """Set up logging configuration."""
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"common_name_import_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_file}")
    return logger


def alter_table_add_common_name_column(conn, table_name, logger):
    """Add common_name column if it doesn't exist and create an index on it."""
    try:
        # Add the column
        conn.execute(f"ALTER TABLE {table_name} ADD COLUMN common_name TEXT")
        logger.info("Added common_name column successfully")

        # Create the index
        conn.execute(
            f"""
        CREATE INDEX IF NOT EXISTS idx_{table_name}_common_name 
        ON {table_name}(common_name)
        """
        )
        logger.info("Created index on common_name successfully")

    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e):
            logger.info("common_name column already exists")
        else:
            raise


def process_tar_file(tar_path, uuid_shard_pairs, logger):
    """Process all needed records from a single tar file."""
    try:
        results = []
        with tarfile.open(tar_path, "r") as tar:
            for uuid, _ in uuid_shard_pairs:
                common_name_file = f"{uuid}.common_name.txt"
                try:
                    member = tar.getmember(common_name_file)
                    f = tar.extractfile(member)
                    if f:
                        content = f.read().decode("utf-8").strip()
                        if content:
                            results.append((content, uuid))
                            logger.debug(
                                f"Found common name for UUID {uuid}: {content}"
                            )
                        else:
                            logger.warning(f"Empty content for UUID {uuid}")
                except KeyError:
                    logger.warning(
                        f"File not found in tar for UUID {uuid}: {common_name_file}"
                    )
        return results
    except Exception as e:
        logger.error(f"Error processing tar file {tar_path}: {str(e)}")
        return []


def update_common_names(conn, table_name, tar_dir, logger):
    """Update common names for all records in the database."""
    # Get all UUIDs and shard_ids where common_name is NULL
    cursor = conn.execute(
        f"SELECT uuid, shard_id FROM {table_name} WHERE common_name IS NULL"
    )
    rows = cursor.fetchall()

    logger.info(f"Processing {len(rows)} records...")

    # Group records by shard_id
    shard_groups = defaultdict(list)
    for uuid, shard_id in rows:
        shard_groups[shard_id].append((uuid, shard_id))

    logger.info(f"Found {len(shard_groups)} unique shards to process")

    processed = 0
    skipped = 0
    error_count = 0

    # Process one tar file at a time
    for shard_id, uuid_pairs in tqdm(shard_groups.items(), desc="Processing shards"):
        tar_path = os.path.join(tar_dir, f"shard-{shard_id}.tar")

        try:
            if not os.path.exists(tar_path):
                logger.error(f"Tar file not found: {tar_path}")
                error_count += len(uuid_pairs)
                continue

            # Get all common names from this tar file
            updates = process_tar_file(tar_path, uuid_pairs, logger)

            if updates:
                try:
                    conn.executemany(
                        f"UPDATE {table_name} SET common_name = ? WHERE uuid = ?",
                        updates,
                    )
                    conn.commit()
                    processed += len(updates)
                    skipped += len(uuid_pairs) - len(updates)
                    logger.info(
                        f"Committed {len(updates)} updates from shard {shard_id}"
                    )
                except Exception as e:
                    error_count += len(updates)
                    logger.error(f"Error committing batch: {str(e)}")
                    conn.rollback()
            else:
                skipped += len(uuid_pairs)
                logger.warning(f"No successful updates from shard {shard_id}")

        except Exception as e:
            error_count += len(uuid_pairs)
            logger.error(f"Error processing shard {shard_id}: {str(e)}")
            continue

        # Log progress periodically
        if processed > 0 and processed % 10000 == 0:
            logger.info(
                f"Progress update - Processed: {processed}, "
                f"Skipped: {skipped}, Errors: {error_count}, "
                f"Completion: {((processed + skipped + error_count) / len(rows)) * 100:.2f}%"
            )

    logger.info(f"\nFinal Statistics:")
    logger.info(f"Successfully processed: {processed} records")
    logger.info(f"Skipped: {skipped} records")
    logger.info(f"Errors: {error_count} records")
    logger.info(f"Total records examined: {len(rows)}")
    logger.info(
        f"Completion percentage: {((processed + skipped + error_count) / len(rows)) * 100:.2f}%"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Add and populate common_name column in SQLite database"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--table-name", required=True, help="Name of the table in SQLite"
    )
    parser.add_argument(
        "--tar-dir", required=True, help="Directory containing tar files"
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)

    logger.info("Starting common name import process")
    logger.info(f"Database path: {args.db_path}")
    logger.info(f"Table name: {args.table_name}")
    logger.info(f"Tar directory: {args.tar_dir}")
    logger.info(f"Log directory: {args.log_dir}")

    # Connect to SQLite database with optimized settings
    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -2000000")  # Use 2GB of cache

    try:
        # Add the common_name column and index
        alter_table_add_common_name_column(conn, args.table_name, logger)

        # Update common names
        update_common_names(conn, args.table_name, args.tar_dir, logger)

        logger.info("Common name update completed successfully")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
