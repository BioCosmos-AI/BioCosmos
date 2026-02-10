import os
import sqlite3
import tarfile
import argparse
import logging
from datetime import datetime
import time
from tqdm import tqdm


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"common_name_import_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )
    return logging.getLogger(__name__)


def get_tar_files(tar_dir):
    """Get sorted list of all tar files."""
    tar_files = [f for f in os.listdir(tar_dir) if f.endswith(".tar")]
    tar_files.sort()
    return tar_files


def process_tar_file(tar_path, batch_size, logger):
    """Process a single tar file and yield batches of updates."""
    updates = []

    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            logger.info(f"Processing {len(members)} files in {tar_path}")

            for member in tqdm(
                members, desc=f"Processing {os.path.basename(tar_path)}"
            ):
                if member.name.endswith(".common_name.txt"):
                    try:
                        uuid = member.name.split(".")[0]
                        f = tar.extractfile(member)
                        if f:
                            content = f.read().decode("utf-8").strip()
                            if content:
                                updates.append((content, uuid))

                                # Yield batch if we've accumulated enough
                                if len(updates) >= batch_size:
                                    yield updates
                                    updates = []
                    except Exception as e:
                        logger.error(f"Error processing {member.name}: {str(e)}")

            # Yield any remaining updates
            if updates:
                yield updates

    except Exception as e:
        logger.error(f"Error processing tar file {tar_path}: {str(e)}")
        if updates:  # Yield any updates we managed to get before the error
            yield updates


def update_database(conn, table_name, updates, logger):
    """Update database with a batch of updates."""
    try:
        conn.executemany(
            f"UPDATE {table_name} SET common_name = ? WHERE uuid = ?", updates
        )
        conn.commit()
        return len(updates)
    except Exception as e:
        logger.error(f"Error updating database: {str(e)}")
        conn.rollback()
        return 0


def main():
    parser = argparse.ArgumentParser(
        description="Batch import of common names from tar files to SQLite database"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--table-name", required=True, help="Name of table in SQLite")
    parser.add_argument(
        "--tar-dir", required=True, help="Directory containing tar files"
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Number of records to update in a single batch",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_dir)
    start_time = time.time()

    # Log start information
    logger.info("Starting batch import of common names")
    logger.info(
        f"Parameters: db={args.db_path}, table={args.table_name}, "
        f"tar_dir={args.tar_dir}, batch_size={args.batch_size}"
    )

    try:
        # Connect to database
        logger.info("Connecting to database")
        conn = sqlite3.connect(args.db_path, timeout=60.0)

        # Set performance pragmas
        conn.execute("PRAGMA synchronous = NORMAL")
        conn.execute("PRAGMA cache_size = -2000000")  # 2GB cache
        conn.execute("PRAGMA temp_store = MEMORY")
        conn.commit()

        # Get list of tar files
        tar_files = get_tar_files(args.tar_dir)
        logger.info(f"Found {len(tar_files)} tar files to process")

        # Process each tar file
        total_processed = 0
        total_updated = 0

        for tar_file in tar_files:
            tar_path = os.path.join(args.tar_dir, tar_file)
            file_start = time.time()
            file_updates = 0

            # Process tar file in batches
            for batch in process_tar_file(tar_path, args.batch_size, logger):
                updated = update_database(conn, args.table_name, batch, logger)
                total_updated += updated
                file_updates += updated

            file_time = time.time() - file_start
            logger.info(
                f"Processed {tar_file}: {file_updates} updates in {file_time:.2f}s "
                f"({file_updates/file_time:.2f} updates/s)"
            )

            total_processed += 1
            if total_processed % 10 == 0:
                elapsed = time.time() - start_time
                logger.info(
                    f"Progress: {total_processed}/{len(tar_files)} tar files, "
                    f"{total_updated} total updates, "
                    f"{total_updated/elapsed:.2f} overall updates/s"
                )

        # Log final statistics
        elapsed = time.time() - start_time
        logger.info("\nFinal Statistics:")
        logger.info(f"Processed {total_processed} tar files")
        logger.info(f"Total updates: {total_updated}")
        logger.info(f"Processing time: {elapsed:.2f} seconds")
        logger.info(f"Overall throughput: {total_updated/elapsed:.2f} updates/second")

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
