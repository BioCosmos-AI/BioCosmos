import os
import sqlite3
import requests
import json
from tqdm import tqdm
import argparse
from datetime import datetime
import logging


def setup_logging(log_dir):
    """Set up logging to both file and console."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"species_lookup_{timestamp}.log")

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Setup file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Setup logger
    logger = logging.getLogger("species_lookup")
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def get_species_from_gbif(common_name):
    """Query GBIF API for species information using common name."""
    base_url = "https://api.gbif.org/v1/species/search"
    params = {
        "q": common_name,
        "rank": "SPECIES",
        "datasetKey": "d7dddbf4-2cf0-4f39-9b2a-bb099caae36c",  # gbif backbone dataset
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        # Look for exact match in vernacular names
        for result in data.get("results", []):
            vernacular_names = result.get("vernacularNames", [])
            for vname in vernacular_names:
                if (
                    vname.get("vernacularName", "").lower() == common_name.lower()
                    and vname.get("language") == "eng"
                ):
                    return (
                        result.get("species"),
                        f"{base_url}?{response.url.split('?')[1]}",
                    )

        return None, f"{base_url}?{response.url.split('?')[1]}"

    except requests.exceptions.RequestException as e:
        return None, str(e)


def update_missing_species(conn, table_name, logger):
    """Update missing species names using common names."""
    # Get records with missing species names
    cursor = conn.execute(
        f"SELECT uuid, shard_id, common_name FROM {table_name} WHERE species_name IS NULL AND common_name IS NOT NULL"
    )
    rows = cursor.fetchall()

    logger.info(f"Found {len(rows)} records with missing species names")

    # Initialize counters
    ambiguous_count = 0
    not_found_count = 0
    updated_count = 0

    # Process in batches for efficiency
    batch_size = 1000
    updates = []

    for uuid, shard_id, common_name in tqdm(rows):
        species_name, api_url = get_species_from_gbif(common_name)

        if species_name:
            updates.append((species_name, uuid))
            updated_count += 1

            # Log successful but potentially ambiguous matches
            logger.info(
                f"Matched - UUID: {uuid}, Shard: {shard_id}, "
                f"Common Name: {common_name}, Species: {species_name}, "
                f"API URL: {api_url}"
            )
            ambiguous_count += 1
        else:
            # Log failed matches
            logger.warning(
                f"No match found - UUID: {uuid}, Shard: {shard_id}, "
                f"Common Name: {common_name}, API URL: {api_url}"
            )
            not_found_count += 1

        # Execute batch update
        if len(updates) >= batch_size:
            conn.executemany(
                f"UPDATE {table_name} SET species_name = ? WHERE uuid = ?", updates
            )
            conn.commit()
            updates = []

    # Final batch update
    if updates:
        conn.executemany(
            f"UPDATE {table_name} SET species_name = ? WHERE uuid = ?", updates
        )
        conn.commit()

    # Log final statistics
    logger.info(f"\nFinal Statistics:")
    logger.info(f"Total records processed: {len(rows)}")
    logger.info(f"Successfully updated: {updated_count}")
    logger.info(f"Potentially ambiguous matches: {ambiguous_count}")
    logger.info(f"No matches found: {not_found_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Fill missing species names using GBIF API"
    )
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--table-name", required=True, help="Name of the table in SQLite"
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")

    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(args.log_dir)

    # Connect to SQLite database with optimized settings
    conn = sqlite3.connect(args.db_path)
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = MEMORY")
    conn.execute("PRAGMA cache_size = -2000000")  # Use 2GB of cache

    try:
        update_missing_species(conn, args.table_name, logger)
        logger.info("Species name update completed successfully")
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise
    finally:
        conn.close()


if __name__ == "__main__":
    main()
