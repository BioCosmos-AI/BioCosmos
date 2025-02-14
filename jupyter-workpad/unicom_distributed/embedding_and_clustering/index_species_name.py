import sqlite3
import argparse


def add_species_index(conn, table_name):
    """Create an index on species_name column."""
    print("Creating index on species_name...")
    conn.execute(
        f"""
    CREATE INDEX IF NOT EXISTS idx_{table_name}_species_name 
    ON {table_name}(species_name)
    """
    )
    conn.commit()
    print("Index created successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Add species_name index to SQLite database"
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
        add_species_index(conn, args.table_name)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
