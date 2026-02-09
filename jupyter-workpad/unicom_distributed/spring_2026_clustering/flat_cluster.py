import glob
import os
import argparse
import pandas as pd
import numpy as np
import faiss
import re
import matplotlib
matplotlib.use('Agg')  # no display
import matplotlib.pyplot as plt
import logging
from logging import Logger


def to_snake_case(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^a-z0-9\s]", "", title)  # remove special chars
    title = re.sub(r"\s+", "_", title.strip())  # spaces to underscores
    return title

def setup_logging() -> Logger:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
    )
    return logging.getLogger(__name__)


def get_df(logger: Logger, embedding_dir: str = "", taxa_metadata_csv: str = "", drop_null_species: bool = False) -> pd.DataFrame:
    shard_files = sorted(glob.glob(f"{embedding_dir}/embeddings_train_shard_*.parquet"))
    logger.info(f"Found {len(shard_files)} shards")

    dfs: list[pd.DataFrame] = []
    for f in shard_files:
        df: pd.DataFrame = pd.read_parquet(f, engine="pyarrow")
        # Extract shard ID from filename
        shard_id = f.split('_')[-1].replace('.parquet', '')  # e.g., '000000'
        df['shard_id'] = shard_id
        dfs.append(df)

    full_df = pd.concat(dfs, ignore_index=False)  # keep index (UUIDs)

    # This only needs to be True for image-ONLY embed (which is older)
    # Can be false for text+image embeds (average & concat)
    if drop_null_species:
        if not taxa_metadata_csv:
            raise Exception("Missing taxa metadata csv")
        logger.info(f"Loading metadata from {taxa_metadata_csv}")
        meta_df: pd.DataFrame = pd.read_csv(taxa_metadata_csv)
        
        # Filter to rows with species populated
        valid_meta = meta_df[
            meta_df['species'].notna() &
            (meta_df['species'] != '')
        ]
        logger.info(f"Metadata rows with not null/empty species: {len(valid_meta)}")
        
        valid_uuids = set(valid_meta['treeoflife_id'])
        
        before_count = len(full_df)
        full_df = full_df[full_df.index.isin(valid_uuids)]
        logger.info(f"Filtered {before_count} -> {len(full_df)} rows (dropped {before_count - len(full_df)})")

    logger.info(f"Total rows: {len(full_df)}")
    return full_df

def cluster_different_ks(logger: Logger, df: pd.DataFrame, cosine: bool = True) -> tuple:
    embedding_cols = [c for c in df.columns if c.startswith('dim_')]
    embeddings = df[embedding_cols].values.astype('float32')

    if cosine:
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

    dim: int = embeddings.shape[1]

    # ToL10M has ~454k taxa . . .used a centrailizing poing for  k determination
    ks = [80_000, 400_000, 454_000, 700_000, 1_100_000]
    inertias = []
    for k in ks:
        logger.info(f"Training k={k}...")
        kmeans = faiss.Kmeans(dim, k, niter=20, nredo=3, gpu=True)
        kmeans.train(embeddings)
        distances, _ = kmeans.index.search(embeddings, 1)
        inertia = distances.sum()
        logger.info(f"k={k}: {inertia}")
        inertias.append(inertia)

    return ks, inertias

def plot_k_inertia(title: str, ks: list[int], inertias: list, plot_dir: str):
    plt.plot(ks, inertias, marker='o')
    plt.xlabel('k')
    plt.ylabel('inertia')
    plt.axvline(x=454_000, color='r', linestyle='--', label='454k taxa')
    plt.title(title)
    plt.legend()
    plt.savefig(f"{plot_dir}/{to_snake_case(title)}.png")
    print(f"Saved plot to {plot_dir}/{to_snake_case(title)}.png")


def cluster(logger: Logger, title: str, df: pd.DataFrame, k: int, cosine: bool = True) -> pd.DataFrame:
    embedding_cols = [c for c in df.columns if c.startswith('dim_')]
    embeddings = df[embedding_cols].values.astype('float32')

    if cosine:
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

    dim: int = embeddings.shape[1]

    logger.info(f"Training k={k}...")
    kmeans = faiss.Kmeans(dim, k, niter=20, nredo=5, gpu=True)
    kmeans.train(embeddings)

    _, labels = kmeans.index.search(embeddings, 1)
    labels = labels.flatten()  # shape (n,) with values 0 to k-1
    
    logger.info(f"Clustering complete.")
    
    cluster_assignment = to_snake_case(title)
    result_df = pd.DataFrame({
        'uuid': df.index,
        'shard_id': df['shard_id'],
        cluster_assignment: labels
    })
    
    # potentially interesting cluster stats
    counts = np.bincount(labels)
    logger.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, median={np.median(counts):.0f}, mean={counts.mean():.1f}")
    logger.info(f"Clusters with <10 samples: {(counts < 10).sum()}")
    logger.info(f"Clusters with >1000 samples: {(counts > 1000).sum()}")
    
    return result_df

def save_assignments(result_df: pd.DataFrame, output_path: str, logger: Logger) -> None:
    """This APPENDS cluster assignments (which have unique column name) to existing CSV or creates new one."""

    logger.info("Attempting to write to . . . {output_path}")
    
    if os.path.exists(output_path):
        logger.info(f"Loading existing CSV: {output_path}")
        existing_df: pd.DataFrame = pd.read_csv(output_path)
        
        # Get the new column name (not uuid or shard_id)
        new_col = [c for c in result_df.columns if c not in ['uuid', 'shard_id']][0]
        
        if new_col in existing_df.columns:
            logger.warning(f"Column '{new_col}' already exists, overwriting")
            existing_df = existing_df.drop(columns=[new_col])
        
        # Merge on uuid
        merged_df = existing_df.merge(
            result_df[['uuid', new_col]], 
            on='uuid', 
            how='left'
        )
        logger.info(f"Merged column '{new_col}' onto existing {len(existing_df)} rows")
        
    else:
        logger.info(f"Creating new CSV: {output_path}")
        merged_df = result_df
    
    merged_df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path} ({len(merged_df)} rows, {len(merged_df.columns)} columns)")


def main():
    parser = argparse.ArgumentParser(
        description="Flat clustering of ToL10M embeddings"
    )
    # e.g., image-only embeddings: "/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/embeddings"
    parser.add_argument(
        "--title", required=True, help="Title of generated plot (if plot only) OR will be used to derive cluster assignment column name (--plot-elbow-only=False)"
    )
    parser.add_argument(
        "--embedding-dir", required=True, help="Directory containing parquet files which contain the image embeddings"
    )
    parser.add_argument(
        "--cosine", default=True, required=True, help="Directory containing parquet files which contain the image embeddings"
    )
    parser.add_argument(
        "--plot-dir", required=True, help="Where to write logs"
    )
    parser.add_argument(
        "--drop-null-species", default=False, help="Maps .parquet data against metadata taxa csv and removes any null species from clustering results"
    )
    parser.add_argument(
        "--taxa-metadata-csv", default="", help="Used to drop null species from clustering if drop-null-species=True"
    )
    parser.add_argument(
        "--plot-elbow-only", default=False, help="Plot the elbow k vs inertia graph (ONLY)"
    )
    parser.add_argument(
        "--k", default=0, help="The k number of clusters (only relevent if --plot-elbow-only=False)"
    )
    parser.add_argument(
        "--csv-output-file", default="", help="The CSV to WRITE cluster assignements (only relevent if --plot-elbow-only=False)"
    )

    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"LOGGING STARTED for {args.title}")

    df: pd.DataFrame = get_df(logger=logger, embedding_dir=args.embedding_dir, taxa_metadata_csv=args.taxa_metadata_csv, drop_null_species=args.drop_null_species)
    if args.plot_elbow_only:
        logger.info("Generating k comparision elbow plot only . . .")
        ks, inertias = cluster_different_ks(logger, df=df, cosine=args.cosine)
        plot_k_inertia(title=args.title, ks=ks, inertias=inertias, plot_dir=args.plot_dir)
    else:
        logger.info("Generating cluster assignemnts for k={args.k}")
        result_df = cluster(logger=logger, title=args.title, df=df, k=args.k, cosine=args.cosine)
        save_assignments(result_df=result_df, output_path=args.csv_output_file, logger=logger)
        # write labels to CSV

if __name__ == "__main__":
    main()