import glob
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


def get_df(logger: Logger, embedding_dir: str = "") -> pd.DataFrame:
    shard_files = sorted(glob.glob(f"{embedding_dir}/embeddings_train_shard_*.parquet"))
    logger.info(f"Found {len(shard_files)} shards")

    dfs: list[pd.DataFrame] = []
    for f in shard_files:
        df: pd.DataFrame = pd.read_parquet(f, engine="pyarrow")
        # Extract shard ID from filename
        shard_id = f.split('_')[-1].replace('.parquet', '')  # e.g., '000000'
        df['shard_id'] = shard_id
        dfs.append(df)

    # TODO: Remove NULL/empty species from DF
    full_df = pd.concat(dfs, ignore_index=False)  # keep index (UUIDs)
    logger.info(f"Total rows: {len(full_df)}")
    return full_df

def cluster(df: pd.DataFrame, cosine: bool = False) -> tuple:
    embedding_cols = [c for c in df.columns if c.startswith('dim_')]
    embeddings = df[embedding_cols].values.astype('float32')

    if cosine:
        embeddings = np.ascontiguousarray(embeddings)
        faiss.normalize_L2(embeddings)

    dim: int = embeddings.shape[1]

    ks = [350_000, 400_000, 454_000, 500_000, 550_000, 600_000, 650_000]
    inertias = []
    for k in ks:
        print(f"Training k={k}...")
        kmeans = faiss.Kmeans(dim, k, niter=20, nredo=1, gpu=True)
        kmeans.train(embeddings)
        distances, _ = kmeans.index.search(embeddings, 1)
        inertia = distances.sum()
        print(f"k={k}: {inertia}")
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


def main():
    parser = argparse.ArgumentParser(
        description="Flat clustering of ToL10M embeddings"
    )
    # e.g., image-only embeddings: "/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train/embeddings"
    parser.add_argument(
        "--title", required=True, help="Title of generated plot (if plot only)"
    )
    parser.add_argument(
        "--embedding-dir", required=True, help="Directory containing parquet files which contain the image embeddings"
    )
    parser.add_argument(
        "--cosine", default=False, required=True, help="Directory containing parquet files which contain the image embeddings"
    )
    parser.add_argument(
        "--plot-dir", required=True, help="Where to write logs"
    )

    args = parser.parse_args()

    logger = setup_logging()
    logger.info(f"LOGGING STARTED for {args.title}")

    df: pd.DataFrame = get_df(logger=logger, embedding_dir=args.embedding_dir)
    ks, inertias = cluster(df=df, cosine=args.cosine)
    plot_k_inertia(title=args.title, ks=ks, inertias=inertias, plot_dir=args.plot_dir)



if __name__ == "__main__":
    main()