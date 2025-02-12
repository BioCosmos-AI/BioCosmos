
import os, glob
import sys
import torch
import webdataset as wds
import pandas as pd
from PIL import Image
import torch.nn.functional as F
import open_clip
import logging

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()

device = "cuda" if torch.cuda.is_available() else "cpu"

tol_path = '/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/'


def get_model(model_name='ViT-H-14-378-quickgelu'):
    """Load the specified CLIP model and preprocessing transforms."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="dfn5b", device=device
    )
    return model, preprocess

def get_image_embedding_clip(img, model, preprocess):
    """Generate embeddings for an image using the provided CLIP model."""
    try:
        img = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            image_embedding = model.encode_image(img)
        return image_embedding.cpu().numpy()
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def get_split_files(split):
    """Get all the embeddings parquet file names within a specific split"""
    tol_emb_path = f'/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/{split}/embeddings/*'
    emb_parquets = glob.glob(tol_emb_path)
    emb_tups = []
    for ep in emb_parquets:
        parquet_num = ep.split('_')[-1].split('.parquet')[0]
        emb_tups.append((parquet_num, ep))
    return emb_tups

def get_random_split_row(emb_split):
    """get a random row from the parquet file"""
    emb_path = emb_split[1]
    df = pd.read_parquet(emb_path, engine='pyarrow')
    return df.sample()


def extract_image_from_wds(shard_path, uuid_key):
    """given path to shard and key, find the image file and return"""
    dataset = wds.WebDataset(shard_path).decode("pil").to_tuple("jpg", "__key__")

    for image, key in dataset:
        if key == uuid_key:
            return image  # Return the PIL Image

    raise FileNotFoundError(f"Image with UUID {uuid_key} not found in {shard_path}")


def compare_random_img_to_embedding(model, preprocess, emb_split, split_row, split):
    emb_num = emb_split[0]
    tol_path = f'/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/{split}'
    shard_file = f'shard-{emb_num}.tar'
    shard_path = os.path.join(tol_path, shard_file)

    # first get the index and the embeddings for this random img
    key = split_row.index[0]
    parquet_embedding = torch.tensor(split_row.iloc[0].values, dtype=torch.float32)

    target_img = extract_image_from_wds(shard_path, key)
    img_embedding = get_image_embedding_clip(target_img, model)

    cosine_similarity = F.cosine_similarity(parquet_embedding.unsqueeze(0), torch.tensor(img_embedding))

    return key, cosine_similarity.item()


def main():
    # args formatting check
    if len(sys.argv) != 2:
        logger.error("Usage: python validate_embeddings.py <split>")
        sys.exit(1)

    split = sys.argv[1]

    if split not in ["train", "train_small", "val"]:
        logger.error("Invalid split. Choose from: train, train_small, val")
        sys.exit(1)

    # using best performing model
    model_name = "ViT-H-14-378-quickgelu"
    model, preprocess = get_model(model_name)

    logger.info("getting split file names")
    emb_splits = get_split_files(split) # get the split num + path to split

    results = []
    for emb_split in emb_splits:
        logger.info(f"commencing with split file: {emb_split}")
        split_row = get_random_split_row(emb_split)
        logger.info(f"obtained a sample row, now comparing the img to its embedding")
        uuid, sim = compare_random_img_to_embedding(model, preprocess, emb_split, split_row, split)
        logger.info(f"UUID: {uuid}; COSINE SIMILARITY: {sim}")
        results.append((uuid, sim))

    df = pd.DataFrame(results, columns=["UUID", "Cosine Similarity"])

    # Save to CSV
    df.to_csv("similarity_results.csv", index=False)

if __name__ == "__main__":
    main()
