
import os, glob
import sys
import torch
import webdataset as wds
import pandas as pd
from PIL import Image
from scipy.io import savemat
from torchvision import transforms
import open_clip
import multiprocessing
import logging

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model(model_name='ViT-H-14-378-quickgelu'):
    """Load the specified CLIP model and preprocessing transforms."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="dfn5b", device=device
    )
    return model, preprocess

def get_image_embeddings_clip(images, model):
    """Generate embeddings for an image using the provided CLIP model."""
    try:
        # stack the images for batching purpose
        images = torch.stack(images).to(device)  
        with torch.no_grad():
            image_embeddings = model.encode_image(images)
        return image_embeddings.cpu().numpy()
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return None

def validate_shard_files(shard_files, embedded_files):
    embedded_nums=[f.split('.')[1].split('_')[-1] for f in embedded_files]

    return [shard_file for shard_file in shard_files if shard_file.split('.tar')[0].split('-')[-1] not in embedded_nums]

def process_shards(split, model, preprocess, output_format):
    """Process shards to extract image embeddings in batches and save them."""
    logger.info("Beginning to process shards")
    
    try:
        # get shard
        shards_dir = f"/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/{split}"
        #shard_pattern = os.path.join(shards_dir, "shard-*.tar")
        shard_files = glob.glob(os.path.join(shards_dir, "shard-*.tar"))
    except Exception as e:
        logger.error(f"Can't use the given shard dir due to error: {e}")

    embeddings_dir = shards_dir + '/embeddings'
    embeddings_files = glob.glob(os.path.join(embeddings_dir, f'*.{output_format}'))
    shard_files = validate_shard_files(shard_files, embeddings_files)
    logger.info(f"analyzing the following shard files: {shard_files}")
    
    # set up multiprocessing with gpus
    gpu_ids = list(range(torch.cuda.device_count()))
    logger.info(f"We're working with the following gpus: {gpu_ids}")
    pool = multiprocessing.Pool(len(gpu_ids))

    args = [(shard, gpu_ids[i % len(gpu_ids)], model, preprocess, shards_dir, split, output_format) for i, shard in enumerate(shard_files)]  
    
    pool.starmap(process_shard_in_parallel, args)
    
def process_shard_in_parallel(shard, gpu_id, model, preprocess, shards_dir, split, output_format, batch_size=128):
    device=f"cuda:{gpu_id}"
    
    logger.info(f"Now processing shard: {shard} on device: {gpu_id}")
    shard_no = shard.split('.tar')[0].split('-')[-1]
    logger.info(f'shard_no is : {shard_no}')
    output_file = os.path.join(shards_dir, f"embeddings/embeddings_{split}_shard_{shard_no}_pp.{output_format}")

    dataset = wds.WebDataset(shard).decode("pil").to_tuple("jpg", "__key__").batched(batch_size)

    embeddings = []
    keys = []

    # go through wds by batch
    for batch in dataset:
        #logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        #logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        if not batch:  # Skip empty batches
            logger.warning("Encountered an empty batch, skipping.")
            continue

        try:
            images, batch_keys = batch
        except Exception as e:
            logger.error(f"Malformed batch encountered: {batch}, Error: {e}")


        try:
            #preprocess batch of images, then get embeddings
            preprocessed_images = [preprocess(img) for img in images]
                
            batch_embeddings = get_image_embeddings_clip(preprocessed_images, model)

            if batch_embeddings is not None:
                embeddings.extend(batch_embeddings)
                keys.extend(batch_keys)
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            continue
            
        torch.cuda.empty_cache()

    # save as df
    df = pd.DataFrame(embeddings, index=keys, columns=[f"dim_{i}" for i in range(len(embeddings[0]))])

    # save into file
    if output_format == "mat":
        savemat(output_file, {"embeddings": df.values, "keys": df.index.tolist()})
    elif output_format == "parquet":
        df.to_parquet(output_file, index=True, engine='pyarrow')

    logger.info(f"{shard_no} embeddings saved to {output_file}")

def main():
    # args formatting check
    if len(sys.argv) != 2:
        logger.error("Usage: python process_embeddings.py <split>")
        sys.exit(1)

    split = sys.argv[1]

    if split not in ["train", "train_small", "val"]:
        logger.error("Invalid split. Choose from: train, train_small, val")
        sys.exit(1)

    # using best performing model
    model_name = "ViT-H-14-378-quickgelu"
    model, preprocess = get_model(model_name)

    output_format = "parquet"  # could be mat if you want .mat files
    multiprocessing.set_start_method("spawn")
    process_shards(split, model, preprocess, output_format)

if __name__ == "__main__":
    main()
