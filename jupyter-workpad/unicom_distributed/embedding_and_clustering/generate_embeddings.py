import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import open_clip
import pandas as pd
from PIL import Image
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path
import argparse
import logging


def setup_logging(rank, output_dir):
    log_dir = Path(output_dir)
    log_file = log_dir / f"embedding_gen_rank_{rank}.log"

    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    logging.info(f"Logging initialized for rank {rank}. Log file: {log_file}")


def setup_distributed(output_dir):
    rank = int(os.environ["SLURM_PROCID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    local_rank = int(os.environ["SLURM_LOCALID"])

    # Get the hostname of the first node
    nodes = os.environ["SLURM_NODELIST"]
    if "[" in nodes:
        # Handle node ranges like "c0800a-s[11,17,23]"
        prefix = nodes.split("[")[0]
        node_nums = nodes.split("[")[1].split("]")[0].split(",")
        master_addr = f"{prefix}{node_nums[0]}"
    else:
        # Handle single node or comma-separated list
        master_addr = nodes.split(",")[0]

    master_port = int(os.environ.get("MASTER_PORT", "12355"))

    logging.info(f"Using master node: {master_addr}")
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)

    # Add some debugging info
    logging.info(f"SLURM_NODELIST: {os.environ['SLURM_NODELIST']}")
    logging.info(f"SLURM_PROCID: {rank}")
    logging.info(f"MASTER_ADDR: {master_addr}")
    logging.info(f"MASTER_PORT: {master_port}")

    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)

    setup_logging(rank, output_dir)

    logging.info(
        f"Initialized distributed process: rank {rank}/{world_size} on {master_addr}"
    )

    return rank, world_size, local_rank


class DistributedVLM4BioDataset(Dataset):
    """Distributed dataset for loading and preprocessing image-text pairs.

    This dataset class handles loading and preprocessing of image-text pairs for distributed training
    across multiple GPUs/nodes. It filters out invalid image paths and provides access to image data,
    captions, and metadata.

    Args:
        csv_path (str): Path to CSV file containing image paths and metadata
        base_path (str): Base directory containing the image files
        preprocess (callable): Image preprocessing function from CLIP model
        cohort (str): Dataset split to use ('train' or 'test')

    The CSV file should contain columns:
        - image_path: Relative path to image from base_path
        - caption: Text caption/description of the image
        - split: Dataset split ('train' or 'test')
        - scientific_name: Scientific name of the specimen
        - category: Category/class of the specimen

    Returns:
        dict: Dictionary containing:
            - image: Preprocessed image tensor
            - text: Tokenized text tensor
            - image_path: Full path to image file
            - scientific_name: Scientific name string
            - category: Category string
    """

    def __init__(self, csv_path, base_path, preprocess, cohort):
        df = pd.read_csv(csv_path)
        self.df = df
        self.base_path = Path(base_path)
        self.preprocess = preprocess

        valid_indices = []
        total = len(df)
        for idx, row in enumerate(df.iterrows()):
            file_path = self.base_path / "images" / row[1]["fileNameAsDelivered"]
            if file_path.exists():
                try:
                    with Image.open(file_path) as img:
                        img.verify()
                    valid_indices.append(row[0])
                    if idx % 100 == 0:  # Progress logging every 100 images
                        logging.info(f"Validated {idx}/{total} images")
                except Exception as e:
                    logging.warning(f"Invalid image file {file_path}: {e}")
            else:
                logging.warning(f"File not found: {file_path}")

        self.df = self.df.loc[valid_indices]
        logging.info(
            f"Dataset initialized with {len(self.df)} valid samples out of {total} total"
        )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = Path(
            self.base_path / "images" / row["fileNameAsDelivered"]
        ).resolve()

        image = self.preprocess(Image.open(img_path))
        text = self.tokenizer([row["scientificName"]])[
            0
        ]  # Use scientificName as caption # TODO: USE THE LLM-Generated Caption instead!

        return {
            "image": image,
            "text": text,
            "image_path": str(img_path),
            "scientific_name": row["scientificName"],  # Update to match new CSV
            "category": "",  # Assuming category is not provided in the new CSV
        }


def main(args):
    """Main function to generate embeddings using distributed training.

    Args:
        args: Namespace containing:
            model_name (str): Name of the CLIP model to use
            csv_path (str): Path to CSV file containing image metadata
            base_path (str): Base path to image directory
            batch_size (int): Batch size for data loading
            cohort (str): Dataset split to use ('train' or 'test')
            output_dir (str): Directory to save embeddings

    The function:
    1. Sets up distributed training across multiple GPUs/nodes
    2. Initializes the CLIP model and dataset
    3. Generates image and text embeddings in batches
    4. Saves embeddings and metadata to .mat files
    """
    rank, world_size, local_rank = setup_distributed(args.output_dir)

    logging.info(
        f"Process starting - Rank: {rank}, World Size: {world_size}, Local Rank: {local_rank}"
    )

    try:
        model, _, preprocess = open_clip.create_model_and_transforms(args.model_name)
        logging.info("Model created successfully")

        model = model.to(local_rank)
        model = DistributedDataParallel(model, device_ids=[local_rank])
        model.eval()
        logging.info("Model moved to device and wrapped in DDP")

        tokenizer = open_clip.get_tokenizer(args.model_name)
        DistributedVLM4BioDataset.tokenizer = tokenizer

        dataset = DistributedVLM4BioDataset(
            args.csv_path, args.base_path, preprocess, args.cohort
        )
        logging.info(f"Dataset initialized with {len(dataset)} samples")

        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
        logging.info(f"DistributedSampler initialized for rank {rank}")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
        )
        logging.info(f"DataLoader created with {len(dataloader)} batches")

        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=4,
        )
        logging.info(f"DataLoader created with {len(dataloader)} batches")
        dist.barrier()

        node_name = os.environ["SLURMD_NODENAME"]
        embeddings_dir = (
            Path(args.output_dir)
            / f"embeddings_{args.cohort}_{args.model_name}"
            / f"{node_name}_{rank}"
        )
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                images = batch["image"].to(local_rank)
                texts = batch["text"].to(local_rank)

                image_emb = model.module.encode_image(images)
                text_emb = model.module.encode_text(texts)

                image_emb /= image_emb.norm(dim=-1, keepdim=True)
                text_emb /= text_emb.norm(dim=-1, keepdim=True)

                save_dict = {
                    "image_embeddings": image_emb.cpu().numpy(),
                    "text_embeddings": text_emb.cpu().numpy(),
                    "image_paths": batch["image_path"],
                    "scientific_names": batch["scientific_name"],
                    "categories": batch["category"],
                }

                save_path = embeddings_dir / f"batch_{batch_idx}.mat"
                savemat(save_path, save_dict)

                if batch_idx % 10 == 0:
                    logging.info(f"Processed batch {batch_idx}/{len(dataloader)}")
                    dist.barrier()

        logging.info("Finished generating embeddings")
        dist.barrier()

    except Exception as e:
        logging.error(f"Error in generate_embeddings: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--base_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="ViT-H-14-378-quickgelu")
    parser.add_argument("--cohort", type=str, default="train")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    main(args)

# Example usage with SLURM/srun:
# 1. Basic usage with default parameters
#    srun -n 4 python generate_embeddings.py \
#        --csv_path /path/to/metadata.csv \
#        --base_path /path/to/image/directory \
#        --output_dir /path/to/save/embeddings

# 2. Specify different model and cohort
#    srun -n 8 python generate_embeddings.py \
#        --csv_path /path/to/metadata.csv \
#        --base_path /path/to/image/directory \
#        --output_dir /path/to/save/embeddings \
#        --model_name "ViT-L-14" \
#        --cohort "test"

# 3. Adjust batch size for memory constraints
#    srun -n 4 python generate_embeddings.py \
#        --csv_path /path/to/metadata.csv \
#        --base_path /path/to/image/directory \
#        --output_dir /path/to/save/embeddings \
#        --batch_size 16

# Note: This script requires SLURM environment variables to be set
# Make sure to run with srun and specify the number of processes with -n
