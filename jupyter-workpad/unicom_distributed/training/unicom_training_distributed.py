import os
import sqlite3
import argparse
import logging
import numpy as np
import pandas as pd
import json
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast

# from torch.amp import GradScaler
from tqdm import tqdm
import open_clip
import webdataset as wds
from functools import partial
import random
from datetime import datetime
import math
from collections import defaultdict
from PIL import Image

from eval_utils import create_tester_embed_dataset

from partial_fc import CombinedMarginLoss, PartialFC_V2

print(torch.cuda.get_arch_list())

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "18000"  # 30 minutes in seconds


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"unicom_training_distributed_{timestamp}.log")

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove existing handlers if any
    if logger.hasHandlers():
        logger.handlers.clear()

    # Add handlers
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info(f"Logging initialized. Log file: {log_file}")

    return logger


def load_data_from_sqlite(db_path, logger=None):
    """Load relevant data from SQLite database into memory."""
    if logger:
        logger.info(f"Loading data from {db_path}")

    try:
        connection = sqlite3.connect(db_path, timeout=900.0)
        if logger:
            logger.info(f"Connected to database")

        # Query for the relevant columns
        query = """
        SELECT uuid, shard_id, species_name, single_node_cluster 
        FROM image_embeddings 
        WHERE single_node_cluster IS NOT NULL
            AND species_name IS NOT NULL 
            AND species_name != ''
        """
        # Aribtrary 100000 records sample for testing
        # AND rowid >= 60001
        # AND rowid <= 70000
        # """
        if logger:
            logger.info("Executing query to retrieve clustering data")

        # Read all data at once (each process reads independently)
        df = pd.read_sql_query(query, connection)

        connection.close()

        if logger:
            logger.info(f"Loaded {len(df)} total records from database")

        # Create pseudo-class labels by combining species_name and single_node_cluster
        df["pseudo_class"] = (
            df["species_name"] + "_" + df["single_node_cluster"].astype(str)
        )

        # Create a mapping from pseudo_class to integer label
        pseudo_classes = sorted(df["pseudo_class"].unique())
        class_to_idx = {cls: idx for idx, cls in enumerate(pseudo_classes)}

        # Add integer labels to DataFrame
        df["label"] = df["pseudo_class"].map(class_to_idx)

        if logger:
            logger.info(f"Created {len(pseudo_classes)} unique pseudo-classes")

        return df, class_to_idx

    except Exception as e:
        if logger:
            logger.error(f"Error loading data from SQLite: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
        return pd.DataFrame(), {}


class TreeOfLifeWebDataset:
    """WebDataset iterator for TreeOfLife10M with clustering info that evenly distributes images across GPUs."""

    def __init__(
        self, webdataset_path, df, transform=None, world_size=1, rank=0, batch_size=64
    ):
        self.webdataset_path = webdataset_path
        self.df = df
        self.transform = transform
        self.world_size = world_size
        self.rank = rank
        self.batch_size = batch_size

        # Create mapping from uuid to label
        self.uuid_to_label = dict(zip(df["uuid"], df["label"]))

        # For even distribution across GPUs, we'll use a simple approach:
        # Each GPU gets a subset of images based on its rank and the total number of GPUs
        # We'll filter by the image index (via the UUID) rather than by shard
        self.rank_uuids = []
        all_uuids = sorted(df["uuid"].unique())

        # Distribute UUIDs evenly to ranks
        for i, uuid in enumerate(all_uuids):
            if i % world_size == rank:
                self.rank_uuids.append(uuid)

        # Store the UUIDs for this rank in a set for faster lookups
        self.rank_uuid_set = set(self.rank_uuids)

        # Get all shard IDs (we'll still use all shards but filter by UUID)
        self.shard_ids = sorted(df["shard_id"].unique())

        # Estimate number of samples per rank
        self.num_samples = len(self.rank_uuids)

    def create_loader(self, epoch=0):
        """Create a WebDataset pipeline for training with even image distribution."""
        # Set random seed based on epoch for consistent shuffling
        seed = epoch + self.rank * 10000
        random.seed(seed)

        # Create URLs for all shards
        all_urls = [
            f"{self.webdataset_path}/shard-{shard_id}.tar"
            for shard_id in self.shard_ids
        ]

        if self.rank == 0:  # Only print from rank 0 to avoid console spam
            print(
                f"Using {len(all_urls)} shards, distributing {self.num_samples} images to rank {self.rank}"
            )

        # Create a selection function that only keeps images assigned to this rank
        def select_by_rank(sample):
            # Extract the UUID from the key
            uuid = sample["__key__"].split(".")[0]
            # Keep only if this UUID is assigned to this rank
            return uuid in self.rank_uuid_set

        # Create the pipeline
        ds = wds.DataPipeline(
            wds.SimpleShardList(all_urls),
            wds.detshuffle(100, seed=seed),  # Deterministic shuffle of shards
            # No need for split_by_node or split_by_worker - our selection function handles distribution
            wds.tarfile_to_samples(handler=wds.warn_and_continue),
            wds.select(select_by_rank),  # Select only images for this rank
            wds.decode("pilrgb"),
            wds.map(
                lambda sample: (
                    self.transform(sample["jpg"]) if self.transform else sample["jpg"],
                    self.uuid_to_label.get(sample["__key__"].split(".")[0], 0),
                )
            ),
            wds.shuffle(100, seed=seed),
            wds.batched(
                self.batch_size, partial=False
            ),  # <-- Partial == True causes issues with NCCL, so this means we're losing <= 15 samples at boundary batches
        )

        return ds

    def __len__(self):
        """Return the number of batches on this rank."""
        return max(1, self.num_samples // self.batch_size)


def train_epoch(
    model,
    data_loader,
    module_partial_fc,
    optimizer,
    scheduler,
    scaler,
    epoch,
    args,
    logger,
    rank=0,
    dataset=None,
):
    """Train one epoch."""
    model.train()

    losses = []
    batch_time = AverageMeter()
    data_time = AverageMeter()

    end = time.time()

    for i, batch in enumerate(data_loader):
        data_time.update(time.time() - end)

        # WebDataset returns (images, labels) where each is already a batch
        images, labels = batch

        # Ensure correct types and device
        if not isinstance(images, torch.Tensor):
            images = torch.stack(images)
        if not isinstance(labels, torch.Tensor):
            labels = torch.tensor(labels)

        images = images.cuda(non_blocking=True)
        labels = labels.long().cuda(non_blocking=True)

        # Forward pass with automatic mixed precision
        # I had trouble with mixed precision training, so I'm not using it.  This should be fine though, just slower.

        if args.use_amp:
            with autocast(enabled=args.use_amp):
                embeddings = model(images)
                loss = module_partial_fc(embeddings, labels)
        else:
            embeddings = model(images)
            loss = module_partial_fc(embeddings, labels)

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        if (i + 1) % args.gradient_acc == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Only step() when the optimizer above also steps
            if scheduler is not None:
                scheduler.step()

        # Record loss
        losses.append(loss.item())

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress
        if rank == 0 and i % args.log_freq == 0:
            logger.info(
                # f"Epoch: [{epoch}][{i}/{len(data_loader)}] "
                f"Epoch: [{epoch}][{i}/{len(dataset)} batches] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {loss.item():.4f} ({np.mean(losses):.4f}) "
                f"LR {optimizer.param_groups[0]['lr']:.8f}"
            )

    return np.mean(losses)


def save_checkpoint(
    model, partial_fc, optimizer, scheduler, epoch, class_to_idx, checkpoint_dir, logger
):
    """Save training checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"unicom_checkpoint_{epoch:03d}.pt")

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.module.state_dict(),
        "partial_fc_state_dict": partial_fc.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "class_to_idx": class_to_idx,
    }

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")

    # Save a latest link
    latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
    torch.save(checkpoint, latest_path)

    return checkpoint_path


def load_checkpoint(
    checkpoint_path,
    model,
    partial_fc,
    optimizer,
    scheduler,
    load_optimizer=True,
    logger=None,
):
    """Load training checkpoint."""
    if logger:
        logger.info(f"Loading checkpoint from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Just use model directly instead of model.module
    model.load_state_dict(checkpoint["model_state_dict"])

    if "partial_fc_state_dict" in checkpoint:
        # SKIPPING THIS as it interferes with reloading the model/weights from checkpoint
        # partial_fc.load_state_dict(checkpoint["partial_fc_state_dict"])
        pass

    # Optimizer is very memory intensive.
    if load_optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if (
        scheduler
        and "scheduler_state_dict" in checkpoint
        and checkpoint["scheduler_state_dict"]
    ):
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    start_epoch = checkpoint["epoch"] + 1
    class_to_idx = checkpoint.get("class_to_idx", None)

    if logger:
        logger.info(f"Resumed from epoch {checkpoint['epoch']}")

    return start_epoch, class_to_idx


class AverageMeter:
    """Compute and store the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    parser = argparse.ArgumentParser(
        description="UNICOM Training on TreeOfLife with Clustering"
    )

    # Data parameters
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument(
        "--webdataset-path", required=True, help="Path to WebDataset tar files"
    )

    # Output parameters
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save output models"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True, help="Directory to save checkpoints"
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")

    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=2.8e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.05, help="Weight decay")
    parser.add_argument(
        "--gradient-acc", type=int, default=1, help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=1,
        help="Checkpoint saving frequency in epochs",
    )
    parser.add_argument(
        "--log-freq", type=int, default=10, help="Logging frequency in iterations"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=8,
        help="Number of data loading workers per GPU",
    )

    parser.add_argument(
        "--use-amp",
        action="store_true",
        default=True,
        help="Use automatic mixed precision",
    )

    # UNICOM specific parameters
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.125,
        help="Sample rate for random class selection",
    )
    parser.add_argument(
        "--num-feat",
        type=int,
        default=None,
        help="Number of features for random feature selection",
    )
    parser.add_argument(
        "--margin-loss-s", type=float, default=64.0, help="Scale for margin loss"
    )
    parser.add_argument(
        "--margin-loss-m1", type=float, default=1.0, help="Margin parameter m1"
    )
    parser.add_argument(
        "--margin-loss-m2", type=float, default=0.3, help="Margin parameter m2"
    )
    parser.add_argument(
        "--margin-loss-m3", type=float, default=0.0, help="Margin parameter m3"
    )

    # Misc parameters
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--resume", type=str, help="Path to checkpoint for resuming training"
    )
    parser.add_argument(
        "--use-checkpoint-classes",
        action="store_true",
        default=False,
        help="Use class mapping from checkpoint instead of recreating it",
    )
    parser.add_argument(
        "--lr-pfc-weight", type=float, default=10.0,
        help="The weight to apply to the learning rate for the Partial FC layer during training. Sure, when fine-tuning a pre-trained neural network, it is usually recommended to adjust the learning rates of different layers in order to achieve better performance. For example, the learning rate of the backbone layers (i.e., the pre-trained layers) should be set lower because they already have learned features, while the learning rate of the Partial FC layer should be set higher, as it needs to adapt to the new task"
    )
    # Roman's in-training eval
    parser.add_argument(
        "--tester-path", required=True, help="Path where tester .csv file was saved "
    )

    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Initialize distributed training
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(local_rank)

    # Set random seeds for reproducibility
    seed = args.seed + global_rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set up logging (only for rank 0)
    logger = setup_logging(args.log_dir) if global_rank == 0 else None
    if global_rank == 0:
        logger.info(f"World size: {world_size}, Global rank: {global_rank}")
        logger.info(f"Args: {args}")

    # Check if we should load class mapping from checkpoint first
    class_to_idx = None
    if args.resume and args.use_checkpoint_classes:
        if os.path.isfile(args.resume):
            if global_rank == 0:
                logger.info(f"Loading class mapping from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location="cpu")
            if "class_to_idx" in checkpoint:
                class_to_idx = checkpoint["class_to_idx"]
                if global_rank == 0:
                    logger.info(f"Loaded {len(class_to_idx)} classes from checkpoint")
            else:
                if global_rank == 0:
                    logger.warning("No class mapping found in checkpoint")

    # Load data from SQLite
    start_time = time.time()
    if class_to_idx is None:
        # Fresh start - create class mapping from scratch
        df, class_to_idx = load_data_from_sqlite(
            args.db_path, logger if global_rank == 0 else None
        )
    else:
        # Resume from checkpoint - use existing class mapping
        df, _ = load_data_from_sqlite(
            args.db_path, logger if global_rank == 0 else None
        )

        # Create pseudo_class labels
        df["pseudo_class"] = (
            df["species_name"] + "_" + df["single_node_cluster"].astype(str)
        )

        # Map pseudo_class to labels using checkpoint's class_to_idx
        # Any pseudo_class not in checkpoint gets filtered out
        df["label"] = df["pseudo_class"].map(class_to_idx)
        df = df.dropna(subset=["label"])
        df["label"] = df["label"].astype(int)

        if global_rank == 0:
            logger.info(
                f"After mapping to checkpoint classes: {len(df)} samples remain"
            )

    if global_rank == 0:
        logger.info(f"Data loading took {time.time() - start_time:.2f} seconds")
        logger.info(f"Total classes: {len(class_to_idx)}")

    # Wait for all processes to complete data loading
    dist.barrier()

    # Save class mapping (only from rank 0)
    if global_rank == 0:
        with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
            json.dump(class_to_idx, f)

    # Get the CLIP model and transforms
    if global_rank == 0:
        logger.info("Loading pretrained ViT-H-14-quickgelu model (dfn5b)")

    model, _, preprocess_train = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu", pretrained="dfn5b"
    )

    # Get embedding dimension from the model
    embedding_dim = model.visual.output_dim

    if global_rank == 0:
        logger.info(f"Model created. Embedding dimension: {embedding_dim}")

    # Use only the vision encoder part
    model = model.visual
    model.cuda()

    # Create optimizer
    # optimizer = optim.AdamW(
    #     model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    # )

    # Create learning rate scheduler
    steps_per_epoch = len(df) // (world_size * args.batch_size) + 1
    total_steps = args.epochs * steps_per_epoch

    # Create mixed precision scaler
    scaler = GradScaler(enabled=args.use_amp)

    # Define margin loss for UNICOM
    margin_loss = CombinedMarginLoss(
        s=args.margin_loss_s,
        m1=args.margin_loss_m1,
        m2=args.margin_loss_m2,
        m3=args.margin_loss_m3,
        interclass_filtering_threshold=0.0,
    )

    # Create Partial FC module - use class_to_idx length to match checkpoint exactly
    if global_rank == 0:
        logger.info(f"Creating PartialFC with {len(class_to_idx)} classes")

    module_partial_fc = PartialFC_V2(
        margin_loss=margin_loss,
        embedding_size=embedding_dim,
        num_classes=len(class_to_idx),  # Always use checkpoint's class count
        sample_rate=args.sample_rate,
        fp16=args.use_amp,
        sample_num_feat=args.num_feat,
    )

    module_partial_fc.cuda()

    optimizer = optim.AdamW(
        params=[
            {"params": model.parameters(), "lr": args.lr},  # model backbone group
            {"params": module_partial_fc.parameters(), "lr": args.lr * args.lr_pfc_weight},  # partial fc params
        ],
        lr=args.lr,  
        weight_decay=args.weight_decay,
    )

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )


    # Wait for all processes to sync at this point
    dist.barrier()

    # Load checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.isfile(args.resume):
            start_epoch, loaded_class_to_idx = load_checkpoint(
                args.resume,
                model,
                module_partial_fc,
                optimizer,
                scheduler,
                load_optimizer=True,
                logger=logger if global_rank == 0 else None,
            )

            # Clear memory because loading from checkpoint is very memory intensive
            torch.cuda.empty_cache()
            import gc

            gc.collect()

            # Verify class mappings match
            if loaded_class_to_idx and len(loaded_class_to_idx) != len(class_to_idx):
                if global_rank == 0:
                    logger.warning(
                        f"Class count mismatch: checkpoint has {len(loaded_class_to_idx)}, "
                        f"current has {len(class_to_idx)}"
                    )
        else:
            if global_rank == 0:
                logger.warning(f"Checkpoint not found: {args.resume}")

    # Wrap model with DDP
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )

    # Create dataset
    dataset = TreeOfLifeWebDataset(
        args.webdataset_path,
        df,
        transform=preprocess_train,
        world_size=torch.cuda.device_count(),  # Number of GPUs per node
        rank=local_rank,  # Local rank within node
        batch_size=args.batch_size,
    )

    if global_rank == 0:
        logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")

    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        # Create a new loader for each epoch to get different shuffling
        train_loader = dataset.create_loader(epoch=epoch)

        # Train one epoch
        train_loss = train_epoch(
            model,
            train_loader,
            module_partial_fc,
            optimizer,
            scheduler,
            scaler,
            epoch,
            args,
            logger if global_rank == 0 else None,
            rank=global_rank,
            dataset=dataset,
        )

        dist.barrier()

        if global_rank == 0:
            logger.info(f"Epoch {epoch} completed. Avg loss: {train_loss:.4f}")
            topk_accuracies = create_tester_embed_dataset(model, preprocess_train, args.tester_path)
            logger.info(f"Top-k accuracies (true labels) at epoch {epoch}: {topk_accuracies}")


            # Save checkpoint
            if (epoch + 1) % args.checkpoint_freq == 0 or epoch == args.epochs - 1:
                save_checkpoint(
                    model,
                    module_partial_fc,
                    optimizer,
                    scheduler,
                    epoch,
                    class_to_idx,
                    args.checkpoint_dir,
                    logger,
                )
        dist.barrier()

    # Save final model
    if global_rank == 0:
        logger.info("Saving final model...")
        torch.save(
            model.module.state_dict(),
            os.path.join(args.output_dir, "unicom_final_model.pt"),
        )
        logger.info("Training completed!")


if __name__ == "__main__":
    main()
