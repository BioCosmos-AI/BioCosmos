import os
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
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import GradScaler, autocast
from scipy.io import loadmat

# from torch.amp import GradScaler
from tqdm import tqdm
import open_clip
import webdataset as wds
import random
from datetime import datetime
import math
from collections import defaultdict
from PIL import Image
import PIL

from partial_fc import CombinedMarginLoss, PartialFC_V2
from eval_utils import *

os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"
os.environ["TORCH_DISTRIBUTED_TIMEOUT"] = "1800"  # 30 minutes in seconds


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


def load_data_from_mat(mat_path, logger=None):
    """Load relevant data from mat file into memory."""
    if logger:
        logger.info(f"Loading data from {mat_path}")

    try:
        data = loadmat(mat_path)
        loaded_embeddings = data['embeddings']

        # Create a DataFrame with each list as a column
        df = pd.DataFrame({
            'image_filename': data['image_filename'],
            'scientific_name': data['scientific_name'],
            'taxa': data['taxa'],
            'embeddings': [sub_array for sub_array in loaded_embeddings],
            'cluster_id': data['cluster_id'][0],
            'image_path': data['image_path']
        })
        
        # strip() on all string columns
        df.loc[:, :] = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

        if logger:
            logger.info(f"Loaded {len(df)} total records from database")

        # Create pseudo-class labels by combining species_name and single_node_cluster
        df["pseudo_class"] = df['cluster_id']

        if logger:
            logger.info(f"Loaded {df['pseudo_class'].nunique()} unique pseudo-classes")

        return df

    except Exception as e:
        if logger:
            logger.error(f"Error loading data from mat: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
        return pd.DataFrame()


class VLM4BioDataset:
    """Dataset iterator for VLM4Bio with clustering info that evenly distributes images across GPUs."""

    def __init__(
        self, df, transform=None
    ):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row["image_path"].strip()
        img_filename = row["image_filename"].strip()
        species = row["scientific_name"].strip()
        cluster_id = row["cluster_id"]
        try:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
            return image, cluster_id, img_filename, species
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        return ds


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
    tracked_set,
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
        images, labels, img_filenames, species = batch

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
            
        
        # fails for distributed case --> classes are distributed across ranks
        # for idx, filename in enumerate(img_filenames):
        #     if filename in tracked_set:
        #         target_label = labels[idx].item()
        #         center_vec = module_partial_fc.weight[target_label]
        #         emb_vec = F.normalize(embeddings[idx].unsqueeze(0), dim=1)
        #         cos_sim = F.cosine_similarity(center_vec.unsqueeze(0), emb_vec, dim=1).item()
        #         logger.info(f"[Rank {rank}] Epoch {epoch}, Tracked image {filename}, Species {species[idx]}: Cosine sim = {cos_sim:.4f}")
        
        # With this we can ONLY see the classes on rank 0 --> if we want to coalesce there are other ways
        if rank == 0 and logger is not None:
            # This rank owns global classes in [class_start, class_start + num_local)
            class_start = int(module_partial_fc.class_start)
            num_local   = int(module_partial_fc.num_local)
            class_end   = class_start + num_local

            for j, filename in enumerate(img_filenames):
                # Only care about tracked files
                if filename not in tracked_set:
                    continue

                # Dataset labels are global ids
                global_lbl = int(labels[j])

                # If this global class is NOT on this rank, skip
                if not (class_start <= global_lbl < class_end):
                    continue

                # Map global -> local index into this rank's shard
                local_idx = global_lbl - class_start

                # Get the local center vector (full shard, not the per-step sampled subset)
                center_vec = module_partial_fc.weight[local_idx].detach()

                # Normalize both vectors for cosine similarity
                emb_vec_n    = F.normalize(embeddings[j].detach().unsqueeze(0), dim=1)
                center_vec_n = F.normalize(center_vec.unsqueeze(0), dim=1)

                cos_sim = F.cosine_similarity(center_vec_n, emb_vec_n, dim=1).item()

                # species is typically a list of strings from the collate
                sp = species[j] if isinstance(species, (list, tuple)) else species

                logger.info(
                    f"[Rank {rank}] Epoch {epoch} Tracked '{filename}' "
                    f"(class {global_lbl}, species {sp}): cos={cos_sim:.4f}"
                )

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        if (i + 1) % args.gradient_acc == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if scheduler is not None:
            scheduler.step()

        # Record loss
        losses.append(loss.item())

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Log progress
        if rank == 0 and logger is not None and i % args.log_freq == 0:
            names = [pg.get("name", str(k)) for k, pg in enumerate(optimizer.param_groups)]
            lrs   = [pg["lr"] for pg in optimizer.param_groups]
            lr_str = " ".join(f"{n}LR={lr:.6e}" for n, lr in zip(names, lrs)) # wanna get learning rate for all

            logger.info(
                f"Epoch: [{epoch}][{i}/{len(data_loader)}] "
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}) "
                f"Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                f"Loss {loss.item():.4f} ({np.mean(losses):.4f}) "
                f"{lr_str}"
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
        partial_fc.load_state_dict(checkpoint["partial_fc_state_dict"])

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

    parser.add_argument(
        "--mat-path", required=True, help="Path where .mat file was saved "
    )
    # Output parameters
    parser.add_argument(
        "--output-dir", required=True, help="Directory to save output models"
    )
    parser.add_argument(
        "--checkpoint-dir", required=True, help="Directory to save checkpoints"
    )
    parser.add_argument("--log-dir", required=True, help="Directory for log files")
    parser.add_argument(
        "--tester-path", required=True, help="Path where tester .csv file was saved "
    )
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size per GPU")
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate")
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
        # default=False,  # I had trouble with this == True
        default=True,
        help="Use automatic mixed precision",
    )

    # UNICOM specific parameters
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=0.1,
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

    parser.add_argument(
        "--lr-pfc-weight", type=float, default=5.0,
        help="The weight to apply to the learning rate for the Partial FC layer during training. Sure, when fine-tuning a pre-trained neural network, it is usually recommended to adjust the learning rates of different layers in order to achieve better performance. For example, the learning rate of the backbone layers (i.e., the pre-trained layers) should be set lower because they already have learned features, while the learning rate of the Partial FC layer should be set higher, as it needs to adapt to the new task"
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
    #if args.resume and args.use_checkpoint_classes:
    #    if os.path.isfile(args.resume):
    #        if global_rank == 0:
    #            logger.info(f"Loading class mapping from checkpoint: {args.resume}")
    #        checkpoint = torch.load(args.resume, map_location="cpu")
    #        if "class_to_idx" in checkpoint:
    #            class_to_idx = checkpoint["class_to_idx"]
    #            if global_rank == 0:
    #                logger.info(f"Loaded {len(class_to_idx)} classes from checkpoint")
    #        else:
    #            if global_rank == 0:
    #                logger.warning("No class mapping found in checkpoint")

    # Load data from .mat file
    df = load_data_from_mat(args.mat_path, logger if global_rank == 0 else None)
    
    start_time = time.time()

    df["label"] = df["cluster_id"]

    if global_rank == 0:
        logger.info(f"Data loading took {time.time() - start_time:.2f} seconds")
        #logger.info(f"Total classes: {len(class_to_idx)}")

        logger.info(f"Checking info on CUDA. Cuda device name: {torch.cuda.get_device_name(0)}. Torch cuda version: {torch.version.cuda}")

    # Wait for all processes to complete data loading
    dist.barrier()
    # Save class mapping (only from rank 0)
    # if global_rank == 0:
    #     with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
    #         json.dump(class_to_idx, f)

    # Get the CLIP model and transforms
    if global_rank == 0:
        logger.info("Loading pretrained ViT-H-14-378-quickgelu model (dfn5b)")

    # The function returns model, tokenizer (which we don't need), and transform
    model, _, preprocess_train = open_clip.create_model_and_transforms(
        "ViT-H-14-378-quickgelu", pretrained="dfn5b"
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
    # ) --> we didn't optimize partial_fc!!!! 

    # Create Partial FC module for UNICOM training
    module_partial_fc = PartialFC_V2(
        margin_loss=margin_loss,
        embedding_size=embedding_dim,
        num_classes=len(df["label"].unique()),
        sample_rate=args.sample_rate,
        fp16=args.use_amp,  # Enable/disable mixed precision
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


                # opt = torch.optim.AdamW(
                # params=[
                #     {"params": backbone.parameters()},
                #     {"params": module_partial_fc.parameters(), "lr": args.lr * args.lr_pfc_weight}],
                # lr=args.lr, weight_decay=args.weight_decay)


    # Create learning rate scheduler
    steps_per_epoch = len(df) // (world_size * args.batch_size) + 1
    total_steps = args.epochs * steps_per_epoch

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        # max_lr=args.lr, # could be same lr for both or different for each part
        max_lr=[args.lr, args.lr * args.lr_pfc_weight],
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy="cos",
    )

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

    # Wait for all processes to sync at this point
    dist.barrier()

    # Load checkpoint if specified
    start_epoch = 0

    # Wrap model with DDP
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    
    # select random 10 images - this will be used to test cluster based eval per epoch
    if global_rank == 0:
        tracked_filenames = df['image_filename'].drop_duplicates().sample(n=10, random_state=42).tolist()
    else:
        tracked_filenames = [None] * 10
        
    #tracked_filenames = dist.broadcast_object_list(tracked_filenames, src=0)
    #tracked_filenames = tracked_filenames if isinstance(tracked_filenames, list) else tracked_filenames

    dist.broadcast_object_list(tracked_filenames, src=0)

    dataset = VLM4BioDataset(
        df,
        transform=preprocess_train,
    )

    # Create sampler
    # Using webdataset's own distributed sampler
    sampler = DistributedSampler(
        dataset, num_replicas=world_size, rank=global_rank, shuffle=True, drop_last=True, seed=args.seed
    )
    
    train_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )


    # Create WebLoader directly without a sampler
    # train_loader = wds.WebLoader(
    #     dataset,
    #     batch_size=args.batch_size,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    #     shuffle=False,  # WebDataset handles shuffling differently
    # )

    # train_loader = train_loader.shuffle(1000)  # Buffer size of 1000

    # def set_epoch(epoch):
    #     # Set a different seed for each epoch for shuffling
    #     random.seed(args.seed + epoch + global_rank * 100)

    # train_loader.set_epoch = set_epoch

    if global_rank == 0:
        logger.info(f"Starting training from epoch {start_epoch} to {args.epochs}")

    # Main training loop
    for epoch in range(start_epoch, args.epochs):
        # Create a new loader for each epoch to get different shuffling
        # train_loader = dataset.create_loader(epoch=epoch)
        sampler.set_epoch(epoch)

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
            tracked_filenames,
            rank=global_rank,
            dataset=dataset,
        )

        if global_rank == 0:
            logger.info(f"Finished training epoch {epoch}")
        

        # dist barrier ensures no other ranks get ahead of global rank 0
        dist.barrier()
        if global_rank == 0 :#and epoch % args.eval_freq == 0:
            topk_accuracies = create_tester_embed_dataset(model, preprocess_train, args.tester_path)
            logger.info(f"Top-k accuracies (true labels) at epoch {epoch}: {topk_accuracies}")


            logger.info(f"Epoch {epoch} completed. Avg loss: {train_loss:.4f}")

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
