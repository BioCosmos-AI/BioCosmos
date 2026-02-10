#!/usr/bin/env python3
import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import seaborn as sns
import argparse
import time
from tqdm import tqdm
import logging
import tarfile
import io
from PIL import Image
import tempfile
from sklearn.metrics import pairwise_distances

# Try to import GPU libraries, but fall back gracefully if not available
try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cuml
    from cuml.manifold import TSNE as cuTSNE

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


def setup_logging(log_dir):
    """Set up logging configuration."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"tsne_viz_{timestamp}.log")

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


def check_gpu():
    """Check if GPU is available and return info."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        gpu_info = {
            "available": True,
            "device_name": torch.cuda.get_device_name(0),
            "device_count": torch.cuda.device_count(),
            "cuml_available": CUML_AVAILABLE,
        }
        return gpu_info
    else:
        return {"available": False, "cuml_available": CUML_AVAILABLE}


def load_checkpoint(checkpoint_path, logger):
    """Load checkpoint data from the specified path."""
    try:
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)
        logger.info(
            f"Successfully loaded checkpoint containing {len(checkpoint.get('species_processed', []))} processed species"
        )
        return checkpoint
    except Exception as e:
        logger.error(f"Error loading checkpoint: {str(e)}")
        return None


def load_species_data(db_path, species_list, logger):
    """Load data for multiple species from the SQLite database."""
    logger.info(f"Loading data for {len(species_list)} species from {db_path}")

    # Fix capitalization for "abaeis nicippe" -> "Abaeis nicippe"
    corrected_species_list = []
    for species in species_list:
        if species.lower() == "abaeis nicippe":
            corrected_species_list.append("Abaeis nicippe")
            logger.info(
                f"Corrected species name from 'abaeis nicippe' to 'Abaeis nicippe'"
            )
        else:
            corrected_species_list.append(species)

    try:
        # Format the species list for SQL IN clause
        species_str = ", ".join(f"'{species}'" for species in corrected_species_list)

        # Connect to database
        connection = sqlite3.connect(db_path, timeout=300.0)

        # Query for our specific species only, include shard_id
        query = f"""
        SELECT uuid, shard_id, species_name, embedding, single_node_cluster
        FROM image_embeddings
        WHERE species_name IN ({species_str})
        """

        logger.info("Starting database query...")
        logger.info(f"Query: {query}")

        start_time = time.time()
        df = pd.read_sql_query(query, connection)
        query_time = time.time() - start_time
        logger.info(
            f"Query completed in {query_time:.2f} seconds, retrieved {len(df)} rows"
        )

        # Log the species counts
        species_counts = df["species_name"].value_counts()
        logger.info(f"Species counts in query result: {species_counts.to_dict()}")

        connection.close()

        if len(df) == 0:
            logger.warning("No data found for any of the requested species")
            return None

        # Process species data
        species_data = {}
        for species_name in corrected_species_list:
            species_df = df[df["species_name"] == species_name].copy()

            if len(species_df) == 0:
                logger.warning(f"No data found for species: {species_name}")
                continue  # Skip to next species instead of returning None

            # Parse embeddings from binary
            logger.info(
                f"Processing embeddings for {species_name} ({len(species_df)} rows)"
            )

            # Use a different approach to handle embeddings and maintain proper indexing
            valid_embeddings = []
            valid_df = pd.DataFrame()  # Create a new, filtered DataFrame

            # Reset the index to ensure safe iteration
            species_df = species_df.reset_index(drop=True)

            for i, row in tqdm(
                species_df.iterrows(),
                total=len(species_df),
                desc=f"Processing {species_name}",
            ):
                try:
                    if row["embedding"] is not None:
                        embedding = np.frombuffer(row["embedding"], dtype=np.float32)
                        if not np.isnan(embedding).any() and not (embedding == 0).all():
                            valid_embeddings.append(embedding)
                            valid_df = pd.concat(
                                [valid_df, pd.DataFrame([row])], ignore_index=True
                            )
                except Exception as e:
                    logger.warning(
                        f"Error processing embedding for UUID {row['uuid']}: {str(e)}"
                    )

            if len(valid_embeddings) == 0:
                logger.warning(f"No valid embeddings found for species: {species_name}")
                continue  # Skip to next species instead of returning None

            # Create embeddings array
            embeddings_array = np.array(valid_embeddings)

            logger.info(
                f"Processed {len(valid_df)} valid embeddings for {species_name}"
            )
            species_data[species_name] = {
                "df": valid_df,
                "embeddings": embeddings_array,
            }

        # Log how many species we successfully processed
        logger.info(
            f"Successfully processed data for {len(species_data)} species out of {len(corrected_species_list)} requested"
        )

        # Return species_data even if it's not complete for all requested species
        return species_data if len(species_data) > 0 else None

    except Exception as e:
        logger.error(f"Error loading species data: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def generate_tsne(embeddings, use_gpu, logger, perplexity=30, n_components=2):
    """Generate t-SNE representation using GPU if available."""
    logger.info(
        f"Generating {n_components}D t-SNE projection for {len(embeddings)} samples"
    )
    start_time = time.time()

    if len(embeddings) <= 1:
        logger.warning("Cannot generate t-SNE with only one sample")
        return None

    # Adjust perplexity if needed (must be smaller than n_samples - 1)
    perplexity = min(perplexity, len(embeddings) - 1)

    tsne_result = None

    # Try GPU implementation first if requested
    if use_gpu and CUML_AVAILABLE:
        try:
            logger.info("Attempting to use GPU-accelerated t-SNE via cuML")
            tsne = cuTSNE(
                n_components=n_components, random_state=42, perplexity=perplexity
            )
            tsne_result = tsne.fit_transform(embeddings)

            # Convert to numpy if needed
            if hasattr(tsne_result, "get"):
                tsne_result = tsne_result.get()
            elif hasattr(tsne_result, "to_numpy"):
                tsne_result = tsne_result.to_numpy()

            logger.info("Successfully used GPU-accelerated t-SNE")
        except Exception as e:
            logger.warning(f"GPU t-SNE failed, falling back to CPU: {str(e)}")
            import traceback

            logger.warning(traceback.format_exc())
            tsne_result = None

    # Fall back to CPU if GPU failed or wasn't requested
    if tsne_result is None:
        try:
            logger.info("Using CPU-based t-SNE")
            from sklearn.manifold import TSNE

            tsne = TSNE(
                n_components=n_components, random_state=42, perplexity=perplexity
            )
            tsne_result = tsne.fit_transform(embeddings)
            logger.info("Successfully used CPU-based t-SNE")
        except Exception as e:
            logger.error(f"CPU t-SNE failed: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())
            return None

    elapsed = time.time() - start_time
    logger.info(f"t-SNE completed in {elapsed:.2f} seconds")

    return tsne_result


def get_cluster_representatives(df, embeddings, tsne_result, logger):
    """Find representative samples for each cluster (closest to centroid)."""
    logger.info("Finding representative samples for each cluster")

    # If no clusters present, return empty results
    if (
        "single_node_cluster" not in df.columns
        or df["single_node_cluster"].isna().all()
    ):
        logger.warning("No cluster information available - can't find representatives")
        return {}

    # Reset index to ensure alignment
    df = df.reset_index(drop=True)

    # Get unique clusters - skip NA values
    clusters = sorted([c for c in df["single_node_cluster"].unique() if not pd.isna(c)])
    representatives = {}

    # Create a copy of df with tsne coordinates
    df_with_tsne = df.copy()
    df_with_tsne["x"] = tsne_result[:, 0]
    df_with_tsne["y"] = tsne_result[:, 1]

    for cluster_id in clusters:
        logger.info(f"Finding representative for cluster {cluster_id}")

        # Get indices for this cluster
        cluster_mask = df_with_tsne["single_node_cluster"] == cluster_id
        if not cluster_mask.any():
            logger.warning(f"No samples in cluster {cluster_id}")
            continue

        cluster_indices = np.where(cluster_mask)[0]

        # If only one sample, it's the representative
        if len(cluster_indices) == 1:
            idx = cluster_indices[0]
            representatives[cluster_id] = {
                "index": idx,
                "uuid": df_with_tsne.iloc[idx]["uuid"],
                "shard_id": df_with_tsne.iloc[idx]["shard_id"],
                "position": (df_with_tsne.iloc[idx]["x"], df_with_tsne.iloc[idx]["y"]),
            }
            continue

        # Get embeddings for this cluster
        cluster_embeddings = embeddings[cluster_indices]

        # Calculate cluster centroid
        centroid = np.mean(cluster_embeddings, axis=0)

        # Find the sample closest to the centroid
        distances = np.sqrt(np.sum((cluster_embeddings - centroid) ** 2, axis=1))
        closest_idx = cluster_indices[np.argmin(distances)]

        # Get position in t-SNE space
        position = (
            df_with_tsne.iloc[closest_idx]["x"],
            df_with_tsne.iloc[closest_idx]["y"],
        )

        representatives[cluster_id] = {
            "index": closest_idx,
            "uuid": df_with_tsne.iloc[closest_idx]["uuid"],
            "shard_id": df_with_tsne.iloc[closest_idx]["shard_id"],
            "position": position,
        }

    logger.info(f"Found {len(representatives)} representative samples")
    return representatives


def extract_image_from_tar(uuid, shard_id, data_dir, logger):
    """Extract an image from a WebDataset tar file based on UUID and shard_id."""
    # Format shard filename with leading zeros
    try:
        shard_num = int(shard_id)
        shard_filename = f"shard-{shard_num:06d}.tar"
    except (ValueError, TypeError):
        logger.error(f"Invalid shard_id: {shard_id}")
        return None

    tar_path = os.path.join(data_dir, shard_filename)

    logger.info(f"Extracting image for UUID {uuid} from shard {shard_filename}")

    try:
        # Check if tar file exists
        if not os.path.exists(tar_path):
            logger.error(f"Tar file not found: {tar_path}")
            return None

        # Open tar file
        with tarfile.open(tar_path, "r") as tar:
            # Try different image extensions
            for ext in [".jpg", ".jpeg", ".png"]:
                image_name = f"{uuid}{ext}"
                try:
                    # Extract the file if it exists
                    member = tar.getmember(image_name)
                    f = tar.extractfile(member)
                    if f:
                        img_data = f.read()
                        img = Image.open(io.BytesIO(img_data))
                        return img
                except (KeyError, tarfile.ReadError) as e:
                    # File not found with this extension, try next
                    continue

            # If we got here, we didn't find any image file
            logger.warning(f"No image file found for UUID {uuid} in {tar_path}")
            return None

    except Exception as e:
        logger.error(f"Error extracting image: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return None


def generate_visualization_with_images(
    species_name,
    species_data,
    representatives,
    data_dir,
    output_dir,
    logger,
    jitter_amount=0.05,
):
    """Generate and save t-SNE visualization for a species with representative images."""
    logger.info(f"Generating visualization with images for {species_name}")

    valid_df = species_data["df"].copy()

    # Add jitter to spread out points
    # Calculate range of x and y to scale jitter appropriately
    x_range = valid_df["x"].max() - valid_df["x"].min()
    y_range = valid_df["y"].max() - valid_df["y"].min()

    # Apply jitter proportional to the range
    valid_df["x_jittered"] = valid_df["x"] + np.random.normal(
        0, jitter_amount * x_range, len(valid_df)
    )
    valid_df["y_jittered"] = valid_df["y"] + np.random.normal(
        0, jitter_amount * y_range, len(valid_df)
    )

    logger.info(
        f"Applied jitter of {jitter_amount * 100}% of data range to spread points"
    )

    # Set up plot style
    plt.figure(figsize=(14, 12))
    sns.set_style("whitegrid")

    # Check if we have cluster information
    cluster_column = "single_node_cluster"
    if cluster_column in valid_df.columns and valid_df[cluster_column].notna().any():
        # Count unique clusters
        unique_clusters = valid_df[cluster_column].dropna().unique()
        num_clusters = len(unique_clusters)

        logger.info(f"Found {num_clusters} clusters for {species_name}")

        # Create color palette
        if num_clusters > 10:
            colors = sns.color_palette("husl", num_clusters)
        else:
            colors = sns.color_palette("Set1", num_clusters)

        cmap = ListedColormap(colors)

        # Create scatter plot with cluster colors using jittered coordinates
        scatter = plt.scatter(
            valid_df["x_jittered"],
            valid_df["y_jittered"],
            c=valid_df[cluster_column],
            cmap=cmap,
            s=50,  # Smaller point size to avoid overcrowding
            alpha=0.6,  # More transparency to see overlapping points
            edgecolor="k",
            linewidth=0.5,
        )

        # Add legend
        legend_elements = []
        for i, cluster_id in enumerate(sorted(unique_clusters)):
            if cluster_id == -1:
                label = "Outliers"
            else:
                label = f"Cluster {int(cluster_id)}"
            legend_elements.append(
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=colors[i % len(colors)],
                    markersize=10,
                    label=f"{label} ({sum(valid_df[cluster_column] == cluster_id)} samples)",
                )
            )

        plt.legend(
            handles=legend_elements,
            title="Clusters",
            loc="best",
            fontsize="medium",
            title_fontsize="large",
        )
    else:
        # If no cluster information, just plot points
        logger.warning(
            f"No cluster information found for {species_name}, using uniform color"
        )
        plt.scatter(
            valid_df["x_jittered"],
            valid_df["y_jittered"],
            s=50,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.5,
        )

    # Calculate appropriate image size based on number of clusters
    img_zoom = max(0.1, min(0.7, 1.0 / (max(len(representatives), 1) ** 0.5)))
    logger.info(f"Using image zoom factor: {img_zoom:.2f}")

    # Add representative images
    if representatives:
        unique_clusters = (
            valid_df[cluster_column].dropna().unique()
            if cluster_column in valid_df.columns
            else []
        )
        for cluster_id, rep in representatives.items():
            logger.info(f"Adding representative image for cluster {cluster_id}")

            # Extract image from tar file
            img = extract_image_from_tar(rep["uuid"], rep["shard_id"], data_dir, logger)

            if img is not None:
                # Create OffsetImage
                imagebox = OffsetImage(img, zoom=img_zoom)

                # Create annotation box - use original coordinates, not jittered ones
                if len(unique_clusters) > 0:
                    edge_color = colors[
                        list(sorted(unique_clusters)).index(cluster_id) % len(colors)
                    ]
                else:
                    edge_color = "black"

                ab = AnnotationBbox(
                    imagebox,
                    rep["position"],
                    pad=0.0,
                    frameon=True,
                    bboxprops=dict(edgecolor=edge_color, linewidth=3),
                )

                # Add to plot
                plt.gca().add_artist(ab)
            else:
                logger.warning(
                    f"Could not add image for cluster {cluster_id} (UUID: {rep['uuid']})"
                )
    else:
        logger.warning("No representative images found to display")

    # Add sample count information to the title
    cluster_counts = (
        valid_df[cluster_column].value_counts().to_dict()
        if cluster_column in valid_df.columns
        else {}
    )
    cluster_info = ", ".join(
        [f"Cluster {int(k)}: {v}" for k, v in cluster_counts.items()]
    )

    # Titles and labels
    plt.title(
        f"t-SNE Visualization of {species_name}\n({len(valid_df)} samples)", fontsize=16
    )
    plt.xlabel("t-SNE dimension 1", fontsize=14)
    plt.ylabel("t-SNE dimension 2", fontsize=14)

    # Add a text box with cluster counts
    if cluster_info:
        plt.figtext(
            0.5,
            0.01,
            f"Cluster counts: {cluster_info}",
            ha="center",
            fontsize=12,
            bbox={"facecolor": "white", "alpha": 0.5, "pad": 5},
        )

    plt.tight_layout()

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    safe_name = species_name.replace(" ", "_").lower()
    output_path = os.path.join(output_dir, f"{safe_name}_tsne_with_images_jittered.png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved visualization to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate t-SNE visualizations with images for specific species"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/blue/arthur.porto-biocosmos/tdeatherage3.gatech/embeddings/image_embeddings.sqlite",
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default="/home/tdeatherage3.gatech/logs/clustering_single_60302562/checkpoints/checkpoint.pkl",
        help="Path to checkpoint file",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/blue/arthur.porto-biocosmos/data/datasets/TreeOfLife-10M/dataset/evobio10m-CVPR-2024/224x224/train",
        help="Directory containing WebDataset tar files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./tsne_visualizations",
        help="Directory to save visualizations",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory for log files"
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        default=True,
        help="Use GPU acceleration if available",
    )
    parser.add_argument(
        "--species",
        nargs="+",
        default=[
            "Abagrotis alternata",
            "Abaeis nicippe",  # Corrected capitalization
            "Hemicircus canente",
            "Hemigomphus comitatus",
            "Zyrphelis crenata",
            "Zygaena oxytropis",
        ],
        help="List of species to visualize",
    )

    args = parser.parse_args()

    # Set up logging
    logger = setup_logging(args.log_dir)

    # Check GPU availability
    gpu_info = check_gpu()
    if gpu_info["available"]:
        logger.info(f"GPU available: {gpu_info['device_name']}")
        if gpu_info["cuml_available"]:
            logger.info("cuML is available for GPU acceleration")
        else:
            logger.info(
                "cuML is not available, will use GPU only for PyTorch operations"
            )
    else:
        logger.warning("GPU not available, will use CPU processing only")
        if args.use_gpu:
            logger.info("--use-gpu flag is set but no GPU detected")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint for verification
    checkpoint = load_checkpoint(args.checkpoint_path, logger)
    if checkpoint:
        if "species_processed" in checkpoint:
            processed_species = set(checkpoint["species_processed"])
            for species in args.species:
                if species in processed_species:
                    logger.info(f"Species '{species}' found in processed species list")
                else:
                    logger.warning(
                        f"Warning: Species '{species}' not found in processed species list"
                    )

    # Load data for all species at once to minimize database connections
    species_data = load_species_data(args.db_path, args.species, logger)

    if not species_data:
        logger.error("Failed to load any species data. Exiting.")
        return

    # Process each species
    for species_name in list(
        species_data.keys()
    ):  # Convert to list to avoid modification during iteration
        logger.info(
            f"Processing {species_name} with {len(species_data[species_name]['embeddings'])} embeddings"
        )

        # Generate t-SNE
        tsne_result = generate_tsne(
            species_data[species_name]["embeddings"],
            use_gpu=args.use_gpu and gpu_info["available"],
            logger=logger,
        )

        if tsne_result is None:
            logger.error(
                f"Failed to generate t-SNE for {species_name}. Skipping visualization."
            )
            continue

        # Add t-SNE coordinates to dataframe
        species_data[species_name]["df"]["x"] = tsne_result[:, 0]
        species_data[species_name]["df"]["y"] = tsne_result[:, 1]

        # Find representative samples for each cluster
        representatives = get_cluster_representatives(
            species_data[species_name]["df"],
            species_data[species_name]["embeddings"],
            tsne_result,
            logger,
        )

        # Generate visualization with images
        output_path = generate_visualization_with_images(
            species_name,
            species_data[species_name],
            representatives,
            args.data_dir,
            args.output_dir,
            logger,
        )

    logger.info("All visualizations completed!")


if __name__ == "__main__":
    main()
