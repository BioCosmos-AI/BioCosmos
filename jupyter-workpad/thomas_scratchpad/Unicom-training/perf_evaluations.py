import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import clip
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np


class WarpModule(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)


class TestDataset(Dataset):
    def __init__(self, csv_path, base_path="../Image-Captioning", transform=None):
        # Read CSV and filter for test split
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == "test"].reset_index(drop=True)
        # Filter out rows with NaN scientific names -- Not sure why we have any of those?
        self.df = self.df.dropna(subset=["scientific_name"])

        self.base_path = Path(base_path)
        self.transform = transform

        # Get unique categories and species for label encoding
        self.categories = sorted(self.df["category"].unique())
        self.species = sorted(self.df["scientific_name"].str.lower().unique())

        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.species_to_idx = {sp: idx for idx, sp in enumerate(self.species)}

        # Print some dataset statistics
        print("\nDataset Statistics:")
        print(f"Total test samples: {len(self.df)}")
        print(f"Categories: {len(self.categories)}")
        print(f"Species: {len(self.species)}")
        for cat in self.categories:
            cat_count = len(self.df[self.df["category"] == cat])
            print(f"\n{cat}:")
            print(f"  Total samples: {cat_count}")
            species_in_cat = self.df[self.df["category"] == cat][
                "scientific_name"
            ].nunique()
            print(f"  Unique species: {species_in_cat}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert relative path to absolute path
        image_path = self.base_path / row["image_path"]
        category = row["category"]
        species = row["scientific_name"].lower()  # Case insensitive

        try:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        category_label = self.category_to_idx[category]
        species_label = self.species_to_idx[species]

        return image, category_label, species_label, category, species


def load_trained_model(checkpoint_path):
    # Load CLIP and modify
    model, transform = clip.load("ViT-B/32")
    model = model.visual
    model = model.float()
    model = WarpModule(model)

    # Load trained weights unless we want the base (untrained) model
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])

    return model, transform


def plot_training_log(log_file):
    # Read the log file
    df = pd.read_csv(log_file)

    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot loss
    sns.lineplot(data=df, x="global_step", y="loss", ax=ax1)
    ax1.set_title("Training Loss over Time")
    ax1.set_xlabel("Steps")
    ax1.set_ylabel("Loss")

    # Plot learning rates
    sns.lineplot(
        data=df,
        x="global_step",
        y="learning_rate_backbone",
        label="Backbone LR",
        ax=ax2,
    )
    sns.lineplot(
        data=df, x="global_step", y="learning_rate_pfc", label="PFC LR", ax=ax2
    )
    ax2.set_title("Learning Rates over Time")
    ax2.set_xlabel("Steps")
    ax2.set_ylabel("Learning Rate")

    plt.tight_layout()
    plt.savefig("training_plots.png")
    plt.close()


@torch.no_grad()
def evaluate_retrieval(model, test_loader):
    model.eval()

    # Lists to store embeddings and metadata
    all_embeddings = []
    all_category_labels = []
    all_species_labels = []
    all_categories = []
    all_species = []

    # Get embeddings for all test images
    for images, cat_labels, sp_labels, categories, species in test_loader:
        images = images.cuda()
        embeddings = model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_category_labels.append(cat_labels)
        all_species_labels.append(sp_labels)
        all_categories.extend(categories)
        all_species.extend(species)

    # Concatenate all embeddings and labels
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_category_labels = torch.cat(all_category_labels, dim=0)
    all_species_labels = torch.cat(all_species_labels, dim=0)

    # Compute similarity matrix
    similarity = torch.mm(all_embeddings, all_embeddings.t())

    # Remove diagonal elements
    mask = torch.eye(similarity.shape[0], dtype=bool)
    similarity[mask] = -float("inf")

    # Compute metrics for different k values
    ks = [1, 5, 10]
    metrics = {}

    for k in ks:
        _, indices = similarity.topk(k, dim=1)

        # Per-category metrics
        category_metrics = {}
        for category in set(all_categories):
            category_mask = [c == category for c in all_categories]
            category_indices = [i for i, m in enumerate(category_mask) if m]

            if not category_indices:
                continue

            correct = 0
            for idx in category_indices:
                retrieved_categories = [all_categories[i] for i in indices[idx]]
                if category in retrieved_categories:
                    correct += 1

            recall = correct / len(category_indices)
            category_metrics[category] = {
                "recall": recall,
                "count": len(category_indices),
            }

        # Per-species metrics
        species_metrics = {}
        for species in set(all_species):
            species_mask = [s == species for s in all_species]
            species_indices = [i for i, m in enumerate(species_mask) if m]

            if not species_indices:
                continue

            correct = 0
            for idx in species_indices:
                retrieved_species = [all_species[i] for i in indices[idx]]
                if species in retrieved_species:
                    correct += 1

            recall = correct / len(species_indices)
            species_metrics[species] = {
                "recall": recall,
                "count": len(species_indices),
                "category": all_categories[
                    species_indices[0]
                ],  # Store category for grouping
            }

        # Overall metrics
        category_correct = 0
        species_correct = 0
        total = len(all_category_labels)

        for i, (cat_label, sp_label) in enumerate(
            zip(all_category_labels, all_species_labels)
        ):
            retrieved_cat_labels = all_category_labels[indices[i]]
            retrieved_sp_labels = all_species_labels[indices[i]]

            if cat_label in retrieved_cat_labels:
                category_correct += 1
            if sp_label in retrieved_sp_labels:
                species_correct += 1

        metrics[f"recall@{k}"] = {
            "overall_category": category_correct / total,
            "overall_species": species_correct / total,
            "per_category": category_metrics,
            "per_species": species_metrics,
        }

    return metrics


def print_metrics(metrics, output_file=None):
    def write_line(line, file=None):
        print(line)
        if file:
            file.write(line + "\n")

    f = open(output_file, "w") if output_file else None

    for k, results in metrics.items():
        write_line(f"\n{k.upper()}:", f)
        write_line(f"Overall Category Recall: {results['overall_category']:.4f}", f)
        write_line(f"Overall Species Recall: {results['overall_species']:.4f}", f)

        write_line("\nPer Category Results:", f)
        write_line("--------------------", f)
        for category, cat_results in results["per_category"].items():
            write_line(f"{category}:", f)
            write_line(f"  Recall: {cat_results['recall']:.4f}", f)
            write_line(f"  Count: {cat_results['count']}", f)

        write_line("\nPer Species Results:", f)
        write_line("-------------------", f)
        # Group species by category
        species_by_category = {}
        for species, sp_results in results["per_species"].items():
            category = sp_results["category"]
            if category not in species_by_category:
                species_by_category[category] = []
            species_by_category[category].append((species, sp_results))

        for category in sorted(species_by_category.keys()):
            write_line(f"\n{category}:", f)
            for species, sp_results in sorted(species_by_category[category]):
                write_line(f"  {species}:", f)
                write_line(f"    Recall: {sp_results['recall']:.4f}", f)
                write_line(f"    Count: {sp_results['count']}", f)

    if f:
        f.close()


def analyze_species_performance(csv_path, metrics):
    # Read full dataset to get training counts
    df = pd.read_csv(csv_path)

    # Get training counts per species (case insensitive)
    train_counts = (
        df[df["split"] == "train"]["scientific_name"]
        .str.lower()
        .value_counts()
        .to_dict()
    )

    # Get species performance from metrics
    species_metrics = metrics["recall@1"]["per_species"]

    # Create analysis dataframe
    analysis_data = []
    for species, metrics in species_metrics.items():
        species_lower = species.lower()
        analysis_data.append(
            {
                "species": species_lower,
                "category": metrics["category"],
                "recall": metrics["recall"],
                "test_count": metrics["count"],
                "train_count": train_counts.get(species_lower, 0),
            }
        )

    df_analysis = pd.DataFrame(analysis_data)

    # Print overall statistics
    print("\nOverall Analysis:")
    print(f"Total number of species analyzed: {len(df_analysis)}")

    # Print per-category statistics
    for category in df_analysis["category"].unique():
        cat_data = df_analysis[df_analysis["category"] == category]
        print(f"\n{category} Analysis:")
        print(f"Number of species: {len(cat_data)}")
        print("\nTraining sample distribution:")
        print(f"Min samples: {cat_data['train_count'].min()}")
        print(f"Max samples: {cat_data['train_count'].max()}")
        print(f"Mean samples: {cat_data['train_count'].mean():.2f}")
        print(f"Median samples: {cat_data['train_count'].median()}")

    # Plotting
    import matplotlib.pyplot as plt
    import seaborn as sns

    # TEMPORARY!
    print("\nDetailed Training Sample Distribution Analysis")
    print("============================================")

    for category in sorted(df_analysis["category"].unique()):
        cat_data = df_analysis[df_analysis["category"] == category]
        print(f"\n{category}:")
        print("-" * (len(category) + 1))
        print(f"Number of species: {len(cat_data)}")
        print("\nTraining sample statistics:")
        print(f"Min: {cat_data['train_count'].min()}")
        print(f"Max: {cat_data['train_count'].max()}")
        print(f"Mean: {cat_data['train_count'].mean():.2f}")
        print(f"Median: {cat_data['train_count'].median():.2f}")

        # Distribution of training samples
        print("\nTraining sample distribution:")
        dist = cat_data["train_count"].value_counts().sort_index()
        print(dist)

        # Species with least training samples
        print("\nSpecies with fewest training samples:")
        bottom_5_samples = cat_data.nsmallest(5, "train_count")
        for _, row in bottom_5_samples.iterrows():
            print(
                f"{row['species']}: {row['train_count']} samples, Recall@1: {row['recall']:.3f}"
            )

        # Species with most training samples
        print("\nSpecies with most training samples:")
        top_5_samples = cat_data.nlargest(5, "train_count")
        for _, row in top_5_samples.iterrows():
            print(
                f"{row['species']}: {row['train_count']} samples, Recall@1: {row['recall']:.3f}"
            )

        # Performance analysis
        print("\nPerformance analysis:")
        print(f"Average recall: {cat_data['recall'].mean():.3f}")
        print(f"Median recall: {cat_data['recall'].median():.3f}")
        print(
            f"Correlation with training samples: {cat_data['train_count'].corr(cat_data['recall']):.3f}"
        )

        # Binned analysis
        bins = [0, 10, 25, 50, 100, float("inf")]
        labels = ["1-10", "11-25", "26-50", "51-100", "100+"]
        cat_data["sample_bin"] = pd.cut(
            cat_data["train_count"], bins=bins, labels=labels
        )

        print("\nPerformance by training sample size:")
        bin_stats = cat_data.groupby("sample_bin").agg(
            {"recall": ["count", "mean", "std"], "species": "count"}
        )
        print(bin_stats)

        print("\n" + "=" * 50)

    # 1. Scatter plots by category
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    for i, category in enumerate(sorted(df_analysis["category"].unique())):
        cat_data = df_analysis[df_analysis["category"] == category]
        sns.scatterplot(
            data=cat_data, x="train_count", y="recall", alpha=0.6, ax=axs[i]
        )
        sns.regplot(
            data=cat_data,
            x="train_count",
            y="recall",
            scatter=False,
            color="red",
            ax=axs[i],
        )

        # Set x-axis limits based on category data with a small buffer
        x_min = cat_data["train_count"].min()
        x_max = cat_data["train_count"].max()
        buffer = (x_max - x_min) * 0.05  # 5% buffer
        axs[i].set_xlim(max(0, x_min - buffer), x_max + buffer)

        axs[i].set_title(f"{category} - Recall vs Training Samples")
        axs[i].set_xlabel("Number of Training Samples")
        axs[i].set_ylabel("Recall@1")

        # Add correlation coefficient
        corr = cat_data["train_count"].corr(cat_data["recall"])
        axs[i].text(
            0.05,
            0.95,
            f"Correlation: {corr:.3f}",
            transform=axs[i].transAxes,
            bbox=dict(facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig("species_performance_by_category_scatter.png")
    plt.close()

    def make_bucket_labels(min_val, max_val):
        edges = [0, 5, 10, 20, 50, float("inf")]
        labels = ["1-5", "6-10", "11-20", "21-50", "50+"]
        # Adjust edges based on actual data
        valid_edges = [e for e in edges if e <= max_val]
        if valid_edges[-1] != float("inf"):
            valid_edges.append(float("inf"))
        valid_labels = labels[: len(valid_edges) - 1]
        return valid_edges, valid_labels

    # 2. Box plots by category
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))  # Create new figure for box plots

    for i, category in enumerate(sorted(df_analysis["category"].unique())):
        cat_data = df_analysis[df_analysis["category"] == category]

        # Create custom bins based on category's data
        edges, labels = make_bucket_labels(
            cat_data["train_count"].min(), cat_data["train_count"].max()
        )

        cat_data["bucket"] = pd.cut(
            cat_data["train_count"], bins=edges, labels=labels, duplicates="drop"
        )

        # Only plot non-empty buckets
        bucket_counts = cat_data["bucket"].value_counts()
        valid_buckets = bucket_counts[bucket_counts > 0].index
        plot_data = cat_data[cat_data["bucket"].isin(valid_buckets)]

        if len(plot_data) > 0:
            sns.boxplot(data=plot_data, x="bucket", y="recall", ax=axs[i])
            axs[i].set_title(f"{category} - Recall Distribution by Training Samples")
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=45)

            # Add sample counts
            for j, bucket in enumerate(valid_buckets):
                count = bucket_counts[bucket]
                axs[i].text(j, -0.05, f"n={count}", ha="center")

    plt.tight_layout()
    plt.savefig("species_performance_by_category_boxes.png")
    plt.close()

    # Save detailed statistics to file
    with open("species_performance_analysis.txt", "w") as f:
        f.write("Performance Analysis by Category\n")
        f.write("==============================\n\n")

        for category in sorted(df_analysis["category"].unique()):
            cat_data = df_analysis[df_analysis["category"] == category]

            f.write(f"\n{category} Analysis:\n")
            f.write(f"Number of species: {len(cat_data)}\n")
            f.write("\nTraining sample distribution:\n")
            f.write(f"Min samples: {cat_data['train_count'].min()}\n")
            f.write(f"Max samples: {cat_data['train_count'].max()}\n")
            f.write(f"Mean samples: {cat_data['train_count'].mean():.2f}\n")
            f.write(f"Median samples: {cat_data['train_count'].median()}\n")

            corr = cat_data["train_count"].corr(cat_data["recall"])
            f.write(f"\nCorrelation between training samples and recall: {corr:.4f}\n")

            f.write("\nTop 5 performing species:\n")
            top_5 = cat_data.nlargest(5, "recall")
            for _, row in top_5.iterrows():
                f.write(
                    f"  {row['species']}: {row['recall']:.3f} (train samples: {row['train_count']})\n"
                )

            f.write("\nBottom 5 performing species:\n")
            bottom_5 = cat_data.nsmallest(5, "recall")
            for _, row in bottom_5.iterrows():
                f.write(
                    f"  {row['species']}: {row['recall']:.3f} (train samples: {row['train_count']})\n"
                )

            f.write("\n" + "=" * 50 + "\n")

    return df_analysis


# Add to main():


def compare_base_to_trained_performance(csv_path, metrics_base, metrics_trained):
    # Read full dataset
    df = pd.read_csv(csv_path)

    # Get genus from scientific name
    def extract_genus(scientific_name):
        return scientific_name.lower().split()[0]

    # Function to create analysis dataframe
    def create_analysis_df(metrics):
        analysis_data = []
        for species, metrics in metrics["recall@1"]["per_species"].items():
            species_lower = species.lower()
            genus = extract_genus(species_lower)
            analysis_data.append(
                {
                    "species": species_lower,
                    "genus": genus,
                    "category": metrics["category"],
                    "recall": metrics["recall"],
                    "test_count": metrics["count"],
                }
            )
        return pd.DataFrame(analysis_data)

    # Create dataframes for both models
    df_base = create_analysis_df(metrics_base)
    df_trained = create_analysis_df(metrics_trained)

    # Merge the dataframes
    df_comparison = df_base.merge(
        df_trained, on=["species", "genus", "category"], suffixes=("_base", "_trained")
    )

    # Calculate improvements
    df_comparison["species_improvement"] = (
        df_comparison["recall_trained"] - df_comparison["recall_base"]
    )

    # Compute genus-level metrics
    genus_base = df_base.groupby(["category", "genus"])["recall"].mean().reset_index()
    genus_trained = (
        df_trained.groupby(["category", "genus"])["recall"].mean().reset_index()
    )

    genus_comparison = genus_base.merge(
        genus_trained, on=["category", "genus"], suffixes=("_base", "_trained")
    )
    genus_comparison["genus_improvement"] = (
        genus_comparison["recall_trained"] - genus_comparison["recall_base"]
    )

    # Print analysis
    print("\nComparison Analysis of Base vs Trained Model")
    print("==========================================")

    for category in sorted(df_comparison["category"].unique()):
        print(f"\n{category}:")
        print("-" * (len(category) + 1))

        # Species-level analysis
        cat_data = df_comparison[df_comparison["category"] == category]
        print("\nSpecies-level Analysis:")
        print(f"Number of species: {len(cat_data)}")
        print("\nRecall Statistics:")
        print(
            f"Base Model - Mean: {cat_data['recall_base'].mean():.3f}, Median: {cat_data['recall_base'].median():.3f}"
        )
        print(
            f"Trained Model - Mean: {cat_data['recall_trained'].mean():.3f}, Median: {cat_data['recall_trained'].median():.3f}"
        )
        print(f"Average Improvement: {cat_data['species_improvement'].mean():.3f}")

        # Species with most improvement
        print("\nTop 5 Most Improved Species:")
        top_5 = cat_data.nlargest(5, "species_improvement")
        for _, row in top_5.iterrows():
            print(
                f"{row['species']}: {row['species_improvement']:.3f} "
                f"(Base: {row['recall_base']:.3f} → Trained: {row['recall_trained']:.3f})"
            )

        # Species with least improvement
        print("\nTop 5 Least Improved Species:")
        bottom_5 = cat_data.nsmallest(5, "species_improvement")
        for _, row in bottom_5.iterrows():
            print(
                f"{row['species']}: {row['species_improvement']:.3f} "
                f"(Base: {row['recall_base']:.3f} → Trained: {row['recall_trained']:.3f})"
            )

        # Genus-level analysis
        cat_genus_data = genus_comparison[genus_comparison["category"] == category]
        print("\nGenus-level Analysis:")
        print(f"Number of genera: {len(cat_genus_data)}")
        print("\nRecall Statistics:")
        print(
            f"Base Model - Mean: {cat_genus_data['recall_base'].mean():.3f}, Median: {cat_genus_data['recall_base'].median():.3f}"
        )
        print(
            f"Trained Model - Mean: {cat_genus_data['recall_trained'].mean():.3f}, Median: {cat_genus_data['recall_trained'].median():.3f}"
        )
        print(f"Average Improvement: {cat_genus_data['genus_improvement'].mean():.3f}")

        # Genera with most improvement
        print("\nTop 5 Most Improved Genera:")
        top_5_genus = cat_genus_data.nlargest(5, "genus_improvement")
        for _, row in top_5_genus.iterrows():
            print(
                f"{row['genus']}: {row['genus_improvement']:.3f} "
                f"(Base: {row['recall_base']:.3f} → Trained: {row['recall_trained']:.3f})"
            )

        print("\n" + "=" * 50)

    # Create visualizations
    import matplotlib.pyplot as plt
    import seaborn as sns

    # 1. Species-level improvement distributions by category
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df_comparison, x="category", y="species_improvement")
    plt.title("Species-level Improvement Distribution by Category")
    plt.ylabel("Recall Improvement (Trained - Base)")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("species_improvement_distribution.png")
    plt.close()

    # 2. Genus-level improvement distributions by category
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=genus_comparison, x="category", y="genus_improvement")
    plt.title("Genus-level Improvement Distribution by Category")
    plt.ylabel("Recall Improvement (Trained - Base)")
    plt.axhline(y=0, color="r", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("genus_improvement_distribution.png")
    plt.close()

    # Save detailed statistics to file
    with open("model_comparison_analysis.txt", "w") as f:
        f.write("Model Comparison Analysis\n")
        f.write("=======================\n")

        for category in sorted(df_comparison["category"].unique()):
            # ... [Similar printing code as above, but writing to file] ...
            pass

    return df_comparison, genus_comparison


def main():
    # Paths
    csv_path = "../Image-Captioning/vlm4bio_captions_with_split.csv"
    base_path = "../Image-Captioning"
    checkpoint_path = "checkpoints/best_checkpoint.pt"

    # Load model and create dataset
    model, transform = load_trained_model(checkpoint_path)
    model = model.cuda()

    test_dataset = TestDataset(
        csv_path=csv_path, base_path=base_path, transform=transform
    )

    test_loader = DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)

    # Evaluate trained
    metrics_trained = evaluate_retrieval(model, test_loader)

    #   print_metrics(metrics)
    df_analysis = analyze_species_performance(csv_path, metrics_trained)

    # Evaluate Base
    base_model, transform = load_trained_model(None)
    base_model = base_model.cuda()

    metrics_base = evaluate_retrieval(base_model, test_loader)

    compare_base_to_trained_performance(csv_path, metrics_base, metrics_trained)


# Use after training:
# plot_training_log('logs/training_log_TIMESTAMP.csv')
