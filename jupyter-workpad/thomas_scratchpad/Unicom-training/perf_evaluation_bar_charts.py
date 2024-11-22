import torch
import open_clip
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torch.nn.functional as F


def normalize_text(text):
    """Normalize text for comparison by removing whitespace and converting to lowercase"""
    return " ".join(text.lower().strip().split())


class TestDataset(Dataset):
    def __init__(self, csv_path, base_path="../Image-Captioning", transform=None):
        df = pd.read_csv(csv_path)
        self.df = df[df["split"] == "train"].reset_index(drop=True)
        self.df = self.df.dropna(subset=["scientific_name"])

        self.base_path = Path(base_path)
        self.transform = transform

        # Normalize scientific names and extract genus
        self.df["scientific_name"] = self.df["scientific_name"].apply(normalize_text)
        self.df["genus"] = self.df["scientific_name"].str.split().str[0]
        self.df = self.df.dropna(subset=["genus"])

        # Create label mappings with normalized names
        self.categories = sorted(self.df["category"].unique())
        self.species = sorted(self.df["scientific_name"].unique())
        #         print(self.df['genus'].unique())
        self.genera = sorted(self.df["genus"].unique())

        self.category_to_idx = {cat: idx for idx, cat in enumerate(self.categories)}
        self.species_to_idx = {sp: idx for idx, sp in enumerate(self.species)}
        self.genus_to_idx = {gen: idx for idx, gen in enumerate(self.genera)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.base_path / row["image_path"]

        try:
            image = Image.open(image_path)
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise

        return (
            image,
            self.category_to_idx[row["category"]],
            self.species_to_idx[row["scientific_name"]],
            self.genus_to_idx[row["genus"]],
            row["category"],
            row["scientific_name"],
            row["genus"],
        )


# TODO: Fix how this function is called, model name is misleading
def load_model(model_name, checkpoint_path=None):
    if checkpoint_path:
        # Load base ViT-B/32 model for checkpoints
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B/32")
        checkpoint = torch.load(checkpoint_path)

        state_dict = model.visual.state_dict()
        for k in state_dict.keys():
            state_dict[k] = checkpoint["model_state_dict"][f"model.{k}"]

        model.visual.load_state_dict(state_dict)
        model = model.visual
        model = model.float()
    else:
        # For pretrained models, use requested architecture
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
        model = model.visual
        model = model.float()

    return model, preprocess


@torch.no_grad()
def evaluate_model(model, test_loader):
    model.eval()
    model = model.cuda()

    all_embeddings = []
    all_categories = []
    all_species = []
    all_genera = []

    for batch in test_loader:
        images = batch[0].cuda()
        embeddings = model(images)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        all_embeddings.append(embeddings.cpu())
        all_categories.extend(batch[4])
        all_species.extend(batch[5])
        all_genera.extend(batch[6])

    all_embeddings = torch.cat(all_embeddings, dim=0)
    similarity = torch.mm(all_embeddings, all_embeddings.t())
    mask = torch.eye(similarity.shape[0], dtype=bool)
    similarity[mask] = -float("inf")

    _, indices = similarity.topk(1, dim=1)

    results = {cat: {"species": [], "genus": []} for cat in set(all_categories)}

    for idx, (category, species, genus) in enumerate(
        zip(all_categories, all_species, all_genera)
    ):
        retrieved_species = all_species[indices[idx][0]]
        retrieved_genus = all_genera[indices[idx][0]]

        results[category]["species"].append(species == retrieved_species)
        results[category]["genus"].append(genus == retrieved_genus)

    # Calculate recall for each category
    metrics = {}
    for category, data in results.items():
        metrics[category] = {
            "species_recall": sum(data["species"]) / len(data["species"]),
            "genus_recall": sum(data["genus"]) / len(data["genus"]),
        }

    return metrics


def plot_comparison(all_metrics, output_prefix="recall_train_only"):
    categories = list(next(iter(all_metrics.values())).keys())
    models = list(all_metrics.keys())

    # Plot species recall
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(categories))
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        species_recalls = [
            all_metrics[model][cat]["species_recall"] * 100 for cat in categories
        ]
        offset = (i - len(models) / 2) * width + width / 2
        ax.bar(x + offset, species_recalls, width, label=model)

    ax.set_ylabel("Species Recall@1 (%)")
    ax.set_title("Species Recall by Model and Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_species.png")
    plt.close()

    # Plot genus recall
    fig, ax = plt.subplots(figsize=(12, 6))

    for i, model in enumerate(models):
        genus_recalls = [
            all_metrics[model][cat]["genus_recall"] * 100 for cat in categories
        ]
        offset = (i - len(models) / 2) * width + width / 2
        ax.bar(x + offset, genus_recalls, width, label=model)

    ax.set_ylabel("Genus Recall@1 (%)")
    ax.set_title("Genus Recall by Model and Category")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_genus.png")
    plt.close()


def main():
    csv_path = "../Image-Captioning/vlm4bio_captions_with_split.csv"

    # Model configurations
    models = {
        "OpenCLIP": ("ViT-H-14-378-quickgelu", None),
        #         'BioCLIP': ('hf-hub:microsoft/bioclip-large-722k', None),
        "BioCLIP": ("hf-hub:imageomics/bioclip", None),
        "Cluster_100": (
            "ViT-H-14-378-quickgelu",
            "checkpoints_ViT-H-14-378-quickgelu_cluster_100/best_checkpoint.pt",
        ),
        "Cluster_500": (
            "ViT-H-14-378-quickgelu",
            "checkpoints_ViT-H-14-378-quickgelu_cluster_500/best_checkpoint.pt",
        ),
        "Cluster_1000": (
            "ViT-H-14-378-quickgelu",
            "checkpoints_ViT-H-14-378-quickgelu_cluster_1000/best_checkpoint.pt",
        ),
        "Cluster_2000": (
            "ViT-H-14-378-quickgelu",
            "checkpoints_ViT-H-14-378-quickgelu_cluster_2000/best_checkpoint.pt",
        ),
        "Cluster_3000": (
            "ViT-H-14-378-quickgelu",
            "checkpoints_ViT-H-14-378-quickgelu_cluster_3000/best_checkpoint.pt",
        ),
        #         'Original': ('ViT-H-14-378-quickgelu', 'checkpoints/best_checkpoint.pt')
    }

    all_metrics = {}

    for model_name, (architecture, checkpoint_path) in models.items():
        print(f"Evaluating {model_name}...")
        model, preprocess = load_model(architecture, checkpoint_path)

        test_dataset = TestDataset(csv_path=csv_path, transform=preprocess)
        test_loader = DataLoader(
            test_dataset, batch_size=32, num_workers=2, shuffle=False
        )

        metrics = evaluate_model(model, test_loader)
        all_metrics[model_name] = metrics

    plot_comparison(all_metrics)


if __name__ == "__main__":
    main()
