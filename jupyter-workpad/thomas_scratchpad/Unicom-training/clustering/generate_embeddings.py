import os
import torch
import open_clip
import pandas as pd
from PIL import Image
from pathlib import Path
import numpy as np
from scipy.io import savemat
from torch.utils.data import Dataset, DataLoader

MODEL_NAME = "ViT-H-14-378-quickgelu"


class VLM4BioDataset(Dataset):
    def __init__(self, csv_path, base_path, preprocess, cohort):
        df = pd.read_csv(csv_path)
        self.df = df[
            df["split"] == cohort
        ]  # IMPORTANT! Only train on rows labeled "train" or "test" depending on cohort
        self.base_path = Path(base_path)
        self.preprocess = preprocess

        # Check for missing images and report/filter
        print("Checking for missing images...")
        missing_images = []
        for idx, row in self.df.iterrows():
            img_path = self.base_path / row["image_path"]
            if not img_path.exists():
                missing_images.append((idx, str(img_path)))

        if missing_images:
            print(f"Found {len(missing_images)} missing images")
            print("First few missing images:")
            for idx, path in missing_images[:5]:
                print(f"Row {idx}: {path}")

            # Filter out rows with missing images
            valid_indices = [
                idx
                for idx, row in self.df.iterrows()
                if (self.base_path / row["image_path"]).exists()
            ]
            self.df = self.df.loc[valid_indices]
            print(f"Filtered dataset from {len(df)} to {len(self.df)} rows")
        else:
            print("All images found successfully!")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        # Convert to absolute path
        img_path = Path(self.base_path / row["image_path"]).resolve()

        try:
            image = self.preprocess(Image.open(img_path))
            text = self.tokenizer([row["caption"]])[0]
            #             print("Image Path", str(img_path))
            return {
                "image": image,
                "text": text,
                "image_path": str(img_path),
                "scientific_name": row["scientific_name"],
                "category": row["category"],
            }
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            return None


def process_and_save_embeddings(
    csv_path, base_path, batch_size=32, save_every=1000, cohort="train"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    model = model.to(device)
    model.eval()

    # Add tokenizer to dataset class
    VLM4BioDataset.tokenizer = tokenizer

    dataset = VLM4BioDataset(csv_path, base_path, preprocess, cohort)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    image_embeddings = []
    text_embeddings = []
    metadata = []
    batch_count = 0
    file_count = 0

    save_dir = f"embeddings_{cohort}_{MODEL_NAME}"
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if batch is None:
                continue

            images = batch["image"].to(device)
            texts = batch["text"].to(device)

            image_emb = model.encode_image(images)
            text_emb = model.encode_text(texts)

            image_emb /= image_emb.norm(dim=-1, keepdim=True)
            text_emb /= text_emb.norm(dim=-1, keepdim=True)

            image_embeddings.extend(image_emb.cpu().numpy())
            text_embeddings.extend(text_emb.cpu().numpy())
            metadata.extend(
                [
                    {"image_path": p, "scientific_name": sn, "category": c}
                    for p, sn, c in zip(
                        batch["image_path"], batch["scientific_name"], batch["category"]
                    )
                ]
            )

            batch_count += 1

            if batch_count * batch_size >= save_every:
                save_batch_to_mat(
                    image_embeddings,
                    text_embeddings,
                    metadata,
                    file_count,
                    cohort,
                    MODEL_NAME,
                )

                image_embeddings = []
                text_embeddings = []
                metadata = []
                batch_count = 0
                file_count += 1

        # Save any remaining
        if image_embeddings:
            save_batch_to_mat(
                image_embeddings,
                text_embeddings,
                metadata,
                file_count,
                cohort,
                MODEL_NAME,
            )


def save_batch_to_mat(
    image_embeddings, text_embeddings, metadata, file_count, cohort, model_name
):
    save_dict = {
        "image_embeddings": np.array(image_embeddings).astype("float32"),
        "text_embeddings": np.array(text_embeddings).astype("float32"),
        "image_paths": [m["image_path"] for m in metadata],
        "scientific_names": [m["scientific_name"] for m in metadata],
        "categories": [m["category"] for m in metadata],
    }

    save_dir = f"./embeddings_{cohort}_{model_name}"
    savemat(f"{save_dir}/embeddings_batch_{file_count}.mat", save_dict)
    print(f"Saved batch {file_count}")


if __name__ == "__main__":
    process_and_save_embeddings(
        csv_path="../../Image-Captioning/vlm4bio_captions_with_split.csv",
        base_path="../../Image-Captioning",
        batch_size=32,
        save_every=1000,
    )
