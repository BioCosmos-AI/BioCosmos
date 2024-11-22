import os
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AdamW,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model
from PIL import Image
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import cv2
from skimage import measure
from torchvision.transforms import Resize
import traceback


class FishVistaDataset(Dataset):
    def __init__(self, csv_file, image_dir, seg_dir, trait_map_path, processor):
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.seg_dir = seg_dir
        self.processor = processor
        #         self.resize = Resize((512, 512))  # Resize

        with open(trait_map_path, "r") as f:
            self.trait_map = json.load(f)

        # Filter out rows with missing image or segmentation files
        valid_rows = []
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.image_dir, row["filename"])
            seg_path = os.path.join(
                self.seg_dir, os.path.splitext(row["filename"])[0] + ".png"
            )
            if os.path.exists(img_path) and os.path.exists(seg_path):
                valid_rows.append(idx)
            else:
                print(f"Skipping row {idx}: Image or segmentation file not found.")

        self.data = self.data.iloc[valid_rows].reset_index(drop=True)
        print(f"Dataset contains {len(self.data)} valid images.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_name = self.data.iloc[idx]["filename"]
            img_path = os.path.join(self.image_dir, img_name)
            seg_name = os.path.splitext(img_name)[0] + ".png"
            seg_path = os.path.join(self.seg_dir, seg_name)

            image = Image.open(img_path)
            image_np = np.array(image)
            seg_mask = np.array(Image.open(seg_path))

            # Get all traits present in the image
            traits = [
                self.trait_map[str(i)]
                for i in np.unique(seg_mask)
                if str(i) in self.trait_map and i != 0
            ]

            results = []
            for trait in traits:
                # Create a prompt for the REFERRING_EXPRESSION_SEGMENTATION task
                # prompt = f"<REFERRING_EXPRESSION_SEGMENTATION> a fish {trait}"
                prompt = f"<BIO_TRAIT_SEG> a fish {trait}"

                # Create polygon for the trait
                trait_id = [k for k, v in self.trait_map.items() if v == trait][0]
                trait_mask = (seg_mask == int(trait_id)).astype(np.uint8)
                contours, _ = cv2.findContours(
                    trait_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )

                polygons = []
                for contour in contours:
                    if len(contour) >= 2:
                        epsilon = 0.01 * cv2.arcLength(contour, True)
                        approx = cv2.approxPolyDP(contour, epsilon, True)

                        if len(approx) >= 2:
                            h, w = seg_mask.shape[:2]
                            polygon = approx.reshape(-1).tolist()
                            polygon = [
                                (
                                    max(0, min(999, int(p * 999 / h)))
                                    if i % 2 == 0
                                    else max(0, min(999, int(p * 999 / w)))
                                )
                                for i, p in enumerate(polygon)
                            ]

                            #                             poly_str = f"{trait}"
                            poly_str = ""
                            for i in range(0, len(polygon), 2):
                                poly_str += f"<loc_{polygon[i+1]}><loc_{polygon[i]}>"
                            polygons.append(poly_str)

                target = " ".join(polygons)

                try:
                    inputs = self.processor(
                        text=prompt, images=image_np, return_tensors="pt", padding=True
                    )
                    inputs = {
                        k: v.squeeze(0) for k, v in inputs.items()
                    }  # Remove batch dimension
                    results.append((inputs, target, image))
                except Exception as e:
                    print(
                        f"Error processing trait {trait} for image {img_path}: {str(e)}"
                    )

            return results

        except Exception as e:
            print(f"Error processing item {idx}: {str(e)}")
            print(f"Image path: {img_path}")
            print(f"Segmentation mask path: {seg_path}")
            return None


def collate_fn(batch, processor):
    # Flatten the batch of results
    flattened_batch = [
        item for sublist in batch if sublist is not None for item in sublist
    ]

    if len(flattened_batch) == 0:
        return None

    input_ids = pad_sequence(
        [item[0]["input_ids"] for item in flattened_batch],
        batch_first=True,
        padding_value=0,
    )
    attention_mask = pad_sequence(
        [item[0]["attention_mask"] for item in flattened_batch],
        batch_first=True,
        padding_value=0,
    )
    pixel_values = torch.stack([item[0]["pixel_values"] for item in flattened_batch])

    targets = [item[1] for item in flattened_batch]
    images = [item[2] for item in flattened_batch]
    target_encoding = processor(
        text=targets, images=images, return_tensors="pt", padding=True
    )
    target_ids = target_encoding.input_ids

    print("Sample target:", targets[0])
    print("Sample target_ids:", target_ids[0][:10])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "pixel_values": pixel_values,
        "labels": target_ids,
    }


def train(
    model, train_dataloader, test_dataloader, optimizer, scheduler, device, num_epochs
):
    model.train()
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter {name} requires gradient")

    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(
            tqdm(train_dataloader, desc=f"Epoch {epoch + 1}")
        ):
            if batch is None:
                continue  # Skip this batch if it's None

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                labels=labels,
            )
            loss = outputs.loss

            print(f"Batch {batch_idx}, Loss: {loss.item()}")
            print(f"Input shape: {input_ids.shape}, Label shape: {labels.shape}")
            print(f"Sample input: {input_ids[0][:10]}")
            print(f"Sample label: {labels[0][:10]}")

            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(
                        f"Parameter {name} grad norm: {param.grad.norm().item() if param.grad is not None else 'None'}"
                    )

            scheduler.step()
            optimizer.zero_grad()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item()}")

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss}")

        # Evaluation loop
        model.eval()
        total_eval_loss = 0
        with torch.no_grad():
            for batch in tqdm(
                test_dataloader, desc=f"Evaluation after Epoch {epoch + 1}"
            ):
                if batch is None:
                    continue  # Skip this batch if it's None

                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                )
                loss = outputs.loss
                total_eval_loss += loss.item()

        avg_eval_loss = total_eval_loss / len(test_dataloader)
        print(f"Epoch {epoch + 1}, Average Evaluation Loss: {avg_eval_loss}")

    # Return the final training and evaluation loss
    return avg_loss, avg_eval_loss


def main():
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Florence-2 model and processor
    model_id = "microsoft/Florence-2-base"
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(
        device
    )

    print("Processor version:", processor.__class__.__name__)
    #     print("Tokenizer config:", processor.tokenizer.get_config())

    # Set up LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=[
            "q_proj",
            "o_proj",
            "k_proj",
            "v_proj",
            "linear",
            "Conv2d",
            "lm_head",
            "fc2",
        ],
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        bias="none",
        inference_mode=False,
        use_rslora=True,
        init_lora_weights="gaussian",
        revision="refs/pr/6",
    )
    model = get_peft_model(model, lora_config)

    # Set up dataset and dataloader
    train_dataset = FishVistaDataset(
        csv_file="./fish-vista/segmentation_train.csv",
        image_dir="./fish-vista/AllImages",
        seg_dir="./fish-vista/segmentation_masks/images",
        trait_map_path="./fish-vista/segmentation_masks/seg_id_trait_map.json",
        processor=processor,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    test_dataset = FishVistaDataset(
        csv_file="./fish-vista/segmentation_test.csv",
        image_dir="./fish-vista/AllImages",
        seg_dir="./fish-vista/segmentation_masks/images",
        trait_map_path="./fish-vista/segmentation_masks/seg_id_trait_map.json",
        processor=processor,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda batch: collate_fn(batch, processor),
    )

    avg_traits_per_image = 7  # estimate given the range of 1-9 traits

    # Calculate estimated number of training steps
    num_images = len(train_dataset)
    batch_size = 4
    num_epochs = 5
    estimated_steps_per_epoch = (num_images * avg_traits_per_image) // batch_size
    num_training_steps = num_epochs * estimated_steps_per_epoch

    # Set up optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    #     # Set up optimizer and scheduler
    #     optimizer = AdamW(model.parameters(), lr=5e-5)
    #     num_epochs = 2
    #     num_training_steps = num_epochs * len(train_dataloader)
    #     scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Train the model
    train(
        model,
        train_dataloader,
        test_dataloader,
        optimizer,
        scheduler,
        device,
        num_epochs,
    )

    # Save the fine-tuned model
    model.save_pretrained("./fine_tuned_florence2")
    processor.save_pretrained("./fine_tuned_florence2")


if __name__ == "__main__":
    main()
