import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("caption_generation.log"), logging.StreamHandler()],
)


class CaptionGenerator:
    def __init__(self, model_name="OpenGVLab/InternVL2-2B", device="cuda"):
        self.device = device
        self.model = (
            AutoModel.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=True,
                trust_remote_code=True,
            )
            .eval()
            .to(device)
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        # Set up image transformation pipeline
        self.transform = self.build_transform(input_size=448)

    def build_transform(self, input_size):
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        return T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize(
                    (input_size, input_size), interpolation=InterpolationMode.BICUBIC
                ),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def find_closest_aspect_ratio(
        self, aspect_ratio, target_ratios, width, height, image_size
    ):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif (
                ratio_diff == best_ratio_diff
                and area > 0.5 * image_size * image_size * ratio[0] * ratio[1]
            ):
                best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(
        self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True
    ):
        # This was a bit of guessing and testing bssed on the internvl documentation
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = set(
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []

        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            split_img = resized_img.crop(box)
            processed_images.append(split_img)

        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)

        return processed_images

    def process_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        processed_images = self.dynamic_preprocess(image)
        pixel_values = torch.stack([self.transform(img) for img in processed_images])
        return pixel_values.to(torch.bfloat16).to(self.device)

    def generate_caption(self, image_path, scientific_name=None):
        pixel_values = self.process_image(image_path)

        if scientific_name:
            prompt = f"<image>\nThis is an image of {scientific_name}. Please provide a detailed one-sentence caption describing the visual appearance of this specimen."
        else:
            prompt = "<image>\nPlease provide a detailed one-sentence caption describing what you see in this image."

        generation_config = dict(max_new_tokens=100, do_sample=False)
        response = self.model.chat(
            self.tokenizer, pixel_values, prompt, generation_config
        )
        return response


def load_metadata(category_dir):
    """Load and validate metadata from the CSV file."""
    metadata_file = os.path.join(category_dir, "metadata", "metadata_10k.csv")
    try:
        df = pd.read_csv(metadata_file)
        return {
            row["fileNameAsDelivered"]: row["scientificName"]
            for _, row in df.iterrows()
            if pd.notna(row["fileNameAsDelivered"])
        }
    except Exception as e:
        logging.error(f"Error loading metadata from {metadata_file}: {str(e)}")
        return {}


def process_vlm4bio_dataset(
    data_dir, output_path, subset=["Bird", "Fish", "Butterfly"]
):
    generator = CaptionGenerator()
    results = []
    error_log = []

    for category in subset:
        category_dir = os.path.join(data_dir, "datasets", category)
        images_dir = os.path.join(category_dir, "images")

        if not os.path.exists(images_dir):
            logging.error(f"Images directory not found: {images_dir}")
            continue

        # Load metadata
        metadata_dict = load_metadata(category_dir)
        if not metadata_dict:
            logging.warning(f"No metadata found for category: {category}")

        # Get list of actual images
        actual_images = set(
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )  # Being flex.  I saw .JPG and .jpg in the dataset.

        # Some stats
        referenced_images = set(metadata_dict.keys())
        missing_images = referenced_images - actual_images
        unlisted_images = actual_images - referenced_images

        logging.info(f"\nCategory: {category}")
        logging.info(f"Total images in directory: {len(actual_images)}")
        logging.info(f"Total images in metadata: {len(referenced_images)}")
        logging.info(
            f"Missing images (in metadata but not in directory): {len(missing_images)}"
        )
        logging.info(
            f"Unlisted images (in directory but not in metadata): {len(unlisted_images)}"
        )

        # Process each existing image
        for img_name in tqdm(actual_images, desc=f"Processing {category}"):
            image_path = os.path.join(images_dir, img_name)
            scientific_name = metadata_dict.get(img_name)

            try:
                caption = generator.generate_caption(image_path, scientific_name)

                results.append(
                    {
                        "category": category,
                        "image_name": img_name,
                        "scientific_name": scientific_name,
                        "caption": caption,
                        "image_path": image_path,
                        "has_metadata": img_name in metadata_dict,
                    }
                )

            except Exception as e:
                error_msg = f"Error processing {img_name}: {str(e)}"
                logging.error(error_msg)
                error_log.append(
                    {"category": category, "image_name": img_name, "error": str(e)}
                )

            # Save progress periodically
            if len(results) % 100 == 0:
                save_results(results, error_log, output_path)

    # Save final results
    save_results(results, error_log, output_path)
    logging.info(f"Processing completed. Results saved to {output_path}")


def save_results(results, error_log, output_path):
    """Save results and error log to files."""
    # Save main results
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)

    # Save error log
    error_df = pd.DataFrame(error_log)
    error_path = output_path.replace(".csv", "_errors.csv")
    error_df.to_csv(error_path, index=False)


if __name__ == "__main__":
    # I set this up in another script . .
    # Code is coupled currently to the local structure of how I organized VLM4Bio
    data_dir = "data/VLM4Bio"
    output_path = "vlm4bio_captions.csv"

    process_vlm4bio_dataset(data_dir, output_path)
