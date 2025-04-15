import argparse
import os
import sys
import json
from tqdm import tqdm
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pandas as pd
from skimage import measure

DEBUG = False


def load_florence2_model():
    model_id = "microsoft/Florence-2-large"
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_id, trust_remote_code=True, torch_dtype="auto"
        )
        .eval()
        .cuda()
    )
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    return model, processor


def run_florence2(model, processor, image, task_prompt, text_input=None):
    prompt = task_prompt if text_input is None else task_prompt + " " + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(
        "cuda", torch.float16
    )
    generated_ids = model.generate(
        input_ids=inputs["input_ids"].cuda(),
        pixel_values=inputs["pixel_values"].cuda(),
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
    return processor.post_process_generation(
        generated_text, task=task_prompt, image_size=(image.width, image.height)
    )


def debug_print(message):
    if DEBUG:
        print(message)


def calculate_trait_match_score(generated_text, target_trait):
    generated_text = generated_text.lower()
    target_trait = target_trait.lower()

    if target_trait in generated_text:
        return 1.0

    # Check for partial matches
    trait_words = target_trait.split()
    for word in trait_words:
        if word in generated_text:
            return 0.5

    return 0.0


def find_polygon_from_segmap(segmap, target_class):
    contours = measure.find_contours(segmap == target_class, 0.5)
    if contours:
        return [(int(y), int(x)) for x, y in contours[0]]
    return []


def polygon_to_bbox(polygon):
    x_coords, y_coords = zip(*polygon)
    return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]


def bbox_to_florence2_format(bbox, image_width, image_height):
    x1, y1, x2, y2 = bbox
    # Convert to [0, 999] range
    x1_norm = int(x1 * 999 / image_width)
    y1_norm = int(y1 * 999 / image_height)
    x2_norm = int(x2 * 999 / image_width)
    y2_norm = int(y2 * 999 / image_height)
    return f"<loc_{x1_norm}><loc_{y1_norm}><loc_{x2_norm}><loc_{y2_norm}>"


def save_annotated_image(image, bbox, target_trait, predicted_description, output_path):
    draw = ImageDraw.Draw(image)

    # Draw bounding box
    draw.rectangle(bbox, outline="red", width=2)

    # Prepare text
    text = f"Correct: {target_trait}\nPredicted: {predicted_description}"

    # Use a default font
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    # Calculate text position (above the bounding box)
    text_x = bbox[0]
    text_y = max(0, bbox[1] - 60)  # 60 pixels above, or at top if not enough space

    # Draw text background
    text_bbox = draw.multiline_textbbox((text_x, text_y), text, font=font)
    draw.rectangle(text_bbox, fill="white")

    # Draw text
    draw.multiline_text((text_x, text_y), text, fill="black", font=font)

    # Save the image
    image.save(output_path)


def main(args):
    model, processor = load_florence2_model()

    # Load the trait map
    with open(args.trait_map_path, "r") as f:
        id_trait_map = json.load(f)

    train_df = pd.read_csv(args.dataset_csv)

    # Create a subdirectory for this specific run
    run_dir = os.path.join(args.result_dir, f"{args.trait_option}_n{args.num_queries}")
    os.makedirs(run_dir, exist_ok=True)

    # Create images subdirectory if visual output is enabled
    if args.visual_output:
        images_dir = os.path.join(run_dir, "images")
        os.makedirs(images_dir, exist_ok=True)

    out_file_name = os.path.join(
        run_dir, f"referring_florence2_{args.trait_option}_num_{args.num_queries}.jsonl"
    )

    total_score = 0
    total_count = 0
    skipped_rows = 0
    processed_rows = 0

    with open(out_file_name, "w") as writer:
        for idx in tqdm(range(len(train_df))):
            if processed_rows >= args.num_queries:
                break

            img_filename = train_df.iloc[idx].filename
            img_mask_filename = os.path.splitext(img_filename)[0] + ".png"

            # Check if both image and mask files exist
            image_path = os.path.join(args.image_dir, img_filename)
            seg_mask_path = os.path.join(args.segmentation_dir, img_mask_filename)

            if not os.path.exists(image_path) or not os.path.exists(seg_mask_path):
                skipped_rows += 1
                continue

            try:
                # Load image
                image = Image.open(image_path)

                # Load segmentation mask
                seg_mask = np.array(Image.open(seg_mask_path))
            except (IOError, SyntaxError) as e:
                print(f"Error loading image or mask for {img_filename}: {e}")
                skipped_rows += 1
                continue

            # Find present traits
            present_traits = [
                id_trait_map[str(trait_id)]
                for trait_id in np.unique(seg_mask)
                if str(trait_id) in id_trait_map
            ]

            if args.trait_option not in present_traits:
                skipped_rows += 1
                continue

            # Get polygon for the target trait
            target_class = next(
                int(k) for k, v in id_trait_map.items() if v == args.trait_option
            )
            polygon = find_polygon_from_segmap(seg_mask, target_class)

            if not polygon:
                skipped_rows += 1
                continue

            # Convert polygon to bounding box
            bbox = polygon_to_bbox(polygon)

            # Convert bbox to Florence-2 format
            florence2_bbox = bbox_to_florence2_format(bbox, image.width, image.height)

            task_prompt = "<REGION_TO_DESCRIPTION>"
            florence_output = run_florence2(
                model, processor, image, task_prompt, text_input=florence2_bbox
            )

            result = {
                "image-path": image_path,
                "target-trait": args.trait_option,
                "florence-output": florence_output,
            }

            generated_text = florence_output["<REGION_TO_DESCRIPTION>"]
            match_score = calculate_trait_match_score(generated_text, args.trait_option)

            result["match_score"] = match_score
            total_score += match_score
            total_count += 1

            json.dump(result, writer)
            writer.write("\n")

            # Save annotated image if visual output is enabled
            if args.visual_output:
                output_image_path = os.path.join(
                    images_dir, f"{os.path.splitext(img_filename)[0]}_annotated.jpg"
                )
                save_annotated_image(
                    image.copy(),
                    bbox,
                    args.trait_option,
                    generated_text,
                    output_image_path,
                )

            processed_rows += 1

    if total_count > 0:
        average_score = total_score / total_count
        print(f"Average match score for {args.trait_option}: {average_score}")

    print(f"Total rows processed: {processed_rows}")
    print(f"Total rows skipped: {skipped_rows}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", "-m", type=str, default="florence-2", help="multimodal-model"
    )
    parser.add_argument(
        "--task_option", "-t", type=str, default="referring", choices=["referring"]
    )
    parser.add_argument(
        "--trait_option",
        "-r",
        type=str,
        default="Head",
        choices=[
            "Head",
            "Eye",
            "Dorsal fin",
            "Pectoral fin",
            "Pelvic fin",
            "Anal fin",
            "Caudal fin",
            "Adipose fin",
            "Barbel",
        ],
    )
    parser.add_argument(
        "--result_dir",
        "-o",
        type=str,
        default="results/referring",
        help="path to output",
    )
    parser.add_argument(
        "--num_queries",
        "-n",
        type=int,
        default=5,
        help="number of images to query from dataset",
    )
    parser.add_argument(
        "--image_dir", type=str, required=True, help="path to image directory"
    )
    parser.add_argument(
        "--segmentation_dir",
        type=str,
        required=True,
        help="path to segmentation masks directory",
    )
    parser.add_argument(
        "--trait_map_path",
        type=str,
        required=True,
        help="path to seg_id_trait_map.json file",
    )
    parser.add_argument(
        "--dataset_csv", type=str, required=True, help="path to dataset CSV file"
    )
    parser.add_argument(
        "--visual_output",
        action="store_true",
        help="Enable visual output of annotated images",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")

    args = parser.parse_args()

    DEBUG = args.debug

    main(args)
