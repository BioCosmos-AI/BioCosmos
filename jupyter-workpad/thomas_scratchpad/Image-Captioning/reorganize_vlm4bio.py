import os
import shutil
import glob
import pandas as pd
from pathlib import Path


def reorganize_vlm4bio_dataset(base_dir="data/VLM4Bio/datasets"):
    """
    Reorganize the VLM4Bio dataset into a cleaner structure with consolidated
    images and metadata directories for each category.
    """
    # Get all category directories (Bird, Fish, Butterfly)
    categories = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    for category in categories:
        category_path = os.path.join(base_dir, category)

        # Create new directory structure
        new_images_dir = os.path.join(category_path, "images")
        new_metadata_dir = os.path.join(category_path, "metadata")

        os.makedirs(new_images_dir, exist_ok=True)
        os.makedirs(new_metadata_dir, exist_ok=True)

        # Process all chunk directories
        chunk_dirs = glob.glob(os.path.join(category_path, "chunk_*"))

        for chunk_dir in chunk_dirs:
            # Move images
            image_files = []
            for ext in ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]:
                image_files.extend(glob.glob(os.path.join(chunk_dir, ext)))
                image_files.extend(glob.glob(os.path.join(chunk_dir, "**", ext)))

            for image_file in image_files:
                if os.path.isfile(image_file):
                    filename = os.path.basename(image_file)
                    dest_path = os.path.join(new_images_dir, filename)

                    # Handle duplicate filenames by adding a suffix
                    base, ext = os.path.splitext(filename)
                    counter = 1
                    while os.path.exists(dest_path):
                        dest_path = os.path.join(
                            new_images_dir, f"{base}_{counter}{ext}"
                        )
                        counter += 1

                    shutil.copy2(image_file, dest_path)

        # Clean up old chunk directories (optional)
        # Uncomment the following lines if you want to remove the old structure
        # for chunk_dir in chunk_dirs:
        #     shutil.rmtree(chunk_dir)

    print("Dataset reorganization completed!")


def verify_reorganization(base_dir="data/VLM4Bio/datasets"):
    """
    Verify the reorganization by printing statistics about the new structure.
    """
    categories = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]

    for category in categories:
        category_path = os.path.join(base_dir, category)
        images_dir = os.path.join(category_path, "images")
        #         metadata_dir = os.path.join(category_path, "metadata")

        num_images = len(glob.glob(os.path.join(images_dir, "*")))
        #         num_metadata = len(glob.glob(os.path.join(metadata_dir, "*.csv")))

        print(f"\n{category}:")
        print(f"  - Number of images: {num_images}")


#         print(f"  - Number of metadata files: {num_metadata}")

if __name__ == "__main__":
    # Run the reorganization
    reorganize_vlm4bio_dataset()

    # Verify the results
    verify_reorganization()
