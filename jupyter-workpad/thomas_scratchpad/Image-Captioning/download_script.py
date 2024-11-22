import os
from huggingface_hub import hf_hub_download, snapshot_download
from transformers import AutoTokenizer, AutoModel
import torch


# First, let's download and set up the model using Python instead of CLI
def download_internvl2_model(
    model_name="OpenGVLab/InternVL2-2B", cache_dir="pretrained"
):
    print(f"Downloading {model_name} model...")

    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)

    # Download model and tokenizer using transformers
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        cache_dir=cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, cache_dir=cache_dir
    )

    return model, tokenizer


# Download VLM4Bio dataset
def download_vlm4bio_dataset(base_dir="data"):
    """
    Download the VLM4Bio dataset from Hugging Face using Python API
    """
    print("Downloading VLM4Bio dataset...")

    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)

    try:
        # Download dataset
        dataset_path = snapshot_download(
            repo_id="sammarfy/VLM4Bio",
            repo_type="dataset",
            local_dir=os.path.join(base_dir, "VLM4Bio"),
            ignore_patterns=["*.git*", "*.md", "LICENSE"],
        )
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        raise


# Example usage in Jupyter notebook:
if __name__ == "__main__":
    # Set up paths
    base_dir = "data"

    # Download model and dataset
    print("Downloading required files...")

    # Download model
    model, tokenizer = download_internvl2_model()

    # Download dataset
    if not os.path.exists(os.path.join(base_dir, "VLM4Bio")):
        data_dir = download_vlm4bio_dataset(base_dir)
    else:
        data_dir = os.path.join(base_dir, "VLM4Bio")
        print(f"Using existing dataset at: {data_dir}")
