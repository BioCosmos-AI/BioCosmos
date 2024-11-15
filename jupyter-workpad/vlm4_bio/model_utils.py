#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, CLIPImageProcessor
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import open_clip
from huggingface_hub import hf_hub_download


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



def get_model(model_name):
    """
    Get a model and its transform function based on model name

    Parameters
    ----------
    model_name : str

    Returns
    -------
    model : Transformer model
    processing : transform function for the model

    """ 
    
    if model_name == 'florence':
        model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch_dtype, trust_remote_code=True)
        processing = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    elif model_name == 'bioclip':
        model, _, processing = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')
    elif model_name == 'arborclip':
        model_name = "ChihHsuan-Yang/ArborCLIP"
        filename = "arborclip-vit-b-16-from-bioclip-epoch-8.pt"

        # Download the file from the repository
        weights_path = hf_hub_download(repo_id=model_name, filename=filename)

        # Initialize the base BioCLIP model using OpenCLIP
        model, _, processing = open_clip.create_model_and_transforms('hf-hub:imageomics/bioclip')

        # Load the fine-tuned weights into the model
        checkpoint = torch.load(weights_path, map_location='cpu')

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    elif model_name == 'internvl':
        model = AutoModel.from_pretrained(
            'OpenGVLab/InternViT-300M-448px',
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True).cuda().eval()

        processing = CLIPImageProcessor.from_pretrained('OpenGVLab/InternViT-300M-448px')
    else: # else use openclip
        model, _, processing = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="laion2b_s34b_b79k", device=device)

    model.to(device)
    model = model.eval()
    return model, processing


def get_image_embeddings_florence(image_filename, model, processor):
    """
    Obtain image embeddings, given the image name and the model and transform function

    Parameters
    ----------
    image_filename : str
    model : florence transformer model
    processor : florence transform function

    Returns
    -------
    image embeddings as np array

    """
    # folder where images are stored
    image_folder = "downloaded_images"

    try:
        # Create the full image path by combining the folder and filename
        image_path = os.path.join(image_folder, image_filename)

        # Open the image
        image = Image.open(image_path)

        # Prepare inputs using the processor
        inputs = processor(images=image, return_tensors="pt").to(device)

        # Ensure inputs are converted to float16 if using mixed precision
        inputs = {k: v.to(dtype=torch.float16) if model.dtype == torch.float16 else v for k, v in inputs.items()}

        # Get image embeddings using the private method
        with torch.no_grad():  # Avoid computing gradients since we are not training
            image_embeddings = model._encode_image(inputs["pixel_values"])

        # Convert embeddings to CPU tensor (to avoid keeping them on GPU)
        return image_embeddings.cpu().numpy()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None  # Return None for failed image processing


def get_image_embeddings_clip(image_filename, model, preprocess):
    """
    Obtain image embeddings, given the image name and the model and transform function

    Parameters
    ----------
    image_filename : str
    model : clip transformer model
    processor : clip transform function

    Returns
    -------
    image embeddings as np array

    """
    # folder where images are stored
    image_folder = "downloaded_images"

    try:
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path)
        image = preprocess(image).unsqueeze(0).to(device)

        with torch.no_grad():
            image_embeddings = model.encode_image(image)

        return image_embeddings.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None  # Return None for failed image processing


def get_image_embeddings_intern(image_filename, model, preprocess):
    """
    Obtain image embeddings, given the image name and the model and transform function


    Parameters
    ----------
    image_filename : str
    model : internvl model
    processor : internvl transform function

    Returns
    -------
    image embeddings as np array

    """
    image_folder = "downloaded_images"

    try:
        image_path = os.path.join(image_folder, image_filename)
        image = Image.open(image_path).convert('RGB')
        pixel_values = preprocess(images=image, return_tensors='pt').pixel_values

        with torch.no_grad():
            pixel_values = pixel_values.to(torch.bfloat16).cuda()
            image_embeddings = model(pixel_values).pooler_output
        return image_embeddings.to(torch.float32).cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None  # Return None for failed image processing
    
def get_clip_text_embeddings(caption, model, process):
    """
    Get embeddings for a text caption using a CLIP model

    Parameters
    ----------
    caption : str
        DESCRIPTION.
    model : CLIP model
    process : CLIP transform function

    Returns
    -------
    text embeddings as np array

    """
#     text = process(caption)
    tokens = open_clip.tokenize([caption]).cuda()
    with torch.no_grad():
        text_embeddings = model.encode_text(tokens)

    return text_embeddings.cpu().numpy()

    
