#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import requests
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModel, CLIPImageProcessor
from PIL import Image
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
import open_clip
from huggingface_hub import hf_hub_download


device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32



def download_metadata_csv(metadata_url, metadata_file_path):
    """
    Download metadata csv from supplied url

    Parameters
    ----------
    metadata_url : str
    metadata_file_path : str

    Returns
    -------
    None.

    """
    if not os.path.exists(metadata_file_path):
        response = requests.get(metadata_url)
        if response.status_code == 200:
            with open(metadata_file_path, 'wb') as f:
                f.write(response.content)
            print(f"Metadata CSV file saved as {metadata_file_path}")
        else:
            print(f"Failed to download the metadata file. Status code: {response.status_code}")
            return
    else:
        print(f"Metadata file already exists at {metadata_file_path}")


def convert_metadata_to_dataframe(metadata_file_path):
    """
    Converts supplied metadata csv to a pandas df

    Parameters
    ----------
    metadata_file_path : str

    Returns
    -------
    df : pandas dataframe

    """
    df = pd.read_csv(metadata_file_path)
    df.rename(columns={"fileNameAsDelivered": "image_name", "scientificName": "scientific_name"}, inplace=True)
    return df


def download_images(df, chunk_base_url, chunk_count, output_dir):
    """
    Download images from VLM4Bio dataset on huggingface
    (Performs worse than git lfs)

    Parameters
    ----------
    df : dataframe
        pandas df with image names.
    chunk_base_url : str
    chunk_count : int
    output_dir : str

    Returns
    -------
    None.

    """
    os.makedirs(output_dir, exist_ok=True)
    for _, row in df.iterrows():
        image_filename = row['image_name']
        image_downloaded = False
        for chunk_index in range(chunk_count):
            image_url = f"{chunk_base_url}chunk_{chunk_index}/{image_filename}"
            response = requests.get(image_url)
            if response.status_code == 200:
                image_path = os.path.join(output_dir, image_filename)
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded and saved: {image_filename} from chunk_{chunk_index}")
                image_downloaded = True
                break
            else:
                print(f"Image not found in chunk_{chunk_index}: {image_filename}")
        if not image_downloaded:
            print(f"Failed to download {image_filename} from all chunks.")


def filter_existing_images_with_scientific_name(df, output_dir, taxa='Fish'):
    """
    Checks if image already exists in df

    Parameters
    ----------
    df : dataframe
    output_dir : str
    taxa : str, optional
        The default is 'Fish'.

    Returns
    -------
    df_cleaned : dataframe

    """
    df['file_exists'] = df['image_name'].apply(lambda x: os.path.exists(os.path.join(output_dir, x)))
    df_cleaned = df[df['file_exists']].copy()
    df_cleaned.drop(columns=['file_exists'], inplace=True)
    df_cleaned['category'] = taxa
    return df_cleaned


def save_dataframe_to_csv(df, csv_file_path="cleaned_images_with_scientific_names.csv"):
    """
    Save cleaned dataframe to csv file

    Parameters
    ----------
    df : dataframe
    csv_file_path : str, optional
        The default is "cleaned_images_with_scientific_names.csv".

    Returns
    -------
    None.

    """
    df.to_csv(csv_file_path, index=False)
    print(f"DataFrame saved to {csv_file_path}")
    

def make_dataframe_from_metadata_vlm4bio(dataset='metadata_10k', output_dir='downloaded_images', download=False):
    """
    Using metadata from VLM4Bio, generate a dataframe containing information about the images for the dataset

    Parameters
    ----------
    dataset : str, optional
        Specific dataset defined by metadata file in VLM4Bio. The default is 'metadata_10k'.
    output_dir : str, optional
        Directory into which images would be saved if download were true. The default is 'downloaded_images'.
    download : Boolean, optional
        Flag to check if images are to be downloaded from HuggingFace. The default is False.


    Returns
    -------
    full_species_df - Dataframe containing data loaded from metadata across VLM4Bio taxa - Fish, Bird, Butterfly

    """
    taxa = ['Fish', 'Bird', 'Butterfly']
    cleaned_dfs = []
    for t in taxa:
      metadata_url = f"https://huggingface.co/datasets/sammarfy/VLM4Bio/resolve/main/datasets/{t}/metadata/{dataset}.csv"
      

      metadata_file_path = f"{dataset}_{t}.csv"
      download_metadata_csv(metadata_url, metadata_file_path)
      df = convert_metadata_to_dataframe(metadata_file_path)
      df_cleaned = filter_existing_images_with_scientific_name(df, output_dir, t)
      cleaned_dfs.append(df_cleaned)
      save_dataframe_to_csv(df_cleaned, f"cleaned_images_with_scientific_names_{t}.csv")
      
      # if you want to download images from VLM4Bio using this function run this
      if download:
          chunk_base_url = f"https://huggingface.co/datasets/sammarfy/VLM4Bio/resolve/main/datasets/{t}/"
          chunk_count = 5  # We have chunk_0 to chunk_4
          download_images(df, chunk_base_url, chunk_count, output_dir)


    full_species_df = pd.concat(cleaned_dfs)
    return full_species_df
