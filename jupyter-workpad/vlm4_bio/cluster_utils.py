#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import requests
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from PIL import Image
import torch
import numpy as np
from scipy.io import loadmat, savemat
import faiss
import clip

np.random.seed(5)

"""
THE ASSUMPTION GOING FORWARD FOR THE DATAFRAMES AND .mat FILES IS THAT THEY CONTAIN THE FOLLOWING COLUMNS:
    
image_name, scientific_name, category, caption, image_embeddings, text_embeddings

"""

def load_df(mat_filename):
    """
    Load dataframe for UNICOM clustering from .mat file
    
    Parameters
    ----------
    mat_filename : str
        DESCRIPTION.

    Returns
    -------
    df : Dataframe
        Dataframe of the df file including image_name, scientific_name, category, caption, image/text_embeddings fields.
    """
    
    data = loadmat(f"{mat_filename}.mat")
    img_embeddings = data['image_embeddings']
    text_embeddings = data['text_embeddings']
        
    # Create a DataFrame with each list as a column
    df = pd.DataFrame({
        'image_name': data['image_name'],
        'scientific_name': data['scientific_name'],
        'category': data['category'],
        'caption': data['caption'],
        'image_embeddings': [sub_array for sub_array in img_embeddings],
        'text_embeddings': [sub_array for sub_array in text_embeddings]
    })
    
    return df



def save_df(mat_filename, df):
    """
    Save dataframe for UNICOM clustering as .mat file
    
    Parameters
    ----------
    mat_filename : str
        Filename being used.
    df : str
        DESCRIPTION.
    Returns
    -------

    """

    # Convert the embeddings to a numpy array
    np_img_embeds_list = np.array(df['image_embeddings'].tolist())
    np_text_embeds_list = np.array(df['text_embeddings'].tolist())

    # Ensure embeddings are of the correct type (float32)
    img_embeddings = np_img_embeds_list.astype('float32')
    text_embeddings = np_text_embeds_list.astype('float32')
    
    save_vars = {
        'scientific_name': df['scientific_name'].tolist(),
        'category': df['category'].tolist(),
        'image_name': df['image_name'].tolist(),
        'caption': df['caption'],
        'image_embeddings': img_embeddings,
        'text_embeddings': text_embeddings
    }
    
    # Save as .mat file
    savemat(f'{mat_filename}.mat', save_vars)
    


def set_clusters_on_df(df, cluster_column, ncentroids=3000, niter=20):
    """
    Using Faiss, generate clusters and set their ids for each row in df.

    Parameters
    ----------
    df : Dataframe
        pandas df used to cluster.
    cluster_column : str
        Column on which to cluster.
    ncentroids : int, optional
        number of cluster centers. The default is 3000.
    niter : int, optional
        Number of iterations to generate clusters. The default is 20.

    Returns
    -------
    df : Dataframe
        pandas df to return after attaching cluster centers.

    """
    
    # get embeddings from df
    np_embeds_list = np.array(df[cluster_column].tolist())

    embeddings = np.squeeze(np_embeds_list, axis=1).astype('float32')

    # number of clusters - cannot exceed number of samples
    verbose = True

    # num samples, and dimensionality of embeds
    n, d = embeddings.shape

    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
    kmeans.train(embeddings)

    centroids = kmeans.centroids
    D, I = kmeans.index.search(embeddings, 1)
    df['cluster_id'] = I.flatten()
    df['square_dist'] = D.flatten()
    return df

def get_embedding_size(df, col):
    return df[col].iloc[0].shape[1]

def set_embed_avg_col(df):
    """
    Add a new column with the average of image and text embeddings

    Parameters
    ----------
    df : Dataframe
        Dataframe with embeddings.

    Returns
    -------
    df : Dataframe
        Dataframe with new avg_embedding column.

    """
    df['avg_embedding'] = (df['image_embeddings'] + df['text_embeddings']) / 2
    return df


def set_embed_concat_col(df):
    """
    Add a new column with the concatenation of image and text embeddings

    Parameters
    ----------
    df : Dataframe
        Dataframe with embeddings.

    Returns
    -------
    df : Dataframe
        Dataframe with new concatenated_embeds column.

    """
    df['concatenated_embeds'] = None

    for index, row in df.iterrows():
        img, text = row['image_embeddings'], row['text_embeddings']
        row['concatenated_embeds'] =  np.concatenate((img, text), axis=1)
        df.at[index, 'concatenated_embeds'] = row['concatenated_embeds']
    return df
    


class VLM4BioEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, df, transform=None, visual=False):
        self.img_dir = 'downloaded_images/'
        self.df = df
        self.transform = transform
        if 'cluster_id' in df.columns:
            self.num_classes = self.df['cluster_id'].nunique()
        self.visual = visual

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image and apply transformations
        row = self.df.iloc[idx]
        image_path = os.path.join(self.img_dir, row['image_name']).strip()
        label = row.get('cluster_id', -1)
        try:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
                
            if self.visual:
                return image, label
            
            text = clip.tokenize([row['caption']])[0]
            return image, text, label
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            if self.visual:
                return None, None
            return None, None, None
        