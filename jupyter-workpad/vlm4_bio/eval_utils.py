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


from time import time

def get_test_train_split(df_cleaned, test_size=0.2, stratify_by_scientific_name=False):
    """
    Split a dataframe into test and train sets

    Parameters
    ----------
    df_cleaned : dataframe
        DESCRIPTION.
    test_size : float, optional
        size of the test set. The default is 0.1.
    stratify_by_scientific_name : Boolean, optional
        Allow for stratification (even representation across classes). The default is False.

    Returns
    -------
    train_df : TYPE
        DESCRIPTION.
    test_df : TYPE
        DESCRIPTION.

    """
    if stratify_by_scientific_name:
        # if we want to get at least one from each we can filter out all options with only 1 image
        species_counts = df_cleaned['scientific_name'].value_counts()
        valid_species = species_counts[species_counts > 1].index
        df_filtered = df_cleaned[df_cleaned['scientific_name'].isin(valid_species)]
        # with stratification, we need to specify the test_size because we need to hit a minimum
        calc_min = len(valid_species)/len(df_filtered)
        min_split = max(test_size, calc_min)
        print(f'The split for stratification is: {min_split}')

        train_df, test_df = train_test_split(df_filtered, test_size=min_split, stratify=df_filtered['scientific_name'])
    else:
        train_df, test_df = train_test_split(df_cleaned, test_size=test_size)
        return train_df, test_df
    
    
def calculate_similarity_and_distance(test_embeddings, train_embeddings, model):
    
    # we don't need the dim=1, in both cases its 1
    test_embeddings, train_embeddings = np.squeeze(test_embeddings, axis=1), np.squeeze(train_embeddings, axis=1)

    # if florence, avg across new dim 1, size = 577 (patches on img)
    if model == 'florence':
        test_embeddings = np.mean(test_embeddings, axis=1)
        train_embeddings = np.mean(train_embeddings, axis=1)

    similarity_matrix = cosine_similarity(test_embeddings, train_embeddings)
    distance_matrix = euclidean_distances(test_embeddings, train_embeddings)

    return similarity_matrix, distance_matrix


def find_best_match_vectorized(test_embeddings, train_df, label_column, model):
    
    # Get train embeddings and labels as numpy arrays
    train_embeddings = np.array(train_df['image_embeddings'].tolist())
    train_labels = train_df[label_column].values

    # Calculate similarity and distance for all test embeddings at once
    similarity_matrix, distance_matrix = calculate_similarity_and_distance(test_embeddings, train_embeddings, model)

    # Get the labels with the highest individual similarity score and lowest distance
    highest_similarity_idx = similarity_matrix.argmax(axis=1)
    highest_similarity_labels = train_labels[highest_similarity_idx]
    highest_similarity_scores = similarity_matrix.max(axis=1)

    lowest_dist_idx = distance_matrix.argmin(axis=1)
    lowest_dist_labels = train_labels[highest_similarity_idx]
    lowest_dist_scores = similarity_matrix.min(axis=1)

    # create a repeated matrix of train labels to match the size of similarity matrix (test_size x train_size)
    repeated_train_labels = np.tile(train_labels, (similarity_matrix.shape[0], 1))

    # Create a DataFrame with each test embedding's comparison to all train labels
    # Flatten the similarity matrix and repeated labels to match
    similarity_df = pd.DataFrame({
        'test_idx': np.repeat(range(similarity_matrix.shape[0]), similarity_matrix.shape[1]),
        label_column: repeated_train_labels.flatten(),
        'similarity_score': similarity_matrix.flatten()
    })
    avg_similarity_df = similarity_df.groupby(['test_idx', label_column], as_index=False).mean()
    idx_max_avg = avg_similarity_df.groupby('test_idx')['similarity_score'].idxmax()
    highest_avg_sim_labels = avg_similarity_df.loc[idx_max_avg, label_column].values
    highest_avg_sim_scores = avg_similarity_df.loc[idx_max_avg, 'similarity_score'].values

    distance_df = pd.DataFrame({
        'test_idx': np.repeat(range(similarity_matrix.shape[0]), similarity_matrix.shape[1]),
        label_column: repeated_train_labels.flatten(),
        'distance_score': distance_matrix.flatten()
    })
    avg_dist_df = distance_df.groupby(['test_idx', label_column], as_index=False).mean()
    idx_min_avg = avg_dist_df.groupby('test_idx')['distance_score'].idxmin()
    lowest_avg_dist_labels = avg_dist_df.loc[idx_min_avg, label_column].values
    lowest_avg_dist_scores = avg_dist_df.loc[idx_min_avg, 'distance_score'].values

    return highest_similarity_labels,highest_similarity_scores,highest_avg_sim_labels,highest_avg_sim_scores,lowest_dist_labels,lowest_dist_scores,lowest_avg_dist_labels,lowest_avg_dist_scores

def apply_best_match_vectorized(test_df, train_df, label_column, model):
    # Convert all test image embeddings to numpy arrays
    test_embeddings = np.array(test_df['image_embeddings'].tolist())

    # Vectorized function to find the best match for each test embedding
    highest_individual_name, highest_individual_score, highest_mean_name, highest_mean_score,lowest_dist_name, lowest_dist_score, lowest_avg_dist_name, lowest_avg_dist_score = find_best_match_vectorized(
        test_embeddings, train_df, label_column, model
    )

    # Assign results back to the test DataFrame
    test_df['highest_individual_name'] = highest_individual_name
    test_df['highest_individual_score'] = highest_individual_score
    test_df['highest_mean_name'] = highest_mean_name
    test_df['highest_mean_score'] = highest_mean_score
    test_df['lowest_dist_name'] = lowest_dist_name
    test_df['lowest_dist_score'] = lowest_dist_score
    test_df['lowest_avg_dist_name'] = lowest_avg_dist_name
    test_df['lowest_avg_dist_score'] = lowest_avg_dist_score

    # Calculate accuracy metrics
    accuracy_individual = np.mean(test_df['highest_individual_name'] == test_df[label_column])
    accuracy_mean = np.mean(test_df['highest_mean_name'] == test_df[label_column])
    accuracy_dist = np.mean(test_df['lowest_dist_name'] == test_df[label_column])
    accuracy_avg_dist = np.mean(test_df['lowest_avg_dist_name'] == test_df[label_column])

    print(f"Accuracy for model {model} on column {label_column} based on highest individual cosine similarity: {accuracy_individual * 100:.2f}%")
    print(f"Accuracy for model {model} on column {label_column} based on highest mean cosine similarity: {accuracy_mean * 100:.2f}%")
    print(f"Accuracy for model {model} on column {label_column} based on lowest individual euclidean distance: {accuracy_individual * 100:.2f}%")
    print(f"Accuracy for model {model} on column {label_column} based on lowest mean euclidean distance: {accuracy_mean * 100:.2f}%")

    return test_df, accuracy_individual, accuracy_mean, accuracy_dist, accuracy_avg_dist


def extract_genus(scientific_name):
    
    try:
        return scientific_name.split()[0]
    except Exception as e:
        print(scientific_name)
        raise



def vlm_species_eval(full_species_df, model, transform, get_embeds):
    """
    Evaluation loop to calculate cosine similarity and euclidean distance in n=1 NN method

    Parameters
    ----------
    full_species_df : Dataframe
        Dataframe containing all info about the VLM4Bio datasets, with image file names and scientific_names.
    model : model
        model used to evaluate the data.
    transform : function
        transform function used to preprocess data for the model.
    get_embeds : function
        function used to get image embeddings for the model.

    Returns
    -------
    eval_df : Dataframe
        output dataframe including all the accuracy calculations.

    """
    full_species_df['category'] = full_species_df['category'].apply(lambda x: x.strip())

    eval_df = pd.DataFrame(columns=['model', 'taxa', 'column', 'accuracy_individual', 'accuracy_avg', 'distance_individual',	'distance_avg'])
    rows = []
    model = model.float()
    analysis_taxa = ['Fish', 'Bird', 'Butterfly', 'All']

    full_species_df = full_species_df[full_species_df['scientific_name'].str.strip() != '']
    full_species_df['genus'] = full_species_df['scientific_name'].apply(extract_genus)

    # note the amount of time necessary to generate all embeddings
    start = time.time()

    full_species_df['image_embeddings'] = full_species_df['image_name'].apply(lambda x: get_embeds(x, model, transform))
    full_species_df.dropna(subset=['scientific_name'], inplace=True)

    time_length = time.time() - start
    print(f'Took {time_length} seconds to generate embeddings')

    # we're gonna compare species by species within each taxa, and against each
    for taxa in analysis_taxa: # cleaned_dfs can have full species appended to it too
        print(f"WORKING ON TAXA: {taxa}")

        df = full_species_df if taxa == 'All' else full_species_df[full_species_df['category'] == taxa]
        train_df, test_df = get_test_train_split(df, test_size=0.2, stratify_by_scientific_name=False)

        start = time.time()
        test_df, accuracy_individual, accuracy_avg, distance_individual, distance_avg = apply_best_match_vectorized(test_df, train_df, 'scientific_name', 'trained_vlm_unicom')
        time_length = time.time() - start
        print(f"Time for analysis for model trained_vlm_unicom: {time_length}")
        rows.append({'model': 'trained_vlm_unicom', 'taxa': taxa, 'column': 'species', 'accuracy_individual': f"{accuracy_individual * 100:.2f}%", 'accuracy_avg': f"{accuracy_avg * 100:.2f}%", 'distance_individual': f"{distance_individual * 100:.2f}%", 'distance_avg': f"{distance_avg * 100:.2f}%"})
    
        test_df, accuracy_individual, accuracy_avg, distance_individual, distance_avg = apply_best_match_vectorized(test_df, train_df, 'genus', 'trained_vlm_unicom')
        rows.append({'model': 'trained_vlm_unicom', 'taxa': taxa, 'column': 'genus', 'accuracy_individual': f"{accuracy_individual * 100:.2f}%", 'accuracy_avg': f"{accuracy_avg * 100:.2f}%", 'distance_individual': f"{distance_individual * 100:.2f}%", 'distance_avg': f"{distance_avg * 100:.2f}%"})
        if taxa == 'All': # only relevant when comparing against other taxa
            test_df, accuracy_individual, accuracy_avg, distance_individual, distance_avg = apply_best_match_vectorized(test_df, train_df, 'category', 'trained_vlm_unicom')
            rows.append({'model': 'trained_vlm_unicom', 'taxa': taxa, 'column': 'taxa', 'accuracy_individual': f"{accuracy_individual * 100:.2f}%", 'accuracy_avg': f"{accuracy_avg * 100:.2f}%", 'distance_individual': f"{distance_individual * 100:.2f}%", 'distance_avg': f"{distance_avg * 100:.2f}%"})

    eval_df = pd.concat([eval_df, pd.DataFrame(rows)], ignore_index=True)
    return eval_df


