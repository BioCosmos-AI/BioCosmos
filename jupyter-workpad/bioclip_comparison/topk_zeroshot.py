#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# TODO: these are not correct right now 
from .model_utils import load_df


def zero_shot_accuracy(test_loader, class_prototypes, topk=(1, 3, 5)):
    """
    Method to compute accuracy for top k when comparing test embeddings to class prototypes

    Parameters
    ----------
    test_loader : torch.utils.data.DataLoader
        DataLoader for test dataset.
    class_prototypes : Dict[str: torch.tensor]
        Class prototypes of the .
    topk : tuple, optional
        Tuple detailing which topk analysis is to be conducted. The default is (1, 3, 5).

    Returns
    -------
    accuracies : Dict[int: float]
        Accuracies for ks in topk of test embeddings to class prototype embeddings.

    """

    # assume the prototype embeddings exist in a stack, with the index of the prototype being the same as its class idx
    prototype_embeddings = torch.stack(list(class_prototypes.values())).squeeze(1)  # Shape: [num_classes, embedding_dim]

    correct = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for embeddings, labels in test_loader:

            # Compute similarity between test embeddings and class prototypes

            similarities = F.cosine_similarity(embeddings.unsqueeze(1), prototype_embeddings.unsqueeze(0), dim=-1) #  Shape [batch_size, num_classes]

            # Get top-k predictions
            _, predictions = similarities.topk(max(topk), dim=-1)  # Shape: [batch_size, max(topk)]

            # set predictions for each top k (1, 3, 5)
            for k in topk:
                correct[k] += (predictions[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    # Compute accuracy for each k
    accuracies = {k: correct[k] / total for k in topk}
    return accuracies


def get_test_train_split(df_cleaned, min_count_per_class=10, test_size=0.1):
    """
    Modified get_test_train_split function. Always stratified, get a minimum count of each class.

    Parameters
    ----------
    df_cleaned : pandas Dataframe
        DESCRIPTION.
    min_count_per_class: int
        minimum number of instances (images) for each class. The default is 10.
    test_size : float, optional
        ratio size for the test split. The default is 0.1.

    Returns
    -------
    train_df : pandas Dataframe
        the training split.
    test_df : pandas Dataframe
        the test split.

    """

    species_counts = df_cleaned['scientific_name'].value_counts()
    valid_species = species_counts[species_counts > min_count_per_class].index
    df_filtered = df_cleaned[df_cleaned['scientific_name'].isin(valid_species)]
    calc_min = len(valid_species)/len(df_filtered)
    min_split = max(test_size, calc_min)

    train_df, test_df = train_test_split(df_filtered, test_size=min_split, stratify=df_filtered['scientific_name'])
    return train_df, test_df

def get_class_prototypes(train_split):
    """
    Given the train split dataframe, create the class prototypes by getting mean embeddings

    Parameters
    ----------
    train_split : pandas Dataframe
        The train split for the data, used to create the class prototypes.

    Returns
    -------
    class_prototypes : Dict[str: torch.tensor]
        DESCRIPTION.

    """
    train_split["embedding_tensor"] = train_split["image_embeddings"].apply(torch.tensor)
    class_prototypes = (
        train_split.groupby("scientific_name")["embedding_tensor"]
        .apply(lambda x: torch.stack(list(x.squeeze(0))).mean(dim=0))
        .to_dict()
    )
    return class_prototypes


class EmbeddingsDataset(torch.utils.data.Dataset):
    """
    Mostly a redo of previous datasets. This is assuming the dataframe already contains the image embeddings 
    """
    def __init__(self, df, cls_to_idx, transform=None):
        self.df = df
        self.transform = transform
        self.cls_to_idx = cls_to_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load image and apply transformations
        row = self.df.iloc[idx]
        image_embeddings = row['image_embeddings']
        cls = row['scientific_name']
        # image embeddings are size [1, embed_dim] --> squeeze first dim
        return image_embeddings.squeeze(0), self.cls_to_idx[cls]



if __name__ == "__main__":
    # get existing df with image embeddings
    embeds_df = load_df('ViT-H-14-embeddings')
    
    # split data to train and test
    train_split, test_split = get_test_train_split(embeds_df)
    
    # get prototypes from training splits
    class_prototypes = get_class_prototypes(train_split)
    
    
    prototype_labels = list(class_prototypes.keys())
    prototype_to_idx = {cls: i for i, cls in enumerate(prototype_labels)}
    
    dataset = EmbeddingsDataset(test_split, prototype_to_idx)

    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    topk_accuracies = zero_shot_accuracy(test_loader, class_prototypes, prototype_to_idx, topk=(1, 3, 5))
    print(f"Top-k Accuracies: {topk_accuracies}")

