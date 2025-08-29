import os, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from PIL import Image

### SPLIT DATA, TAKE AVG EMBEDDING, TOPK ANALYSIS

def get_test_train_split(df_cleaned, test_size=0.1, stratify_by_scientific_name=False):
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

def get_class_prototypes(train_split, embedding_column):
    """
    Given the train split dataframe, create the class prototypes by getting mean embeddings

    Parameters
    ----------
    train_split : pandas Dataframe
        The train split for the data, used to create the class prototypes.

    Returns
    -------
    class_prototypes : Dict[str: torch.Tensor]
        Class prototype vectors per species.
    """
    class_prototypes = (
        train_split.groupby("scientific_name")[embedding_column]
        .apply(lambda x: torch.stack([torch.tensor(e) for e in x]).mean(dim=0))
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
        embedding = torch.tensor(row['trained_embedding'])
        clas = row['scientific_name']
        # image embeddings are size [1, embed_dim] --> squeeze first dim
        return embedding, self.cls_to_idx[clas]

def image_path_to_embed(image_path, model, preprocess):
    # folder where images are stored
    try:
        image = Image.open(image_path)
        img = preprocess(image).unsqueeze(0)
        img = img.cuda()

        with torch.no_grad():
            image_embeddings = model(img)
        return image_embeddings.cpu().numpy()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None  # Return None for failed image processing

def zero_shot_accuracy(test_loader, cntrl_p, topk=(1, 3, 5)):
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
    control_prototype_stack = torch.stack(list(cntrl_p.values())).squeeze(1)  # Shape: [num_classes, embedding_dim]

    correct_c = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for embedding, labels in test_loader:

            # Compute similarity between test embeddings and class prototypes
            print('shape of control prototype stack is: ', control_prototype_stack.shape)
            print('embedding shape is', embedding.shape)
            c_similarities = F.cosine_similarity(embedding, control_prototype_stack.unsqueeze(0), dim=-1) #  Shape [batch_size, num_classes]

            #  Get top-k predictions
            _, c_predictions = c_similarities.topk(max(topk), dim=-1)  # Shape: [batch_size, max(topk)]
            

            # set predictions for each top k (1, 3, 5)
            for k in topk:
                correct_c[k] += (c_predictions[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    # Compute accuracy for each k
    accuracies_c = {k: correct_c[k] / total for k in topk}
    return accuracies_c

def create_tester_embed_dataset(model, preprocess, tester_path):
    df = pd.read_csv(tester_path)
    df['trained_embedding'] = df.apply(lambda row: image_path_to_embed(row['image_path'].strip(), model, preprocess), axis=1)
    print(f"The tester df looks like this: {df.head()}")                                   
    train_split, test_split = get_test_train_split(df)
    print(f"size of train split: {len(train_split)}")
    class_prototypes = get_class_prototypes(train_split, 'trained_embedding')
    prototype_labels = list(class_prototypes.keys())
    prototype_to_idx = {clas: i for i, clas in enumerate(prototype_labels)}
    
    # set up the dataset loader
    dataset = EmbeddingsDataset(test_split, prototype_to_idx)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)
    
    topk_accuracies = zero_shot_accuracy(test_loader, class_prototypes, topk=(1, 3, 5))
    return topk_accuracies
