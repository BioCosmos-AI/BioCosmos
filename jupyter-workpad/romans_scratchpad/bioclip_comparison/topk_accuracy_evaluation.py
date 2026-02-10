import os, argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import logging

log_format = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_format)
logger = logging.getLogger()

def save_comparison(control, test, output):
    """
    
    """

    logger.info("Saving accuracies to csv")
    combined_acc = {k: [] for k in control.keys()}
    for k in combined_acc.keys():
        combined_acc[k].append(control[k])
        combined_acc[k].append(test[k])

    df = pd.DataFrame(combined_acc, index=['Control', 'Test'])
    df.to_csv(output)


def zero_shot_accuracy(test_loader, cntrl_p, test_p, topk=(1, 3, 5)):
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
    logger.info(f"cntrp_p size is {len(cntrl_p)}")
    control_prototype_stack = torch.stack(list(cntrl_p.values()))  # Shape: [num_classes, embedding_dim]
    test_prototype_stack = torch.stack(list(test_p.values()))

    correct_c = {k: 0 for k in topk}
    correct_t = {k: 0 for k in topk}
    total = 0

    with torch.no_grad():
        for control_e, test_e, labels in test_loader:

            # Compute similarity between test embeddings and class prototypes

            c_similarities = F.cosine_similarity(control_e.unsqueeze(1), control_prototype_stack.unsqueeze(0), dim=-1) #  Shape [batch_size, num_classes]
            t_similarities = F.cosine_similarity(test_e.unsqueeze(1), test_prototype_stack.unsqueeze(0), dim=-1)

            #  Get top-k predictions
            _, c_predictions = c_similarities.topk(max(topk), dim=-1)  # Shape: [batch_size, max(topk)]
            _, t_predictions = t_similarities.topk(max(topk), dim=-1)


            # set predictions for each top k (1, 3, 5)
            for k in topk:
                correct_c[k] += (c_predictions[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
                correct_t[k] += (t_predictions[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()

            total += labels.size(0)

    # Compute accuracy for each k
    accuracies_c = {k: correct_c[k] / total for k in topk}
    accuracies_t = {k: correct_t[k] / total for k in topk}
    return accuracies_c, accuracies_t

def load_parquet_dir(path):
    """Load and concatenate all parquet files in a directory"""
    dfs = []
    logger.info(f"loading parquet from {path}")
    for fname in tqdm(sorted(os.listdir(path)), desc=f"Loading from {path}"):
        if fname.endswith(".parquet"):
            full_path = os.path.join(path, fname)
            df = pd.read_parquet(full_path, engine='pyarrow')
            df["embedding"] = list(df.to_numpy())
            df = df[["embedding"]]  
            dfs.append(df)
    logger.info(f"Length of dfs {len(dfs)}")
    logger.info(f"size of embedding: {df['embedding'].iloc[0].shape}")
    return pd.concat(dfs)


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

    species_counts = df_cleaned['species'].value_counts()
    valid_species = species_counts[species_counts > min_count_per_class].index
    df_filtered = df_cleaned[df_cleaned['species'].isin(valid_species)]
    calc_min = len(valid_species)/len(df_filtered)
    min_split = max(test_size, calc_min)

    train_df, test_df = train_test_split(df_filtered, test_size=min_split, stratify=df_filtered['species'])
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
        train_split.groupby("species")[embedding_column]
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
        control_embs, test_embs = torch.tensor(row['control_emb']), torch.tensor(row['test_emb'])
        cls = row['species']
        # image embeddings are size [1, embed_dim] --> squeeze first dim
        return control_embs, test_embs, self.cls_to_idx[cls]

def main():

    parser = argparse.ArgumentParser(
        description="Evaluation of trained model in UNICOM method"
    )
    parser.add_argument("--control-model-parquet-dir", required=True, help="Directory containing generated embedding .parquet files for control model")
    parser.add_argument("--test-model-parquet-dir", required=True, help="Directory containing generated embedding .parquet files for test model")
    parser.add_argument("--mapping-csv", required=True, help="Directory containing csv mapping uuid : species")
    parser.add_argument("--output-dir", required=True, help="Directory where accuracy results are output")

    args = parser.parse_args()

    logger.info(f"Loading parquet files for control and test model into df")

    control_df = load_parquet_dir(args.control_model_parquet_dir)
    test_df = load_parquet_dir(args.test_model_parquet_dir)

    # create a mapping df to map from uuid : species
    mapping_df = pd.read_csv(args.mapping_csv).dropna(subset=["species"])
    uuid_to_species = mapping_df.set_index("treeoflife_id")["species"].to_dict() 

    # check that indices are the same across control and test files
    if set(control_df.index) != set(test_df.index):
        raise ValueError("Control and test embeddings have different UUID sets!")

    # sprt to ensure indices are aligned
    control_df = control_df.sort_index()
    test_df = test_df.sort_index()

    # create a column of species in same order of uuid -- for species
    species_col = pd.Series(control_df.index).map(uuid_to_species)
    full_df = pd.DataFrame({
        "uuid": control_df.index,
        "species": species_col
    })

    # add embeddings as arrays -- ARRAY IS BEST OPTION?
    full_df["control_emb"] = list(control_df['embedding'])
    full_df["test_emb"] = list(test_df['embedding'])

    len_full = len(full_df)

    # Unmapped UUID - drop it
    full_df = full_df.dropna(subset=["species"]).reset_index(drop=True)

    logger.info(f"Dropped {len_full - len(full_df)} rows data with NA species")

    # test train split - stratified
    train_split, test_split = get_test_train_split(full_df)

    # get the class prototypes (avg) for each class across control and test model embeddings
    class_prototypes_control = get_class_prototypes(train_split, 'control_emb')
    class_prototypes_test = get_class_prototypes(train_split, 'test_emb')
    
    # get the labels in integer form
    prototype_labels = list(class_prototypes_control.keys())
    prototype_to_idx = {cls: i for i, cls in enumerate(prototype_labels)}
    
    # set up the dataset loader
    dataset = EmbeddingsDataset(test_split, prototype_to_idx)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # run the control 
    topk_accuracies_control, topk_accuracies_test = zero_shot_accuracy(test_loader, class_prototypes_control, class_prototypes_test, topk=(1, 3, 5))
    logger.info(f"Top-k Accuracies for control model: {topk_accuracies_control}")
    logger.info(f"Top-k Accuracies for test model: {topk_accuracies_test}")

    # set these two dict to a df
    save_comparison(topk_accuracies_control, topk_accuracies_test, args.output_dir)

if __name__ == "__main__":
    main()
