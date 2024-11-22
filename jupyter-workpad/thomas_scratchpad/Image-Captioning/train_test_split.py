import pandas as pd
import numpy as np


def create_train_test_split(csv_path, train_ratio=0.7, random_seed=42):
    # Read the CSV
    df = pd.read_csv(csv_path)

    # Create splits per category to ensure balanced representation
    train_dfs = []
    test_dfs = []

    # Get unique categories
    categories = df["category"].unique()

    # Set random seed for reproducibility
    np.random.seed(random_seed)

    for category in categories:
        # Get rows for this category
        category_df = df[df["category"] == category].copy()

        # Get number of samples for train/test split
        n_samples = len(category_df)
        n_train = int(n_samples * train_ratio)

        # Create random permutation
        shuffle_idx = np.random.permutation(n_samples)

        # Split into train and test
        train_mask = np.zeros(n_samples, dtype=bool)
        train_mask[shuffle_idx[:n_train]] = True

        # Mark rows using boolean indexing
        category_df["split"] = "test"  # Default all to test
        category_df.loc[train_mask, "split"] = "train"

        # Append to lists
        train_dfs.append(category_df[category_df["split"] == "train"])
        test_dfs.append(category_df[category_df["split"] == "test"])

    # Combine all splits
    train_df = pd.concat(train_dfs)
    test_df = pd.concat(test_dfs)

    # Create final DataFrame with split information
    df["split"] = "test"  # Default all to test
    df.loc[train_df.index, "split"] = "train"  # Mark training examples

    # Save the marked dataset
    df.to_csv("vlm4bio_captions_with_split.csv", index=False)

    # Print statistics
    print("\nDataset split statistics:")
    print("-------------------------")
    for category in categories:
        train_count = len(train_df[train_df["category"] == category])
        test_count = len(test_df[test_df["category"] == category])
        print(f"\n{category}:")
        print(f"Training samples: {train_count}")
        print(f"Test samples: {test_count}")

    return df
