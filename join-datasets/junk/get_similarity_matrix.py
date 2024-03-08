import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations

# Function to compute similarity between a pair of rows
def compute_similarity(pair, custom_similarity_function, kwargs, df):
    i, j = pair
    similarity = custom_similarity_function(df.iloc[i], df.iloc[j], **kwargs)
    return i, j, similarity

# Function to build the similarity matrix using threading
def build_custom_matrix_threaded(df: pd.DataFrame, custom_similarity_function: Callable, kwargs: dict) -> np.ndarray:
    n = len(df)
    similarity_matrix = np.zeros((n, n), dtype=float)

    # Generate all unique pairs to compute similarity
    pairs = list(combinations(range(n), 2))

    # Use a thread pool to compute similarities in parallel
    with ThreadPoolExecutor() as executor:
        # Submit all the tasks and get a dictionary to keep track of them
        future_to_pair = {executor.submit(compute_similarity, pair, custom_similarity_function, kwargs, df): pair for pair in pairs}

        # As each thread completes, get the result and update the similarity matrix
        for future in as_completed(future_to_pair):
            i, j, similarity = future.result()
            # Fill in the matrix symmetrically
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    # Set the diagonal to 1 after all computations are done
    np.fill_diagonal(similarity_matrix, 1)
    return similarity_matrix

# Example usage
# similarity_matrix = build_custom_matrix_threaded(df, custom_similarity_function, kwargs)
