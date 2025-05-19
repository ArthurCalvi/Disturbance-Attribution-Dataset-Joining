import pandas as pd
import geopandas as gpd
import numpy as np 
from shapely.geometry import Polygon
from scipy.spatial import ConvexHull
from datetime import datetime, timedelta
import math 
from typing import List, Dict, Tuple, Set, Optional, Callable
import scipy.stats as stats
z = stats.norm.ppf(0.75)

def calculate_temporal_centroid(start_date, end_date):
    # Calculate the midpoint of the period
    return start_date + (end_date - start_date) / 2

def get_temporal_range(gdf):
    # Assuming you have a 'start_date' and 'end_date' column
    min_date = gdf['start_date'].min()
    max_date = gdf['end_date'].max()
    return min_date, max_date

def is_temporally_similar(event1, event2, temporal_threshold):
    # Check if temporal condition is met
    return abs((event1.centroid_date - event2.centroid_date).days) < temporal_threshold

def modify_dsbuffer(dsbuffer : Dict[str, int or None], granularity : int) -> Dict[str, int or None]:
    for key in dsbuffer.keys():
        if dsbuffer[key] is not None:
            dsbuffer[key] = min(granularity, dsbuffer[key])
    return dsbuffer

def get_spatial_join(dataset: gpd.GeoDataFrame, reference: gpd.GeoDataFrame, t: str, spatial_buffer: int) -> gpd.GeoDataFrame:
    """Perform a spatial join between a dataset and a reference dataset according to the type of the dataset (point or polygon)

    Args:
        dataset (gpd.GeoDataFrame): left dataset
        reference (gpd.GeoDataFrame): right dataset
        t (str): type of the dataset (point or polygon)
        spatial_buffer (int): spatial buffer for the spatial join in meters

    Returns:
        gpd.GeoDataFrame: the spatial join between the dataset and the reference dataset
    """
    if t == 'point':
        return dataset.sjoin_nearest(reference.compute(), max_distance=spatial_buffer, distance_col='sd')
    
    elif t == 'polygon':
        return dataset.sjoin(reference)

def classify_source_type(gdf : gpd.GeoDataFrame, area_threshold = 101, name=None) -> str:
    """Classify the type of the dataset (point or polygon) according to the area threshold

    Args:
        gdf (gpd.GeoDataFrame): dataset
        area_threshold (int, optional): area threshold in km2. Defaults to 101.

    Returns:
        str, float: type of the dataset (point or polygon), alpha a scaling factor 
    """

    assert gdf.crs == 'epsg:2154', "crs is not epsg:2154"
    """area threshold should be viewed as the granularity level of the attribution"""
    
    areas = gdf.geometry.area / 1e6
    # mean = np.mean(areas)
    # std = np.std(areas)
    p95 = np.percentile(areas, 95) #change to p95 
    # print(f"[in km2] - mean: {mean}, std: {std}, min: {mi}, max: {ma}, median: {median}, p90: {p90}")
    median = np.median(areas)
    
    if  p95 >= area_threshold:
        # alpha = np.sqrt(area_threshold) / np.sqrt((p95) / np.pi) #scaling factor
        if name is not None:
            print(f"dataset {name} is of spatial entity type")
            print(f'-> 95% of the areas are below : {p95 :.2f} km2, median : {median :.2f} km2')
        return 'spatial_entity', p95
    else:
        return 'point/polygon',  None
    
def get_subset_dataset(dataset: gpd.GeoDataFrame, start_year: datetime, end_year: datetime) -> gpd.GeoDataFrame:
    return dataset[(dataset['start_date'] <= end_year) & (dataset['end_date'] >= start_year)]

def is_dataset_valid(name: str, dataset: gpd.GeoDataFrame, ref=False) -> bool:
    assert isinstance(dataset, gpd.GeoDataFrame), f"dataset '{name}' is not a GeoDataFrame"
    assert 'geometry' in dataset.columns, f"dataset '{name}' has no geometry column"
    assert dataset.crs is not None, f"dataset '{name}' has no crs"
    assert dataset.index.dtype == int, f"dataset '{name}' has no valid index column"
    if not ref:
        assert 'class' in dataset.columns, f"dataset '{name}' has no class column"
        dataset['class'] = dataset['class'].astype(str)

    #check if the 'start_date' and 'end_date' columns are present, if not check if a column 'year' is present
    if 'start_date' not in dataset.columns:
        assert 'year' in dataset.columns, f"dataset '{name}' has no start_date or year column"

        #if year in column, create a start_date and end_date column as the first and last day of this year
        dataset['start_date'] = pd.to_datetime(dataset['year'], format='%Y')
        dataset['end_date'] = pd.to_datetime(dataset['year'], format='%Y') + pd.offsets.YearEnd(0)
        


    assert 'start_date' in dataset.columns, f"dataset '{name}' has no start_date column"
    assert 'end_date' in dataset.columns, f"dataset '{name}' has no end_date column"
    dataset['start_date'] = pd.to_datetime(dataset['start_date'], format='%Y-%m-%d')
    dataset['end_date'] = pd.to_datetime(dataset['end_date'], format='%Y-%m-%d')
    dataset['dataset'] = name
    
    return dataset 

# def exponential_decay(days: int, half_life: int) -> float:
#     return np.exp(-np.log(2) * days / half_life)

# Define a Gaussian function
def gaussian(x, mean, std):
    return np.exp(-0.5 * ((x - mean) / std)**2)

# Define a step function
def step(x, start, end):
    return np.where((x >= start) & (x <= end), 1, 0)

def offset_gaussian(x, offset, decrease):
    if x <= offset:
        return 1  # Plateau value
    else:
        # Gaussian part
        c = decrease / 2  # Standard deviation set so 2c equals decrease
        return math.exp(-((x - offset) ** 2) / (2 * c ** 2))

# Define a sigmoid function
def sigmoid(x, x0, k):
    return 1 / (1 + np.exp(-k * (x - x0)))

# Define an exponential decay function
def exponential_decay(x, x0, k):
    return np.exp(-k * (x - x0))

# Define a constant function
def constant(x, value):
    return np.full_like(x, value)

def weighting_function(x, x0, k):
    """
    x0 : offset
    k : half_life 
    """
    if x <= x0:
        return 1
    else:
        return np.exp(-np.log(2) * (x - x0) / k)
    
dfunction = {
    'gaussian': gaussian,
    'step': step,
    'exponential_decay': exponential_decay,
    'weighting_function': weighting_function, 
    'offset_gaussian': offset_gaussian
}

def combine_functions(funcs, scores):
    """Return a callable function combining the given functions with respective scores."""
    return lambda x: sum(f(x) * score for f, score in zip(funcs, scores)) 

def get_combined_weighting(ddataset_profile, ddisturbance_profile, disturbance_classes_composition, t):
    """Return a callable function for combined spatial or temporal weighting."""
    dataset_func_type, dataset_params = ddataset_profile
    dataset_func = dfunction[dataset_func_type]  # Get the function from the global scope

    disturbance_funcs_scores = [(ddisturbance_profile[disturbance_class][t], score) for disturbance_class, score in disturbance_classes_composition.items()]
    disturbance_funcs = [lambda x, f=dfunction[fn], p=p: f(x, **p) for (fn, p), _ in disturbance_funcs_scores]
    scores = [score for _, score in disturbance_funcs_scores]
    
    combined_disturbance_func = combine_functions(disturbance_funcs, scores)
    # mean_std_gaussian = np.sum([s* p['std'] for (fn, p), s in disturbance_funcs_scores])
    # reg = min(1, z *  mean_std_gaussian / list(dataset_params.values())[1])
    # print('regularization parameter : ', reg)
    return lambda x: (dataset_func(x, **dataset_params) + combined_disturbance_func(x))/2 #* reg

def calculate_iou(poly1, poly2):
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    return intersection / union if union != 0 else 0

def calculate_d_star(area : int) -> float:
    """Calculate the equivalent distance in meters for a given area

    Args:
        area (int): in m2 
        alpha (int, optional): scaling factor. Defaults to 1.

    Returns:
        float: equivalent distance in meters
    """
    # return alpha * np.sqrt(area / np.pi)
    return np.sqrt(area / np.pi)

def build_spatial_matrix(gdf: gpd.GeoDataFrame, dtypes_: Dict[str, str], final_weighting_dict: Dict[str, dict]) -> np.ndarray:
    """Build a spatial matrix from a GeoDataFrame

    Args:
        gdf (gpd.GeoDataFrame): GeoDataFrame
        dtypes_ (Dict[str, str]): dictionary of dataset types
        f (Callable): function to convert distance to similarity
    
    Returns:
        np.ndarray: spatial matrix
    """
    num_elements = len(gdf)
    distance_matrix = np.zeros((num_elements, num_elements))

    if not any([dtypes_[name] == 'spatial_entity' for name in gdf.dataset.unique()]):
        # Calculate distances only for the upper triangle of the matrix
        for i in range(num_elements):
            for j in range(i + 1, num_elements):  # Start from i+1 to skip the diagonal
                # Calculate distance between points
                distance = gdf.geometry.iloc[i].centroid.distance(gdf.geometry.iloc[j].centroid)
                # Convert distance to similarity
                weight = (final_weighting_dict[gdf.iloc[i]['dataset']][gdf.iloc[i]['class']]['spatial'](distance) + final_weighting_dict[gdf.iloc[j]['dataset']][gdf.iloc[j]['class']]['spatial'](distance))/2
                distance_matrix[i][j] = weight
                distance_matrix[j][i] = weight

    else :
        # Calculate distances only for the upper triangle of the matrix
        for i in range(num_elements):
            for j in range(i + 1, num_elements):  # Start from i+1 to skip the diagonal
                d_i = gdf.iloc[i]['dataset']
                d_j = gdf.iloc[j]['dataset']
                t_i = dtypes_[d_i]
                t_j = dtypes_[d_j]

                if t_i == 'point/polygon' and t_j == 'point/polygon':
                    distance = gdf.geometry.iloc[i].centroid.distance(gdf.geometry.iloc[j].centroid)
                elif t_i == 'spatial_entity' and t_j == 'spatial_entity':
                    area_i = gdf.geometry.iloc[i].area
                    area_j = gdf.geometry.iloc[j].area
                    d_star_i = calculate_d_star(area_i)
                    d_star_j = calculate_d_star(area_j)
                    iou = calculate_iou(gdf.geometry.iloc[i], gdf.geometry.iloc[j])
                    distance = (d_star_i + d_star_j) / 2 / iou if iou > 0 else (d_star_i + d_star_j) / 2
                else:
                    if t_i == 'spatial_entity':
                        area_i = gdf.geometry.iloc[i].area
                        distance = calculate_d_star(area_i)
                    else:
                        area_j = gdf.geometry.iloc[j].area
                        distance = calculate_d_star(area_j)

                # print(f"distance between {gdf.iloc[i]['dataset']} and {gdf.iloc[j]['dataset']} is {distance}")
                weight = (final_weighting_dict[gdf.iloc[i]['dataset']][gdf.iloc[i]['class']]['spatial'](distance) + final_weighting_dict[gdf.iloc[j]['dataset']][gdf.iloc[j]['class']]['spatial'](distance))/2
                distance_matrix[i][j] = weight 
                distance_matrix[j][i] = weight 

        # Fill the diagonal with the maximum similarity score, e.g., 1
    np.fill_diagonal(distance_matrix, 1)
    return distance_matrix

#import fuzz
from thefuzz import fuzz 
# def compute_tree_coherence(row1 : pd.Series, row2 : pd.Series) -> float:
#     """Compute the tree coherence between two rows

#     Args:
#         row1 (pd.Series): first row
#         row2 (pd.Series): second row

#     Returns:
#         float: tree coherence
#     """

#     for essence1 in row1['essence'].split(','):
#         for essence2 in row2['essence'].split(','):
#             if fuzz.token_set_ratio(essence1.lower(), essence2.lower()) > 80:
#                 return 1
        
#     if row1['tree_type'].lower() == row2['tree_type'].lower():
#         return 0.75 
    
#     if row1['tree_type'].lower() == 'mixed' or row2['tree_type'].lower() == 'mixed':
#         return 0.5

#     return 0.25 


# def compute_class_similarity(row1 : pd.Series, row2 : pd.Series, dclass_score : Dict[str, Dict[str, Dict[str, float]]] ) -> float:
#     """Compute the similarity between two classes

#     Args:
#         row1 (pd.Series): first row
#         row2 (pd.Series): second row
#         dclass_score (Dict[str, Dict[str, Dict[str, float]]]): dictionary containing the class scores
    
#     Returns:
#         float: similarity score
#     """

#     # If both classes are from the same dataset and have the same class, return 1.0
#     if row1['dataset'] == row2['dataset'] and row1['class'] == row2['class']:
#         return 1.0

#     # Get the score dictionaries for the two classes
#     scores1 = dclass_score.get(row1['dataset'], {}).get(row1['class'], {})
#     scores2 = dclass_score.get(row2['dataset'], {}).get(row2['class'], {})
    
#     # Find common final classes between the two classes
#     common_final_classes = set(scores1.keys()).intersection(set(scores2.keys()))
    
#     # Calculate the sum of the means of the scores for each common final class
#     similarity_score = sum((scores1[final_class] + scores2[final_class]) / 2 for final_class in common_final_classes)
    
#     return similarity_score

import networkx as nx
import numpy as np
from tqdm import tqdm 

def build_graph(dataset_loc, dataset_glob,  sindex, spatial_threshold, temporal_threshold, attribution, filter_class, G=None):
    
    if G is None:
        G = nx.Graph()
        # Construct the similarity matrix using spatial index
    for event1 in tqdm(dataset_loc.itertuples(index=True)):
        i = event1.Index
        # Find nearby events within spatial threshold
        possible_matches_index = list(sindex.intersection(event1.geometry.buffer(spatial_threshold).bounds))
        possible_matches = dataset_glob.iloc[possible_matches_index]

        for event2 in possible_matches.itertuples(index=True):
            j = event2.Index
            if i != j and is_temporally_similar(event1, event2, temporal_threshold):
                ds = event1.geometry.centroid.distance(event2.geometry.centroid)
                dt = abs(event1.centroid_date - event2.centroid_date).days 
                we = compute_tree_coherence(event1, event2, filter_class, attr_class='_3')
                wc = compute_class_similarity(event1, event2, attribution.dclass_score, attr_class='_3')
                ws = np.mean([attribution.final_weighting_dict[event1.dataset][event1._3]['spatial'](ds), attribution.final_weighting_dict[event2.dataset][event2._3]['spatial'](ds)])
                wt = np.mean([attribution.final_weighting_dict[event1.dataset][event1._3]['temporal'](dt), attribution.final_weighting_dict[event2.dataset][event2._3]['temporal'](dt)])
                G.add_edge(i, j, ws=ws, wt=wt, we=we, wc=wc)
    
    return G 

from thefuzz import fuzz 
def compute_tree_coherence(row1 : pd.Series, row2 : pd.Series, filter_class : dict, attr_class='class') -> float:
    """Compute the tree coherence between two rows

    Args:
        row1 (pd.Series): first row
        row2 (pd.Series): second row

    Returns:
        float: tree coherence
    """
    if getattr(row1, attr_class) in filter_class and getattr(row2, attr_class) in filter_class:
        return 1.
    
    for essence1 in row1.essence.split(','):
        for essence2 in row2.essence.split(','):
            if fuzz.token_set_ratio(essence1.lower(), essence2.lower()) > 80:
                return 1
        
    if row1.tree_type.lower() == row2.tree_type.lower():
        return 0.75 
    
    if row1.tree_type.lower() == 'mixed' or row2.tree_type.lower() == 'mixed':
        return 0.5

    return 0.25 

from typing import Dict
def compute_class_similarity(row1 : pd.Series, row2 : pd.Series, dclass_score : Dict[str, Dict[str, Dict[str, float]]], attr_class = 'class' ) -> float:
    """Compute the similarity between two classes

    Args:
        row1 (pd.Series): first row
        row2 (pd.Series): second row
        dclass_score (Dict[str, Dict[str, Dict[str, float]]]): dictionary containing the class scores
    
    Returns:
        float: similarity score
    """

    # If both classes are from the same dataset and have the same class, return 1.0
    if row1.dataset == row2.dataset and getattr(row1, attr_class) == getattr(row2, attr_class):
        return 1.0

    # Get the score dictionaries for the two classes
    scores1 = dclass_score.get(row1.dataset, {}).get(getattr(row1, attr_class), {})
    scores2 = dclass_score.get(row2.dataset, {}).get(getattr(row2, attr_class), {})
    
    # Find common final classes between the two classes
    common_final_classes = set(scores1.keys()).intersection(set(scores2.keys()))
    
    # Calculate the sum of the means of the scores for each common final class
    similarity_score = sum((scores1[final_class] + scores2[final_class]) / 2 for final_class in common_final_classes)
    
    return similarity_score

    
# Function to build the similarity matrix
def build_custom_matrix(df : pd.DataFrame, custom_similarity_function : Callable, kwargs: dict) -> np.ndarray:
    """Build a similarity matrix using a custom similarity function

    Args:
        df (pd.DataFrame): DataFrame containing the data to be compared
        custom_similarity_function (Callable): custom similarity function
        kwargs (dict): optional arguments for the custom similarity function

    Returns:
        np.ndarray: similarity matrix
    """

    n = len(df)
    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((n, n))
    
    # Iterate over each pair of rows
    for i in range(n):
        for j in range(i+1, n):  # Use range(i, n) to avoid redundant computations
            # Compute the tree coherence for each pair of rows
            similarity = custom_similarity_function(df.iloc[i], df.iloc[j], **kwargs)
            # Fill in the matrix, it's symmetric so we can do both i,j and j,i
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    np.fill_diagonal(similarity_matrix, 1)

    return similarity_matrix


import numpy as np


def build_temporal_matrix(data, final_weighting_dict):

    #temporal
    start_date_matrix = data['start_date'].values.astype('datetime64[D]').reshape(-1, 1)
    day_diff_matrix_sd = np.abs((start_date_matrix - start_date_matrix.T) / np.timedelta64(1, 'D'))
  
    end_date_matrix = data['end_date'].values.astype('datetime64[D]').reshape(-1, 1)
    day_diff_matrix_ed = np.abs((end_date_matrix - end_date_matrix.T) / np.timedelta64(1, 'D'))

    m = (day_diff_matrix_sd + day_diff_matrix_ed)/2

    n = len(data)
    for i in range(n):
        for j in range(i + 1, n):
            weight = (final_weighting_dict[data.iloc[i]['dataset']][data.iloc[i]['class']]['temporal'](m[i][j]) + final_weighting_dict[data.iloc[j]['dataset']][data.iloc[j]['class']]['temporal'](m[j][i]))/2
            m[i][j] = weight
            m[j][i] = weight

    #fill diagonal with 1
    np.fill_diagonal(m, 1)

    # return  np.array(list(map(lambda row: list(map(f, row)), m)))
    return m 


def compute_similarity_matrix(data : gpd.GeoDataFrame, dtypes_: Dict[str, str], dcustom_similarity_function : Dict[str, Tuple[Callable, dict, float]], final_weighting_dict : Dict[str, dict], weights=[1,1,1,1]) \
    -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Compute all the different similarity matrices and combine them into a single matrix

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the data to be compared
        dtypes_ (Dict[str, str]): dictionary of dataset types
        dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): Dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
        final_weighting_dict (Dict[str, dict]): dictionary containing the final weighting functions for each dataset and class

    Returns:
        Tuple[np.ndarray, Dict[str, np.ndarray]] : the combined single similarity matrix and the dictionary containing all the similarity matrices
    """
    #MANDATORY SIMILARITY FACTORS
    
    #spatial 
    data = data.fillna('None')
    matrices = {}
    matrices['spatial'] = np.nan_to_num(build_spatial_matrix(data, dtypes_=dtypes_, final_weighting_dict=final_weighting_dict), nan=0).clip(0,1)

    #temporal
    matrices['temporal'] = np.nan_to_num(build_temporal_matrix(data, final_weighting_dict=final_weighting_dict), nan=0).clip(0,1)

    #CUSTOM  SIMILARITY FACTORS
    for name, (custom_function, kwargs, _) in dcustom_similarity_function.items():
        matrices[name] = np.nan_to_num(build_custom_matrix(data, custom_function, kwargs), nan=0).clip(0,1)

    return np.average(np.array(list(matrices.values())), axis=0, weights=weights), matrices

from sklearn.cluster import SpectralClustering

from collections import defaultdict


def get_predominant_class(l : List[Tuple[str, float, float]]) -> str:
    """Get the predominant class from a list of classes

    Args:
        l (List[Tuple[str, float, float]]): list of classes
    
    Returns:
        str: predominant class
    """
    if len(l) == 1 or len(np.unique([l[0] for l in l])) == 1:
        return l[0][0]
    elif len(np.unique([l[0] for l in l])) > 1:
        # Step 1: Group by class and calculate sum of similarities and count
        class_stats = defaultdict(lambda: {'count': 0, 'comp_similarity_sum': 0.0})
        for entry in l:
            class_name, similarity, comp = entry
            class_stats[class_name]['count'] += 1
            class_stats[class_name]['comp_similarity_sum'] += comp * similarity

        # Step 2: Calculate average similarity for each class
        for class_name, stats in class_stats.items():
            stats['average_composed_similarity'] = stats['comp_similarity_sum'] / stats['count']

        # Step 3: Determine the class with the highest frequency
        # Step 4: If there's a tie, use average similarity
        sorted_classes = sorted(class_stats.items(), key=lambda item: (-item[1]['count'], -item[1]['average_composed_similarity']))
        return sorted_classes[0][0]
    else :
        return 'multi-factor'
    
from collections import defaultdict

# def get_cluster(data : gpd.GeoDataFrame, dcustom_similarity_function : Dict[str, Tuple[Callable, dict, float]],
#                  dtypes_: Dict[str, str], final_weighting_dict : Dict[str, dict],
#                  doa : Dict[str, float], dclass_score : Dict[str, Dict[str, Dict[str, float]]], threshold=0.5) -> gpd.GeoDataFrame:
    
#     """Get the cluster from a GeoDataFrame

#     Args:
#         data (gpd.GeoDataFrame): GeoDataFrame containing the data to be compared
#         dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
#         dtypes_ (Dict[str, str]): dictionary of dataset types
#         doa (Dict[str, float]): dictionary of scaling factors
#         threshold (float, optional): threshold for the similarity score. Defaults to 0.5.
    
#     Returns:
#         gpd.GeoDataFrame: cluster
#     """

#     #build dclass as a dict of list containing the first attribute of the dict of dclass_score
#     dclass = {dataset: {k: list(v.keys()) for k, v in dclass_score[dataset].items()} for dataset in dclass_score}

#     similarity_matrix, _ = compute_similarity_matrix(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, final_weighting_dict=final_weighting_dict)


#     #number of unique class,dataset - maximum number of entries of a final class among the datasets + 1
#     dfinal_class = {}
#     dfinal_class_datasets = defaultdict(list)
#     for i in range(1, len(data)):
#         dataset = data.iloc[i].dataset
#         cls = dclass[dataset][data.iloc[i]['class']]
#         for c in cls:
#             if dataset not in dfinal_class_datasets[c]:
#                 dfinal_class_datasets[c].append(dataset)
#                 dfinal_class[c] = dfinal_class.get(c, 0) + 1

#     uc = len(data.iloc[1:].groupby(['class', 'dataset']))
#     n_clusters = max(1, uc - max(dfinal_class.values()) + 1)

#     # Perform spectral clustering
#     sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
#     labels = sc.fit_predict(similarity_matrix)
#     data['labels'] = labels

#     # Unique cluster labels
#     cluster_labels = data['labels'].iloc[1:]
#     unique_labels = np.unique(cluster_labels)

#     # Dictionary to hold the sum of scores and the count for each cluster
#     cluster_sums = {label: {'sum': 0, 'count': 0, 'class': []} for label in unique_labels}

#     # Sum scores and counts for each cluster
#     for i, (score, label) in enumerate(zip(similarity_matrix[0][1:], cluster_labels)):
#         ws = score * doa[data['dataset'].iloc[i+1]]
#         cluster_sums[label]['sum'] += ws 
#         cluster_sums[label]['count'] += 1
#         cluster_sums[label]['class'].extend([(c,ws,comp) for c, comp in dclass_score[data['dataset'].iloc[i+1]][data['class'].iloc[i+1]].items()])

#     # Compute average score for each cluster
#     average_scores = {label: (cluster_sums[label]['sum'] / cluster_sums[label]['count']) for label in cluster_sums}
#     timeperiod_group = {}
#     indexes_group = {}
#     data_ = data.iloc[1:].copy()
#     data_['start_date'] = pd.to_datetime(data_['start_date'], format='%Y-%m-%d')
#     data_['end_date'] = pd.to_datetime(data_['end_date'], format='%Y-%m-%d')

#     #get date of the median start and end date for each cluster
#     for group in data_['labels'].unique():
#         group_df = data_[data_['labels'] == group]
#         min_start = group_df['start_date'].mean()
#         max_end = group_df['end_date'].mean()
#         timeperiod_group[group] = (min_start, max_end)
#         indexes_group[group] = group_df.index.tolist()



#     for label in average_scores:
#         average_scores[label] = (average_scores[label], get_predominant_class(cluster_sums[label]['class']), timeperiod_group[label], indexes_group[label])

#     # Filter the dictionary based on similarity score > 0.5
#     filtered_d = {k: v for k, v in average_scores.items() if v[0] > threshold}

#     # Convert the filtered dictionary into a DataFrame
#     df = pd.DataFrame.from_dict(filtered_d, orient='index', columns=['Similarity', 'Class', 'TimePeriod', 'Indexes'])

#     # Ensure the dates are in the correct format (if they are strings)
#     df['Start_Date'], df['End_Date'] = zip(*df['TimePeriod'])
#     df = df.drop('TimePeriod', axis=1)

#     # Convert the string dates to datetime objects if needed
#     df['Start_Date'] = pd.to_datetime(df['Start_Date'], dayfirst=True)
#     df['End_Date'] = pd.to_datetime(df['End_Date'], dayfirst=True)

#     # df[['geometry', 'detection_year','tree_type', 'essence', 'index_reference']] = data.iloc[0][['geometry', 'year','tree_type', 'essence', 'index_reference']]
#     df['index_reference'] = data.iloc[0]['index_reference']
#     return df.sort_values(by='Similarity', ascending=False)

# def wrapper_get_cluster(data: gpd.GeoDataFrame, dtypes_, dcustom_similarity_function, doa, dclass_score, final_weighting_dict, threshold):
#     """Wrapper function for get_cluster"""

#     if len(data) == 1:
#         return pd.DataFrame({'Similarity':[0.], 'Class':['Unknown'], 'Start_Date':[data.iloc[0].start_date], 'End_Date':[data.iloc[0].end_date], 'index_reference':[data.iloc[0].index_reference]})
#     else:
#         return get_cluster(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, doa=doa, dclass_score=dclass_score, final_weighting_dict=final_weighting_dict, threshold=threshold)

# def get_cluster(data : gpd.GeoDataFrame, dcustom_similarity_function : Dict[str, Tuple[Callable, dict, float]],
#                  dtypes_: Dict[str, str], final_weighting_dict : Dict[str, dict],
#                  doa : Dict[str, float], dclass_score : Dict[str, Dict[str, Dict[str, float]]], threshold=0.5) -> gpd.GeoDataFrame:
    
#     """Get the cluster from a GeoDataFrame

#     Args:
#         data (gpd.GeoDataFrame): GeoDataFrame containing the data to be compared
#         dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
#         dtypes_ (Dict[str, str]): dictionary of dataset types
#         doa (Dict[str, float]): dictionary of scaling factors
#         threshold (float, optional): threshold for the similarity score. Defaults to 0.5.
    
#     Returns:
#         gpd.GeoDataFrame: cluster
#     """

#     #build dclass as a dict of list containing the first attribute of the dict of dclass_score
#     dclass = {dataset: {k: list(v.keys()) for k, v in dclass_score[dataset].items()} for dataset in dclass_score}

#     similarity_matrix, _ = compute_similarity_matrix(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, final_weighting_dict=final_weighting_dict)


#     #number of unique class,dataset - maximum number of entries of a final class among the datasets + 1
#     dfinal_class = {}
#     dfinal_class_datasets = defaultdict(list)
#     for i in range(1, len(data)):
#         dataset = data.iloc[i].dataset
#         cls = dclass[dataset][data.iloc[i]['class']]
#         for c in cls:
#             if dataset not in dfinal_class_datasets[c]:
#                 dfinal_class_datasets[c].append(dataset)
#                 dfinal_class[c] = dfinal_class.get(c, 0) + 1

#     uc = len(data.iloc[1:].groupby(['class', 'dataset']))
#     n_clusters = max(1, uc - max(dfinal_class.values()) + 1)

#     # Perform spectral clustering
#     sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
#     labels = sc.fit_predict(similarity_matrix)
#     data['labels'] = labels

#     # Unique cluster labels
#     cluster_labels = data['labels']
#     unique_labels = np.unique(cluster_labels)

#     # Dictionary to hold the sum of scores and the count for each cluster
#     cluster_sums = {label: {'sum': 0, 'count': 0, 'class': []} for label in unique_labels}

#     # Sum scores and counts for each cluster
#     for i, (score, label) in enumerate(zip(similarity_matrix[0][1:], cluster_labels)):
#         ws = score * doa[data['dataset'].iloc[i+1]]
#         cluster_sums[label]['sum'] += ws 
#         cluster_sums[label]['count'] += 1
#         cluster_sums[label]['class'].extend([(c,ws,comp) for c, comp in dclass_score[data['dataset'].iloc[i+1]][data['class'].iloc[i+1]].items()])

#     # Compute average score for each cluster
#     average_scores = {label: (cluster_sums[label]['sum'] / cluster_sums[label]['count']) for label in cluster_sums}
#     timeperiod_group = {}
#     indexes_group = {}
#     data_ = data.copy()
#     data_['start_date'] = pd.to_datetime(data_['start_date'], format='%Y-%m-%d')
#     data_['end_date'] = pd.to_datetime(data_['end_date'], format='%Y-%m-%d')

#     #get date of the median start and end date for each cluster
#     for group in data_['labels'].unique():
#         group_df = data_[data_['labels'] == group]
#         min_start = group_df['start_date'].mean()
#         max_end = group_df['end_date'].mean()
#         timeperiod_group[group] = (min_start, max_end)
#         indexes_group[group] = group_df.index.tolist()



#     for label in average_scores:
#         average_scores[label] = (average_scores[label], get_predominant_class(cluster_sums[label]['class']), timeperiod_group[label], indexes_group[label])

#     # Filter the dictionary based on similarity score > 0.5
#     filtered_d = {k: v for k, v in average_scores.items() if v[0] > threshold}

#     # Convert the filtered dictionary into a DataFrame
#     df = pd.DataFrame.from_dict(filtered_d, orient='index', columns=['Similarity', 'Class', 'TimePeriod', 'Indexes'])

#     # Ensure the dates are in the correct format (if they are strings)
#     df['Start_Date'], df['End_Date'] = zip(*df['TimePeriod'])
#     df = df.drop('TimePeriod', axis=1)

#     # Convert the string dates to datetime objects if needed
#     df['Start_Date'] = pd.to_datetime(df['Start_Date'], dayfirst=True)
#     df['End_Date'] = pd.to_datetime(df['End_Date'], dayfirst=True)

#     return df.sort_values(by='Similarity', ascending=False)

# def wrapper_get_cluster(data: gpd.GeoDataFrame, dtypes_, dcustom_similarity_function, doa, dclass_score, final_weighting_dict, threshold):
#     """Wrapper function for get_cluster"""

#     if len(data) == 1:
#         return pd.DataFrame({'Similarity':[0.], 'Class':['Unknown'], 'Start_Date':[data.iloc[0].start_date], 'End_Date':[data.iloc[0].end_date], 'index_reference':[data.iloc[0].index_reference]})
#     else:
#         return get_cluster(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, doa=doa, dclass_score=dclass_score, final_weighting_dict=final_weighting_dict, threshold=threshold)

from tqdm import tqdm
import networkx as nx
# def build_graph(dataset_loc, dataset_glob,  sindex, spatial_threshold, temporal_threshold, attribution, G=None):
    
#     if G is None:
#         G = nx.Graph()
#         # Construct the similarity matrix using spatial index
#     for event1 in tqdm(dataset_loc.itertuples(index=True)):
#         i = event1.Index
#         # Find nearby events within spatial threshold
#         possible_matches_index = list(sindex.intersection(event1.geometry.buffer(spatial_threshold).bounds))
#         possible_matches = dataset_glob.iloc[possible_matches_index]

#         for event2 in possible_matches.itertuples(index=True):
#             j = event2.Index
#             if i != j and is_temporally_similar(event1, event2, temporal_threshold):
#                 ds = event1.geometry.centroid.distance(event2.geometry.centroid)
#                 dt = abs(event1.centroid_date - event2.centroid_date).days 
#                 weight = np.mean([attribution.final_weighting_dict[event1.dataset][event1._3]['spatial'](ds), attribution.final_weighting_dict[event2.dataset][event2._3]['spatial'](ds), attribution.final_weighting_dict[event1.dataset][event1._3]['temporal'](dt), attribution.final_weighting_dict[event2.dataset][event2._3]['temporal'](dt)])
#                 G.add_edge(i, j, weight = weight)
    
#     return G 

from collections import defaultdict
from typing import Callable, Dict, Tuple
from sklearn.cluster import SpectralClustering, DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def get_temporal_period(cluster : gpd.GeoDataFrame, final_weighting_dict : Dict[str, dict]) -> (tuple, Tuple[pd.Timestamp, pd.Timestamp], float):

    if len(cluster) == 1:
        return None, (cluster['start_date'].iloc[0], cluster['end_date'].iloc[0]), 1.0
    else :
        upper_bound = 2 * 365 + abs(cluster['start_date'].min() - cluster['end_date'].max()).days
        # Determine the overall time range for the cluster
        start_date = cluster['start_date'].min() - pd.Timedelta(days=upper_bound)
        end_date = cluster['end_date'].max() + pd.Timedelta(days=upper_bound)
        time_range = pd.date_range(start=start_date, end=end_date, freq='2W')

        # Initialize an array to hold the sum of profiles
        sum_profiles = np.zeros(len(time_range))

        for event in cluster.itertuples():
            temporal_profile = final_weighting_dict[event.dataset][event._4]['temporal']
            centroid_date = calculate_temporal_centroid(event.start_date, event.end_date)

            # Days difference from each point in the time range to the centroid
            days_from_centroid = (time_range - centroid_date).days

            # Evaluate the temporal profile
            evaluated_profile = np.array([temporal_profile(abs(day)) for day in days_from_centroid])

            # Accumulate the sum
            sum_profiles += evaluated_profile

        # Average the profiles
        average_profile = sum_profiles / len(cluster)

        # Plotting
        # Assuming 'average_profile' is your array and 'time_range' is your corresponding time axis

        # Find the index of the maximum value in the average profile (peak)
        peak_index = np.argmax(average_profile)
        confidence_threshold = np.percentile(average_profile, 90)
        # Initialize variables to store the desired x-axis values
        x_val_left = None
        x_val_right = None

        # Search to the left of the peak
        for i in range(peak_index, -1, -1):  # Iterate backwards from the peak
            if average_profile[i] <= confidence_threshold:
                x_val_left = time_range[i]
                break

        # Search to the right of the peak
        for i in range(peak_index, len(average_profile)):
            if average_profile[i] <= confidence_threshold:
                x_val_right = time_range[i]
                break

        return (time_range, average_profile), (x_val_left, x_val_right), confidence_threshold

from scipy.spatial import distance   
from shapely.geometry import Polygon, MultiPolygon

def get_spatial_polygon(cluster : gpd.GeoDataFrame, final_weighting_dict: Dict[str, dict]) -> Tuple[Tuple[np.ndarray, np.ndarray], MultiPolygon, float]:

    if len(cluster) == 1:
        return None, cluster.geometry.iloc[0], 1.0 
    else : 
        overall_centroid = cluster.geometry.centroid.unary_union.centroid
        grid_size = 30 # Adjust as needed for resolution
        minx, miny, maxx, maxy = cluster.geometry.total_bounds
        width = maxx - minx
        height = maxy - miny

        # Set half_width to be half of the larger dimension of the bounding box
        half_width = max(width, height) 
        x = np.linspace(overall_centroid.x - half_width, overall_centroid.x + half_width, grid_size)
        y = np.linspace(overall_centroid.y - half_width, overall_centroid.y + half_width, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Initialize a 2D array to hold the sum of profiles
        sum_profiles = np.zeros((grid_size, grid_size))

        # Iterate through each event and accumulate its spatial profile
        for event in cluster.itertuples():
            centroid = event.geometry.centroid.coords[0]
            #plot centroid
            spatial_profile_func = final_weighting_dict[event.dataset][event._4]['spatial']

            # Calculate distances from the centroid to each point on the grid
            distances = distance.cdist([(centroid[0], centroid[1])], np.vstack([xx.ravel(), yy.ravel()]).T).reshape(grid_size, grid_size)

            # Evaluate the spatial profile for these distances
            evaluated_profile = np.array([spatial_profile_func(x) for x in np.ravel(distances)]).reshape(grid_size, grid_size)

            # Accumulate the sum
            sum_profiles += evaluated_profile

        # Average the profiles
        average_profile = sum_profiles / len(cluster)

        threshold = np.percentile(average_profile, 90)

        plt.ioff()
        contour = plt.contour(xx, yy, average_profile, levels=[threshold], colors='k', ) #hold='on'
        plt.close()
        polygons = []
        for path in contour.collections[0].get_paths():
            vertices = path.vertices
            polygon = Polygon(vertices)
            polygons.append(polygon)

        # Combine all polygons into a MultiPolygon
        combined_polygon = MultiPolygon(polygons).simplify(10)

        return (xx, yy, average_profile),combined_polygon, threshold
    

def get_cluster(data : gpd.GeoDataFrame, 
                dcustom_similarity_function : Dict[str, Tuple[Callable, dict, float]],
                dtypes_: Dict[str, str], 
                final_weighting_dict : Dict[str, dict],
                doa : Dict[str, float], 
                dclass_score : Dict[str, Dict[str, Dict[str, float]]], 
                weights=[1,1,1,1], 
                method='DBSCAN') -> Tuple[gpd.GeoDataFrame, Tuple[np.ndarray, np.ndarray]]:
    
    """Get the cluster from a GeoDataFrame

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the data to be compared
        dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
        dtypes_ (Dict[str, str]): dictionary of dataset types
        doa (Dict[str, float]): dictionary of scaling factors
        threshold (float, optional): threshold for the similarity score. Defaults to 0.5.
    
    Returns:
        cluster (gpd.GeoDataFrame): GeoDataFrame containing the clusters
        similarity_matrix (np.ndarray): similarity matrix
        labels (np.ndarray): labels of the cluster
    """

    #build dclass as a dict of list containing the first attribute of the dict of dclass_sco
    similarity_matrix, _ = compute_similarity_matrix(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, final_weighting_dict=final_weighting_dict, weights=weights)

    # Perform spectral clustering
    # DBSCAN 
    if method == 'DBSCAN':
        distance_matrix = 1 - similarity_matrix
        dbscan = DBSCAN(eps=0.23, min_samples=2, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix) #distance matrix
    elif method == 'SpectralClustering':
        spectral = SpectralClustering(affinity='precomputed', assign_labels='cluster_qr', random_state=0)
        labels = spectral.fit_predict(similarity_matrix) #similarity matrix 

    # Unique cluster labels
    data['labels'] = labels
    cluster_labels = data['labels']
    unique_labels = np.unique(cluster_labels)

    # Dictionary to hold the sum of scores and the count for each cluster
    cluster_sums = {label: {'sum': 0, 'count': 0, 'class': []} for label in unique_labels}

    # Sum scores and counts for each cluster
    for i, (score, label) in enumerate(zip(similarity_matrix[0], cluster_labels)):
        ws = score * doa[data['dataset'].iloc[i]]
        cluster_sums[label]['sum'] += ws 
        cluster_sums[label]['count'] += 1
        cluster_sums[label]['class'].extend([(c,ws,comp) for c, comp in dclass_score[data['dataset'].iloc[i]][data['class'].iloc[i]].items()])

    # Compute average score for each cluster
    average_scores = {label: (cluster_sums[label]['sum'] / cluster_sums[label]['count']) for label in cluster_sums}

    timeperiod_group = {}
    indexes_group = {}
    threshold_group = {}
    polygons_group = {}
    data_ = data.copy()
    data_['start_date'] = pd.to_datetime(data_['start_date'], format='%Y-%m-%d')
    data_['end_date'] = pd.to_datetime(data_['end_date'], format='%Y-%m-%d')

    #get date of the median start and end date for each cluster
    for group in data_['labels'].unique():
        group_df = data_[data_['labels'] == group]
        _, (start, end), temporal_threshold = get_temporal_period(group_df, final_weighting_dict)
        _, polygon, spatial_threshold = get_spatial_polygon(group_df, final_weighting_dict)
        threshold_group[group] = (temporal_threshold, spatial_threshold)
        polygons_group[group] = polygon
        # min_start = group_df['start_date'].mean()
        # max_end = group_df['end_date'].mean()
        timeperiod_group[group] = (start, end)
        indexes_group[group] = group_df.index.tolist()



    for label in average_scores:
        average_scores[label] = (average_scores[label], get_predominant_class(cluster_sums[label]['class']), timeperiod_group[label], indexes_group[label], threshold_group[label], polygons_group[label])

    # Convert the filtered dictionary into a DataFrame
    df = pd.DataFrame.from_dict(average_scores, orient='index', columns=['IntraSimilarity', 'Class', 'TimePeriod', 'Indexes', 'Threshold', 'geometry'])

    # Ensure the dates are in the correct format (if they are strings)
    df['Start_Date'], df['End_Date'] = zip(*df['TimePeriod'])
    df['Temporal_threshold'], df['Spatial_threshold'] = zip(*df['Threshold'])
    df = df.drop(['TimePeriod', 'Threshold'], axis=1)

    # Convert the string dates to datetime objects if needed
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], dayfirst=True)
    df['End_Date'] = pd.to_datetime(df['End_Date'], dayfirst=True)

    return gpd.GeoDataFrame(df.sort_values(by='IntraSimilarity', ascending=False), geometry='geometry', crs=data.crs), (similarity_matrix, labels)


def wrapper_get_cluster(data: gpd.GeoDataFrame, dtypes_, dcustom_similarity_function, doa, dclass_score, final_weighting_dict, threshold, weights):
    """Wrapper function for get_cluster"""

    if len(data) == 1:
        return pd.DataFrame({'Similarity':[0.], 'Class':['Unknown'], 'Start_Date':[data.iloc[0].start_date], 'End_Date':[data.iloc[0].end_date], 'index_reference':[data.iloc[0].index_reference]}), (np.array(1), [1])
    else:
        return get_cluster(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, doa=doa, dclass_score=dclass_score, final_weighting_dict=final_weighting_dict, threshold=threshold, weights=weights) 


#NOISE
import geopandas as gpd
from shapely.geometry import Polygon
from shapely.affinity import translate

def translate_polygon(df, polygon_column, sigma=100):
    def shift_polygon(polygon):
        # Random shifts in meters
        x_shift = np.random.normal(0, sigma)  # Longitude shift
        y_shift = np.random.normal(0, sigma)  # Latitude shift
        return translate(polygon, xoff=x_shift, yoff=y_shift)

    df[polygon_column] = df[polygon_column].apply(shift_polygon)
    return df

import pandas as pd
import numpy as np
from datetime import timedelta

def translate_time(df, start_date_column, end_date_column, sigma=90):
    # Applying a Gaussian disturbance with mean = 0 and std = 90 days
    disturbance_delta = np.random.normal(0, sigma, size=len(df))
    
    df[start_date_column] = df[start_date_column] + pd.to_timedelta(disturbance_delta, unit='d')
    df[end_date_column] = df[end_date_column] + pd.to_timedelta(disturbance_delta, unit='d')

    #format dates as just year, month and day
    df[start_date_column] = df[start_date_column].dt.strftime('%Y-%m-%d')
    df[end_date_column] = df[end_date_column].dt.strftime('%Y-%m-%d')
    return df

#V2
from thefuzz import fuzz 
#import namedtuple
from collections import namedtuple
import numpy as np 
def build_temporal_matrix_v2(data, final_weighting_dict):

    #temporal
    start_date_matrix = data['start_date'].values.astype('datetime64[D]').reshape(-1, 1)
    day_diff_matrix_sd = np.abs((start_date_matrix - start_date_matrix.T) / np.timedelta64(1, 'D'))
  
    end_date_matrix = data['end_date'].values.astype('datetime64[D]').reshape(-1, 1)
    day_diff_matrix_ed = np.abs((end_date_matrix - end_date_matrix.T) / np.timedelta64(1, 'D'))

    m = (day_diff_matrix_sd + day_diff_matrix_ed)/2

    n = len(data)
    rows = list(data.itertuples(index=False))
    for i in range(n):
        for j in range(i + 1, n):
            weight = (final_weighting_dict[rows[i].dataset][rows[i].cause]['temporal'](m[i][j]) + final_weighting_dict[rows[j].dataset][rows[j].cause]['temporal'](m[j][i]))/2
            m[i][j] = weight
            m[j][i] = weight

    #fill diagonal with 1
    np.fill_diagonal(m, 1)

    # return  np.array(list(map(lambda row: list(map(f, row)), m)))
    return m 

def compute_tree_coherence_v2(row1 : namedtuple, row2 : namedtuple, filter_cause : dict) -> float:
    """Compute the tree coherence between two rows

    Args:
        row1 (namedtuple): first row
        row2 (namedtuple): second row

    Returns:
        float: tree coherence
    """
    if row1.cause in filter_cause and row2.cause in filter_cause:
        return 1.
    
    for essence1 in row1.essence.split(','):
        for essence2 in row2.essence.split(','):
            if fuzz.token_set_ratio(essence1.lower(), essence2.lower()) > 80:
                return 1
        
    if row1.tree_type.lower() == row2.tree_type.lower():
        return 0.75 
    
    if row1.tree_type.lower() == 'mixed' or row2.tree_type.lower() == 'mixed':
        return 0.5

    return 0.25 

from typing import Dict
def compute_class_similarity_v2(row1 : namedtuple, row2 : namedtuple, dclass_score : Dict[str, Dict[str, Dict[str, float]]], attr_class = 'class' ) -> float:
    """Compute the similarity between two classes

    Args:
        row1 (namedtuple): first row
        row2 (namedtuple): second row
        dclass_score (Dict[str, Dict[str, Dict[str, float]]]): dictionary containing the class scores
    
    Returns:
        float: similarity score
    """

    # If both classes are from the same dataset and have the same class, return 1.0
    if row1.dataset == row2.dataset and row1.cause == row2.cause:
        return 1.0

    # Get the score dictionaries for the two classes
    scores1 = dclass_score.get(row1.dataset, {}).get(row1.cause, {})
    scores2 = dclass_score.get(row2.dataset, {}).get(row2.cause, {})
    
    # Find common final classes between the two classes
    common_final_classes = set(scores1.keys()).intersection(set(scores2.keys()))
    
    # Calculate the sum of the means of the scores for each common final class
    similarity_score = sum((scores1[final_class] + scores2[final_class]) / 2 for final_class in common_final_classes)
    
    return similarity_score

def compute_spatial_distance_v2(row1 : namedtuple, row2 : namedtuple, final_weighting_dict: Dict[str, dict]) -> float:
    """Compute the spatial distance between two rows 

    Args:
        row1 (namedtuple): first row
        row2 (namedtuple): second row

    Returns:
        float: spatial distance
    """
    distance = row1.geometry.distance(row2.geometry) #not the centroid
    # Convert distance to similarity
    weight = (final_weighting_dict[row1.dataset][row1.cause]['spatial'](distance) + final_weighting_dict[row2.dataset][row2.cause]['spatial'](distance))/2
    return weight

from typing import Callable
def build_custom_matrix_v2(df : pd.DataFrame, custom_similarity_function : Callable, kwargs: dict) -> np.ndarray:
    """Build a similarity matrix using a custom similarity function

    Args:
        df (pd.DataFrame): DataFrame containing the data to be compared
        custom_similarity_function (Callable): custom similarity function
        kwargs (dict): optional arguments for the custom similarity function

    Returns:
        np.ndarray: similarity matrix
    """

    n = len(df)
    # Initialize the similarity matrix with zeros
    similarity_matrix = np.zeros((n, n))

    # Convert DataFrame to a list of tuples for more efficient row access
    rows = list(df.itertuples(index=False))
    
    # Iterate over each pair of rows
    for i in range(n):
        for j in range(i+1, n):  # Use range(i, n) to avoid redundant computations
            # Compute the tree coherence for each pair of rows
            similarity = custom_similarity_function(rows[i], rows[j], **kwargs)
            # Fill in the matrix, it's symmetric so we can do both i,j and j,i
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    np.fill_diagonal(similarity_matrix, 1)

    return similarity_matrix

from typing import Dict, Tuple, Callable, List


def get_matrices_v2(data : pd.DataFrame, dtypes_: Dict[str, str], dcustom_similarity_function : Dict[str, Tuple[Callable, dict]], final_weighting_dict : Dict[str, dict]) \
    -> List[np.ndarray]:
    """Compute all the different similarity matrices and combine them into a single matrix

    Args:
        data (pd.DataFrame): DataFrame containing the data to be compared
        dtypes_ (Dict[str, str]): dictionary of dataset types
        dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): Dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
        final_weighting_dict (Dict[str, dict]): dictionary containing the final weighting functions for each dataset and class

    Returns:
        List[np.ndarray]: list of similarity matrices : temporal, spatial, custom
    """

    #spatial 
    data = data.fillna('None')
    matrices = {}

    #temporal
    matrices['temporal'] = np.nan_to_num(build_temporal_matrix_v2(data, final_weighting_dict=final_weighting_dict), nan=0).clip(0,1)

    #CUSTOM  SIMILARITY FACTORS
    for name, (custom_function, kwargs) in dcustom_similarity_function.items():
        matrices[name] = np.nan_to_num(build_custom_matrix_v2(data, custom_function, kwargs), nan=0).clip(0,1)

    return list(matrices.values()) 

def get_matrices(data : gpd.GeoDataFrame, dtypes_: Dict[str, str], dcustom_similarity_function : Dict[str, Tuple[Callable, dict]], final_weighting_dict : Dict[str, dict]) \
    -> List[np.ndarray]:
    """Compute all the different similarity matrices and combine them into a single matrix

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the data to be compared
        dtypes_ (Dict[str, str]): dictionary of dataset types
        dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): Dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
        final_weighting_dict (Dict[str, dict]): dictionary containing the final weighting functions for each dataset and class

    Returns:
        List[np.ndarray]: list of similarity matrices
    """    
    
    data = data.fillna('None')
    matrices = {}
    #temporal
    matrices['temporal'] = np.nan_to_num(build_temporal_matrix(data, final_weighting_dict=final_weighting_dict), nan=0).clip(0,1)

    #spatial 
    matrices['spatial'] = np.nan_to_num(build_spatial_matrix(data, dtypes_=dtypes_, final_weighting_dict=final_weighting_dict), nan=0).clip(0,1)
    #CUSTOM  SIMILARITY FACTORS
    for name, (custom_function, kwargs) in dcustom_similarity_function.items():
        matrices[name] = np.nan_to_num(build_custom_matrix(data, custom_function, kwargs), nan=0).clip(0,1)

    return list(matrices.values()) 

def build_similarity_v2(matrices : List[np.ndarray], weights : List[float]) -> np.ndarray:
    """Combine the different similarity matrices into a single matrix

    Args:
        matrices (List[np.ndarray]): list of similarity matrices
        weights (List[float]): list of weights for each matrix

    Returns:
        np.ndarray: combined similarity matrix
    """

    return np.average(matrices, axis=0, weights=weights)

from typing import Dict, Tuple
from sklearn.cluster import SpectralClustering, DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def get_temporal_period_v2(cluster : gpd.GeoDataFrame, final_weighting_dict : Dict[str, dict]) -> (tuple, Tuple[pd.Timestamp, pd.Timestamp], float):
    proba = cluster['probabilities'].sum()
    if proba == 0:
        return None, (cluster['start_date'].min(), cluster['end_date'].max()), 0.
    # elif len(cluster) == 1:
    #     return None, (cluster['start_date'].iloc[0], cluster['end_date'].iloc[0]), 1.0
    else :
        upper_bound = 2 * 365 + abs(cluster['start_date'].min() - cluster['end_date'].max()).days
        # Determine the overall time range for the cluster
        start_date = cluster['start_date'].min() - pd.Timedelta(days=upper_bound)
        end_date = cluster['end_date'].max() + pd.Timedelta(days=upper_bound)
        time_range = pd.date_range(start=start_date, end=end_date, freq='2W')

        # Initialize an array to hold the sum of profiles
        sum_profiles = np.zeros(len(time_range))
        total_probability = cluster['probabilities'].sum()  # Total probability for normalization
        #total probability set to 1 if it is 0
        if total_probability == 0:
            total_probability = 1

        for event in cluster.itertuples():
            temporal_profile = final_weighting_dict[event.dataset][event.cause]['temporal'] 
            centroid_date = calculate_temporal_centroid(event.start_date, event.end_date)

            # Days difference from each point in the time range to the centroid
            days_from_centroid = (time_range - centroid_date).days

            # Evaluate the temporal profile
            evaluated_profile = np.array([temporal_profile(abs(day)) for day in days_from_centroid])

            # Accumulate the sum
            sum_profiles += evaluated_profile * event.probabilities

        # Average the profiles
        average_profile = sum_profiles / total_probability

        # Plotting
        # Assuming 'average_profile' is your array and 'time_range' is your corresponding time axis

        # Find the index of the maximum value in the average profile (peak)
        peak_index = np.argmax(average_profile)
        confidence_threshold = np.percentile(average_profile, 90)
        # Initialize variables to store the desired x-axis values
        x_val_left = None
        x_val_right = None

        # Search to the left of the peak
        for i in range(peak_index, -1, -1):  # Iterate backwards from the peak
            if average_profile[i] <= confidence_threshold:
                x_val_left = time_range[i]
                break

        # Search to the right of the peak
        for i in range(peak_index, len(average_profile)):
            if average_profile[i] <= confidence_threshold:
                x_val_right = time_range[i]
                break

        return (time_range, average_profile), (x_val_left, x_val_right), confidence_threshold

from scipy.spatial import distance   
from shapely.geometry import Polygon, MultiPolygon

def get_spatial_polygon_v2(cluster : gpd.GeoDataFrame, final_weighting_dict: Dict[str, dict]) -> Tuple[Tuple[np.ndarray, np.ndarray], MultiPolygon, float]:

    proba = cluster['probabilities'].sum()
    if proba == 0:
        return None, cluster.geometry.unary_union, 0. 
    else :  #i am removing this part for uniformity of the map 
        overall_centroid = cluster.geometry.centroid.unary_union.centroid
        grid_size = 30 # Adjust as needed for resolution
        minx, miny, maxx, maxy = cluster.geometry.total_bounds
        width = maxx - minx
        height = maxy - miny

        # Set half_width to be half of the larger dimension of the bounding box
        half_width = max(width, height) 
        x = np.linspace(overall_centroid.x - half_width, overall_centroid.x + half_width, grid_size)
        y = np.linspace(overall_centroid.y - half_width, overall_centroid.y + half_width, grid_size)
        xx, yy = np.meshgrid(x, y)

        # Initialize a 2D array to hold the sum of profiles
        sum_profiles = np.zeros((grid_size, grid_size))
        total_probability = cluster['probabilities'].sum()  # Sum of all probabilities for normalization
        if total_probability == 0:
            total_probability = 1

        # Iterate through each event and accumulate its spatial profile
        for event in cluster.itertuples():
            centroid = event.geometry.centroid.coords[0]
            #plot centroid
            spatial_profile_func = final_weighting_dict[event.dataset][event.cause]['spatial']

            # Calculate distances from the centroid to each point on the grid
            distances = distance.cdist([(centroid[0], centroid[1])], np.vstack([xx.ravel(), yy.ravel()]).T).reshape(grid_size, grid_size)

            # Evaluate the spatial profile for these distances
            evaluated_profile = np.array([spatial_profile_func(x) for x in np.ravel(distances)]).reshape(grid_size, grid_size)

            # Accumulate the sum
            sum_profiles += evaluated_profile * event.probabilities 

        # Average the profiles
        average_profile = sum_profiles / total_probability

        threshold = np.percentile(average_profile, 90)

        plt.ioff()
        contour = plt.contour(xx, yy, average_profile, levels=[threshold], colors='k') #, hold='on'
        plt.close()
        polygons = []
        for path in contour.collections[0].get_paths():
            vertices = path.vertices
            polygon = Polygon(vertices)
            polygons.append(polygon)

        # Combine all polygons into a MultiPolygon
        combined_polygon = MultiPolygon(polygons).simplify(10)

        return (xx, yy, average_profile),combined_polygon, threshold   


from sklearn.cluster import SpectralClustering, DBSCAN, HDBSCAN
from typing import Dict, Tuple, Any

def get_cluster_v2(data : gpd.GeoDataFrame, 
                similarity_matrix : np.ndarray, 
                final_weighting_dict : Dict[str, dict],
                doa : Dict[str, float], 
                dclass_score : Dict[str, Dict[str, Dict[str, float]]], 
                method : str = 'SpectralClustering',
                method_kwargs : Dict[str, Any] = {'eps': 0.23, 'min_samples': 2}
                ) -> Tuple[gpd.GeoDataFrame, Tuple[np.ndarray, np.ndarray]]:
    
    """Get the cluster from a GeoDataFrame

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the data to be clustered
        similarity_matrix (np.ndarray): similarity matrix
        final_weighting_dict (Dict[str, dict]): dictionary containing the final weighting functions for each dataset and class
        doa (Dict[str, float]): dictionary containing the degree of attribution for each dataset
        dclass_score (Dict[str, Dict[str, Dict[str, float]]]): dictionary containing the class scores
        method (str): clustering method
    Returns:
        cluster (gpd.GeoDataFrame): GeoDataFrame containing the clusters
        similarity_matrix (np.ndarray): similarity matrix
        labels (np.ndarray): labels of the cluster
    """
    
    if method == 'DBSCAN':
        distance_matrix = 1 - similarity_matrix
        dbscan = DBSCAN(metric='precomputed', **method_kwargs)
        labels = dbscan.fit_predict(distance_matrix) #distance matrix
        probabilities = 1
    elif method == 'HDBSCAN':
        distance_matrix = 1 - similarity_matrix
        hdbscan = HDBSCAN(metric='precomputed', **method_kwargs) #min_cluster_size=2
        labels = hdbscan.fit_predict(distance_matrix)
        probabilities = hdbscan.probabilities_
        probabilities = np.where(probabilities == np.inf, 0, probabilities)
    elif method == 'SpectralClustering':
        spectral = SpectralClustering(affinity='precomputed', assign_labels='cluster_qr', random_state=0)
        labels = spectral.fit_predict(similarity_matrix) #similarity matrix 
        probabilities = 1

    # Unique cluster labels
    data['labels'] = labels
    data['probabilities'] = probabilities
    cluster_labels = data['labels']
    unique_labels = np.unique(cluster_labels)

    # Dictionary to hold the sum of scores and the count for each cluster
    cluster_sums = {label: {'sum': 0, 'count': 0, 'class': []} for label in unique_labels}

    # Sum scores and counts for each cluster
    rows = list(data.itertuples(index=False))
    for i, (score, label, probability) in enumerate(zip(similarity_matrix[0], cluster_labels, data['probabilities'])):
        ws = score * doa[rows[i].dataset] * probability
        cluster_sums[label]['sum'] += ws 
        cluster_sums[label]['count'] += 1
        cluster_sums[label]['class'].extend([(c,ws,comp) for c, comp in dclass_score[rows[i].dataset][rows[i].cause].items()])

    # Compute average score for each cluster
    average_scores = {label: (cluster_sums[label]['sum'] / cluster_sums[label]['count']) for label in cluster_sums}

    timeperiod_group = {}
    indexes_group = {}
    threshold_group = {}
    polygons_group = {}
    probabilities_group = {}
    data_ = data.copy()
    data_['start_date'] = pd.to_datetime(data_['start_date'], format='%Y-%m-%d')
    data_['end_date'] = pd.to_datetime(data_['end_date'], format='%Y-%m-%d')

    #get date of the median start and end date for each cluster
    for group in data_['labels'].unique():
        group_df = data_[data_['labels'] == group]
        #identify probabilities for each label
        probabilities_group[group] = group_df['probabilities'].values
        _, (start, end), temporal_threshold = get_temporal_period_v2(group_df, final_weighting_dict)
        _, polygon, spatial_threshold = get_spatial_polygon_v2(group_df, final_weighting_dict)
        threshold_group[group] = (temporal_threshold, spatial_threshold)
        polygons_group[group] = polygon
        timeperiod_group[group] = (start, end)
        indexes_group[group] = group_df.index.tolist()

    for label in average_scores:
        average_scores[label] = (average_scores[label], get_predominant_class(cluster_sums[label]['class']), timeperiod_group[label], indexes_group[label], threshold_group[label], polygons_group[label])

    # Convert the filtered dictionary into a DataFrame
    df = pd.DataFrame.from_dict(average_scores, orient='index', columns=['IntraSimilarity', 'Class', 'TimePeriod', 'Indexes', 'Threshold', 'geometry'])
    df['probabilities'] = probabilities_group

    # Ensure the dates are in the correct format (if they are strings)
    df['Start_Date'], df['End_Date'] = zip(*df['TimePeriod'])
    df['Temporal_threshold'], df['Spatial_threshold'] = zip(*df['Threshold'])
    df = df.drop(['TimePeriod', 'Threshold'], axis=1)

    # Convert the string dates to datetime objects if needed
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], dayfirst=True)
    df['End_Date'] = pd.to_datetime(df['End_Date'], dayfirst=True)

    return gpd.GeoDataFrame(df.sort_values(by='IntraSimilarity', ascending=False), geometry='geometry', crs=data.crs), similarity_matrix, labels
