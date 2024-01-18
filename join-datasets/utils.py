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
def compute_tree_coherence(row1 : pd.Series, row2 : pd.Series) -> float:
    """Compute the tree coherence between two rows

    Args:
        row1 (pd.Series): first row
        row2 (pd.Series): second row

    Returns:
        float: tree coherence
    """

    for essence1 in row1['essence'].split(','):
        for essence2 in row2['essence'].split(','):
            if fuzz.token_set_ratio(essence1.lower(), essence2.lower()) > 80:
                return 1
        
    if row1['tree_type'].lower() == row2['tree_type'].lower():
        return 0.75 
    
    if row1['tree_type'].lower() == 'mixed' or row2['tree_type'].lower() == 'mixed':
        return 0.5

    return 0.25 


def compute_class_similarity(row1 : pd.Series, row2 : pd.Series, dclass_score : Dict[str, Dict[str, Dict[str, float]]] ) -> float:
    """Compute the similarity between two classes

    Args:
        row1 (pd.Series): first row
        row2 (pd.Series): second row
        dclass_score (Dict[str, Dict[str, Dict[str, float]]]): dictionary containing the class scores
    
    Returns:
        float: similarity score
    """

    # If both classes are from the same dataset and have the same class, return 1.0
    if row1['dataset'] == row2['dataset'] and row1['class'] == row2['class']:
        return 1.0

    # Get the score dictionaries for the two classes
    scores1 = dclass_score.get(row1['dataset'], {}).get(row1['class'], {})
    scores2 = dclass_score.get(row2['dataset'], {}).get(row2['class'], {})
    
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

def get_cluster(data : gpd.GeoDataFrame, dcustom_similarity_function : Dict[str, Tuple[Callable, dict, float]],
                 dtypes_: Dict[str, str], final_weighting_dict : Dict[str, dict],
                 doa : Dict[str, float], dclass_score : Dict[str, Dict[str, Dict[str, float]]], threshold=0.5) -> gpd.GeoDataFrame:
    
    """Get the cluster from a GeoDataFrame

    Args:
        data (gpd.GeoDataFrame): GeoDataFrame containing the data to be compared
        dcustom_similarity_function (Dict[str, Tuple[Callable, dict, float]]): dictionary containing the custom similarity functions, the optional arguments and the weight associated with each function
        dtypes_ (Dict[str, str]): dictionary of dataset types
        doa (Dict[str, float]): dictionary of scaling factors
        threshold (float, optional): threshold for the similarity score. Defaults to 0.5.
    
    Returns:
        gpd.GeoDataFrame: cluster
    """

    #build dclass as a dict of list containing the first attribute of the dict of dclass_score
    dclass = {dataset: {k: list(v.keys()) for k, v in dclass_score[dataset].items()} for dataset in dclass_score}

    similarity_matrix, _ = compute_similarity_matrix(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, final_weighting_dict=final_weighting_dict)


    #number of unique class,dataset - maximum number of entries of a final class among the datasets + 1
    dfinal_class = {}
    dfinal_class_datasets = defaultdict(list)
    for i in range(1, len(data)):
        dataset = data.iloc[i].dataset
        cls = dclass[dataset][data.iloc[i]['class']]
        for c in cls:
            if dataset not in dfinal_class_datasets[c]:
                dfinal_class_datasets[c].append(dataset)
                dfinal_class[c] = dfinal_class.get(c, 0) + 1

    uc = len(data.iloc[1:].groupby(['class', 'dataset']))
    n_clusters = max(1, uc - max(dfinal_class.values()) + 1)

    # Perform spectral clustering
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans')
    labels = sc.fit_predict(similarity_matrix)
    data['labels'] = labels

    # Unique cluster labels
    cluster_labels = data['labels']
    unique_labels = np.unique(cluster_labels)

    # Dictionary to hold the sum of scores and the count for each cluster
    cluster_sums = {label: {'sum': 0, 'count': 0, 'class': []} for label in unique_labels}

    # Sum scores and counts for each cluster
    for i, (score, label) in enumerate(zip(similarity_matrix[0][1:], cluster_labels)):
        ws = score * doa[data['dataset'].iloc[i+1]]
        cluster_sums[label]['sum'] += ws 
        cluster_sums[label]['count'] += 1
        cluster_sums[label]['class'].extend([(c,ws,comp) for c, comp in dclass_score[data['dataset'].iloc[i+1]][data['class'].iloc[i+1]].items()])

    # Compute average score for each cluster
    average_scores = {label: (cluster_sums[label]['sum'] / cluster_sums[label]['count']) for label in cluster_sums}
    timeperiod_group = {}
    indexes_group = {}
    data_ = data.copy()
    data_['start_date'] = pd.to_datetime(data_['start_date'], format='%Y-%m-%d')
    data_['end_date'] = pd.to_datetime(data_['end_date'], format='%Y-%m-%d')

    #get date of the median start and end date for each cluster
    for group in data_['labels'].unique():
        group_df = data_[data_['labels'] == group]
        min_start = group_df['start_date'].mean()
        max_end = group_df['end_date'].mean()
        timeperiod_group[group] = (min_start, max_end)
        indexes_group[group] = group_df.index.tolist()



    for label in average_scores:
        average_scores[label] = (average_scores[label], get_predominant_class(cluster_sums[label]['class']), timeperiod_group[label], indexes_group[label])

    # Filter the dictionary based on similarity score > 0.5
    filtered_d = {k: v for k, v in average_scores.items() if v[0] > threshold}

    # Convert the filtered dictionary into a DataFrame
    df = pd.DataFrame.from_dict(filtered_d, orient='index', columns=['Similarity', 'Class', 'TimePeriod', 'Indexes'])

    # Ensure the dates are in the correct format (if they are strings)
    df['Start_Date'], df['End_Date'] = zip(*df['TimePeriod'])
    df = df.drop('TimePeriod', axis=1)

    # Convert the string dates to datetime objects if needed
    df['Start_Date'] = pd.to_datetime(df['Start_Date'], dayfirst=True)
    df['End_Date'] = pd.to_datetime(df['End_Date'], dayfirst=True)

    return df.sort_values(by='Similarity', ascending=False)

def wrapper_get_cluster(data: gpd.GeoDataFrame, dtypes_, dcustom_similarity_function, doa, dclass_score, final_weighting_dict, threshold):
    """Wrapper function for get_cluster"""

    if len(data) == 1:
        return pd.DataFrame({'Similarity':[0.], 'Class':['Unknown'], 'Start_Date':[data.iloc[0].start_date], 'End_Date':[data.iloc[0].end_date], 'index_reference':[data.iloc[0].index_reference]})
    else:
        return get_cluster(data, dtypes_=dtypes_, dcustom_similarity_function=dcustom_similarity_function, doa=doa, dclass_score=dclass_score, final_weighting_dict=final_weighting_dict, threshold=threshold)
