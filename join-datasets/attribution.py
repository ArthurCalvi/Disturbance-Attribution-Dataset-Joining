__version__ = '0.2'

import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
import geopandas as gpd
import dask_geopandas as dgpd
import warnings
from collections import defaultdict
import pandas as pd
import dask.dataframe as dd
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from itertools import islice
from tqdm import tqdm
import numpy as np 
import os 
    
from utils import get_spatial_join, get_subset_dataset, is_dataset_valid, classify_source_type, wrapper_get_cluster, modify_dsbuffer, temporal_weight, spatial_weight, calculate_d_star, get_combined_weighting, dfunction


class Attribution:
    
    version = __version__ #version of the package
    
    def __init__(self, ddataset: Dict[str, gpd.GeoDataFrame], reference: gpd.GeoDataFrame, 
    doa: Dict[str, float], temporal_buffer: int, dsbuffer: Dict[str, int], dtypes: Dict[str, int],
    dclass_score: Dict[str, Dict[str, float]],
    ddataset_profile: Dict[str, dict], ddisturbance_profile: Dict[str, dict],
    granularity: int = 10, 
    start_year: int = 2015, end_year: int = 2020):
        
        """ Attribution class

        Args:
            ddataset (Dict[str, gpd.GeoDataFrame]): dict of datasets : {'name': gpd.GeoDataFrame}
            reference (gpd.GeoDataFrame): reference dataset
            doa (Dict[str, float]): dict of overall accuracy for each dataset : {'name': accuracy}
            temporal_buffer (int): temporal buffer in years
            dsbuffer (Dict[str, int]): spatial buffer in meters for each dataset : {'name': buffer}
            dtypes (Dict[str, int]): dict of data types : {'name': type} with type in ['point', 'polygon']
            dclass_score (Dict[str, Dict[str, float]]): dict of class score for each dataset : {'name': {'class': score}}, score in [0, 1], sum(score) = 1, it reflects the proportion of each class in the dataset
            granularity (int, optional): granularity in meters. Defaults to 10.
            start_year (int, optional): start year. Defaults to 2015.
            end_year (int, optional): end year. Defaults to 2020.

        Raises:
            AssertionError: if the keys of ddataset, doa, dsbuffer and dtypes are not the same
            AssertionError: if the crs of each dataset is not the same as the reference dataset
        """

        self.start_year = datetime(start_year, 1, 1)
        self.end_year = datetime(end_year, 12, 31)
        self.ddataset_profile = ddataset_profile
        self.ddisturbance_profile = ddisturbance_profile
        self.doa = doa
        self.dclass_score = dclass_score
        self.temporal_buffer = temporal_buffer #years
        self.granularity = granularity
        self.dsbuffer = modify_dsbuffer(dsbuffer, self.granularity * 1e3) #ensure that the buffers are below the granularity
        self.dtypes = dtypes
        self.mandatory_columns = ['start_date', 'end_date', 'class', 'dataset', 'year']
        self.ddataset = ddataset
        self.dtypes_ = {}
        self.dp95 = {}
        for name, dataset in self.ddataset.items():
            self.ddataset[name] = get_subset_dataset(is_dataset_valid(name, dataset), self.start_year - timedelta(days = 365.25 * self.temporal_buffer), self.end_year + timedelta(days = 365.25 * self.temporal_buffer))
            t, p95 = classify_source_type(dataset, area_threshold= np.pi * self.granularity ** 2, name=name) #(model -> circle) in km2
            self.dtypes_[name] = t
            self.dp95[name] = p95
        t, _ = classify_source_type(reference, area_threshold= np.pi * self.granularity ** 2) #(model -> circle) in km2
        self.dtypes_['reference'] = t
        
        self.reference = get_subset_dataset(is_dataset_valid('reference', reference, ref=True), self.start_year - timedelta(days = 365.25 * self.temporal_buffer), self.end_year + timedelta(days = 365.25 * self.temporal_buffer))
        self.optional_columns = self.get_optional_columns()

        assert self.ddataset.keys() == self.doa.keys(), "doa keys are not the same as ddataset keys"
        assert self.ddataset.keys() == self.dsbuffer.keys(), "dsbuffer keys are not the same as ddataset keys"
        assert self.ddataset.keys() == self.dtypes.keys(), "dtypes keys are not the same as ddataset keys"
        assert all([self.ddataset[name].crs == self.reference.crs for name in self.ddataset.keys()]), "not all the geodataframe have the same crs"

        # Creating the dictionary to store callable functions
        final_weighting_dict = {}

        for dataset, class_dict in self.dclass_score.items():
            
            final_weighting_dict[dataset] = {}
            for class_name, scores_dict in class_dict.items():
                final_weighting_dict[dataset][class_name] = {
                    'spatial': get_combined_weighting(  self.ddataset_profile[dataset]['spatial'], self.ddisturbance_profile, scores_dict, 'spatial'),
                    'temporal': get_combined_weighting( self.ddataset_profile[dataset]['temporal'], self.ddisturbance_profile, scores_dict, 'temporal')
                }

        self.final_weighting_dict = final_weighting_dict

    def plot_weighting_functions(self):

        """ Plot the weighting functions for each dataset
        """
        x_spatial = np.linspace(0, 7000, 1000)
        x_temporal = np.linspace(0, 5*365, 1000)

        #change font to times new roman
        plt.rcParams["font.family"] = "Times New Roman"

        #plot functions of ddataset profile 
        fig, axes = plt.subplots(2, len(self.ddataset_profile), figsize=(14, 5), sharey=True)
        for i, (dataset, profile_dict) in enumerate(self.ddataset_profile.items()):
            ax = axes[:,i]
            if i == 0:
                ax[0].set_ylabel('Spatial Weighting [1]')
                ax[0].set_xlabel('Distance (m)')
                ax[1].set_ylabel('Temporal Weighting [1]')
                ax[1].set_xlabel('Time (days)')
            ax[0].plot(x_spatial, [dfunction[profile_dict['spatial'][0]](t, **profile_dict['spatial'][1]) for t in x_spatial], label=dataset, alpha=0.75)
            ax[1].plot(x_temporal, [dfunction[profile_dict['temporal'][0]](t, **profile_dict['temporal'][1]) for t in x_temporal], alpha=0.75)
            ax[0].set_title(dataset)

            for a in ax:
                a.grid()
                #remove top and right spines
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

        #plot functions of ddisturbance profile
        fig, axes = plt.subplots(2, len(self.ddisturbance_profile), figsize=(12, 5), sharey=True)

        x_spatial = np.linspace(0, 7000, 1000)
        x_temporal = np.linspace(0, 5*365, 1000)

        for i, (disturbance_class, profile_dict) in enumerate(self.ddisturbance_profile.items()):
            ax = axes[:,i]
            if i == 0:
                ax[0].set_ylabel('Spatial Weighting [1]')
                ax[0].set_xlabel('Distance (m)')
                ax[1].set_ylabel('Temporal Weighting [1]')
                ax[1].set_xlabel('Time (days)')
            ax[0].plot(x_spatial, [dfunction[profile_dict['spatial'][0]](t, **profile_dict['spatial'][1]) for t in x_spatial], label=disturbance_class, alpha=0.75)
            ax[1].plot(x_temporal, [dfunction[profile_dict['temporal'][0]](t, **profile_dict['temporal'][1]) for t in x_temporal], alpha=0.75)
            ax[0].set_title(disturbance_class)

            for a in ax:
                a.grid()
                #remove top and right spines
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

        #plot functions of final weighting dict
        fig, axes = plt.subplots(2, len(self.final_weighting_dict), figsize=(14, 5), sharey=True)

        for i, (dataset, class_dict) in enumerate(self.final_weighting_dict.items()):
            ax = axes[:,i]
            if i == 0:
                ax[0].set_ylabel('Spatial Weighting [1]')
                ax[0].set_xlabel('Distance (m)')
                ax[1].set_ylabel('Temporal Weighting [1]')
                ax[1].set_xlabel('Time (days)')

            for j, (class_name, weighting_dict) in enumerate(class_dict.items()):
                ax[0].plot(x_spatial, [weighting_dict['spatial'](t) for t in x_spatial], label=class_name, alpha=0.75)
                ax[1].plot(x_temporal, [weighting_dict['temporal'](t) for t in x_temporal], alpha=0.75)
                ax[0].legend(fontsize=8)
            ax[0].set_title(dataset)

            for a in ax:
                a.grid()
                #remove top and right spines
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()



    def get_optional_columns(self) -> List[str]:
        cols = []
        for dataset in self.ddataset.values():
            cols.extend(dataset.columns.difference(self.mandatory_columns).values.tolist())

        optional_cols = []
        for col in list(set(cols)):
            if col in self.reference.columns and col != 'geometry':
                self.mandatory_columns.append(col)
            elif col != 'geometry':
                optional_cols.append(col)

        return optional_cols

    #write a method that returns the temporal and spatial join
    def get_spatial_joins(self, ddataset_year: Dict[str, dgpd.GeoDataFrame or gpd.GeoDataFrame], reference_year: dgpd.GeoDataFrame) -> gpd.GeoDataFrame:
        
        #SPATIAL JOIN ~ 10s
        dtsj = []
        for (name, dataset) in ddataset_year.items():
            t = self.dtypes[name]
            dtsj.append(get_spatial_join(dataset, reference_year, t, self.dsbuffer[name]))

        concat1 = dd.concat(dtsj, axis=0).compute()

        #CONCATENATION WITH REF ~ 1s
        col = ['geometry'] + [col + '_left' for col in self.mandatory_columns] + [col for col in self.optional_columns] + ['sd']
        all_index_right = concat1['index_right'].unique()
        concat1 = concat1[['index_right']+col]
        rename = {c: c.split('_left')[0] for c in col}
        concat1 = concat1.rename(columns=rename)
        reference_year['index_right'] = reference_year.index

        #order of reference and concat1 is important. If we want to use iloc[0] on the group to retrieve reference row, we have to stick to this order.
        concat2 = dd.concat([reference_year.loc[all_index_right], concat1], axis=0).compute()

        #MERGING ~ 1s 
        merge = pd.merge(concat2, reference_year.compute(), left_on='index_right', right_index=True)
        drop = ['geometry_y','index_right_y', 'index_right_x'] + [col + '_y' for col in self.mandatory_columns] 
        merge = merge.drop(columns=drop)
        rename = {c: c.split('_x')[0] for c in merge.columns} 
        rename.update({'index_right': 'index_reference'})
        return merge.rename(columns=rename).reset_index()

    def get_temporal_spatial_join(self, year : int, dir_ : str) -> gpd.GeoDataFrame:

        NAME = f'disturbances_{year}_g{self.granularity}_v{self.version}.parquet' 

        if NAME in os.listdir(dir_):
            return gpd.read_parquet(os.path.join(dir_, NAME))
        else : 
            ddataset_year = {}
            for name, dataset in self.ddataset.items():
                t = self.dtypes[name]
                d = dataset[(dataset['start_date'].dt.year >= year - self.temporal_buffer) & (dataset['end_date'].dt.year <= year + self.temporal_buffer)]
                if t == 'point':
                    ddataset_year[name] = d
                elif t == 'polygon':
                    ddataset_year[name] = dgpd.from_geopandas(d, npartitions=10)

            reference_year = self.reference[ self.reference['year'] == year ]
            reference_year = dgpd.from_geopandas(reference_year, npartitions=10)

            dataset_joined = self.get_spatial_joins(ddataset_year, reference_year)
            dataset_joined.to_parquet(os.path.join(dir_, NAME))
            return dataset_joined
    
    def get_clusters(self, year : int, dcustom_similiraity_function : Dict[str, Tuple[Callable, Dict, float]], dir_ : str) -> bool:
        """ Get clusters for a given year

        Args:
            year (int): anchor year for the reference data
            dcustom_similiraity_function (Dict[str, Tuple[Callable, Dict, float]]): dict of custom similarity functions : {'name': (function, kwargs, weight)}
            dir_ (str): directory to save the results

        Returns:
            bool: 
        """
        os.makedirs(dir_, exist_ok=True)
        attribution_year = self.get_temporal_spatial_join(year, dir_=dir_)

        cols = ['index', 'index_reference', 'geometry'] + self.mandatory_columns 
        groups = attribution_year[cols].groupby('index_reference')

        print( f'number of groups : {len(groups)}, estimated time (4 cores) = {len(groups)/1e4 * 5} min')

      
        # dcustom_similarity_function = {'tree specie relatedness': (compute_tree_coherence, {}, 1.0), 'class relatedness': (compute_class_similarity, {'dclass_score': dclass_score}, 1.0)}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            for i in tqdm(range(0, len(groups), 10000)):
                r = Parallel(n_jobs=-1, prefer='processes', verbose=0)(delayed(wrapper_get_cluster)(data[1],
                                                                                                     dtypes_=self.dtypes_, 
                                                                                                     dcustom_similarity_function=dcustom_similiraity_function, 
                                                                                                     doa=self.doa, 
                                                                                                     dclass_score=self.dclass_score, 
                                                                                                     final_weighting_dict=self.final_weighting_dict,
                                                                                                     threshold=0.) for data in islice(groups, i, min(len(groups), i+10000)))
                df = pd.concat(r, axis=0)
                df.to_parquet(os.path.join(dir_, f'tmp_cluster_{i}_g{self.granularity}_v{self.version}.parquet'))

        list_df = []
        for file in os.listdir(dir_):
            if file.startswith('tmp_cluster'):
                list_df.append(pd.read_parquet(os.path.join(dir_, file)))
        df = pd.concat(list_df, axis=0)

        #writing results
        try : 
            df.to_parquet(os.path.join(dir_, f'clusters_{year}_g{self.granularity}_v{self.version}.parquet'))
            for file in os.listdir(dir_):
                if file.startswith('tmp_cluster'):
                    os.remove(os.path.join(dir_, file))
            return True
        except:
            return False