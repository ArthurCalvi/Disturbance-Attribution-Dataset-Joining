__version__ = '0.4'

import warnings
warnings.filterwarnings('ignore') 

import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
import geopandas as gpd
import dask_geopandas as dgpd
import warnings
import pandas as pd
import dask.dataframe as dd
from datetime import datetime, timedelta
from joblib import Parallel, delayed
from itertools import islice
from tqdm import tqdm
import numpy as np 
import os 
import contextily as ctx
from matplotlib_scalebar.scalebar import ScaleBar
import networkx as nx
    
from .utils import (
    get_spatial_join,
    get_subset_dataset,
    is_dataset_valid,
    classify_source_type,
    wrapper_get_cluster,
    modify_dsbuffer,
    calculate_d_star,
    get_combined_weighting,
    dfunction,
    calculate_temporal_centroid,
    is_temporally_similar,
    get_temporal_range,
)
from .constants import custom_color_mapping

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

        #creating dataset and spatial_entity_dataset
        self.dataset, self.spatial_entity_dataset = self.get_datasets()

    def get_islands(self, temporal_threshold : int, spatial_threshold : int) -> gpd.GeoDataFrame:

        if os.path.isfile("../data/results/clusters/islands_{}_{}_{}_{}.parquet".format(self.granularity, spatial_threshold, temporal_threshold, self.version)):
            return gpd.read_parquet("../data/results/clusters/islands_{}_{}_{}_{}.parquet".format(self.granularity, spatial_threshold, temporal_threshold, self.version))
        else : 
            dataset = self.dataset[['geometry', 'centroid_date']]
            spatial_entity_dataset = self.spatial_entity_dataset[['geometry', 'centroid_date']]
            sindex = dataset.sindex
            spatial_entity_sindex = spatial_entity_dataset.sindex

            if os.path.isfile(f"../data/results/graph/graph_{self.granularity}_{spatial_threshold}_{temporal_threshold}_{self.version}.gml"):
                G = nx.read_gml(f"../data/results/graph/graph_{self.granularity}_{spatial_threshold}_{temporal_threshold}_{self.version}.gml")

            else: 
                # Assuming gdf is your GeoDataFrame
                print('Building graph...')
                G = nx.Graph()

                # Construct the similarity matrix using spatial index
                for event1 in tqdm(dataset.itertuples(index=True)):
                    i = event1.Index
                    # Find nearby events within spatial threshold
                    possible_matches_index = list(sindex.intersection(event1.geometry.buffer(spatial_threshold).bounds))
                    possible_matches = dataset.iloc[possible_matches_index]

                    for event2 in possible_matches.itertuples(index=True):
                        j = event2.Index
                        if i != j and is_temporally_similar(event1, event2, temporal_threshold):
                            G.add_edge(i, j)

                nx.write_gml(G, f"../data/results/graph/graph_{self.granularity}_{spatial_threshold}_{temporal_threshold}_{self.version}.gml")

            islands = list(nx.connected_components(G))

            # Create a list to store the sub-GeoDataFrames
            island_gdfs = []

            print('Building islands...')
            # Iterate over each island and create a sub-GeoDataFrame
            for island in tqdm(islands):
                # Select rows from the original GeoDataFrame that correspond to the current island
                island_gdf = self.dataset.iloc[list(island)]
                
                # Append this sub-GeoDataFrame to the list
                island_gdfs.append(island_gdf)

            for i, island_gdf in tqdm(enumerate(island_gdfs)):
                # Calculate the envelope (bounding box) of the cluster
                envelope = island_gdf.unary_union.envelope

                # Calculate the temporal range of the cluster
                cluster_start, cluster_end = get_temporal_range(island_gdf)

                # Find potential matches using spatial index
                possible_matches_index = list(spatial_entity_sindex.intersection(envelope.bounds))
                possible_matches = spatial_entity_dataset.iloc[possible_matches_index]

                if len(possible_matches_index) > 0 :
                    break
                # Initialize an empty list to store events to be added
                events_to_add = []

                # Iterate through each potential match
                for event in possible_matches.itertuples(index=True):
                    # Check spatial intersection
                    spatial_condition = envelope.intersects(event.geometry)
                    # Check temporal intersection
                    event_centroid = event.centroid_date  # Replace with your actual column name
                    temporal_condition = (event_centroid <= cluster_end) and (event_centroid >= cluster_start)

                    # If both conditions are met, add the event to the list
                    if spatial_condition and temporal_condition:
                        events_to_add.append(event.Index)

                # Add the events to the cluster GeoDataFrame
                if len(events_to_add) > 0:
                    additional_events = self.spatial_entity_dataset.loc[events_to_add]
                    island_gdfs[i] = gpd.GeoDataFrame(pd.concat([island_gdf, additional_events]), geometry='geometry', crs=island_gdf.crs)
            
            # Initialize an empty list to store the modified cluster GeoDataFrames          
            modified_gdfs = []

            # Add a 'cluster' column and concatenate
            for i, island_gdf in tqdm(enumerate(island_gdfs)):
                island_gdf['cluster'] = i  # Add a 'cluster' column with the cluster index
                modified_gdfs.append(island_gdf)


            # Ensure the GeoDataFrame has the correct geometry set
            all_clusters_gdf = gpd.GeoDataFrame(pd.concat(modified_gdfs), geometry='geometry').drop(columns=['year'])

            # Save to GeoParquet
            all_clusters_gdf.to_parquet('../data/results/clusters/islands_{}_{}_{}_{}.parquet'.format(self.granularity, spatial_threshold, temporal_threshold, self.version))

        return all_clusters_gdf
        

    def plot_weighting_functions(self, title=False):

        """ Plot the weighting functions for each dataset
        """
        x_spatial = np.linspace(0, 7000, 1000)
        x_temporal = np.linspace(0, 5*365, 1000)

        #change font to times new roman
        plt.rcParams["font.family"] = "Times New Roman"

        #plot functions of ddataset profile 
        dp = {k:v for k,v in self.ddataset_profile.items() if k != 'reference'}
        n = len(dp)
        fig1, axes = plt.subplots(2, n, figsize=(int(n * 1.5), 4), sharey=True)
        for i, (dataset, profile_dict) in enumerate(dp.items()):
            ax = axes[:,i]
            if i == 0:
                ax[0].set_ylabel('Spatial Weighting [1]')
                ax[0].set_xlabel('Distance (m)')
                ax[1].set_ylabel('Temporal Weighting [1]')
                ax[1].set_xlabel('Time (days)')
            ax[0].plot(x_spatial, [dfunction[profile_dict['spatial'][0]](t, **profile_dict['spatial'][1]) for t in x_spatial], label=dataset, alpha=0.75, color=f'C{i}')
            ax[1].plot(x_temporal, [dfunction[profile_dict['temporal'][0]](t, **profile_dict['temporal'][1]) for t in x_temporal], alpha=0.75, color=f'C{i}')
            if dataset == 'reference':
                dataset = 'Hansen'
            ax[0].set_title(dataset.upper())

            for a in ax:
                a.grid()
                #remove top and right spines
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

        if title:
            fig1.suptitle('Spatial and Temporal certainties linked to the dataset and/or the algorithm used.')
        plt.tight_layout()
        plt.show()

        #plot functions of ddisturbance profile
        n = len(self.ddisturbance_profile)
        fig2, axes = plt.subplots(2, n, figsize=(int(n * 1.5), 4), sharey=True)

        x_spatial = np.linspace(0, 7000, 1000)
        x_temporal = np.linspace(0, 5*365, 1000)

        for i, (disturbance_class, profile_dict) in enumerate(self.ddisturbance_profile.items()):
            ax = axes[:,i]
            if i == 0:
                ax[0].set_ylabel('Spatial Weighting [1]')
                ax[0].set_xlabel('Distance (m)')
                ax[1].set_ylabel('Temporal Weighting [1]')
                ax[1].set_xlabel('Time (days)')
            ax[0].plot(x_spatial, [dfunction[profile_dict['spatial'][0]](t, **profile_dict['spatial'][1]) for t in x_spatial], label=disturbance_class, alpha=0.75, color=custom_color_mapping[disturbance_class])
            ax[1].plot(x_temporal, [dfunction[profile_dict['temporal'][0]](t, **profile_dict['temporal'][1]) for t in x_temporal], alpha=0.75, color=custom_color_mapping[disturbance_class])
            ax[0].set_title(disturbance_class)

            for a in ax:
                a.grid()
                #remove top and right spines
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

        if title:
            fig2.suptitle('Spatial and Temporal extent of disturbances.')
        plt.tight_layout()
        plt.show()

        #plot functions of final weighting dict
        fw = {k:v for k,v in self.final_weighting_dict.items() if k != 'reference'}
        n = len(fw)
        fig3, axes = plt.subplots(2, n, figsize=(int(n * 1.5), 4), sharey=True)
        
        for i, (dataset, class_dict) in enumerate(fw.items()):

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
            
            if dataset == 'reference':
                dataset = 'Hansen'
            ax[0].set_title(dataset.upper())

            for a in ax:
                a.grid()
                #remove top and right spines
                a.spines['top'].set_visible(False)
                a.spines['right'].set_visible(False)

        if title:
            fig3.suptitle('Spatial and temporal detection profiles of disturbances')
        plt.tight_layout()
        plt.show()

        return fig1, fig2, fig3

    def plot_dataset_examples(self):
        
        data = []
        for dataset, classes in self.dclass_score.items():
            for class_name, scores in classes.items():
                for final_class, score in scores.items():
                    data.append({'Dataset': dataset, 'Class': class_name, 'Final Class': final_class, 'Score': score})

        df = pd.DataFrame(data)

        ddataset = {name:dataset for name, dataset in self.ddataset.items()}
        ddataset['reference'] = self.reference

        # Creating a figure with 4 columns and 4 rows for the 8 datasets
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(10, 5))

        # Grouping by 'Dataset' and enumerating over these groups
        for i, (dataset, dataset_df) in enumerate(df.groupby('Dataset')):
            # Calculating the position for the plots in the grid
            col = i % 4
            row = i // 4 

            if col == 0:
                # Setting up the labels for the rows
                axs[row, col].set_ylabel('Data example')
                # axs[row + 1, col].set_ylabel('Composition of dataset classes')

            # Setting up the plots
            d = ddataset[dataset].sample(n=1)
            #square buffer around the centroid of the polygon
            area = d.iloc[0].geometry.area
            if area > 10000:
                ed = np.sqrt(area / np.pi)
            else : 
                ed = 150
            
            extent = d.centroid.buffer(3 * ed, cap_style = 3).total_bounds
            #define xlim and ylim
            xlim = (extent[0], extent[2])
            ylim = (extent[1], extent[3])
            axs[row, col].set_xlim(xlim)
            axs[row, col].set_ylim(ylim)
            axs[row, col].add_artist(ScaleBar(1))
            #add scale bar
            d.plot(ax=axs[row, col], alpha=0.5, edgecolor='k')
            ctx.add_basemap(axs[row, col], crs=d.crs.to_string(), source=ctx.providers.Esri.WorldImagery, attribution=False)

            #add tex annotation of d['class'] at the centroid 
            if dataset == 'reference':
                dataset = 'Hansen'
                txt = 'reference' + '\n' + str(d['year'].iloc[0])
            else :
                txt = d['class'].iloc[0]  + '\n' + d['start_date'].dt.strftime('%Y-%m-%d').iloc[0] + '\n' + d['end_date'].dt.strftime('%Y-%m-%d').iloc[0]
            axs[row, col].text(d.centroid.x, d.centroid.y, txt, fontsize=10, color='white')
            axs[row, col].axis('off')
            
            #annotate name of the dataste in the left top corner with transAx coordinates between 0 and 1
            axs[row, col].annotate(dataset.upper(), xy=(0.05, 0.95), xycoords='axes fraction', fontsize=12, ha='left', va='top', color='white')


        # Displaying the updated template
        plt.tight_layout()
        plt.show()

        return fig



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

    def get_datasets(self):
        list_dataset = [df for name, df in self.ddataset.items() if self.dtypes_[name] != 'spatial_entity']
        list_spatial_entity_dataset = [df for name, df in self.ddataset.items() if self.dtypes_[name] == 'spatial_entity']

        dataset = gpd.GeoDataFrame(pd.concat(list_dataset), geometry='geometry', crs=list_dataset[0].crs)[self.mandatory_columns + ['geometry']].reset_index().rename(columns={'index': 'id'})
        dataset['centroid_date'] = dataset.apply(lambda x: calculate_temporal_centroid(x.start_date, x.end_date), axis=1)
        n = dataset.index.max()

        if len(list_spatial_entity_dataset) > 0:
            spatial_entity_dataset = gpd.GeoDataFrame(pd.concat(list_spatial_entity_dataset), geometry='geometry', crs=list_spatial_entity_dataset[0].crs)[self.mandatory_columns + ['geometry']].reset_index().rename(columns={'index': 'id'})
            spatial_entity_dataset.index = spatial_entity_dataset.index + n + 1
            spatial_entity_dataset['centroid_date'] = spatial_entity_dataset.apply(lambda x: calculate_temporal_centroid(x.start_date, x.end_date), axis=1)
        else :
            spatial_entity_dataset = None 
        return dataset, spatial_entity_dataset

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
    
    def get_clusters(self, year : int, dcustom_similiraity_function : Dict[str, Tuple[Callable, Dict, float]], dir_ : str, temporal_threshold : int, spatial_threshold : int) -> bool:
        """ Get clusters for a given year

        Args:
            year (int): anchor year for the reference data
            dcustom_similiraity_function (Dict[str, Tuple[Callable, Dict, float]]): dict of custom similarity functions : {'name': (function, kwargs, weight)}
            dir_ (str): directory to save the results

        Returns:
            bool: 
        """
        os.makedirs(dir_, exist_ok=True)
        all_islands = self.get_islands(temporal_threshold=temporal_threshold, spatial_threshold=spatial_threshold)
        islands = all_islands.groupby('cluster')

        print( f'number of groups : {len(islands)}, estimated time (4 cores) = {len(islands)/1e4 * 5} min')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            for i in tqdm(range(0, len(islands), 10000)):
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
            if file.startswith('tmp_cluster') and file.endswith(f'g{self.granularity}_v{self.version}.parquet'):
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
