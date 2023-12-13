import geopandas as gpd

#Load the national forest inventory 
nfi = gpd.read_parquet('NFI_2003-2021_EPSG4326_FR.parquet').to_crs('epsg:32631')
nfi.loc[nfi['class'] == 'Drought', 'class'] = 'Drought-dieback'

#Figure with a 3 columns and 3 rows. 
#First two rows are dedicated to visualize each class of the NFI 
#Last row is dedicated to violin plots : Y axis : class, X axis : year of the disturbance 