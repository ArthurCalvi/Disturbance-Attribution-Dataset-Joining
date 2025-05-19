# Project Progress

## Initial Setup
- Understood the goal: Extract BBOX-defined excerpts from multiple spatial Parquet datasets.
- Identified target datasets from `join-datasets/constants.py`.
- Defined the BBOX for extraction: (307783.0822, 6340505.4366, 469246.8845, 6419190.9011) in EPSG:2154.
- Planned to create a new script in a dedicated folder for this task.

## Implementation
- Created `data_excerpt_creation/create_data_excerpts.py` script.
- The script defines the BBOX and target CRS (EPSG:2154).
- It dynamically loads `loading_dict` from `join-datasets/constants.py` to find dataset paths.
  - Note: The method for loading `loading_dict` by parsing the file content is fragile. A more robust approach would be to ensure `constants.py` is importable or use `ast.literal_eval`.
- It iterates through each dataset:
    - Constructs absolute paths for input Parquet files and output excerpt files.
    - Creates an output directory `data/excerpts/` (relative to `join-datasets/` parent directory).
    - Loads the Parquet file into a GeoDataFrame.
    - Ensures the GeoDataFrame is in the target CRS, reprojecting if necessary.
    - Clips the GeoDataFrame using the BBOX (`gdf.geometry.intersects(bbox_geometry)`).
    - Saves the clipped excerpt as a Parquet file.
    - Logs information, warnings (e.g., if excerpt > 5MB), and errors throughout the process.
    - Skips datasets if files are not found or if other errors occur during loading/processing.
- Added a summary log for successful and failed/skipped excerpts. 