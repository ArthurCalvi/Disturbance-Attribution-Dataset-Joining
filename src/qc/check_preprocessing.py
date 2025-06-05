"""
Script for Quality Checking (QC) of preprocessed GeoDataFrames.

Generates a plot of the geometries and prints summary statistics.
Can be called from other scripts.
"""
import logging
from pathlib import Path
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import argparse # For command-line arguments

# Configure basic logging for the QC script itself
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - QC_SCRIPT - %(levelname)s - %(message)s"
)

# Define base paths - assumes it might be run from project root or its path is handled by caller
# For direct execution, ensure correct paths or adjust sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
QC_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "qc_preprocessing"

def generate_qc_report(input_parquet_path: str, dataset_name: str):
    """
    Generates a QC report (plot and stats) for a given preprocessed Parquet file.

    Args:
        input_parquet_path (str): Path to the input preprocessed Parquet file.
        dataset_name (str): Name of the dataset (e.g., 'senfseidl', 'cdi') for naming outputs.
    """
    input_path = Path(input_parquet_path)
    if not input_path.exists():
        logging.error(f"Input file for QC does not exist: {input_path}")
        return

    logging.info(f"Starting QC for {dataset_name} from {input_path}")

    try:
        gdf = gpd.read_parquet(input_path)
    except Exception as e:
        logging.error(f"Could not read GeoDataFrame for {dataset_name} from {input_path}: {e}")
        return

    if gdf.empty:
        logging.warning(f"GeoDataFrame for {dataset_name} is empty. No QC report generated.")
        return

    # Ensure output directory exists
    QC_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Print Summary Statistics
    logging.info(f"--- Summary Statistics for {dataset_name} ---")
    logging.info(f"Total number of polygons/events: {len(gdf)}")
    
    if 'geometry' in gdf.columns and not gdf.geometry.empty:
        # Ensure geometry is active for area calculation
        if gdf.geometry.name != 'geometry': # Should be 'geometry' by convention after read_parquet
            gdf = gdf.set_geometry('geometry')
        
        try:
            areas = gdf.area # in square meters if CRS is EPSG:2154
            logging.info(f"Area statistics (in CRS units, likely m^2 for EPSG:2154):")
            logging.info(f"  Min area: {areas.min():.2f}")
            logging.info(f"  Max area: {areas.max():.2f}")
            logging.info(f"  Mean area: {areas.mean():.2f}")
            logging.info(f"  Median area: {areas.median():.2f}")
            logging.info(f"  Std Dev area: {areas.std():.2f}")
            # Area quantiles
            logging.info(f"  Area 25th percentile: {areas.quantile(0.25):.2f}")
            logging.info(f"  Area 75th percentile: {areas.quantile(0.75):.2f}")
        except Exception as e:
            logging.error(f"Could not calculate area statistics for {dataset_name}: {e}")
    else:
        logging.warning(f"No valid 'geometry' column found for {dataset_name} to calculate area stats.")
    logging.info(f"---------------------------------------")

    # 2. Generate and Save Plot
    plot_filename = QC_OUTPUT_DIR / f"{dataset_name}_qc_plot.png"
    logging.info(f"Generating plot for {dataset_name}, saving to: {plot_filename}")
    try:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        
        plot_by_class = False
        if 'class' in gdf.columns and not gdf['class'].empty:
            # Create a display string for lists in 'class' column for stable hashing and display
            # Sorting ensures that ['Biotic', 'Storm'] and ['Storm', 'Biotic'] become the same string
            # Handling None or empty lists within the class column as well.
            gdf['class_display'] = gdf['class'].apply(
                lambda x: ', '.join(sorted(x)) if isinstance(x, list) and x else str(x) if x is not None else 'N/A'
            )
            num_unique_display_classes = gdf['class_display'].nunique()
            
            if 0 < num_unique_display_classes < 20:
                plot_by_class = True
            elif num_unique_display_classes >= 20:
                logging.info(f"Many unique class combinations ({num_unique_display_classes}) for {dataset_name}. Plotting geometries without class-based coloring.")
            # If num_unique_display_classes is 0 (e.g. all 'N/A' from empty lists/None), plot_by_class remains False

        if plot_by_class:
            gdf.plot(column='class_display', ax=ax, aspect='equal', legend=True, categorical=True)
        else:
            gdf.plot(ax=ax, aspect='equal') # Plot without specific class coloring or legend for class
            
        ax.set_title(f"Preprocessed Polygons - {dataset_name} ({len(gdf)} features)")
        plt.tight_layout()
        plt.savefig(plot_filename)
        plt.close(fig) # Close the figure to free memory
        logging.info(f"Successfully saved plot: {plot_filename}")
    except Exception as e:
        logging.error(f"Could not generate or save plot for {dataset_name}: {e}", exc_info=True)

    logging.info(f"Finished QC for {dataset_name}.")


if __name__ == "__main__":
    # This allows the script to be run directly from the command line
    # Example: python src/qc/check_preprocessing.py outputs/preprocessing/senfseidl_processed.parquet senfseidl
    
    parser = argparse.ArgumentParser(description="Generate QC report for a preprocessed GeoDataFrame.")
    parser.add_argument("input_file", type=str, help="Path to the input preprocessed Parquet file.")
    parser.add_argument("dataset_name", type=str, help="Name of the dataset (e.g., senfseidl) for naming outputs.")
    
    args = parser.parse_args()
    
    # If running standalone, ensure PROJECT_ROOT and other paths are correct
    # or that matplotlib can save to the specified QC_OUTPUT_DIR relative to execution path.
    # For simplicity, this assumes that if run standalone, it's from project root, or paths are absolute.

    # Add project root to sys.path to find custom modules if any were needed (not in this simple QC script)
    # sys.path.append(str(PROJECT_ROOT))

    generate_qc_report(args.input_file, args.dataset_name) 