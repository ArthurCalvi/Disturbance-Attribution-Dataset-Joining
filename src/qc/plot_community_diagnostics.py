"""
Script to generate diagnostics for Louvain community detection, 
specifically plotting the distribution of community sizes.
"""
import logging
import argparse
from pathlib import Path
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

QC_OUTPUT_DIR = Path("outputs") / "qc"

def plot_community_size_distribution(gdf: gpd.GeoDataFrame, output_dir: Path) -> None:
    """Generates and saves a histogram of community sizes."""
    if "community_id" not in gdf.columns:
        logging.error("'community_id' column not found in GeoDataFrame. Cannot generate plot.")
        return

    # Ensure community_id is treated as numeric/categorical for grouping, handling NaNs
    # NaNs typically mean the node wasn't assigned to any community.
    # We are interested in the sizes of actual communities.
    valid_communities = gdf[gdf["community_id"].notna()]
    if valid_communities.empty:
        logging.info("No valid communities found (all community_id values are NaN). Skipping plot.")
        return

    community_sizes = valid_communities.groupby("community_id").size()

    if community_sizes.empty:
        logging.info("No communities to plot after grouping (perhaps only NaN community_ids).")
        return

    logging.info(f"Total number of distinct communities found: {len(community_sizes)}")
    logging.info(f"Community size statistics:\n{community_sizes.describe()}")

    # Explicitly log mean, median, min, max
    logging.info(f"  Min community size: {community_sizes.min()}")
    logging.info(f"  Max community size: {community_sizes.max()}")
    logging.info(f"  Mean community size: {community_sizes.mean():.2f}")
    logging.info(f"  Median community size: {community_sizes.median()}")

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / "community_size_distribution.png"

    plt.figure(figsize=(12, 7))
    sns.histplot(community_sizes, kde=False, binwidth=1) # Binwidth 1 for discrete sizes
    plt.title("Distribution of Louvain Community Sizes")
    plt.xlabel("Community Size (Number of Members)")
    plt.ylabel("Number of Communities")
    plt.yscale('log') # Use log scale for y-axis if sizes vary a lot
    plt.grid(True, which="both", ls="--", alpha=0.7)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logging.info(f"Community size distribution plot saved to: {plot_path}")

    # Log specific counts for very small communities
    logging.info(f"Number of communities with 1 member: {(community_sizes == 1).sum()}")
    logging.info(f"Number of communities with 2 members: {(community_sizes == 2).sum()}")
    logging.info(f"Number of communities with < 5 members: {(community_sizes < 5).sum()}")

    # --- Tuning Hints ---
    logging.info("--- Louvain Resolution Tuning Hints ---")
    num_single_member_communities = (community_sizes == 1).sum()
    total_communities = len(community_sizes)
    if total_communities > 0:
        percentage_single_member = (num_single_member_communities / total_communities) * 100
        if percentage_single_member > 20: # If more than 20% of communities are singletons
            logging.info(f"{percentage_single_member:.1f}% of communities have only 1 member. ")
            logging.info("Consider DECREASING 'louvain_resolution' in AttributionParams if you want fewer, larger communities.")
            logging.info("This may help reduce instances of HDBSCAN processing very small (e.g., 1-member) communities.")
        
        if community_sizes.median() < 3 and total_communities > 10: # If median size is very small for a decent number of communities
            logging.info(f"The median community size is {community_sizes.median()}. Many communities are very small.")
            logging.info("If aiming for larger, more consolidated initial groupings, consider DECREASING 'louvain_resolution'.")

    if community_sizes.max() > 0.5 * len(gdf) and total_communities < 10 and total_communities > 0 : # If one community is dominating and few communities exist
        logging.info(f"The largest community has {community_sizes.max()} members, which is a large portion of the total {len(gdf)} events.")
        logging.info("If you need more granular communities, consider INCREASING 'louvain_resolution'.")
    
    logging.info("The ideal 'louvain_resolution' balances community coherence with the needs of subsequent steps like HDBSCAN.")
    logging.info("Experimentation is often key. Values typically range from 0.5 to 2.0, but can vary.")
    logging.info("-------------------------------------")

def main():
    parser = argparse.ArgumentParser(description="Generate Louvain community size diagnostics.")
    parser.add_argument(
        "input_gdf_path", 
        type=Path,
        help="Path to the GeoDataFrame Parquet file containing the 'community_id' column."
    )
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=QC_OUTPUT_DIR,
        help=f"Directory to save the output plot. Defaults to {QC_OUTPUT_DIR}"
    )
    args = parser.parse_args()

    if not args.input_gdf_path.exists():
        logging.error(f"Input GeoDataFrame file not found: {args.input_gdf_path}")
        return

    logging.info(f"Loading GeoDataFrame from: {args.input_gdf_path}")
    try:
        gdf = gpd.read_parquet(args.input_gdf_path)
    except Exception as e:
        logging.error(f"Error loading GeoDataFrame: {e}")
        return

    plot_community_size_distribution(gdf, args.output_dir)

if __name__ == "__main__":
    main() 