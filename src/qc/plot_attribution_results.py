import sys
from pathlib import Path

# Add the project root to sys.path to allow imports from src
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

"""
Script for generating professional GIS-style plots to visualize final disturbance attribution results.
"""
import argparse
import logging
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
from pandas import CategoricalDtype
import numpy as np
from matplotlib import cm
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as ticker
from matplotlib.patches import FancyBboxPatch
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.lines import Line2D

# Import the color constant
from src.config.constants import DISTURBANCE_CLASS_COLORS

# Configure basic logging for the QC script itself
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - AttributionQC - %(levelname)s - %(message)s"
)

def add_north_arrow(ax, x=0.92, y=0.88, size=0.05, text_offset=0.015):
    """Add a professional north arrow to the plot."""
    # Arrow itself - using a simpler, more compatible arrow style
    ax.annotate(
        '',
        xy=(x, y + size / 2),
        xytext=(x, y - size / 2),
        arrowprops=dict(facecolor='black', edgecolor='black', arrowstyle='->', 
                       mutation_scale=20, linewidth=2),
        transform=ax.transAxes,
        zorder=10 # Ensure it's on top
    )
    # "N" label
    ax.text(x, y + size / 2 + text_offset, 'N', 
            fontsize=12, fontweight='bold', 
            ha='center', va='bottom', 
            transform=ax.transAxes, zorder=10)

def add_scale_bar(ax, gdf, length_fraction=0.25, height_fraction=0.01, location=('left', 'bottom'), pad=0.02):
    """Add a dynamic scale bar to the plot based on data extent (assumes CRS in meters)."""
    bounds = gdf.total_bounds
    map_width_m = bounds[2] - bounds[0]
    
    # Determine a nice round number for the scale bar length
    target_scale_bar_map_units = map_width_m * length_fraction
    
    possible_scales = np.array([1, 2, 5, 10, 20, 25, 50, 75, 100, 150, 200, 250, 500, 750,
                               1000, 2000, 5000, 10000, 20000, 25000, 50000, 75000, 100000,
                               200000, 250000, 500000, 1000000]) # in meters
    
    # Find the closest possible scale to the target length
    scale_bar_length_m = possible_scales[np.argmin(np.abs(possible_scales - target_scale_bar_map_units))]
    
    if scale_bar_length_m == 0:
        logging.warning("Calculated scale bar length is 0, cannot draw scale bar.")
        return

    # Convert scale bar length to figure coordinates (approximate)
    scale_bar_fig_width = (scale_bar_length_m / map_width_m) * (ax.get_position().width)
    
    # Position the scale bar
    if location[0] == 'left':
        x0 = pad
    else: # right
        x0 = 1 - pad - scale_bar_fig_width
    if location[1] == 'bottom':
        y0 = pad
    else: # top
        y0 = 1 - pad - height_fraction

    # Create the scale bar rectangle
    rect = patches.Rectangle((x0, y0), scale_bar_fig_width, height_fraction, 
                             facecolor='black', transform=ax.transAxes, clip_on=False, zorder=10)
    ax.add_patch(rect)
    
    # Add text label for the scale bar
    if scale_bar_length_m >= 1000:
        label_text = f"{scale_bar_length_m / 1000:.0f} km"
    else:
        label_text = f"{scale_bar_length_m:.0f} m"
    
    ax.text(x0 + scale_bar_fig_width / 2, y0 + height_fraction + 0.005, label_text, 
            ha='center', va='bottom', fontsize=10, transform=ax.transAxes, zorder=10)

def format_axes_professional(ax, gdf):
    """Apply professional cartographic styling to axes."""
    ax.set_xlabel("Easting (Coordinate Units)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Northing (Coordinate Units)", fontsize=12, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    ax.tick_params(axis='both', which='major', labelsize=10)
    try:
        ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3), useMathText=True)
    except AttributeError:
         ax.ticklabel_format(style='sci', axis='both', scilimits=(-3,3))
    ax.set_facecolor('#f0f0f0') # Light gray background
    ax.set_aspect('equal', adjustable='box')

def create_professional_legend(ax, items, title="Legend", loc='upper left', bbox_to_anchor=(1.02, 1)):
    """Creates a professional legend from a list of (label, color) tuples or Line2D objects."""
    legend_elements = []
    if items and isinstance(items[0], Line2D):
        legend_elements = items
    else:
        for label, color in items:
            legend_elements.append(Line2D([0], [0], marker='s', color='w', 
                                        label=label, markerfacecolor=color, markersize=10))
    
    legend = ax.legend(handles=legend_elements, title=title, 
                      loc=loc, bbox_to_anchor=bbox_to_anchor,
                      frameon=True, fancybox=True, shadow=True,
                      title_fontsize=12, fontsize=10, facecolor='white', framealpha=0.9)
    return legend

def plot_spatial_distribution(
    gdf: gpd.GeoDataFrame,
    column_name: str,
    output_path: Path,
    title: str,
    categorical: bool = True,
    custom_colors: dict = None,
    missing_kwds: dict = None,
    figsize: tuple = (14, 10)
):
    """Generates and saves a professional GIS-style spatial plot of the GeoDataFrame."""
    if column_name not in gdf.columns or gdf[column_name].empty:
        logging.warning(f"Column '{column_name}' not found or empty. Skipping plot: {title}")
        return
    if gdf.geometry.is_empty.all():
        logging.warning(f"All geometries are empty. Skipping plot: {title}")
        return

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    format_axes_professional(ax, gdf)
    
    default_missing_kwds = {"color": "lightgrey", "label": "Missing/NA"}
    if missing_kwds:
        default_missing_kwds.update(missing_kwds)

    legend_items = []
    if categorical and custom_colors:
        if not isinstance(gdf[column_name].dtype, CategoricalDtype):
            gdf[column_name] = gdf[column_name].astype('category')
        
        # Ensure all possible classes (including 'Unknown') are in the categories
        all_classes = list(custom_colors.keys())
        gdf[column_name] = gdf[column_name].cat.set_categories(all_classes)
        unique_classes = gdf[column_name].cat.categories
        # Map colors to the actual data values
        plot_colors = [custom_colors.get(val, default_missing_kwds["color"]) for val in gdf[column_name]]
        gdf.plot(
            ax=ax,
            legend=False, 
            color=plot_colors, # Pass list of colors for each polygon
            edgecolor='none',  # Remove edges for cleaner visualization
            linewidth=0,
            missing_kwds=default_missing_kwds
        )
        class_counts = gdf[column_name].value_counts()
        total_events = len(gdf)
        for cls_name in unique_classes:
            count = class_counts.get(cls_name, 0)
            if count > 0 or cls_name == 'Unknown':
                percentage = (count / total_events) * 100 if total_events > 0 else 0
                label = f"{cls_name} ({count}, {percentage:.1f}%)"
                legend_items.append((label, custom_colors.get(cls_name, default_missing_kwds["color"])))
        if gdf[column_name].isnull().any():
             legend_items.append((default_missing_kwds["label"], default_missing_kwds["color"])) 
        create_professional_legend(ax, legend_items, title="Disturbance Classes")

    elif not categorical:
        # Continuous data (e.g., confidence)
        cmap = 'RdYlBu_r' # Professional color scheme for confidence
        gdf.plot(
            column=column_name,
            ax=ax,
            legend=True,
            cmap=cmap,
            edgecolor='none',  # Remove edges for cleaner visualization
            linewidth=0,
            missing_kwds=default_missing_kwds,
            legend_kwds={'label': "Confidence Score", 'orientation': 'vertical', 'shrink': 0.8, 'aspect': 20}
        )
    else: # Categorical but no custom_colors (fallback)
        gdf.plot(
            column=column_name,
            ax=ax,
            legend=True,
            categorical=True,
            cmap='tab20',
            edgecolor='none',  # Remove edges for cleaner visualization
            linewidth=0,
            missing_kwds=default_missing_kwds,
            legend_kwds={'title': column_name, 'loc': 'upper left', 'bbox_to_anchor': (1.02, 1)}
        )

    add_north_arrow(ax)
    try:
        add_scale_bar(ax, gdf) # Add scale bar if CRS allows calculation
    except Exception as e:
        logging.warning(f"Could not add scale bar: {e}")
    
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout(rect=[0, 0, 0.82, 1]) # Adjust for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logging.info(f"Successfully saved professional plot: {output_path}")

def plot_community_envelopes(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    title: str,
    figsize: tuple = (14, 10)
):
    """
    Generates and saves a GIS-style plot of community envelopes,
    colored by the number of events within each community.
    """
    if 'community_id' not in gdf.columns or gdf['community_id'].isna().all():
        logging.warning("No valid 'community_id' data found. Skipping community envelope plot.")
        return

    # Filter out events not in a community
    communities_gdf = gdf[gdf['community_id'].notna()].copy()
    communities_gdf['community_id'] = communities_gdf['community_id'].astype(int)
    
    # Calculate convex hull for each community
    community_envelopes = communities_gdf.dissolve(by='community_id', aggfunc={'community_id': 'size'})
    community_envelopes = community_envelopes.rename(columns={'community_id': 'event_count'})
    community_envelopes['geometry'] = community_envelopes.geometry.convex_hull
    
    if community_envelopes.empty:
        logging.warning("No community envelopes could be generated. Skipping plot.")
        return
        
    logging.info(f"Generated {len(community_envelopes)} community envelopes for plotting.")

    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor='white')
    format_axes_professional(ax, community_envelopes)

    community_envelopes.plot(
        column='event_count',
        ax=ax,
        legend=True,
        cmap='viridis',
        edgecolor='black',
        linewidth=0.8,
        legend_kwds={'label': "Number of Events in Community", 'orientation': 'vertical', 'shrink': 0.8}
    )

    add_north_arrow(ax)
    try:
        add_scale_bar(ax, gdf) # Use original gdf for scale extent
    except Exception as e:
        logging.warning(f"Could not add scale bar to community plot: {e}")
        
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust for legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logging.info(f"Successfully saved community envelope plot: {output_path}")

def plot_class_pie_chart(
    gdf: gpd.GeoDataFrame,
    class_column_name: str,
    output_path: Path,
    title: str,
    custom_colors: dict = None,
    figsize: tuple = (15, 8) # Adjusted for two subplots
):
    """Generates and saves a professional pie chart and a complementary bar chart of class distributions."""
    if class_column_name not in gdf.columns or gdf[class_column_name].empty:
        logging.warning(f"Class column '{class_column_name}' not found or empty. Skipping pie/bar chart: {title}")
        return

    class_counts = gdf[class_column_name].value_counts()
    if class_counts.empty:
        logging.warning(f"No data to plot in pie/bar chart for {class_column_name}. Skipping: {title}")
        return

    fig, (ax_pie, ax_bar) = plt.subplots(1, 2, figsize=figsize, facecolor='white')
    fig.suptitle(title, fontsize=16, fontweight='bold')

    # Prepare colors
    pie_colors = [custom_colors.get(cls, '#CCCCCC') for cls in class_counts.index] if custom_colors else None

    # --- Pie Chart ---
    wedges, texts, autotexts = ax_pie.pie(
        class_counts,
        labels=None, # Labels will be in the legend
        autopct='%1.1f%%',
        startangle=90,
        colors=pie_colors,
        wedgeprops=dict(width=0.4, edgecolor='w', linewidth=2),
        pctdistance=0.80 # Position percentages inside wedges
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax_pie.set_title("Percentage Distribution", fontsize=14, pad=10)
    
    # Legend for Pie Chart (shared effectively)
    legend_items = []
    total_events = class_counts.sum()
    for cls_name, count in class_counts.items():
        percentage = (count / total_events) * 100
        label = f"{cls_name}: {count} ({percentage:.1f}%)"
        legend_items.append((label, custom_colors.get(cls_name, '#CCCCCC')))
    
    create_professional_legend(fig, legend_items, title="Disturbance Classes", 
                               loc='center left', bbox_to_anchor=(0.92, 0.5))

    # --- Bar Chart ---
    bars = ax_bar.bar(class_counts.index, class_counts.values, color=pie_colors, edgecolor='black', linewidth=0.7)
    ax_bar.set_title("Absolute Counts", fontsize=14, pad=10)
    ax_bar.set_ylabel("Number of Events", fontsize=12, fontweight='bold')
    ax_bar.set_xticklabels(class_counts.index, rotation=45, ha='right', fontsize=10)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.7)
    ax_bar.set_facecolor('#f0f0f0')
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + total_events*0.005, # offset slightly above bar
                    f'{int(yval)}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 0.9, 0.95]) # Adjust layout for super title and legend
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    logging.info(f"Successfully saved professional pie/bar chart: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate professional GIS-style QC plots for final disturbance attribution results.")
    parser.add_argument("input_file", type=str, help="Path to the input final_attributed_disturbances.parquet file.")
    parser.add_argument("--output-dir", type=str, default="outputs/qc_attribution", help="Directory to save the output plots.")
    args = parser.parse_args()

    input_path = Path(args.input_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    logging.info(f"Loading final attributed data from: {input_path}")
    try:
        gdf = gpd.read_parquet(input_path)
    except Exception as e:
        logging.error(f"Failed to load GeoDataFrame from {input_path}: {e}", exc_info=True)
        return

    if gdf.empty:
        logging.warning("Input GeoDataFrame is empty. No plots will be generated.")
        return
    
    # Ensure geometry column exists and is active
    if 'geometry' not in gdf.columns:
        logging.error("'geometry' column not found in the input file.")
        return
    try:
        gdf = gdf.set_geometry('geometry')
    except Exception as e:
        logging.error(f"Failed to set active geometry column: {e}")
        return
    if not isinstance(gdf, gpd.GeoDataFrame):
        logging.error("Data loaded is not a GeoDataFrame after setting geometry.")
        return

    logging.info(f"Loaded {len(gdf)} disturbance events for visualization. CRS: {gdf.crs}")
    
    # --- Plot Community Envelopes ---
    plot_community_envelopes(
        gdf.copy(),
        output_path=output_dir / "community_envelopes_pro.png",
        title="Spatial Distribution of Communities by Event Count"
    )
    
    senf_gdf_for_spatial_plot = gdf[gdf['dataset'] == 'senfseidl']
    if senf_gdf_for_spatial_plot.empty:
        logging.warning("No Senf & Seidl data found for the spatial class distribution plot.")
    else:
        logging.info(f"Creating spatial distribution plot for {len(senf_gdf_for_spatial_plot)} Senf & Seidl events")
        plot_spatial_distribution(
            senf_gdf_for_spatial_plot.copy(), # Pass a copy to avoid SettingWithCopyWarning
            column_name='attributed_class_final',
            output_path=output_dir / "spatial_attributed_classes_senfseidl_pro.png",
            title="Spatial Distribution: Final Attributed Classes (Senf & Seidl)",
            categorical=True,
            custom_colors=DISTURBANCE_CLASS_COLORS
        )

    data_for_confidence_plot = senf_gdf_for_spatial_plot if not senf_gdf_for_spatial_plot.empty else gdf
    title_confidence = "Spatial Distribution: Attribution Confidence"
    if not senf_gdf_for_spatial_plot.empty and len(data_for_confidence_plot) == len(senf_gdf_for_spatial_plot):
        title_confidence += " (Senf & Seidl)"
    elif len(data_for_confidence_plot) < len(gdf):
         title_confidence += " (Filtered Dataset)"
    else:
        title_confidence += " (All Datasets)"
    
    if 'attribution_confidence_final' in data_for_confidence_plot.columns:
        logging.info(f"Creating confidence distribution plot for {len(data_for_confidence_plot)} events")
        plot_spatial_distribution(
            data_for_confidence_plot.copy(),
            column_name='attribution_confidence_final',
            output_path=output_dir / "spatial_attribution_confidence_pro.png",
            title=title_confidence,
            categorical=False, 
            custom_colors=None
        )

    logging.info(f"Creating class distribution chart for {len(gdf)} total events")
    plot_class_pie_chart(
        gdf.copy(),
        class_column_name='attributed_class_final',
        output_path=output_dir / "pie_bar_attributed_classes_pro.png",
        title="Overall Distribution of Final Attributed Classes",
        custom_colors=DISTURBANCE_CLASS_COLORS
    )

    logging.info(f"Professional GIS-style attribution QC plots generated in {output_dir}")

if __name__ == "__main__":
    main() 