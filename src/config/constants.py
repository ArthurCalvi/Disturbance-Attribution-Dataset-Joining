"""
Configuration constants for the disturbance attribution pipeline,
including class mappings and definitions.
"""

# 1. Final Target Disturbance Classes (User's desired output)
FINAL_TARGET_CLASSES = [
    'Fire',
    'Storm',
    'Anthropogenic',
    'Biotic',
    'Drought',
    'Unknown' # Good to have a general unknown/other category
]

# 2. Raw to Final Target Class Mappings
# This dictionary will map raw class labels from each dataset to a FINAL_TARGET_CLASS.
# Keys are dataset names (as used in preprocess_excerpts.py and reliability dicts).
# Values are dictionaries mapping raw class string to a FINAL_TARGET_CLASS string.
RAW_TO_FINAL_TARGET_MAPPINGS = {
    'senfseidl': {
        # Raw classes now reflect the simplified 1,2,3 mapping from the Senf&Seidl cause raster
        # Each raw class can map to multiple final classes
        'Storm,Biotic': ['Biotic', 'Storm'],
        'Fire': ['Fire'],
        'Other': ['Anthropogenic', 'Unknown', 'Drought'],
    },
    'cdi': {
        'drought': ['Drought'], # CDI directly indicates Drought
        '_default_': ['Drought']
    },
    'hm': {
        # Outputs of _get_class in hm.py mapped directly to Final Target Classes
        'Drought-dieback': ['Drought'],
        'Fire': ['Fire'],
        'Storm': ['Storm'],
        'Biotic': ['Biotic'], 
        'Other': ['Anthropogenic', 'Unknown'], # Or 'Anthropogenic' if 'Other' usually implies human activity
    },
    'firepolygons': {
        'Fire': ['Fire'],
        'FIRE': ['Fire'],
        '_default_': ['Fire'] # All firepolygon data is considered Fire
    },
    'forms': {
        'clear-cut': ['Anthropogenic'],
        '_default_': ['Anthropogenic'] # All FORMS data is considered Anthropogenic
    }
    # Add other datasets like 'b deff' if they are included in the final preprocessed set.
}

# INTERMEDIATE_CLASSES and INTERMEDIATE_TO_FINAL_AGGREGATION_MAP are removed as per simplification. 

# 3. Professional GIS Colors for Final Target Disturbance Classes
# Colorblind-friendly palette based on cartographic best practices
# Using ColorBrewer-inspired scheme for maximum accessibility and professional appearance
DISTURBANCE_CLASS_COLORS = {
    'Fire': '#e31a1c',        # Bright red - intuitive for fire
    'Storm': '#1f78b4',       # Strong blue - natural for storm/wind
    'Anthropogenic': '#ff7f00', # Orange - human activities
    'Biotic': '#33a02c',       # Forest green - biological processes
    'Drought': '#6a3d9a',      # Purple - water stress/drought
    'Unknown': '#636363',      # Neutral gray - uncertain cases
    
    # Multi-class combinations (for non-injective mappings)
    'Fire,Drought': '#e31a1c', # Fire dominant - red
    'Storm,Biotic': '#1f78b4', # Storm dominant - blue  
    'Anthropogenic,Unknown,Drought': '#ff7f00', # Anthropogenic dominant - orange
    
    # Additional combinations that might appear
    'Biotic,Storm': '#1f78b4',     # Storm dominant
    'Fire,Storm': '#e31a1c',       # Fire dominant
    'Drought,Unknown': '#6a3d9a',  # Drought dominant
    'Anthropogenic,Unknown': '#ff7f00', # Anthropogenic dominant
} 