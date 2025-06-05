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
        'drought': 'Drought', # CDI directly indicates Drought
        '_default_': 'Drought'
    },
    'hm': {
        # Outputs of _get_class in hm.py mapped directly to Final Target Classes
        'Drought-dieback': 'Drought',
        'Fire': 'Fire',
        'Storm': 'Storm',
        'Biotic': 'Biotic', 
        'Other': 'Unknown', # Or 'Anthropogenic' if 'Other' usually implies human activity
    },
    'firepolygons': {
        'Fire': 'Fire',
        'FIRE': 'Fire',
        '_default_': 'Fire' # All firepolygon data is considered Fire
    },
    'forms': {
        'clear-cut': 'Anthropogenic',
        '_default_': 'Anthropogenic' # All FORMS data is considered Anthropogenic
    }
    # Add other datasets like 'b deff' if they are included in the final preprocessed set.
}

# INTERMEDIATE_CLASSES and INTERMEDIATE_TO_FINAL_AGGREGATION_MAP are removed as per simplification. 