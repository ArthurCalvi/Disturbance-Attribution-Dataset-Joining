DCLASS_SCORE = {
    'senfseidl': {
        'Storm,Biotic': {'biotic-mortality':0.12, 'storm':0.35, 'biotic-dieback':0.35, 'drought-dieback':0.12, 'fire':0.02, 'anthropogenic':0.04},
        'Fire' : {'fire':0.87, 'storm':0.1, 'biotic-dieback':0.01, 'drought-dieback':0.01, 'anthropogenic':0.01},
        'Other' : {'storm':0.06, 'fire':0.03, 'biotic-mortality':0.13, 'biotic-dieback':0.13, 'drought-dieback':0.13, 'anthropogenic':0.52}
    },
    'dfde': {
        'Biotic':{'biotic-mortality':0.25, 'biotic-dieback':0.75},
        'Storm':{'storm':1.},
        'Fire':{'fire':1.},
        'Drought':{'drought-dieback':1.},
        'Other':{'anthropogenic':1.}
    },
    'nfi': {
        'Mortality':{'biotic-mortality':1.0},
        'Storm':{'storm':1.},
        'Fire':{'fire':1.},
        'Other':{'anthropogenic':0.33, 'drought-dieback':0.33, 'biotic-dieback':0.33},
        'Anthropogenic':{'anthropogenic':1.} 
    },
    'hm': {
        'Biotic-mortality':{'biotic-mortality':1.},
        'Biotic-dieback': {'biotic-dieback':1.},
        'Storm':{'storm':1.},
        'Fire':{'fire':1.},
        'Drought-dieback':{'drought-dieback':1.},
        'Other':{'anthropogenic':1.},
    },
    'cdi': {
    'drought': {'drought-dieback':1.},
    },
    'forms': {
    'clear-cut': {'anthropogenic':1.},
    },
    'bdiff': {
        'Fire':{'fire':1.}
    },
    'firepolygons': {
        'Fire':{'fire':1.}
    },
    'reference': {
        'None':{'biotic-mortality':0., 'biotic-dieback':0., 'storm':0., 'fire':0., 'drought-dieback':0., 'anthropogenic':0.},
    }}

custom_color_mapping = {
    'fire': 'orangered',
    'storm': 'royalblue',
    'biotic-dieback': 'gold',
    'biotic-mortality': 'darkgoldenrod',  # Yellow-brown
    'drought-dieback': 'darkkhaki',
    'anthropogenic': 'cadetblue'  # Light blue/grey
}

loading_dict = {
    "dfde" : '../data/processed_datasets/simplified_refined_DFDE_1984_2021_EPSG2154_FR.parquet',
    "hm" : '../data/processed_datasets/simplified_health-monitoring_2007-2023_EPSG2154_FR.parquet',
    "nfi" : '../data/processed_datasets/simplified_PascalSchneider_NFI_2003-2021_EPSG2154_FR.parquet',
    "senfseidl" : "../data/processed_datasets/simplified_SenfSeidl_joined_EPSG2154_FR.parquet", 
    "bdiff" : '../data/processed_datasets/simplified_bdiff_2012_2022_FR_EPSG2154.parquet',
    "cdi" : '../data/processed_datasets/simplified_CDI_2012_2023_EPSG2154_FR.parquet',
    "forms" : '../data/processed_datasets/simplified_FORMS_clearcut_2017_2020_EPSG2154.parquet',
    "firepolygons" : '../data/processed_datasets/simplified_firepolygons_2017_2020_FR_EPSG2154.parquet'
}

temporal_buffer = 2 
dtypes = {'dfde': 'polygon', 'hm': 'point', 'nfi': 'point', 'senfseidl': 'point', 'bdiff': 'polygon', 'firepolygons': 'polygon', 'cdi':'polygon', 'forms':'point'}

ddataset_profile = {
    'dfde': {
        'spatial': ('offset_gaussian', {'offset': 150, 'decrease': 5000}), #offset srt( min(area) / pi), k sqrt(median(area) / pi)
        'temporal': ('step', {'start': 0, 'end': 365})
    },
    'hm': {
        'spatial': ('offset_gaussian', {'offset': 10, 'decrease': 100}),
        'temporal': ('step', {'start': 0, 'end': 365})
    },
    'nfi': {
        'spatial': ('offset_gaussian', {'offset': 600, 'decrease': 25}),
        'temporal': ('step', {'start': 0, 'end': 5 * 365})
    },
    'bdiff': {
        'spatial': ('weighting_function', {'x0': 500, 'k': 500}),
        'temporal': ('gaussian', {'mean': 0, 'std': 30})
    },
    'firepolygons': {
        'spatial': ('offset_gaussian', {'offset': 10, 'decrease': 50}),
        'temporal': ('offset_gaussian', {'offset': 7, 'decrease': 31})
    },
    'senfseidl': {
        'spatial': ('offset_gaussian', {'offset': 30, 'decrease': 5 * 30}),
        'temporal': ('offset_gaussian', {'offset': 1 * 365, 'decrease': 1.5 * 365})
    },
    'forms': {
        'spatial': ('offset_gaussian', {'offset': 10, 'decrease': 5*10}),
        'temporal': ('step', {'start': 0, 'end': 365})
    },
    'cdi': {
        'spatial': ('offset_gaussian', {'offset': 5000, 'decrease': 500}),
        'temporal': ('offset_gaussian', {'offset': 1 * 365, 'decrease': 365})
    },
    'reference': {
        'spatial': ('offset_gaussian', {'offset': 0, 'decrease': 3 * 50}),
        'temporal': ('step', {'start': 0, 'end': 365})
    },
}

#visible disturbances
ddisturbance_profile = {
    'fire': {
        'spatial': ('gaussian', {'mean': 0, 'std': 500}),
        'temporal': ('gaussian', {'mean': 0, 'std': 3*365})
    },
    'storm': {
        'spatial': ('gaussian', {'mean': 0, 'std': 2000}),
        'temporal': ('gaussian', {'mean': 0, 'std': 1.5 * 365})
    },
    'biotic-dieback': {
        'spatial': ('gaussian', {'mean': 0, 'std': 1000}),
        'temporal': ('gaussian', {'mean': 0, 'std': 365})
    },
    'drought-dieback': {
        'spatial': ('gaussian', {'mean': 0, 'std': 2500}),
        'temporal': ('gaussian', {'mean': 0, 'std': 2*365})
    },
    'biotic-mortality': {
        'spatial': ('gaussian', {'mean': 0, 'std': 250}),
        'temporal': ('gaussian', {'mean': 0, 'std': 3*365})
    },
    'anthropogenic': {
        'spatial': ('gaussian', {'mean': 0, 'std': 500}),
        'temporal': ('gaussian', {'mean': 0, 'std': 1 * 365})
    }
}

doa = {'dfde': 1.0, 'hm': 1.0, 'nfi': 1.0, 'senfseidl': .75, 'bdiff': 1.0, 'cdi':.75, 'forms':0.75, 'firepolygons':1.0}
dsbuffer = {'dfde': None, 'hm': 5000, 'nfi': 7000, 'senfseidl': 100, 'bdiff': None, 'cdi':100, 'forms':100, 'firepolygons':100}
