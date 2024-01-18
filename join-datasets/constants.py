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

# DCLASS ={
#     'senfseidl': {
#         'Storm,Biotic': ['biotic-mortality', 'storm', 'biotic-dieback'],
#         'Fire' : ['fire'],
#         'Other' : ['biotic-dieback', 'drought-dieback', 'anthropogenic']
#     },
#     'dfde': {
#         'Biotic':['biotic-mortality', 'biotic-dieback'],
#         'Storm':['storm'],
#         'Fire':['fire'],
#         'Drought':['drought-dieback'],
#         'Other':['anthropogenic']
#     },
#     'nfi': {
#         'Mortality':['biotic-mortality'],
#         'Storm':['storm'],
#         'Fire':['fire'],
#         'Other':['anthropogenic', 'drought-dieback', 'biotic-dieback'],
#         'Anthropogenic':['anthropogenic']
#     },
#     'hm': {
#         'Biotic':['biotic-mortality', 'biotic-dieback'],
#         'Storm':['storm'],
#         'Fire':['fire'],
#         'Drought-dieback':['drought-dieback'],
#         'Other':['anthropogenic']
#     },
#     'bdiff': {
#         'Fire':['fire']
#     },
#     'reference': {
#         'None':['biotic-mortality', 'biotic-dieback', 'storm', 'fire', 'drought-dieback', 'anthropogenic'],
#         None:['biotic-mortality', 'biotic-dieback', 'storm', 'fire', 'drought-dieback', 'anthropogenic']
#     }
# }