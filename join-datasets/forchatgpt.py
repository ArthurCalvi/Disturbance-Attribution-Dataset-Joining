import pandas as pd

group = pd.read_csv('group_sample.csv')

#compute weights
def spatial_weight(x):
    if x <= 1:
        return 1
    else: 
        return 1 - (x-1)/9 
    
def temporal_weight(x):
    if x <= 3:
        return 1 - x/12
    else: 
        return 0.75 * (1 - (x-3)/3)

from thefuzz import fuzz

def compute_tree_coherence(row, reference):

    for essence in reference['essence'].split(','):
        if fuzz.token_set_ratio(row['essence'].lower(), essence.lower()) > 80:
            return 1
        
    if row['tree_type'].lower() == reference['tree_type'].lower():
        return 0.75 
    
    if row['tree_type'].lower() == 'mixed' or reference['tree_type'].lower() == 'mixed':
        return 0.5

    return 0.25 

def compute_weight(row, reference):
    # spatial distance, spatial weight, temporal distance, temporal weight, tree correspondance weight, overall accuracy
    if row['dataset'] == 'senfseidl':
        return 0, 1, 0, 1, 1, 0.91, 0.91  
    elif row['dataset'] == 'dfde':
        sd = (row['geometry'].area / 1e6)** (1/2) / 35
        oa = 0.95
    elif row['dataset'] in ['hm', 'nfi']:
        sd = row['geometry'].centroid.distance(reference['geometry'].centroid) / 1e3
        oa = 0.9

    if row['dataset'] in ['dfde', 'nfi']:
        td = min(abs(reference['year'] - row['start_date'].year), abs(row['end_date'].year - reference['year']))
    elif row['dataset'] == 'hm':
        td = min(abs(reference['year'] - row['year']), abs(row['year'] - reference['year']))
    
    tc = compute_tree_coherence(row[['tree_type', 'essence']], reference[['tree_type', 'essence']])
    sw = spatial_weight(sd)
    
    tw = temporal_weight(td)

    return sd, sw, td, tw, tc, oa, sw * tw * tc * oa

group[['sd', 'sw', 'td', 'tw', 'tc', 'oa', 'p']] = group.apply(lambda x: compute_weight(x, group.iloc[0]), axis=1, result_type='expand')