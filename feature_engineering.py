import pandas as pd
import numpy as np

def calculate_elo(df):
    """Calculate Elo ratings per team for each season independently."""
    elo_dict = {} # Key: (Season, TeamID)
    
    for season in sorted(df['Season'].unique()):
        season_df = df[df['Season'] == season]
        current_elo = {} # Local elo for this season
        
        K = 20
        # Initial ranking for a season is 1500 for everyone
        for _, row in season_df.iterrows():
            w_team, l_team = row['WTeamID'], row['LTeamID']
            w_elo = current_elo.get(w_team, 1500)
            l_elo = current_elo.get(l_team, 1500)
            
            # Expected win probability
            p_w = 1 / (1 + 10 ** ((l_elo - w_elo) / 400))
            
            # Update
            current_elo[w_team] = w_elo + K * (1 - p_w)
            current_elo[l_team] = l_elo - K * (1 - p_w)
            
        # Store for this season
        for team, elo in current_elo.items():
            elo_dict[(season, team)] = elo
            
    return elo_dict

def get_features(train_df, elo_dict, seeds_df):
    # Merge seeds
    train_df = train_df.merge(seeds_df[['Season', 'TeamID', 'Seed']], left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
    train_df.rename(columns={'Seed': 'WSeed'}, inplace=True)
    train_df.drop('TeamID', axis=1, inplace=True)
    
    train_df = train_df.merge(seeds_df[['Season', 'TeamID', 'Seed']], left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
    train_df.rename(columns={'Seed': 'LSeed'}, inplace=True)
    train_df.drop('TeamID', axis=1, inplace=True)
    
    # Fill missing seeds with a high number (unranked)
    train_df['WSeed'] = train_df['WSeed'].fillna(20)
    train_df['LSeed'] = train_df['LSeed'].fillna(20)
    
    # Seed difference
    train_df['SeedDiff'] = train_df['WSeed'] - train_df['LSeed']
    
    # ELO difference
    train_df['WELO'] = train_df['WTeamID'].map(elo_dict)
    train_df['LELO'] = train_df['LTeamID'].map(elo_dict)
    train_df['ELODiff'] = train_df['WELO'] - train_df['LELO']
    
    return train_df
