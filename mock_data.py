import pandas as pd
import numpy as np
import os

def create_mock_data(data_path='data'):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    
    # Mock Teams
    m_teams = pd.DataFrame({
        'TeamID': range(1001, 1011),
        'TeamName': [f'MTeam_{i}' for i in range(1, 11)]
    })
    m_teams.to_csv(os.path.join(data_path, 'MTeams.csv'), index=False)
    
    w_teams = pd.DataFrame({
        'TeamID': range(3001, 3011),
        'TeamName': [f'WTeam_{i}' for i in range(1, 11)]
    })
    w_teams.to_csv(os.path.join(data_path, 'WTeams.csv'), index=False)
    
    # Mock Seeds
    seasons = [2021, 2022, 2023, 2024, 2025]
    m_seeds = []
    w_seeds = []
    for s in seasons:
        for i, tid in enumerate(range(1001, 1011)):
            m_seeds.append({'Season': s, 'Seed': f'W{(i%16)+1:02d}', 'TeamID': tid})
        for i, tid in enumerate(range(3001, 3011)):
            w_seeds.append({'Season': s, 'Seed': f'W{(i%16)+1:02d}', 'TeamID': tid})
            
    pd.DataFrame(m_seeds).to_csv(os.path.join(data_path, 'MNCAATourneySeeds.csv'), index=False)
    pd.DataFrame(w_seeds).to_csv(os.path.join(data_path, 'WNCAATourneySeeds.csv'), index=False)
    
    # Mock Regular Season Results
    results = []
    for s in seasons:
        for _ in range(100):
            w_tid = np.random.choice(range(1001, 1011))
            l_tid = np.random.choice(range(1001, 1011))
            if w_tid != l_tid:
                results.append({'Season': s, 'DayNum': 50, 'WTeamID': w_tid, 'WScore': 80, 'LTeamID': l_tid, 'LScore': 70, 'WLoc': 'H', 'NumOT': 0})
    pd.DataFrame(results).to_csv(os.path.join(data_path, 'MRegularSeasonCompactResults.csv'), index=False)
    
    w_results = []
    for s in seasons:
        for _ in range(100):
            w_tid = np.random.choice(range(3001, 3011))
            l_tid = np.random.choice(range(3001, 3011))
            if w_tid != l_tid:
                w_results.append({'Season': s, 'DayNum': 50, 'WTeamID': w_tid, 'WScore': 80, 'LTeamID': l_tid, 'LScore': 70, 'WLoc': 'H', 'NumOT': 0})
    pd.DataFrame(w_results).to_csv(os.path.join(data_path, 'WRegularSeasonCompactResults.csv'), index=False)

    # Mock Tourney Results
    tourney = []
    for s in seasons:
        for _ in range(20):
            w_tid = np.random.choice(range(1001, 1011))
            l_tid = np.random.choice(range(1001, 1011))
            if w_tid != l_tid:
                tourney.append({'Season': s, 'DayNum': 140, 'WTeamID': w_tid, 'WScore': 80, 'LTeamID': l_tid, 'LScore': 70, 'WLoc': 'N', 'NumOT': 0})
    pd.DataFrame(tourney).to_csv(os.path.join(data_path, 'MNCAATourneyCompactResults.csv'), index=False)
    
    w_tourney = []
    for s in seasons:
        for _ in range(20):
            w_tid = np.random.choice(range(3001, 3011))
            l_tid = np.random.choice(range(3001, 3011))
            if w_tid != l_tid:
                w_tourney.append({'Season': s, 'DayNum': 140, 'WTeamID': w_tid, 'WScore': 80, 'LTeamID': l_tid, 'LScore': 70, 'WLoc': 'N', 'NumOT': 0})
    pd.DataFrame(w_tourney).to_csv(os.path.join(data_path, 'WNCAATourneyCompactResults.csv'), index=False)

    # Mock Sample Submission Stage 2
    matchups = []
    team_ids = list(range(1001, 1011)) + list(range(3001, 3011))
    for i in range(len(team_ids)):
        for j in range(i+1, len(team_ids)):
            t1, t2 = sorted([team_ids[i], team_ids[j]])
            # Check if both teams are in same pool (men or women)
            if (t1 < 2000 and t2 < 2000) or (t1 > 2000 and t2 > 2000):
                matchups.append({'ID': f'2026_{t1}_{t2}', 'Pred': 0.5})
    pd.DataFrame(matchups).to_csv(os.path.join(data_path, 'SampleSubmissionStage2.csv'), index=False)

    print("Mock data created in 'data' directory.")

if __name__ == "__main__":
    create_mock_data()
