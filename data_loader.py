import pandas as pd
import os

class DataLoader:
    def __init__(self, data_path='/Users/ajeemkhank/LEARN/data'):
        self.data_path = data_path

    def load_teams(self):
        m_teams = pd.read_csv(os.path.join(self.data_path, 'MTeams.csv'))
        w_teams = pd.read_csv(os.path.join(self.data_path, 'WTeams.csv'))
        return m_teams, w_teams

    def load_seeds(self):
        m_seeds = pd.read_csv(os.path.join(self.data_path, 'MNCAATourneySeeds.csv'))
        w_seeds = pd.read_csv(os.path.join(self.data_path, 'WNCAATourneySeeds.csv'))
        return m_seeds, w_seeds

    def load_results(self):
        m_results = pd.read_csv(os.path.join(self.data_path, 'MRegularSeasonCompactResults.csv'))
        w_results = pd.read_csv(os.path.join(self.data_path, 'WRegularSeasonCompactResults.csv'))
        return m_results, w_results

    def load_detailed_results(self):
        m_results = pd.read_csv(os.path.join(self.data_path, 'MRegularSeasonDetailedResults.csv'))
        w_results = pd.read_csv(os.path.join(self.data_path, 'WRegularSeasonDetailedResults.csv'))
        return m_results, w_results

    def load_tourney_results(self):
        m_tourney = pd.read_csv(os.path.join(self.data_path, 'MNCAATourneyCompactResults.csv'))
        w_tourney = pd.read_csv(os.path.join(self.data_path, 'WNCAATourneyCompactResults.csv'))
        return m_tourney, w_tourney

    def load_massey(self):
        return pd.read_csv(os.path.join(self.data_path, 'MMasseyOrdinals.csv'))

def prepare_seeds(m_seeds, w_seeds):
    m_seeds['Seed'] = m_seeds['Seed'].apply(lambda x: int(x[1:3]))
    w_seeds['Seed'] = w_seeds['Seed'].apply(lambda x: int(x[1:3]))
    return m_seeds, w_seeds
