import pandas as pd
import numpy as np


# =========================
# SAFE DIVISION
# =========================
def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=(b != 0))

def _finite(s: pd.Series, fill: float = 0.0) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return s


# =========================
# FOUR FACTORS
# =========================
def calculate_four_factors(df):
    # Winner stats
    df['W_eFG'] = safe_divide(df['WFGM'] + 0.5 * df['WFGM3'], df['WFGA'])
    df['W_TO'] = safe_divide(df['WTO'], df['WFGA'] + 0.475 * df['WFTA'] + df['WTO'])
    df['W_OR'] = safe_divide(df['WOR'], df['WOR'] + df['LDR'])
    df['W_FTR'] = safe_divide(df['WFTM'], df['WFGA'])

    # Loser stats
    df['L_eFG'] = safe_divide(df['LFGM'] + 0.5 * df['LFGM3'], df['LFGA'])
    df['L_TO'] = safe_divide(df['LTO'], df['LFGA'] + 0.475 * df['LFTA'] + df['LTO'])
    df['L_OR'] = safe_divide(df['LOR'], df['LOR'] + df['WDR'])
    df['L_FTR'] = safe_divide(df['LFTM'], df['LFGA'])

    # Score margin
    df['W_ScoreDiff'] = df['WScore'] - df['LScore']
    df['L_ScoreDiff'] = df['LScore'] - df['WScore']

    # Safety
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# =========================
# TEAM AGGREGATION
# =========================
def aggregate_team_stats(df):
    # Wins
    w_stats = df.groupby(['Season', 'WTeamID']).agg({
        'W_eFG': 'mean', 'W_TO': 'mean', 'W_OR': 'mean', 'W_FTR': 'mean',
        'L_eFG': 'mean', 'L_TO': 'mean', 'L_OR': 'mean', 'L_FTR': 'mean',
        'W_ScoreDiff': ['mean', 'count']
    }).reset_index()

    w_stats.columns = [
        'Season', 'TeamID',
        'Off_eFG', 'Off_TO', 'Off_OR', 'Off_FTR',
        'Def_eFG', 'Def_TO', 'Def_OR', 'Def_FTR',
        'ScoreMargin', 'Wins'
    ]

    # Losses
    l_stats = df.groupby(['Season', 'LTeamID']).agg({
        'L_eFG': 'mean', 'L_TO': 'mean', 'L_OR': 'mean', 'L_FTR': 'mean',
        'W_eFG': 'mean', 'W_TO': 'mean', 'W_OR': 'mean', 'W_FTR': 'mean',
        'L_ScoreDiff': ['mean', 'count']
    }).reset_index()

    l_stats.columns = [
        'Season', 'TeamID',
        'Off_eFG', 'Off_TO', 'Off_OR', 'Off_FTR',
        'Def_eFG', 'Def_TO', 'Def_OR', 'Def_FTR',
        'ScoreMargin', 'Losses'
    ]

    # Merge win/loss stats
    combined = pd.merge(
        w_stats, l_stats,
        on=['Season', 'TeamID'],
        how='outer',
        suffixes=('_W', '_L')
    )

    stats_cols = [
        'Off_eFG', 'Off_TO', 'Off_OR', 'Off_FTR',
        'Def_eFG', 'Def_TO', 'Def_OR', 'Def_FTR',
        'ScoreMargin'
    ]

    final_rows = []

    for (season, team_id), group in combined.groupby(['Season', 'TeamID']):
        wins = group['Wins'].fillna(0).values[0]
        losses = group['Losses'].fillna(0).values[0]
        total = wins + losses

        if total == 0:
            continue

        row = {
            'Season': season,
            'TeamID': team_id,
            'WinRate': wins / total
        }

        for col in stats_cols:
            val_w = group[f'{col}_W'].fillna(0).values[0]
            val_l = group[f'{col}_L'].fillna(0).values[0]
            row[col] = (val_w * wins + val_l * losses) / total

        final_rows.append(row)

    out = pd.DataFrame(final_rows)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


# =========================
# MASSEY FEATURES
# =========================
def get_massey_features(massey_df, systems=['POM', 'SAG', 'MOR']):
    latest = massey_df.groupby(
        ['Season', 'SystemName', 'TeamID']
    )['RankingDayNum'].max().reset_index()

    massey_latest = massey_df.merge(
        latest,
        on=['Season', 'SystemName', 'TeamID', 'RankingDayNum']
    )

    massey_filtered = massey_latest[
        massey_latest['SystemName'].isin(systems)
    ]

    massey_pivot = massey_filtered.pivot_table(
        index=['Season', 'TeamID'],
        columns='SystemName',
        values='OrdinalRank'
    ).reset_index()

    # Fill missing rankings (conservative)
    for sys in systems:
        if sys in massey_pivot.columns:
            massey_pivot[sys] = _finite(massey_pivot[sys], 100.0).clip(1.0, 500.0)
        else:
            massey_pivot[sys] = 100.0

    return massey_pivot


# =========================
# FEATURE COMBINATION
# =========================
def combine_all_features(train_df, elo_dict, seeds_df, team_stats, massey_pivot):

    def get_team_features(df, prefix, id_col):
        # Seeds
        df = df.merge(
            seeds_df.rename(columns={'TeamID': id_col, 'Seed': f'{prefix}_Seed'}),
            on=['Season', id_col],
            how='left'
        )
        df[f'{prefix}_Seed'] = df[f'{prefix}_Seed'].fillna(10)

        # Elo (previous season)
        df[f'{prefix}_Elo'] = df.apply(
            lambda x: elo_dict.get((x['Season'] - 1, x[id_col]), 1500),
            axis=1
        )

        # Team stats
        stat_cols = [
            'Off_eFG', 'Off_TO', 'Off_OR', 'Off_FTR',
            'Def_eFG', 'Def_TO', 'Def_OR', 'Def_FTR',
            'ScoreMargin', 'WinRate'
        ]

        team_stats_renamed = team_stats.rename(
            columns={c: f'{prefix}_{c}' for c in stat_cols}
        )

        df = df.merge(
            team_stats_renamed.rename(columns={'TeamID': id_col}),
            on=['Season', id_col],
            how='left'
        )

        # Safe fill
        for col in stat_cols:
            c = f'{prefix}_{col}'
            mean_val = df[c].mean()
            df[c] = df[c].fillna(mean_val if not np.isnan(mean_val) else 0)

        # Massey
        if massey_pivot is not None:
            df = df.merge(
                massey_pivot.rename(columns={'TeamID': id_col}),
                on=['Season', id_col],
                how='left'
            )

            for col in ['POM', 'SAG', 'MOR']:
                if col in df.columns:
                    df.rename(columns={col: f'{prefix}_{col}'}, inplace=True)
                    df[f'{prefix}_{col}'] = _finite(df[f'{prefix}_{col}'], 100.0).clip(1.0, 500.0)

        return df

    # Apply for both teams
    df = get_team_features(train_df, 'T1', 'WTeamID')
    df = get_team_features(df, 'T2', 'LTeamID')

    # Differences
    df['EloDiff'] = df['T1_Elo'] - df['T2_Elo']
    df['SeedDiff'] = df['T1_Seed'] - df['T2_Seed']

    stat_cols = [
        'Off_eFG', 'Off_TO', 'Off_OR', 'Off_FTR',
        'Def_eFG', 'Def_TO', 'Def_OR', 'Def_FTR',
        'ScoreMargin', 'WinRate'
    ]

    for col in stat_cols:
        df[f'{col}_Diff'] = df[f'T1_{col}'] - df[f'T2_{col}']

    # Massey ordinals: lower is better => reverse sign (T2 - T1)
    if massey_pivot is not None:
        for col in ['POM', 'SAG', 'MOR']:
            if f'T1_{col}' in df.columns:
                df[f'{col}_Diff'] = _finite(df[f'T2_{col}'], 100.0) - _finite(df[f'T1_{col}'], 100.0)

    # Interaction
    df['Elo_Seed_Interaction'] = df['EloDiff'] * df['SeedDiff']

    # =========================
    # SAFE INTERACTION FEATURES (difference-space only)
    # =========================
    # Keep interactions few and robust; all are computed from already-safe diffs.
    df['EloDiff_x_Off_eFG_Diff'] = df['EloDiff'] * df['Off_eFG_Diff']
    df['EloDiff_x_Def_eFG_Diff'] = df['EloDiff'] * df['Def_eFG_Diff']
    # Note: this is redundant with Elo_Seed_Interaction but empirically helped in CV on this setup.
    df['SeedDiff_x_EloDiff'] = df['SeedDiff'] * df['EloDiff']

    # Light clipping to avoid extreme products (keeps numeric stability even before winsorization)
    interaction_cols = ['EloDiff_x_Off_eFG_Diff', 'EloDiff_x_Def_eFG_Diff', 'SeedDiff_x_EloDiff']
    for c in interaction_cols:
        df[c] = pd.to_numeric(df[c], errors='coerce')
        df[c] = df[c].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        df[c] = df[c].clip(-1e6, 1e6)

    # Final columns
    keep_cols = [
        'Season', 'WTeamID', 'LTeamID',
        'EloDiff', 'SeedDiff', 'Elo_Seed_Interaction'
    ]

    if 'Target' in df.columns:
        keep_cols.append('Target')

    keep_cols += [f'{col}_Diff' for col in stat_cols]
    keep_cols += interaction_cols

    if massey_pivot is not None:
        keep_cols += [
            f'{col}_Diff'
            for col in ['POM', 'SAG', 'MOR']
            if f'{col}_Diff' in df.columns
        ]

    out = df[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # Assert numeric-only
    for c in out.columns:
        if c not in ['Season', 'WTeamID', 'LTeamID', 'Target']:
            out[c] = pd.to_numeric(out[c], errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out