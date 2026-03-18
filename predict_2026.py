import pandas as pd
import numpy as np
import os

from data_loader import DataLoader, prepare_seeds
from feature_engineering import calculate_elo
from advanced_features import (
    calculate_four_factors,
    aggregate_team_stats,
    get_massey_features,
    combine_all_features
)
from model_trainer import ModelTrainer


def run_pipeline(data_path='/Users/ajeemkhank/LEARN/data'):
    loader = DataLoader(data_path)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    outputs_dir = os.path.join(script_dir, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)
    # Small self-validation loop: pick the best recent-era cutoff to reduce distribution shift.
    cutoff_candidates = [2003, 2007]

    print("Loading data...")
    m_seeds, w_seeds = loader.load_seeds()
    m_seeds, w_seeds = prepare_seeds(m_seeds, w_seeds)

    m_detail, w_detail = loader.load_detailed_results()

    m_detail = calculate_four_factors(m_detail)
    w_detail = calculate_four_factors(w_detail)

    m_stats = aggregate_team_stats(m_detail)
    w_stats = aggregate_team_stats(w_detail)

    m_massey = loader.load_massey()
    m_massey_pivot = get_massey_features(m_massey)

    m_tourney, w_tourney = loader.load_tourney_results()
    m_results, w_results = loader.load_results()

    def build_train(results, tourney, seeds, stats, massey, min_season):
        tourney = tourney[tourney["Season"] >= min_season].copy()
        elo = calculate_elo(results)

        rows = []
        for _, r in tourney.iterrows():
            rows.append({'Season': r['Season'], 'WTeamID': r['WTeamID'], 'LTeamID': r['LTeamID'], 'Target': 1})
            rows.append({'Season': r['Season'], 'WTeamID': r['LTeamID'], 'LTeamID': r['WTeamID'], 'Target': 0})

        df = pd.DataFrame(rows)
        df = combine_all_features(df, elo, seeds, stats, massey)
        return df, elo

    best_cv = None
    best_cutoff = None
    best_trainer = None
    best_feature_cols = None
    best_m_elo = None
    best_w_elo = None

    for cutoff in cutoff_candidates:
        print(f"\n=== CV selection: training seasons >= {cutoff} ===")
        m_train, m_elo = build_train(m_results, m_tourney, m_seeds, m_stats, m_massey_pivot, cutoff)
        w_train, w_elo = build_train(w_results, w_tourney, w_seeds, w_stats, None, cutoff)

        full_train = pd.concat([m_train, w_train], ignore_index=True)
        feature_cols = [c for c in full_train.columns if c not in ['Season', 'WTeamID', 'LTeamID', 'Target']]

        X = full_train[feature_cols]
        y = full_train['Target']
        groups = full_train['Season']

        trainer = ModelTrainer(
            seeds=(42, 123, 456),
            n_splits=5,
            power_grid=tuple(np.round(np.arange(1.04, 1.101, 0.01), 2)),
            smooth_grid=tuple(np.round(np.arange(0.02, 0.081, 0.01), 2)),
            weight_grid=(0.5, 0.6, 0.7),
        )
        cv = trainer.train(X, y, groups)
        print(f"CV Brier (tuned) @ cutoff {cutoff}: {cv:.4f}")

        if best_cv is None or cv < best_cv:
            best_cv = cv
            best_cutoff = cutoff
            best_trainer = trainer
            best_feature_cols = feature_cols
            best_m_elo = m_elo
            best_w_elo = w_elo

    trainer = best_trainer
    feature_cols = best_feature_cols
    m_elo = best_m_elo
    w_elo = best_w_elo
    print(f"\n=== Best cutoff selected: {best_cutoff} with CV {best_cv:.4f} ===")
    print(f"CV Brier (tuned): {best_cv:.4f}")

    print("Generating predictions...")

    sample = pd.read_csv(os.path.join(data_path, 'SampleSubmissionStage2.csv'))

    sample['Season'] = sample['ID'].apply(lambda x: int(x.split('_')[0]))
    sample['WTeamID'] = sample['ID'].apply(lambda x: int(x.split('_')[1]))
    sample['LTeamID'] = sample['ID'].apply(lambda x: int(x.split('_')[2]))

    all_seeds = pd.concat([m_seeds, w_seeds])
    all_elo = {**m_elo, **w_elo}
    all_stats = pd.concat([m_stats, w_stats])

    test = combine_all_features(sample, all_elo, all_seeds, all_stats, m_massey_pivot)

    X_test = test[feature_cols]

    sample['Pred'] = trainer.predict(X_test)

    pred_mean = float(sample['Pred'].mean())
    pred_std = float(sample['Pred'].std())
    pred_min = float(sample['Pred'].min())
    pred_max = float(sample['Pred'].max())
    print("Prediction Stats:")
    print(f"Mean: {pred_mean:.6f}")
    print(f"Std:  {pred_std:.6f}")
    print(f"Min:  {pred_min:.6f}")
    print(f"Max:  {pred_max:.6f}")

    out_path = os.path.join(outputs_dir, "submission.csv")
    sample[['ID', 'Pred']].to_csv(out_path, index=False)

    print(f"submission.csv ready at {out_path}")


if __name__ == "__main__":
    run_pipeline()