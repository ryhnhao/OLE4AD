import pandas as pd

def parquet_to_csv(parquet_file_path, csv_file_path):
    df = pd.read_parquet(parquet_file_path)
    df.to_csv(csv_file_path, index=False)

# parquet_file_path = '/data/haoruiyang_pro/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/2024.05.28.20.47.39/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2024.05.28.20.47.39.parquet'
# csv_file_path = 'pdm_corrected_states_CLS_N.csv'
# parquet_to_csv(parquet_file_path, csv_file_path)

# parquet_file_path = '/data/haoruiyang_pro/nuplan/exp/exp/simulation/closed_loop_reactive_agents/2024.05.28.21.38.44/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2024.05.28.21.38.44.parquet'
# csv_file_path = 'pdm_corrected_states_CLS_R.csv'
# parquet_to_csv(parquet_file_path, csv_file_path)

# parquet_file_path = '/data/haoruiyang_pro/nuplan/exp/exp/simulation/open_loop_boxes/pdm_corrected_states/aggregator_metric/open_loop_boxes_weighted_average_metrics_2024.05.28.19.44.39.parquet'
# csv_file_path = 'pdm_corrected_states_OLS.csv'
# parquet_to_csv(parquet_file_path, csv_file_path)

parquet_file_path = '/data/haoruiyang_pro/nuplan/exp/exp/simulation/closed_loop_nonreactive_agents/2024.05.28.23.30.45/aggregator_metric/closed_loop_nonreactive_agents_weighted_average_metrics_2024.05.28.23.30.45.parquet'
csv_file_path = '/data/haoruiyang_pro/tuplan_garage/scripts/analysis/idm_CLS_N.csv'
parquet_to_csv(parquet_file_path, csv_file_path)

parquet_file_path = '/data/haoruiyang_pro/nuplan/exp/exp/simulation/closed_loop_reactive_agents/2024.05.29.00.05.18/aggregator_metric/closed_loop_reactive_agents_weighted_average_metrics_2024.05.29.00.05.18.parquet'
csv_file_path = '/data/haoruiyang_pro/tuplan_garage/scripts/analysis/idm_CLS_R.csv'
parquet_to_csv(parquet_file_path, csv_file_path)

parquet_file_path = '/data/haoruiyang_pro/nuplan/exp/exp/simulation/open_loop_boxes/idm/aggregator_metric/open_loop_boxes_weighted_average_metrics_2024.05.28.23.22.06.parquet'
csv_file_path = '/data/haoruiyang_pro/tuplan_garage/scripts/analysis/idm_OLS.csv'
parquet_to_csv(parquet_file_path, csv_file_path)