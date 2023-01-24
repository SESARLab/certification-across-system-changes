import json
import os
import shutil

import numpy as np
import pandas as pd
import pytest

import entrypoint
from exp_quality_scheme_test import get_good_config


@pytest.mark.parametrize('include_strip_down', [True])
def test_main_func(include_strip_down: bool):
    config_size, config_group_size, datasets_size, dataset_names, config_names, config_group_names, dgt_1, dgt_2, \
        configs_1, configs_2 = get_good_config()

    base_dir = 'Res'
    dataset_dir = os.path.join(base_dir, 'Input')
    output_dir = os.path.join(base_dir, 'Out')

    dataset_dir_1 = os.path.join(dataset_dir, dgt_1.config.directory)
    dataset_dir_2 = os.path.join(dataset_dir, dgt_2.config.directory)

    dirs_to_create = [base_dir, output_dir, dataset_dir_1, dataset_dir_2, dataset_dir]
    for single_dir in dirs_to_create:
        os.makedirs(single_dir, exist_ok=True)

    n_components = 10

    clean_df = pd.DataFrame(np.random.default_rng().choice([16, 17], size=(120, n_components)),)
    anomalous_df = clean_df * 100

    # we also need to add a column 'trace_id'
    clean_df['trace_id'] = 1
    anomalous_df['trace_id'] = 2

    dgt_1.config.directory = dataset_dir_1
    dgt_2.config.directory = dataset_dir_2

    # we also need to create the datasets.
    clean_df.to_csv(os.path.join(dataset_dir_1, 'no-interference.csv'), index=False)
    clean_df.to_csv(os.path.join(dataset_dir_2, 'no-interference.csv'), index=False)

    for i in range(n_components):
        anomalous_df.to_csv(os.path.join(dataset_dir_1, f'{i}.csv'), index=False)
        anomalous_df.to_csv(os.path.join(dataset_dir_2, f'{i}.csv'), index=False)

    overall_config = [
        entrypoint.OverallConfig(dataset_training=dgt_1.config, exp_configs=configs_1),
        entrypoint.OverallConfig(dataset_training=dgt_2.config, exp_configs=configs_2)
    ]

    # export the config.
    config_file_name = os.path.join(dataset_dir, 'output.json')

    with open(config_file_name, 'w') as config_file:
        config_file.write(json.dumps([config.as_dict() for config in overall_config]))

    try:
        entrypoint.main_func(config_file_name=config_file_name, output_directory=output_dir,
                             include_strip_down=include_strip_down)
    finally:
        shutil.rmtree(base_dir)
