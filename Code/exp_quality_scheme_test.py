import itertools
import typing

import numpy as np
import pandas as pd

import base
import dataset_generator
import exp_quality_scheme
import exp_quality_scheme_support


def get_base_dgt_config(dataset_size: int) -> base.DatasetGeneratorForTrainingConfig:
    return base.DatasetGeneratorForTrainingConfig(size=dataset_size, directory='', rng_state_=None, rng_=None,
                                                  truncate_to=100, rng=None, inner_n_jobs=2)


def get_base_quality_scheme_config(dataset_scheme_size: int) -> exp_quality_scheme_support.QualitySchemeConfig:
    return exp_quality_scheme_support.QualitySchemeConfig(
        dataset_config=dataset_generator.DatasetGeneratorConfigLinearProba(
            group='P1.X', name='P1.1', wanted_size=dataset_scheme_size, use_detailed_config_series=True,
        ), directory='', rng_state_=None, rng_=None, rng=None, inner_n_jobs=2)


def get_dg_generator_for_training(
        dataset_size: int,
        size: typing.Tuple[int, int]) -> exp_quality_scheme.DatasetGeneratorForTrainingAndModel:
    clean_df = pd.DataFrame(np.random.default_rng().choice([17, 15], size=size),
                            columns=[f'COMP{i}' for i in range(size[1])])
    anomalous_df = clean_df * 50
    generator_clazz = base.DatasetGeneratorForTrainingEnsemble

    config = get_base_dgt_config(dataset_size=dataset_size)

    return exp_quality_scheme.DatasetGeneratorForTrainingAndModel(config=config, clean_df=clean_df,
                                                                  anomalous_df=anomalous_df,
                                                                  generator_clazz=generator_clazz)


def get_exp(dataset_training_size: int, dataset_scheme_size: int,
            size: typing.Tuple[int, int], fit: bool = False) -> exp_quality_scheme.ExperimentQualityScheme:
    generator = get_dg_generator_for_training(dataset_size=dataset_training_size, size=size)
    if fit:
        generator.fit()
    config = get_base_quality_scheme_config(dataset_scheme_size=dataset_scheme_size)
    return exp_quality_scheme.ExperimentQualityScheme(config=config, dataset_generator_and_model=generator)


def test_dataset_generator_for_training_and_model_basic():
    dg = get_dg_generator_for_training(dataset_size=100, size=(120, 10))
    dg.fit()
    assert np.all((dg.model_metrics_0 > .8) & (dg.model_metrics_0 <= 1))


def test_run_single():
    dataset_training_size = 100
    dataset_scheme_size = 100
    exp = get_exp(dataset_training_size=dataset_training_size, dataset_scheme_size=dataset_scheme_size,
                  size=(120, 10), fit=True)
    result = exp.run_single_execution_single_config()

    config_to_s = exp.config.dataset_config.to_series(include_name_and_dataset=True)

    assert config_to_s.isin(result.evaluation_scheme).all()
    assert config_to_s.isin(result.evaluation_model).all()
    # on the pd.DataFrame is more difficult. Leave it as is.


def test_run_multi_execution_same_config():
    dataset_training_size = 100
    dataset_scheme_size = 100
    exp = get_exp(dataset_training_size=dataset_training_size, dataset_scheme_size=dataset_scheme_size,
                  size=(120, 10), fit=True)
    result = exp.run_multi_execution_on_single_config()
    # pd.DataFrame([result.aggregated_results.aggregated_evaluation_scheme]).to_excel('early_agg.xlsx')
    # result.aggregated_results.evaluation_scheme.to_excel('early.xlsx')
    assert len(result.aggregated_results.individuals) == result.config.n_run_for_each
    assert len(result.aggregated_results.evaluation_scheme) == result.config.n_run_for_each
    assert len(result.aggregated_results.evaluation_model) == result.config.n_run_for_each
    assert np.all(result.aggregated_results.training_dataset == exp.dtg.dataset)


def test_run_multi_config():
    # config_groups = 5
    # config_for_each_group = 5

    config_size = 5

    # dgs = [get_dg_generator_for_training(dataset_size=100, size=(120, 10)) for _ in range(config_groups)]
    # exps = [get_exp_from_dgt(dgt=dgs[i], dataset_scheme_size=120)
    #         for _ in range(config_for_each_group) for i in range(config_groups)]
    #
    # exp_quality_scheme.ExperimentQualityScheme.run_multi_config(configs=None)

    # dgt_configs = [get_base_dgt_config(dataset_size=100) for ]

    dgt = get_dg_generator_for_training(dataset_size=100, size=(120, 10))

    configs = [get_base_quality_scheme_config(dataset_scheme_size=120) for _ in range(config_size)]

    experiments = exp_quality_scheme.ExperimentQualityScheme.run_multi_config(dataset_training=dgt,
                                                                              quality_config=configs)
    assert len(experiments) == config_size
    for single_exp, set_config in zip(experiments, configs):
        assert len(single_exp.aggregated_results.individuals) == set_config.n_run_for_each


def get_good_config():
    dataset_name_1 = 'dir1'
    dataset_name_2 = 'dir2'

    dgt_1 = get_dg_generator_for_training(dataset_size=100, size=(120, 10))
    dgt_1.config.directory = dataset_name_1

    dgt_2 = get_dg_generator_for_training(dataset_size=100, size=(120, 10))
    dgt_2.config.directory = dataset_name_2

    config_size = 5
    config_group_size = 2

    datasets_size = 2

    configs_1 = [get_base_quality_scheme_config(dataset_scheme_size=120) for _ in range(config_size)
                 for _ in range(config_group_size)]

    configs_2 = [get_base_quality_scheme_config(dataset_scheme_size=120) for _ in range(config_size)
                 for _ in range(config_group_size)]

    config_group_names = set()
    config_names = set()
    dataset_names = {dataset_name_1, dataset_name_2}

    i = 0
    # now rename name and groups
    for group_idx in range(config_group_size):

        group_name = f'P{group_idx}.X'
        config_group_names.add(group_name)
        for single_config_idx in range(config_size):
            config_name = f'P{group_idx}.{single_config_idx}'
            config_names.add(config_name)

            configs_1[i].dataset_config.group = group_name
            configs_2[i].dataset_config.group = group_name

            configs_1[i].dataset_config.name = config_name
            configs_2[i].dataset_config.name = config_name

            configs_1[i].dataset_config.dataset_name = dataset_name_1
            configs_1[i].directory = dataset_name_1
            configs_2[i].dataset_config.dataset_name = dataset_name_2
            configs_2[i].directory = dataset_name_2
            i += 1

    return config_size, config_group_size, datasets_size, dataset_names, config_names, config_group_names, dgt_1, dgt_2, configs_1, configs_2


def test_overall():
    # dataset_name_1 = 'dir1'
    # dataset_name_2 = 'dir2'
    #
    # dgt_1 = get_dg_generator_for_training(dataset_size=100, size=(120, 10))
    # dgt_1.config.directory = dataset_name_1
    #
    # dgt_2 = get_dg_generator_for_training(dataset_size=100, size=(120, 10))
    # dgt_2.config.directory = dataset_name_2
    #
    # config_size = 5
    # config_group_size = 2
    #
    # datasets_size = 2
    #
    # configs_1 = [get_base_quality_scheme_config(dataset_scheme_size=120) for _ in range(config_size)
    #              for _ in range(config_group_size)]
    #
    # configs_2 = [get_base_quality_scheme_config(dataset_scheme_size=120) for _ in range(config_size)
    #              for _ in range(config_group_size)]
    #
    # config_group_names = set()
    # config_names = set()
    # dataset_names = {dataset_name_1, dataset_name_2}
    #
    # i = 0
    # # now rename name and groups
    # for group_idx in range(config_group_size):
    #
    #     group_name = f'P{group_idx}.X'
    #     config_group_names.add(group_name)
    #     for single_config_idx in range(config_size):
    #         config_name = f'P{group_idx}.{single_config_idx}'
    #         config_names.add(config_name)
    #
    #         configs_1[i].dataset_config.group = group_name
    #         configs_2[i].dataset_config.group = group_name
    #
    #         configs_1[i].dataset_config.name = config_name
    #         configs_2[i].dataset_config.name = config_name
    #
    #         configs_1[i].dataset_config.dataset_name = dataset_name_1
    #         configs_1[i].directory = dataset_name_1
    #         configs_2[i].dataset_config.dataset_name = dataset_name_2
    #         configs_2[i].directory = dataset_name_2
    #         i += 1

    config_size, config_group_size, datasets_size, dataset_names, config_names, config_group_names, dgt_1, dgt_2, configs_1, configs_2 = get_good_config()

    exps_1 = exp_quality_scheme.ExperimentQualityScheme.run_multi_config(dataset_training=dgt_1,
                                                                         quality_config=configs_1)

    exps_2 = exp_quality_scheme.ExperimentQualityScheme.run_multi_config(dataset_training=dgt_2,
                                                                         quality_config=configs_2)

    all_exps = list(itertools.chain(exps_1, exps_2))

    aggregated = exp_quality_scheme.aggregate_all_results(all_exps)

    print(list(aggregated.grouped_by_dataset.keys()))
    print(list(aggregated.grouped_by_config_group.keys()))
    print(list(aggregated.grouped_by_config.keys()))
    print(list(aggregated.grouped_by_config_group_and_dataset.keys()))

    assert len(aggregated.grouped_by_dataset) == datasets_size
    assert len(aggregated.grouped_by_config_group) == config_group_size
    assert len(aggregated.grouped_by_config) == config_size * config_group_size
    assert len(aggregated.grouped_by_config_group_and_dataset) == config_group_size * datasets_size
    assert len(aggregated.results_of_each) == config_group_size * config_size * datasets_size

    assert set(aggregated.grouped_by_dataset.keys()).issubset(dataset_names)
    assert set(aggregated.grouped_by_config.keys()).issubset(config_names)
    assert set(aggregated.grouped_by_config_group.keys()).issubset(config_group_names)
