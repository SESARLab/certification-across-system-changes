import json
import typing

import numpy as np
import pandas as pd
import pytest

import const
import dataset_generator
from const import COLUMN_NAME_CHANGE_TYPE, CHANGE_CAUSE_NO, COLUMN_NAME_CHANGED_COMP, \
    CHANGE_CAUSE_CODE, COLUMN_NAME_CHANGED_COMP_PRIMARY, COLUMN_NAME_CHANGED_COMP_SECONDARY, CHANGE_CAUSE_CODE_ENV


@pytest.mark.parametrize('config, base_proba, n_components', [
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10
            ),
            .6,
            10
    )
])
def test_probabilities_for_n_components(config: dataset_generator.DatasetGeneratorConfig, base_proba, n_components):
    proba = config.get_probabilities_for_n_components(base_proba=base_proba, n_components=n_components)
    assert len(proba) == n_components
    assert np.sum(proba) == 1.0 or np.sum(proba) == 0.9999999999999999


@pytest.mark.parametrize('config, base_proba, n_components', [
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10
            ),
            .6,
            10
    )
])
def test_get_n_components_involved_in_change(config: dataset_generator.DatasetGeneratorConfig, base_proba,
                                             n_components):
    n_changes = 10
    chosen_n = config.get_n_components_involved_in_change(n_total_components=n_components, base_proba=base_proba,
                                                          n_changes=n_changes)
    assert len(chosen_n) == n_changes
    # check that return number of components in each slot is correct
    assert np.all(np.in1d(chosen_n, np.arange(1, n_components+1)))


@pytest.mark.parametrize('config, base_proba, n_components', [
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10
            ),
            .6,
            10
    )
])
def test_get_components_involved_in_change(config: dataset_generator.DatasetGeneratorConfig, base_proba,
                                           n_components):
    n_changes = 10
    chosen_components = config.get_components_involved_in_change(n_total_components=n_components, base_proba=base_proba,
                                                                 n_changes=n_changes)
    assert len(chosen_components) == n_changes
    # check that the size of each sub-array is not higher than the total number of components.
    n_chosen = np.array(list(map(lambda arr: len(arr), chosen_components)))
    assert np.all(np.in1d(n_chosen, np.arange(n_components)))
    # check that each returned is valid. We can easily flatten.
    assert np.all(np.in1d(np.concatenate(chosen_components), np.arange(n_components)))
    for single_chosen_components in chosen_components:
        assert len(np.unique(single_chosen_components)) == len(single_chosen_components)


@pytest.mark.parametrize('config, primary_proba, n_components', [
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10
            ),
            .6,
            10
    )
])
def test_get_components_in_change_code(config: dataset_generator.DatasetGeneratorConfig, primary_proba, n_components):
    n_changes = 10
    chosen_components = config.get_components_involved_in_change_code(n_total_components=n_components,
                                                                      primary_proba=primary_proba, n_changes=n_changes)
    for component_primarily_involved, component_secondarily_involved, \
            components_regardless_involved in chosen_components.zip():
        # check that size matches
        assert len(component_primarily_involved) + len(component_secondarily_involved) == len(
            components_regardless_involved)
        # that primary and secondary are contained in the overall.
        assert np.all(np.in1d(np.union1d(component_primarily_involved, component_secondarily_involved),
                              components_regardless_involved))
        # that primary and secondary do not have any elements in common.
        assert len(np.intersect1d(component_primarily_involved, component_secondarily_involved)) == 0
        assert len(np.unique(component_primarily_involved)) == len(component_primarily_involved)
        assert len(np.unique(component_secondarily_involved)) == len(component_secondarily_involved)
        # and that values are ok as we did above
        assert np.all(np.in1d(components_regardless_involved, np.arange(n_components)))
        assert len(component_secondarily_involved) <= n_components


@pytest.mark.parametrize('config, expected_n_minor, expected_n_major', [
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10
            ),
            None, None
    ),
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10,
                proba_minor_vs_major=1.0
            ),
            10, 0
    ),
    (
            dataset_generator.DatasetGeneratorConfigLinearProba(
                group='P1.X',
                name='P1.1',
                rng=None,
                wanted_size=10,
                proba_minor_vs_major=0
            ),
            0, 10
    )
])
def test_get_code_extent(config: dataset_generator.DatasetGeneratorConfig, expected_n_minor, expected_n_major):
    n_changes = 10
    code_extent = config.get_code_extent(n_changes=n_changes)
    assert len(code_extent) == n_changes
    assert np.all(np.isin(code_extent, np.array([dataset_generator.CODE_EXTENT_MINOR,
                                                 dataset_generator.CODE_EXTENT_MAJOR])))
    if expected_n_major is not None:
        assert len(np.argwhere(code_extent == dataset_generator.CODE_EXTENT_MAJOR)) == expected_n_major
    if expected_n_minor is not None:
        assert len(np.argwhere(code_extent == dataset_generator.CODE_EXTENT_MINOR)) == expected_n_minor


@pytest.mark.parametrize('target, columns, how_many, expected_extraction', [
    (
            pd.DataFrame(np.repeat(17, repeats=17 * 3).reshape(17, 3), columns=[f'{const.COMP_PREFIX}{i}'
                                                                                for i in range(3)]),
            [0, 1],
            10,
            pd.DataFrame(np.repeat(17, repeats=10 * 2).reshape(10, 2), columns=[f'{const.COMP_PREFIX}{i}'
                                                                                for i in range(2)])
    )
])
def test_dg_random_extract_from(target: pd.DataFrame, columns: typing.List[int], how_many: int,
                                expected_extraction: typing.Optional[pd.DataFrame]):
    got = dataset_generator.DatasetGenerator.random_extract_from(target=target, how_many=how_many,
                                                                 columns=columns)
    assert len(got) == how_many
    if expected_extraction is not None:
        assert np.all(np.equal(got.values, expected_extraction.values))


def get_reasonable_config(wanted_size: int) -> dataset_generator.DatasetGeneratorConfig:
    return dataset_generator.DatasetGeneratorConfigLinearProba(
        group='P1.X', name='P1.1', rng=None, wanted_size=wanted_size, n_jobs_inner=3)


def get_reasonable_config_and_df(
        wanted_size: int, n_components: int
) -> typing.Tuple[dataset_generator.DatasetGenerator, dataset_generator.DatasetGeneratorConfig,
pd.DataFrame, pd.DataFrame]:
    config = get_reasonable_config(wanted_size)

    clean_df = pd.DataFrame(np.repeat(17, repeats=wanted_size * n_components).reshape(wanted_size, n_components))
    anomalous_df = clean_df * 3

    generator = dataset_generator.DatasetGenerator(config=config, clean_df=clean_df, anomalous_df=anomalous_df)

    return generator, config, clean_df, anomalous_df


def test_dg_random_extract_with_anomalies():
    generator, config, clean_df, anomalous_df = get_reasonable_config_and_df(wanted_size=20, n_components=3)

    anomalous_components = [np.array([0, 1]), np.array([0, 2]), np.array([1, 2])]

    extracted = generator.random_extract_with_anomalies(anomalous_components_idx=anomalous_components)

    # check that values lies in source.
    assert len(extracted) == len(anomalous_components)
    assert np.all(np.in1d(extracted.values.flatten(), pd.concat([clean_df, anomalous_df]).values.flatten()))


def test_generate():
    n_components = 3
    print()
    generator, config, clean_df, anomalous_df = get_reasonable_config_and_df(wanted_size=20, n_components=n_components)

    generated = generator.generate(verbose=True)

    # the first thing to do is to check that values for components are
    # actually good.
    components_columns = np.arange(n_components)

    all_values = np.union1d(clean_df.values.flatten(), anomalous_df.values.flatten())

    # we check that values are correct.
    assert np.all(np.in1d(generated[components_columns].values.flatten(), all_values))

    # we check that for rows where there is a change in behavior, there is at least
    # one anomalous value.
    anomalous_rows = generated[(generated[COLUMN_NAME_CHANGE_TYPE] != CHANGE_CAUSE_NO) &
                               (generated[COLUMN_NAME_CHANGE_TYPE] != CHANGE_CAUSE_CODE)].index
    # otherwise indexing is preserved and we have difficulty in using iloc.
    values = generated.iloc[anomalous_rows]  # .reset_index(drop=False)
    # components_changed = generated.loc[anomalous_rows, COLUMN_NAME_CHANGED_COMP]
    components_changed = values[COLUMN_NAME_CHANGED_COMP]
    primary_components_changed = values[COLUMN_NAME_CHANGED_COMP_PRIMARY]
    secondary_components_changed = values[COLUMN_NAME_CHANGED_COMP_SECONDARY]

    assert len(components_changed) == len(primary_components_changed)
    assert len(components_changed) == len(secondary_components_changed)
    assert len(anomalous_rows) == len(components_changed)
    assert len(values) == len(anomalous_rows)

    for index, comp_all, primary_comp, secondary_comp in zip(anomalous_rows.tolist(),
                                                             components_changed,
                                                             primary_components_changed,
                                                             secondary_components_changed):
        decoded_components = sorted(json.loads(comp_all))

        decoded_components_primary = sorted(json.loads(primary_comp))
        decoded_components_secondary = sorted(json.loads(secondary_comp))

        assert np.all(np.in1d(decoded_components, np.arange(n_components)))
        assert np.any(np.in1d(values.loc[index, np.arange(n_components)], anomalous_df.values.flatten()))
        # this does not always happen, only in case of CODE_ENV
        if values.loc[index, CHANGE_CAUSE_CODE] == CHANGE_CAUSE_CODE_ENV:
            assert np.all(np.union1d(decoded_components_primary, decoded_components_secondary) == decoded_components)

    other_rows = np.setdiff1d(np.arange(len(generated)), anomalous_rows)

    # check that, for rows not involved in behavior changes, all behavioral data are correct.
    assert np.all(np.in1d(generated.loc[other_rows, np.arange(n_components)], clean_df.values.flatten()))

    # now check that for rows where there has been a change of type CODE
    # components are valid and only present in PRIMARY.
    code_only_rows = generated[generated[COLUMN_NAME_CHANGE_TYPE] == CHANGE_CAUSE_CODE].index

    for row in code_only_rows:
        decoded_components = json.loads(generated.loc[row, COLUMN_NAME_CHANGED_COMP])
        decoded_components_primary = json.loads(generated.loc[row, COLUMN_NAME_CHANGED_COMP_PRIMARY])
        decoded_components_secondary = json.loads(generated.loc[row, COLUMN_NAME_CHANGED_COMP_SECONDARY])
        assert np.all(np.equal(decoded_components, decoded_components_primary))
        assert len(decoded_components_secondary) == 0
        # values are not anomalous
        assert np.all(np.in1d(generated.loc[row, np.arange(n_components)], clean_df.values.flatten()))


@pytest.mark.parametrize('critical_mask, components, expected', [
    (
        np.array([False, True, False, True, True]),
        [np.array([0, 1]), np.array([0, 2]), np.array([1, 3, 4]), np.array([0, 4]), np.arange(5), np.array([])],
        np.array([True, False, True, True, True, False])
    ),
    (
            np.array([False, False, False, False, False]),
            [np.array([0, 1]), np.array([0, 2]), np.array([1, 3, 4]), np.array([0, 4]), np.arange(5)],
            np.array([False, False, False, False, False])
    )
])
def test_are_critical(critical_mask: np.ndarray, components: typing.List[np.ndarray],
                      expected: typing.List[bool]):
    gen = dataset_generator.DatasetGenerator(config=dataset_generator.DatasetGeneratorConfigLinearProba(
        group='P1.X', name='P1.1', wanted_size=10), clean_df=pd.DataFrame(), anomalous_df=pd.DataFrame())
    gen.components_criticality = critical_mask

    got = gen.are_critical(components)

    print(type(got))

    assert np.all(got == expected)
