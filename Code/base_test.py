import typing

import numpy as np
import pandas as pd
import pytest

import base


def get_generator_with_standard_config(
        size: int,
        n_columns: typing.Optional[int] = None,
        inner_n_jobs: typing.Optional[int] = None, ) -> base.DatasetGeneratorForTrainingEnsemble:
    n_columns = n_columns if n_columns is not None else 25

    config = base.DatasetGeneratorForTrainingConfig(
        size=size, directory='', inner_n_jobs=inner_n_jobs, rng_state_=None, rng_=None, rng=None)

    clean_df = pd.DataFrame(np.random.default_rng().choice([16, 17], size=(size, n_columns)))
    anomalous_df = clean_df * 10
    return base.DatasetGeneratorForTrainingEnsemble(config=config, clean_df=clean_df, anomalous_df=anomalous_df)


@pytest.mark.parametrize('config, clean_df, anomalous_df', [
    (
            base.DatasetGeneratorForTrainingConfig(
                size=15,
                directory='',
                inner_n_jobs=1,
                rng_=None,
                rng=None,
                rng_state_=None,
            ),
            pd.DataFrame([np.arange(10) for _ in range(20)]),
            pd.DataFrame([np.arange(10) for _ in range(20)]) * 100,
    )
])
def test_generator_generate(config: base.DatasetGeneratorForTrainingConfig,
                            clean_df: pd.DataFrame, anomalous_df: pd.DataFrame):
    generator = base.DatasetGeneratorForTrainingEnsemble(config=config, clean_df=clean_df, anomalous_df=anomalous_df)
    generated = generator.generate()

    assert len(generated) == config.size

    for col in generated.columns:
        if col != 'Y':
            assert np.all(np.in1d(generated[generated['Y'] == 1][col].values, clean_df[col].values))
            assert np.all(np.in1d(generated[generated['Y'] == -1][col].values, anomalous_df[col].values))

    expected_neg = np.around(config.perc_of_anomalous_data_points * config.size).astype(int)

    assert len(generated[generated['Y'] == - 1]) == expected_neg
    assert len(generated[generated['Y'] == 1]) == config.size - expected_neg

    # check that it does not crash :)
    generator.get_dataset()


def test_fit_model():
    generator = get_generator_with_standard_config(50)
    generator.generate()
    model, X_test, y_test = generator.get_and_fit_model()

    assert isinstance(model, base.AnomalyModelEnsemble)
    assert len(model.models) == len(generator.clean_df.columns)


# raw_predictions are horizontally stacked, i.e., assume that we have 4 data points and 3 components,
# the predictions will be
#   c1  c2  c3
#   1   -1  -1
#   1    1  -1
#   1    1   1
#   1   -1  -1
# where each column represents the prediction for a component and each row represents the data point.
@pytest.mark.parametrize('raw_predictions, expected_predictions', [
    (
            np.array([[1, 1, 1, -1], [1, 1, 1, -1]]).T,
            np.array([1, 1, 1, -1])
    ),
    (
            np.array([[-1, 1, 1, -1], [1, -1, 1, -1]]).T,
            np.array([-1, -1, 1, -1])
    ),
    (
            np.array([[-1, 1, -1, -1], [1, -1, 1, -1]]).T,
            np.array([-1, -1, -1, -1])
    )
])
def test_from_raw_predictions(raw_predictions: np.ndarray, expected_predictions: np.ndarray):
    generator = get_generator_with_standard_config(50)
    generator.generate()

    model, X_test, y_test = generator.get_and_fit_model()
    model: base.AnomalyModelEnsemble = model

    got_predictions = model.from_raw_predictions(raw_predictions=raw_predictions)
    assert np.all(got_predictions == expected_predictions)


def test_predictions_on_model():
    generator = get_generator_with_standard_config(50, inner_n_jobs=5, n_columns=10)
    generator.generate()
    model, X_test, y_test = generator.get_and_fit_model()

    result = model.predict(X_test)

    assert len(result) == len(X_test)
    # at least 1 prediction is correct.
    assert np.any(result == y_test)


def test_predictions_and_components_on_model():
    generator = get_generator_with_standard_config(50, inner_n_jobs=5, n_columns=3)
    generator.generate()

    model, _, _ = generator.get_and_fit_model()

    # now create an ad hoc X_test

    anomalous_data_points_c1 = np.arange(1, 10)
    anomalous_data_points_c2 = np.arange(1, 5)
    anomalous_data_points_c3 = np.array([7, 8, 9, 11])

    # in total, we have anomalous components at [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]

    X_test = generator.clean_df.copy(deep=True)

    # the first 10 data points are anomalous for the first component.
    X_test.loc[anomalous_data_points_c1, [0]] = generator.clean_df.iloc[anomalous_data_points_c1][[0]] * 100

    # the first 5 data points are anomalous also for the second component.
    X_test.loc[anomalous_data_points_c2, [1]] = generator.clean_df.iloc[anomalous_data_points_c2][[1]] * 100

    # data points [7, 8, 9, 11] are anomalous for the third component.
    X_test.loc[anomalous_data_points_c3, [2]] = generator.clean_df.iloc[anomalous_data_points_c3][[2]] * 100

    # contains the indices where at least one component is anomaly.
    all_anomalous = np.union1d(anomalous_data_points_c1, np.union1d(anomalous_data_points_c2, anomalous_data_points_c3))

    predictions, involved_components = model.predict_and_get_components(X_test.values)

    assert len(predictions) == len(involved_components)
    assert len(predictions) == len(X_test)

    # check that all anomalous data have been caught.
    assert np.all(np.in1d(all_anomalous, np.argwhere(predictions == -1).flatten()))

    # and now check that also components matches.
    expected_anomalous_component = [np.array([]),      # 0
                                    np.array([0, 1]),   # 1
                                    np.array([0, 1]),   # 2
                                    np.array([0, 1]),   # 3
                                    np.array([0, 1]),   # 4
                                    np.array([0]),      # 5
                                    np.array([]),       # 6
                                    np.array([0]),      # 7
                                    np.array([0, 2]),   # 8
                                    np.array([0, 2]),   # 9
                                    np.array([0, 2]),   # 10
                                    np.array([2])]      # 11

    # # note there won't be perfect match because of how data is created.
    # # so we are interested only in some part, that is, those corresponding to the data that we have
    # # injected with anomaly
    # # use all_anomalous which contains such indices.
    # for i, our in enumerate(all_anomalous):
    #     try:
    #         print(f'({our}): Expected: {expected_anomalous_component[i]}, got: {involved_components[our]}, '
    #               f'at {X_test.iloc[our]}')
    #         assert np.all(np.in1d(expected_anomalous_component[i], involved_components[our])) or \
    #             np.any(np.in1d(expected_anomalous_component[i], involved_components[our]))
    #     except IndexError:
    #         pass
    for i in all_anomalous:
        # print(f'at {i}: {expected_anomalous_component[i]}, got: {involved_components[i]}')
        assert np.all(np.in1d(expected_anomalous_component[i], involved_components[i])) or \
               np.any(np.in1d(expected_anomalous_component[i], involved_components[i]))

    non_anomalous = np.setdiff1d(np.arange(len(X_test)), all_anomalous)

    for i in non_anomalous:
        if len(involved_components[i]) != 0:
            print(f'{i}: {involved_components[i]}')
        assert len(involved_components[i]) == 0
