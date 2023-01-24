import abc
import dataclasses
import enum
import json
import os
import typing

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn import base, ensemble, metrics, model_selection

import const


@dataclasses.dataclass
class BaseExpConfig:
    rng: typing.Optional[int]
    rng_state_: np.random.RandomState
    rng_: np.random.Generator
    directory: str
    # times: int
    inner_n_jobs: int

    def __post_init__(self):
        if self.rng_ is None:
            self.rng_ = np.random.default_rng()
        else:
            self.rng_ = np.random.default_rng(self.rng)
            self.rng_state_ = self.rng

    @staticmethod
    def fill_dict(val: dict) -> dict:
        if 'rng' not in val:
            val['rng'] = None
        if 'inner_n_jobs' not in val:
            val['inner_n_jobs'] = None
        val['rng_'] = None
        val['rng_state_'] = None
        return val

    @staticmethod
    def fields_not_included_in_as_dict() -> typing.Sequence[str]:
        return ['rng_state_', 'rng_']


METRIC_NAME_ACCURACY = 'Accc'
METRIC_NAME_AUC = 'AUC'
METRIC_NAME_F1 = 'F1'
METRIC_NAME_PRECISION = 'Prec'
METRIC_NAME_RECALL = 'Rec'

METRICS_ORDER_ALL = [METRIC_NAME_ACCURACY, METRIC_NAME_AUC, METRIC_NAME_F1, METRIC_NAME_PRECISION, METRIC_NAME_RECALL]
METRIC_ORDER_BASIC = [METRIC_NAME_ACCURACY, METRIC_NAME_F1, METRIC_NAME_PRECISION, METRIC_NAME_RECALL]


class MetricType(enum.Enum):
    BASIC = 'BASIC'
    ALL = 'ALL'


class AbstractAnomalyModel(abc.ABC):

    @staticmethod
    def supported_metrics_type() -> MetricType:
        return MetricType.ALL

    @abc.abstractmethod
    def fit(self, X, y=None):
        pass

    @abc.abstractmethod
    def predict(self, X):
        pass

    @abc.abstractmethod
    def predict_and_get_components(self, X) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
        pass

    def evaluate(self, y_test, y_pred=None, X_test=None) -> pd.Series:
        args = {'y_test': y_test}
        if y_pred is not None:
            args['y_pred'] = y_pred
        else:
            args['target_model'] = self
            args['X_test'] = X_test
        output_basic = self.evaluate_basic(**args)
        if self.supported_metrics_type() == MetricType.BASIC:
            output = output_basic.reindex(METRIC_ORDER_BASIC)
        else:
            # should make the type checker happy (but it does not).
            assert hasattr(self, 'decision_function')
            # ignore this error.
            output_additional = self.evaluate_with_decision_function(X_test=X_test, y_test=y_test, target_model=self)
            output = pd.concat([output_basic, output_additional])
            output = output.reindex(METRICS_ORDER_ALL)
        return output

    @staticmethod
    def evaluate_basic(y_test, target_model=None, y_pred=None, X_test=None) -> pd.Series:
        if target_model is None and y_pred is None:
            raise ValueError('Either target model or y_pred must be not None')
        if (target_model is not None) and X_test is None or (target_model is None and X_test is not None):
            raise ValueError('target_model and X_test must be not None')
        eval_metrics_with_y_pred = [
            (METRIC_NAME_ACCURACY, metrics.accuracy_score),
            (METRIC_NAME_F1, metrics.f1_score),
            (METRIC_NAME_PRECISION, metrics.precision_score),
            (METRIC_NAME_RECALL, metrics.recall_score)
        ]
        output = pd.Series(np.zeros(len(eval_metrics_with_y_pred)), index=[m[0] for m in eval_metrics_with_y_pred])
        for metric_name, metric_func in eval_metrics_with_y_pred:
            output[metric_name] = metric_func(
                y_true=y_test,
                y_pred=target_model.predict(X_test) if target_model is not None else y_pred)
        return output

    @staticmethod
    def evaluate_with_decision_function(X_test, y_test, target_model) -> pd.Series:
        eval_metrics_with_decision_func = [
            METRIC_NAME_AUC, metrics.roc_auc_score,
        ]
        output = pd.Series(np.zeros(len(eval_metrics_with_decision_func)),
                           index=[m[0] for m in eval_metrics_with_decision_func])
        for metric_name, metric_func in eval_metrics_with_decision_func:
            output[metric_name] = metric_func(y_true=y_test, y_score=target_model.decision_function(X_test))
        return output


class AnomalyModelEnsemble(AbstractAnomalyModel):

    def __init__(self, model_args, n_components, n_jobs):
        self.models = [ensemble.IsolationForest(**model_args if model_args is not None else {})
                       for _ in range(n_components)]
        self.n_jobs = n_jobs

    def supported_metrics_type(self) -> MetricType:
        return MetricType.BASIC

    def fit(self, X, y=None):

        def fit_inner(m_, X_) -> ensemble.IsolationForest:
            return m_.fit(X_)

        if X.shape[1] != len(self.models):
            raise ValueError('Not enough values to train the models on each component')
        self.models = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(fit_inner)(
            base.clone(m), X[:, i].reshape(-1, 1)) for i, m in enumerate(self.models))

    def raw_predictions_(self, X):

        if X.shape[1] != len(self.models):
            raise ValueError(f'X.shape[1] != len(self.models): got {X.shape[1]} != {len(self.models)}')
        if not isinstance(X, np.ndarray):
            raise ValueError(f'not isinstance(X, np.ndarray got: {not isinstance(X, np.ndarray)}')

        predictions: typing.List[np.ndarray] = joblib.Parallel(n_jobs=self.n_jobs)(joblib.delayed(
            m.predict
        )(X[:, i].reshape(-1, 1)) for i, m in enumerate(self.models))
        return np.vstack(predictions).T

    @staticmethod
    def anomalous_points_(raw_predictions) -> npt.NDArray[bool]:
        # all_predictions = self.raw_predictions_(X)
        # check for each row.
        # return np.any(all_predictions == -1, axis=1)
        return np.any(raw_predictions == -1, axis=1)

    def from_raw_predictions(self, raw_predictions) -> npt.NDArray[int]:
        final_predictions = np.ones(raw_predictions.shape[0])
        final_predictions[self.anomalous_points_(raw_predictions)] = -1
        return final_predictions.astype(int)

    def predict(self, X):
        if X.shape[1] != len(self.models):
            raise ValueError('Mismatched size of X')
        # # by default 1.
        # final_predictions = np.ones(len(X))
        # # we insert -1 for the rows where an anomaly has been signalled.
        # final_predictions[self.anomalous_rows_(X)] = -1
        # return final_predictions
        return self.from_raw_predictions(self.raw_predictions_(X))

    def predict_and_get_components(
            self, X, n_jobs: typing.Optional[int] = None) -> typing.Tuple[np.ndarray, typing.List[np.ndarray]]:
        raw_predictions = self.raw_predictions_(X)

        non_raw_predictions = self.from_raw_predictions(raw_predictions=raw_predictions)

        def _inner(prediction_row_: np.ndarray):
            # here we operate on an individual prediction.
            assert len(prediction_row_.shape) == 1

            components_changed = np.argwhere(prediction_row_ == -1).flatten()

            return components_changed

        components_changed_at_each_row = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_inner)(prediction_row) for prediction_row in raw_predictions)

        return non_raw_predictions, components_changed_at_each_row


@dataclasses.dataclass
class DatasetGeneratorForTrainingConfig(BaseExpConfig):
    # how many repetitions (of course the dataset must be of size * repeats)
    # repeats: int
    size: int
    train_test_split: float = dataclasses.field(default=.75)
    perc_of_anomalous_data_points: float = dataclasses.field(default=.1)
    truncate_to: int = dataclasses.field(default=1000)

    def get_n_anomalous_points(self) -> int:
        return np.around(self.perc_of_anomalous_data_points * self.size / 1).astype(int)

    def get_n_clear_points(self) -> int:
        return self.size - self.get_n_anomalous_points()

    @abc.abstractmethod
    def generate(self):
        pass

    @abc.abstractmethod
    def as_dict(self) -> dict:
        not_included = self.fields_not_included_in_as_dict()
        output = {}

        all_keys = set(self.__annotations__.keys()).union(set(super().__annotations__.keys())) - set(not_included)

        for attr_name in all_keys:
            output[attr_name] = getattr(self, attr_name)

        return output

    @staticmethod
    def from_dict(val: dict):
        filled = DatasetGeneratorForTrainingConfig.fill_dict(val)
        return DatasetGeneratorForTrainingConfig(**filled)

    def read_files(self) -> typing.Tuple[pd.DataFrame, pd.DataFrame]:
        directory = os.path.abspath(self.directory)

        files = os.listdir(directory)
        # we are interested in csv files only.
        files = list(filter(lambda file_name: file_name.endswith('.csv') and file_name != 'no-interference.csv', files))
        files_full_path = list(map(lambda file_name: os.path.join(directory, file_name), files))

        # now, we read the clean dataset, dropping the column
        # 'trace_id' which are not interested in.
        clean_df = pd.read_csv(os.path.join(directory, 'no-interference.csv'), nrows=self.truncate_to).drop(
            'trace_id', axis=1)

        # now we read anomalous data.
        # we are interesting only in one column. Namely,
        # the column containing anomalous data. In each file, we only
        # one column containing anomalous data. The name of this column
        # corresponds to the file name.
        anomalous_df = pd.read_csv(files_full_path[0], nrows=self.truncate_to
                                   )[[os.path.basename(files_full_path[0].replace('.csv', ''))]]
        for file in files_full_path[1:]:
            df = pd.read_csv(file)
            anomalous_df = anomalous_df.join(df[[os.path.basename(file.replace('.csv', ''))]])

        if len(clean_df.columns) != len(anomalous_df.columns):
            raise ValueError('len(clean_df.columns) != len(anomalous_df.columns), got: '
                             '{len(clean_df.columns)} != {len(anomalous_df.columns)}')

        # we now need to rename each column of anomalous_df and clean with COMP{col_idx}
        new_cols = [f'{const.COMP_PREFIX}{i}' for i in range(len(clean_df.columns))]
        clean_df.columns = new_cols
        anomalous_df.columns = new_cols

        if len(anomalous_df.columns) != len(files_full_path):
            raise ValueError(f'There are {len(files_full_path)} in the directory excluding clean, but '
                             f'anomalous df has {len(anomalous_df.columns)} columns')

        return clean_df, anomalous_df


class AbstractDatasetGeneratorForTraining(abc.ABC):

    def __init__(self, config: DatasetGeneratorForTrainingConfig, clean_df: pd.DataFrame, anomalous_df: pd.DataFrame):
        self.config = config
        self.clean_df = clean_df
        self.anomalous_df = anomalous_df
        self.components_list = np.arange(len(self.clean_df.columns))
        self.components_list_name = self.clean_df.columns

        self.components_list_name_dtype = self.components_list_name.to_numpy().dtype
        self.dataset: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def extract_x_y(dataset: pd.DataFrame) -> typing.Tuple[np.ndarray, np.ndarray]:
        y = dataset[['Y']].values
        remaining_columns = [col for col in dataset.columns if col != 'Y']
        X = dataset[remaining_columns].values
        return X, y

    @abc.abstractmethod
    def generate(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def get_model(self) -> AbstractAnomalyModel:
        pass

    def get_dataset(self) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if len(self.dataset) == 0:
            raise ValueError('Dataset not created yet')
        X, y = self.extract_x_y(self.dataset)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=self.config.rng_state_)
        return X_train, X_test, y_train, y_test

    def get_and_fit_model(self) -> typing.Tuple[AbstractAnomalyModel, np.ndarray, np.ndarray]:
        X_train, X_test, _, y_test = self.get_dataset()
        model = self.get_model()

        model.fit(X_train)
        return model, X_test, y_test


class DatasetGeneratorForTrainingEnsemble(AbstractDatasetGeneratorForTraining):

    def generate(self):
        n_anomalous_points_needed = self.config.get_n_anomalous_points()
        n_clean_points_needed = self.config.get_n_clear_points()

        # start the easier way, select random clear points.
        reindex = self.config.rng_.permutation(self.clean_df.index)

        extracted_clean_data = self.clean_df.iloc[reindex[:n_clean_points_needed]]

        #  now, we permute the columns independently.
        anomalous_permuted_df = pd.DataFrame()
        anomalous_permuted_df[self.anomalous_df.columns] = self.config.rng_.permuted(self.anomalous_df, axis=0)

        # and we extract the required data points.
        extracted_anomalous_data = anomalous_permuted_df.iloc[anomalous_permuted_df.index[:n_anomalous_points_needed]]

        extracted_anomalous_data['Y'] = -1
        extracted_clean_data['Y'] = 1

        self.dataset = pd.concat([extracted_clean_data, extracted_anomalous_data])
        return self.dataset

    def get_model(self) -> AbstractAnomalyModel:
        return AnomalyModelEnsemble(
            model_args={'contamination': self.config.perc_of_anomalous_data_points},
            n_components=len(self.components_list), n_jobs=self.config.inner_n_jobs)


EVAL_NAME_AVG = 'AVG'
EVAL_NAME_STD = 'STD'
EVL_NAME_VAR = 'VAR'

COLUMN_NAME_NAME = 'Name'


def to_series_with_name(name: str) -> pd.Series:
    return pd.Series([name], index=[COLUMN_NAME_NAME])


def concat(*, output_avg: pd.Series, output_std: pd.Series, output_var: pd.Series) -> pd.Series:
    return pd.concat([output.rename(lambda col_name: f'{output_prefix}({col_name})')
                      for output, output_prefix in zip([output_avg, output_std, output_var],
                                                       [EVAL_NAME_AVG, EVAL_NAME_STD, EVL_NAME_VAR])])


def get_summary(raw_results: typing.List[pd.Series]) -> typing.Tuple[pd.Series, pd.DataFrame]:
    aggregated = pd.DataFrame(raw_results)
    output_avg = aggregated.mean(axis=0)
    output_std = aggregated.std(axis=0)
    output_var = aggregated.var(axis=0)
    return concat(output_avg=output_avg, output_var=output_var, output_std=output_std), aggregated


@dataclasses.dataclass
class AggregatedOutput:
    raw_list_of_results: pd.DataFrame
    aggregated: pd.Series

    @staticmethod
    def from_list_of_individual(raw_results: typing.List[pd.Series], common_part: pd.Series) -> "AggregatedOutput":
        series, df = get_summary(raw_results)
        df[common_part.index] = common_part
        return AggregatedOutput(raw_list_of_results=df, aggregated=pd.concat([series, common_part]))

    @staticmethod
    def merge(a: "AggregatedOutput", b: "AggregatedOutput",
              l_func: typing.Callable[[str], str], r_func: typing.Callable[[str], str]) -> "AggregatedOutput":

        # first we rename all_summary.
        a_aggregated, b_aggregated = a.aggregated.rename(l_func), b.aggregated.rename(r_func)
        # then aggregated.
        a_raw = a.raw_list_of_results.rename(l_func, axis='columns')
        b_raw = b.raw_list_of_results.rename(r_func, axis='columns')

        return AggregatedOutput(
            aggregated=pd.concat([a_aggregated, b_aggregated]),
            raw_list_of_results=a_raw.join(b_raw))

    @staticmethod
    def from_list_of_others(values: typing.List["AggregatedOutput"]) -> "AggregatedOutput":

        to_aggregate = [value.aggregated for value in values]
        to_concatenate = [value.raw_list_of_results for value in values]

        aggregated_series, _ = get_summary(to_aggregate)

        return AggregatedOutput(raw_list_of_results=pd.concat(to_concatenate), aggregated=aggregated_series)

    def export(self, base_directory_aggregated: str, base_directory_raw: str,
               prefix_func: typing.Optional[typing.Callable[[str], str]] = None):
        prefix_func_ = prefix_func if prefix_func is not None else identity
        export_multi(self.raw_list_of_results, os.path.join(base_directory_raw, prefix_func_('raw')))
        export_multi(self.raw_list_of_results, os.path.join(base_directory_aggregated, prefix_func_('normalized')))


def export_multi(data: pd.DataFrame, base_name: str, include_strip_down: bool = False):
    os.makedirs(os.path.dirname(base_name), exist_ok=True)

    def _inner_export(base_name_: str, data_: pd.DataFrame):
        for full_name, func in [(f'{base_name_}.csv', data_.to_csv), (f'{base_name_}.xlsx', data_.to_excel)]:
            func(full_name)

    if include_strip_down:
        stripped_columns = [col for col in data if isinstance(col, str)
                            and EVAL_NAME_AVG in col and EVL_NAME_VAR not in col and
                            EVAL_NAME_STD not in col]
        stripped_df = data[stripped_columns]

        _inner_export(base_name_=f'{base_name}_stripped', data_=stripped_df)
    _inner_export(base_name_=base_name, data_=data)


def identity(a):
    return a


def df_json(df: typing.List[npt.NDArray[int]]) -> typing.List[str]:
    return [json.dumps(df[i].tolist()) for i in range(len(df))]


def extract_components(comp: typing.List[npt.NDArray[int]],
                       idx: typing.Union[typing.List[int], npt.NDArray[int]]) -> typing.List[npt.NDArray[int]]:
    return [comp[i] for i in idx]
