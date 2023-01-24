import abc
import dataclasses
import enum
import itertools
import json
import multiprocessing
import os.path
import typing

import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn import metrics, preprocessing

import base
import const
import dataset_generator
import situation

T = typing.TypeVar('T')


class Comparable2(abc.ABC):

    def compare_to(self, other) -> pd.Series:
        pass


@dataclasses.dataclass
class BasicResultContainer:
    accuracy: float
    f1: float

    def to_series(self, measure_name: str) -> pd.Series:
        return pd.Series([self.accuracy, self.f1],
                         index=[const.apply_col_name(const.OUTPUT_COLUMN_NAME_ACC, measure_name),
                                const.apply_col_name(const.OUTPUT_COLUMN_NAME_F1, measure_name)])


@dataclasses.dataclass
class ResultContainer(BasicResultContainer):
    precision: float
    recall: float

    def to_series(self, measure_name: str) -> pd.Series:
        this_series = pd.Series([self.precision, self.recall],
                                index=[const.apply_col_name(const.OUTPUT_COLUMN_NAME_PREC, measure_name),
                                       const.apply_col_name(const.OUTPUT_COLUMN_NAME_REC, measure_name), ])
        return pd.concat([super().to_series(measure_name), this_series])


def binarizer(a_true: npt.NDArray[T], b_pred: npt.NDArray[T]) -> typing.Tuple[npt.NDArray[T], npt.NDArray[T]]:
    assert len(a_true.shape) == 1
    assert len(b_pred.shape) == 1

    highest = np.max(np.concatenate([a_true, b_pred]))

    a_true_binarized = np.zeros(highest + 1)
    b_pred_binarized = np.zeros(highest + 1)

    a_true_binarized[a_true] = 1
    b_pred_binarized[b_pred] = 1

    return a_true_binarized, b_pred_binarized


def multi_binarizer(a_true: typing.List[npt.NDArray[T]], b_pred: typing.List[npt.NDArray[T]]
                    ) -> typing.Tuple[npt.NDArray[T], npt.NDArray[T]]:
    assert len(a_true) == len(b_pred)

    binarizer_transformer = preprocessing.MultiLabelBinarizer().fit(itertools.chain(a_true, b_pred))

    a_true_binarized = binarizer_transformer.transform(a_true)
    b_pred_binarized = binarizer_transformer.transform(b_pred)

    return a_true_binarized, b_pred_binarized


class BinarizerType(enum.Enum):
    NONE = 'NONE'
    SIMPLE = 'SIMPLE'
    MULTI = 'MULTI'

    def binarize(self, a_true: typing.Union[npt.NDArray[T], typing.List[npt.NDArray[T]]],
                 b_pred: typing.Union[npt.NDArray[T], typing.List[npt.NDArray[T]]]
                 ) -> typing.Tuple[npt.NDArray[T], npt.NDArray[T]]:
        if self == BinarizerType.NONE:
            return a_true, b_pred
        elif self == BinarizerType.SIMPLE:
            return binarizer(a_true, b_pred)
        else:
            return multi_binarizer(a_true, b_pred)


def generic_compare_to(binarize: BinarizerType, a_true: np.ndarray, b_pred: np.ndarray,
                       average: str = 'binary', ) -> ResultContainer:
    """
    NOTE: an accuracy or another metric > 1 appears also where they both *not* catch the incorrect situation.
    :param binarize:
    :param a_true:
    :param b_pred:
    :param average:
    :return:
    """
    if len(a_true) == 0 and len(b_pred) == 0:
        # if there are no components at all, there is nothing to retrieve
        return ResultContainer(
            accuracy=np.nan,
            precision=np.nan,
            recall=np.nan,
            f1=np.nan,
        )

    a_true, b_pred = binarize.binarize(a_true, b_pred)

    return ResultContainer(
        accuracy=metrics.accuracy_score(y_true=a_true, y_pred=b_pred),
        precision=metrics.precision_score(y_true=a_true, y_pred=b_pred, average=average),
        recall=metrics.recall_score(y_true=a_true, y_pred=b_pred, average=average),
        f1=metrics.f1_score(y_true=a_true, y_pred=b_pred, average=average)
    )


@dataclasses.dataclass
class NonePartialFullReCertificationOutput(Comparable2):
    none: np.ndarray
    partial: np.ndarray
    full: np.ndarray

    def compare_to(self, other: "NonePartialFullReCertificationOutput") -> pd.Series:
        none = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.none, b_pred=other.none).to_series(
            measure_name=const.OUTPUT_COLUMN_NAME_RECERT_NO)
        partial = generic_compare_to(
            binarize=BinarizerType.SIMPLE, a_true=self.partial, b_pred=other.partial).to_series(
            measure_name=const.OUTPUT_COLUMN_NAME_RECERT_PARTIAL)
        full = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.full, b_pred=other.full).to_series(
            measure_name=const.OUTPUT_COLUMN_NAME_RECERT_FULL)
        return pd.concat([none, partial, full])


@dataclasses.dataclass
class SituationsOutput(Comparable2):
    """
    Wraps the indices of each detected situation.
    """
    S0: np.ndarray
    S1: np.ndarray
    S2: np.ndarray
    S3: np.ndarray

    def compare_to(self, other) -> pd.Series:
        s0 = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.S0, b_pred=other.S0).to_series(
            const.OUTPUT_COLUMN_NAME_S0_OK)
        s1 = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.S1, b_pred=other.S1).to_series(
            const.OUTPUT_COLUMN_NAME_S1_OK)
        s2 = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.S2, b_pred=other.S2).to_series(
            const.OUTPUT_COLUMN_NAME_S2_OK)
        s3 = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.S3, b_pred=other.S3).to_series(
            const.OUTPUT_COLUMN_NAME_S3_OK)

        return pd.concat([s0, s1, s2, s3])

    @staticmethod
    def from_series(series: pd.Series) -> "SituationsOutput":
        return SituationsOutput(
            S0=np.argwhere(series.values == str(situation.S0())).flatten(),
            S1=np.argwhere(series.values == str(situation.S1())).flatten(),
            S2=np.argwhere(series.values == str(situation.S2())).flatten(),
            S3=np.argwhere(series.values == str(situation.S3())).flatten(),
        )


@dataclasses.dataclass
class ChangesOfTypeOutput(Comparable2):
    changes_all: np.ndarray
    changes_code: np.ndarray
    changes_code_env: np.ndarray
    changes_env: np.ndarray

    def compare_to(self, other) -> pd.Series:
        changes_all = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.changes_all,
                                         b_pred=other.changes_all).to_series(
            const.OUTPUT_COLUMN_NAME_CHANGE_TYPE_ALL)
        changes_code = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.changes_code,
                                          b_pred=other.changes_code).to_series(
            const.OUTPUT_COLUMN_NAME_CHANGE_TYPE_CODE)
        changes_code_env = generic_compare_to(
            binarize=BinarizerType.SIMPLE, a_true=self.changes_code_env, b_pred=other.changes_code_env).to_series(
            const.OUTPUT_COLUMN_NAME_CHANGE_TYPE_CODE_ENV)
        changes_env = generic_compare_to(binarize=BinarizerType.SIMPLE, a_true=self.changes_env,
                                         b_pred=other.changes_env).to_series(
            const.OUTPUT_COLUMN_NAME_CHANGE_TYPE_ENV)
        return pd.concat([changes_all, changes_code, changes_code_env, changes_env])


@dataclasses.dataclass
class ChangedComponentsPerSituationOutput(Comparable2):
    s_less: typing.List[np.ndarray]

    def compare_to(self, other) -> pd.Series:
        assert len(self.s_less) == len(other.s_less)
        return generic_compare_to(binarize=BinarizerType.MULTI, a_true=self.s_less, b_pred=other.s_less,
                                  average='macro').to_series(const.OUTPUT_COLUMN_NAME_CORRECT_COMPONENTS)


def get_partial_and_full_our(source: pd.Series) -> NonePartialFullReCertificationOutput:
    """
    Counts the number of no action, partial, and full re-certification performed
    according to our scheme (exact and applied). It takes as input the right column.
    :param source:
    :return:
    """
    source_values = source.values
    none = np.argwhere(source_values == str(situation.S0())).flatten()
    partial = np.argwhere((source_values == str(situation.S1())) | (source_values == str(situation.S2()))).flatten()
    full = np.argwhere(source_values == str(situation.S3())).flatten()
    return NonePartialFullReCertificationOutput(
        none=none,
        partial=partial,
        full=full)


def get_partial_and_full_stota(source: pd.DataFrame) -> NonePartialFullReCertificationOutput:
    """
    no changes -> no changes
    minor code not touching critical componentes -> partial
    major code w/ and w/o touching critical components, minor code touching critical componentes -> full
    (also vuln but we did not consider it).
    :param source:
    :return:
    """

    # none: = no changes.
    none = source[source[const.COLUMN_NAME_CHANGE_TYPE_STOTA] == const.CHANGE_CAUSE_NO].index
    # partial: =
    # - code change with minor change
    partial = source[(source[const.COLUMN_NAME_CHANGE_TYPE_STOTA] == const.CHANGE_CAUSE_CODE) &
                     (source[const.COLUMN_NAME_CODE_EXTENT_STOTA] == const.CODE_EXTENT_MINOR) &
                     (source[const.COLUMN_NAME_CRITICAL_STOTA] == const.CRITICAL_FALSE)].index
    # full: =
    # - code change with major change
    # - code change with impact on critical components
    full = source[((source[const.COLUMN_NAME_CHANGE_TYPE_STOTA] == const.CHANGE_CAUSE_CODE) &
                   (source[const.COLUMN_NAME_CODE_EXTENT_STOTA] == const.CODE_EXTENT_MAJOR)) |
                  ((source[const.COLUMN_NAME_CHANGE_TYPE_STOTA] == const.CHANGE_CAUSE_CODE) &
                   (source[const.COLUMN_NAME_CRITICAL_STOTA] == const.CRITICAL_TRUE))].index
    return NonePartialFullReCertificationOutput(
        none=none,
        partial=partial,
        full=full)


def get_changes_of_type(eval_type: const.EvalType, dataset: pd.DataFrame) -> ChangesOfTypeOutput:
    changes_caught = dataset[dataset[eval_type.col_name(const.COLUMN_NAME_CHANGE_TYPE)] !=
                             const.CHANGE_CAUSE_NO].index
    # retrieve the changes caught for each type.
    changes_code = dataset[dataset[eval_type.col_name(const.COLUMN_NAME_CHANGE_TYPE)] ==
                           const.CHANGE_CAUSE_CODE].index
    changes_code_env = dataset[dataset[eval_type.col_name(const.COLUMN_NAME_CHANGE_TYPE)] ==
                               const.CHANGE_CAUSE_CODE_ENV].index
    changes_env = dataset[dataset[eval_type.col_name(const.COLUMN_NAME_CHANGE_TYPE)] ==
                          const.CHANGE_CAUSE_ENV].index
    return ChangesOfTypeOutput(
        changes_all=changes_caught,
        changes_code=changes_code,
        changes_code_env=changes_code_env,
        changes_env=changes_env)


def get_involved_components(eval_type: const.EvalType, dataset: pd.DataFrame
                            ) -> ChangedComponentsPerSituationOutput:
    components = [np.array(json.loads(val)) for val in dataset[eval_type.col_name(const.COLUMN_NAME_CHANGED_COMP)]]
    return ChangedComponentsPerSituationOutput(s_less=components)


def evaluate_single(eval_type: const.EvalType, dataset: pd.DataFrame) -> pd.Series:
    if eval_type == const.EvalType.GT:
        raise ValueError('const.EvalType.GT is not supported')

    situations_retrieved_raw = dataset[eval_type.col_name(const.COLUMN_NAME_SITUATION)]
    situations_real_raw = dataset[const.EvalType.GT.col_name(const.COLUMN_NAME_SITUATION)]

    situations_retrieved = SituationsOutput.from_series(dataset[eval_type.col_name(const.COLUMN_NAME_SITUATION)])
    situations_real = SituationsOutput.from_series(
        dataset[const.EvalType.GT.col_name(const.COLUMN_NAME_SITUATION)])

    # retrieve all the changes caught.
    changes_retrieved = get_changes_of_type(eval_type=eval_type, dataset=dataset)
    changes_real = get_changes_of_type(eval_type=const.EvalType.GT, dataset=dataset)

    if eval_type == const.EvalType.OUR:
        # retrieve the indices of the row where we performed none, partial, full re-certification.
        recertification_type_retrieved = get_partial_and_full_our(situations_retrieved_raw)
        recertification_type_real = get_partial_and_full_our(situations_real_raw)
    else:
        # just to be explicit
        assert eval_type == const.EvalType.STOTA
        # retrieve the indices of the row where we performed none, partial, full re-certification.
        recertification_type_retrieved = get_partial_and_full_stota(dataset)
        # recertification_type_retrieved = get_partial_and_full_stota(situations_retrieved_raw)
        recertification_type_real = get_partial_and_full_our(situations_real_raw)

    # and now it's time to make comparisons.
    # how many changes we missed for each type?
    comparison_changes_real_vs_retrieved = changes_real.compare_to(changes_retrieved)
    # how many re-certifications we missed?
    comparison_re_cert_type_real_vs_retrieved = recertification_type_real.compare_to(recertification_type_retrieved)
    # how many situations we missed?
    comparison_situations_real_vs_our_retrieved = situations_real.compare_to(situations_retrieved)

    components_changed_real = get_involved_components(const.EvalType.GT, dataset=dataset)
    components_changed_retrieved = get_involved_components(eval_type, dataset=dataset)
    comparison_components_changed = components_changed_real.compare_to(components_changed_retrieved)

    return pd.concat([comparison_situations_real_vs_our_retrieved, comparison_changes_real_vs_retrieved,
                      comparison_re_cert_type_real_vs_retrieved, comparison_components_changed])


def extract_columns(source, prefix) -> typing.List[str]:
    cols = [col for col in source if col.startswith(prefix)]
    # now, we remove Our
    cols = list(
        map(lambda col: col.replace(f'{prefix}', '').replace(')', '').replace('(', '') if isinstance(col, str) else col,
            cols))
    return cols


@dataclasses.dataclass
class ApplyOurOutput:
    any_changes_idx: np.ndarray
    code_and_code_env_idx: np.ndarray
    code_idx: np.ndarray
    env_idx: np.ndarray
    code_env_idx: np.ndarray
    df: pd.DataFrame


def apply_our(dataset: pd.DataFrame,
              generator: dataset_generator.DatasetGenerator,
              y_pred: np.ndarray, changed_components: typing.List[np.ndarray]) -> ApplyOurOutput:
    """

    :param dataset:
    :param generator:
    :param y_pred: array of predictions -1: yes there has been a change here, 1: no there has not been a change here.
    :param changed_components: an array as returned by the model
    :return:
    """

    assert np.all((y_pred == 1) | (y_pred == -1))
    if len(y_pred) != len(dataset):
        raise ValueError(f'len(y_pred) != len(dataset): got {len(y_pred)} != {len(dataset)}')
    if len(y_pred) != len(changed_components):
        raise ValueError(f'len(y_pred) != len(changed_components): got {len(y_pred)} != {len(changed_components)}')

    any_behavioral_change_idx = np.argwhere(y_pred == -1).flatten()

    # now let's start annotating the dataset.
    # we mark any row signalled as behavioral change as CHANGE_CAUSE_ENV.
    # we will do further refinement later on.
    dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] = const.CHANGE_CAUSE_NO
    dataset.loc[any_behavioral_change_idx, const.COLUMN_NAME_CHANGE_TYPE_OUR] = const.CHANGE_CAUSE_ENV

    # we mark as detected CODE_ENV the rows where there has been a code change
    # and a detected behavioral change
    dataset.loc[(dataset[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE_ENV) &
                (dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_ENV),
    const.COLUMN_NAME_CHANGE_TYPE_OUR] = const.CHANGE_CAUSE_CODE_ENV
    # we mark qw detected CODE the rows where there has been a code change
    # and NOT detected as a behavior change (ENV or CODE_ENV).
    dataset.loc[(dataset[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE) &
                (dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] != const.CHANGE_CAUSE_ENV) &
                (dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] != const.CHANGE_CAUSE_CODE_ENV), [
        const.COLUMN_NAME_CHANGE_TYPE_OUR]] = const.CHANGE_CAUSE_CODE
    # now, we add another column containing the extent of the code change (where the change
    # has been detected as CODE or CODE_ENV by us).
    dataset[const.COLUMN_NAME_CODE_EXTENT_OUR] = const.CODE_EXTENT_NO
    # to do this, we need to recover the corresponding column at the corresponding position.
    code_and_code_env_by_us_idx = dataset[
        (dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_CODE) |
        (dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_CODE_ENV)].index
    dataset.loc[code_and_code_env_by_us_idx, const.COLUMN_NAME_CODE_EXTENT_OUR] = dataset.loc[
        code_and_code_env_by_us_idx, const.COLUMN_NAME_CODE_EXTENT]
    # now, we add the column criticality, for the changes
    # of type CODE, CODE_ENV, and ENV.
    # we cannot do this at once, because we can recover the list of involved components, and, in turn,
    # if at least one of them is critical, for changes whose detected type is CODE_ENV and ENV
    # (thanks to the model).
    # while for changes of detected type CODE, we need to copy from the corresponding column,
    # just as we did for the extent above.

    components_to_add = generator.are_critical(changed_components)
    # these elements must be placed at locations where we detected a change of type ENV or CODE_ENV
    env_or_code_env_idx: npt.NDArray[int] = dataset.loc[(dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR]
                                                         == const.CHANGE_CAUSE_ENV) |
                                                        (dataset[
                                                             const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_CODE_ENV)].index.values

    components_to_add = components_to_add[env_or_code_env_idx]
    assert len(components_to_add) == len(env_or_code_env_idx)

    dataset[const.COLUMN_NAME_CRITICAL_OUR] = const.CRITICAL_FALSE
    dataset.loc[env_or_code_env_idx, const.COLUMN_NAME_CRITICAL_OUR] = components_to_add
    # next, we extract the indices where our scheme detected CODE.
    code_only_by_us_idx = dataset[dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_CODE
                                  ].index
    dataset.loc[code_only_by_us_idx, const.COLUMN_NAME_CRITICAL_OUR] = dataset.loc[code_only_by_us_idx,
    const.COLUMN_NAME_CRITICAL]
    # as a final step, we write the components involved in each change according to us.
    # by default, no components.
    dataset[const.COLUMN_NAME_CHANGED_COMP_OUR] = '[]'
    # for the changes of type CODE, we just copy.
    dataset.loc[code_only_by_us_idx, const.COLUMN_NAME_CHANGED_COMP_OUR] = dataset.loc[code_only_by_us_idx,
    const.COLUMN_NAME_CHANGED_COMP]
    # for the changes of type ENV and CODE_ENV, we use ours.
    env_idx = dataset[dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_ENV].index
    code_env_idx = dataset[dataset[const.COLUMN_NAME_CHANGE_TYPE_OUR] == const.CHANGE_CAUSE_CODE_ENV].index

    involved_env_components = base.extract_components(changed_components, env_idx.values)
    involved_code_env_components = base.extract_components(changed_components, code_env_idx.values)

    dataset.loc[env_idx, const.COLUMN_NAME_CHANGED_COMP_OUR] = base.df_json(involved_env_components)
    dataset.loc[code_env_idx, const.COLUMN_NAME_CHANGED_COMP_OUR] = base.df_json(
        involved_code_env_components)

    # finally copy novelty. It is always false but we need it nevertheless.
    dataset[const.COLUMN_NAME_NOVELTY_OUR] = const.NOVELTY_FALSE

    any_changes_idx = np.union1d(env_idx.values, code_and_code_env_by_us_idx)

    return ApplyOurOutput(
        df=dataset,
        env_idx=env_idx,
        code_env_idx=code_env_idx,
        code_and_code_env_idx=code_and_code_env_by_us_idx,
        any_changes_idx=any_changes_idx,
        code_idx=code_only_by_us_idx)


def apply_state_of_the_art(dataset: pd.DataFrame,
                           generator: dataset_generator.DatasetGenerator) -> pd.DataFrame:
    # we now apply the state of the art.
    dataset[const.COLUMN_NAME_CHANGE_TYPE_STOTA] = const.CHANGE_CAUSE_NO
    # all the changes marked as CODE and CODE_ENV are successfully marked as CODE by stota.
    code_ground_truth_idx = dataset[
        (dataset[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE) |
        (dataset[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE_ENV)].index
    dataset.loc[code_ground_truth_idx, const.COLUMN_NAME_CHANGE_TYPE_STOTA] = const.CHANGE_CAUSE_CODE
    # now, for the components involved stota can retrieve the
    # components *directly* involved only. Luckily, we have this information.
    # as default, we use the empty list.
    dataset[const.COLUMN_NAME_CHANGED_COMP_STOTA] = '[]'
    dataset.loc[code_ground_truth_idx, const.COLUMN_NAME_CHANGED_COMP_STOTA] = dataset.loc[
        code_ground_truth_idx, const.COLUMN_NAME_CHANGED_COMP_PRIMARY]
    # now, for the criticality, stota can retrieve the criticality of the components
    # *directly* involved in the change only.
    changed_components = [np.array(json.loads(a)) for a in
                          dataset.loc[code_ground_truth_idx, const.COLUMN_NAME_CHANGED_COMP_PRIMARY]]

    dataset[const.COLUMN_NAME_CRITICAL_STOTA] = const.CRITICAL_FALSE
    dataset.loc[code_ground_truth_idx, const.COLUMN_NAME_CRITICAL_STOTA] = generator.are_critical(
        changed_components)

    # we also copy the entity of changes CODE
    dataset[const.COLUMN_NAME_CODE_EXTENT_STOTA] = const.CODE_EXTENT_NO
    dataset.loc[code_ground_truth_idx, const.COLUMN_NAME_CODE_EXTENT_STOTA] = dataset.loc[
        code_ground_truth_idx, const.COLUMN_NAME_CODE_EXTENT]

    # finally copy novelty. It is always false but we need it nevertheless.
    dataset[const.COLUMN_NAME_NOVELTY_STOTA] = const.NOVELTY_FALSE

    return dataset


@dataclasses.dataclass
class SingleExecutionOutput:
    dataset: pd.DataFrame
    evaluation_scheme: pd.Series
    evaluation_model: pd.Series

    @staticmethod
    def from_double(dataset: pd.DataFrame, evaluation_our: pd.Series, evaluation_stota: pd.Series,
                    evaluation_model: pd.Series, config: dataset_generator.DatasetGeneratorConfig):
        # we rename evaluation_our and evaluation_stota s.t. when
        # we merge we do not have any issue.
        evaluation_our_ = evaluation_our.rename(lambda col: const.EvalType.OUR.col_name(col))
        evaluation_stota_ = evaluation_stota.rename(lambda col: const.EvalType.STOTA.col_name(col))
        evaluation_model_ = evaluation_model.rename(lambda col: const.prefix(const.PREFIX_MODEL, col))

        config_s = config.to_series()

        # immediately append config
        dataset[config_s.index] = config_s

        return SingleExecutionOutput(
            dataset=dataset, evaluation_scheme=pd.concat([config_s, evaluation_our_, evaluation_stota_]),
            evaluation_model=pd.concat([config_s, evaluation_model_])
        )


@dataclasses.dataclass
class QualitySchemeConfig(base.BaseExpConfig):
    dataset_config: dataset_generator.DatasetGeneratorConfig
    # dataset_training_config: base.DatasetGeneratorForTrainingConfig
    n_run_for_each: int = dataclasses.field(default=10)
    # this is related to how much cores can be allocated in parallel
    # for running n_run_for_each tasks.
    max_n_jobs: int = dataclasses.field(default=multiprocessing.cpu_count() - 2)

    def to_series_short(self) -> pd.Series:
        this_series = pd.Series([self.directory], index=['Dataset'])
        return pd.concat([self.dataset_config.to_series_short(), this_series])

    @staticmethod
    def from_dict(val: dict) -> "QualitySchemeConfig":
        filled = QualitySchemeConfig.fill_dict(val)

        dataset_config = dataset_generator.DatasetGeneratorConfig.from_dict(filled['dataset_config'])

        filled['dataset_config'] = dataset_config

        return QualitySchemeConfig(**filled)

    def as_dict(self) -> dict:
        all_keys = set(self.__annotations__.keys()).union(set(super().__annotations__.keys())) - set(
            self.fields_not_included_in_as_dict())

        output = {}
        for attr_name in all_keys:
            val = getattr(self, attr_name)
            if isinstance(val, dataset_generator.DatasetGeneratorConfig):
                val = val.as_dict()
            output[attr_name] = val

        return output


PairSeriesDf: typing.TypeAlias = typing.Tuple[pd.Series, pd.DataFrame]

BaseExecutionOutputVeryRawOutput: typing.TypeAlias = typing.Tuple[PairSeriesDf, PairSeriesDf]


@dataclasses.dataclass
class BaseExecutionOutput:
    evaluation_scheme: pd.DataFrame
    evaluation_model: pd.DataFrame

    aggregated_evaluation_scheme: pd.Series
    aggregated_evaluation_model: pd.Series

    @staticmethod
    def from_raw_list_to_very_raw(evaluation_scheme_list: typing.List[pd.Series],
                                  evaluation_model_list: typing.List[pd.Series]) -> BaseExecutionOutputVeryRawOutput:
        aggregated_evaluation_scheme, evaluation_scheme = base.get_summary(evaluation_scheme_list)
        aggregated_evaluation_model, evaluation_model = base.get_summary(evaluation_model_list)

        return (aggregated_evaluation_scheme, evaluation_scheme), (aggregated_evaluation_model, evaluation_model)

    @staticmethod
    def from_raw_lists(evaluation_scheme_list: typing.List[pd.Series],
                       evaluation_model_list: typing.List[pd.Series]):
        (aggregated_evaluation_scheme, evaluation_scheme), (aggregated_evaluation_model, evaluation_model) = \
            BaseExecutionOutput.from_raw_list_to_very_raw(evaluation_scheme_list=evaluation_scheme_list,
                                                          evaluation_model_list=evaluation_model_list)

        return BaseExecutionOutput(
            evaluation_scheme=evaluation_scheme, evaluation_model=evaluation_model,
            aggregated_evaluation_model=aggregated_evaluation_model,
            aggregated_evaluation_scheme=aggregated_evaluation_scheme)

    def append_series(self, series: pd.Series) -> "BaseExecutionOutput":
        self.evaluation_model[series.index] = series
        self.evaluation_scheme[series.index] = series
        self.aggregated_evaluation_model = pd.concat([series, self.aggregated_evaluation_model])
        self.aggregated_evaluation_scheme = pd.concat([series, self.aggregated_evaluation_scheme])
        return self

    def export(self, base_name_scheme: str, base_name_model: str, base_name_scheme_aggregated: str,
               base_name_model_aggregated: str, include_strip_down: bool = False):
        base.export_multi(data=self.evaluation_scheme, base_name=base_name_scheme,
                          include_strip_down=include_strip_down)
        base.export_multi(data=self.evaluation_model, base_name=base_name_model,
                          include_strip_down=include_strip_down)
        base.export_multi(data=pd.DataFrame([self.aggregated_evaluation_scheme]),
                          base_name=base_name_scheme_aggregated, include_strip_down=include_strip_down)
        base.export_multi(data=pd.DataFrame([self.aggregated_evaluation_model]),
                          base_name=base_name_model_aggregated, include_strip_down=include_strip_down)


@dataclasses.dataclass
class SingleConfigMultiExecutionOutput(BaseExecutionOutput):
    individuals: typing.List[SingleExecutionOutput]
    training_dataset: pd.DataFrame

    @staticmethod
    def from_individual(individuals: typing.List[SingleExecutionOutput],
                        training_dataset: pd.DataFrame) -> "SingleConfigMultiExecutionOutput":
        evaluation_scheme_list = []
        evaluation_model_list = []

        for individual in individuals:
            evaluation_scheme_list.append(individual.evaluation_scheme)
            evaluation_model_list.append(individual.evaluation_model)

        (aggregated_evaluation_scheme, evaluation_scheme), (aggregated_evaluation_model, evaluation_model) = \
            BaseExecutionOutput.from_raw_list_to_very_raw(evaluation_scheme_list=evaluation_scheme_list,
                                                          evaluation_model_list=evaluation_model_list)

        return SingleConfigMultiExecutionOutput(
            evaluation_scheme=evaluation_scheme, evaluation_model=evaluation_model,
            aggregated_evaluation_scheme=aggregated_evaluation_scheme,
            aggregated_evaluation_model=aggregated_evaluation_model,
            individuals=individuals, training_dataset=training_dataset
        )


@dataclasses.dataclass
class OverallResult:
    grouped_by_dataset: typing.Dict[str, BaseExecutionOutput]
    grouped_by_config_group: typing.Dict[str, BaseExecutionOutput]
    grouped_by_config_group_and_dataset: typing.Dict[str, BaseExecutionOutput]
    grouped_by_config: typing.Dict[str, BaseExecutionOutput]

    # the aggregated result of each experiment.
    results_of_each: typing.List[typing.Tuple[QualitySchemeConfig, SingleConfigMultiExecutionOutput]]

    def export(self, base_directory: str, include_strip_down: bool = False):

        name_dir_dataset_training = 'DatasetsTraining'

        name_dir_detailed_results_main = 'Detailed'

        name_dir_aggregated_datasets = 'Aggregated_Datasets'
        name_dir_aggregated_configs = 'Aggregated_Configs'
        name_dir_aggregated_configs_group = 'Aggregated_ConfigsGroup'
        name_dir_aggregated_configs_group_datasets = 'Aggregated_ConfigsGroup_Datasets'
        name_dir_super_aggregated = 'Aggregated_Super'

        name_subdir_in_aggregated_detailed = 'Detailed'
        name_subdir_in_aggregated_raw = 'Raw'

        postfix_scheme = 'scheme'
        postfix_model = 'model'
        postfix_dataset = 'dataset'

        name_group_by_config = 'by_config'
        name_group_by_config_group = 'by_config_group'
        name_group_by_dataset = 'by_dataset'
        name_group_by_config_group_dataset = 'by_config_group_dataset'

        dir_super_aggregated = os.path.join(base_directory, name_dir_super_aggregated)

        dataset_training_directory = os.path.join(base_directory, name_dir_dataset_training)

        # here we export the training dataset. Note files are actually the same, but whatever.
        for single_result_of_each in self.results_of_each:
            current_dataset_directory = os.path.join(
                dataset_training_directory, os.path.basename(os.path.normpath(single_result_of_each[0].directory)))
            base.export_multi(data=single_result_of_each[1].training_dataset, base_name=os.path.join(
                current_dataset_directory, 'training'), include_strip_down=include_strip_down)

        # now, individual detailed results
        # are exported in a directory named after the config, included in each dataset name.
        def group_by_dataset(pair: typing.Tuple[QualitySchemeConfig, SingleConfigMultiExecutionOutput]) -> str:
            return pair[0].dataset_config.dataset_name

        detailed_results_dir = os.path.join(base_directory, name_dir_detailed_results_main)

        for dataset_name, list_of_results in itertools.groupby(sorted(self.results_of_each, key=group_by_dataset),
                                                               key=group_by_dataset):
            dataset_name: str
            list_of_results: typing.Iterable[typing.Tuple[QualitySchemeConfig, SingleConfigMultiExecutionOutput]] = \
                list_of_results

            current_sub_dir = os.path.join(detailed_results_dir, dataset_name)

            for single_pair in list_of_results:

                # here we have the results of an individual config
                # over an individual target system.
                current_sub_sub_dir = os.path.join(current_sub_dir, single_pair[0].dataset_config.name)

                for i, single_result in enumerate(single_pair[1].individuals):
                    # now, here, we really have the result of an individual execution of an individual config.
                    # we export the generated dataset.
                    base.export_multi(data=single_result.dataset, include_strip_down=include_strip_down,
                                      base_name=os.path.join(current_sub_sub_dir, f'{i}_{postfix_dataset}'), )
                    # we export the results of the scheme.
                    base.export_multi(data=pd.DataFrame([single_result.evaluation_scheme]),
                                      base_name=os.path.join(current_sub_sub_dir, f'{i}_{postfix_scheme}'),
                                      include_strip_down=include_strip_down)
                    # we export the results of the model
                    base.export_multi(data=pd.DataFrame([single_result.evaluation_model]),
                                      base_name=os.path.join(current_sub_sub_dir, f'{i}_{postfix_model}'),
                                      include_strip_down=include_strip_down)

        def _export_single(key_: str, val_: BaseExecutionOutput, sub_sub_dir_name_: str):
            current_dir_ = os.path.join(os.path.join(base_directory, sub_sub_dir_name_), key_)

            current_dir_raw_ = os.path.join(current_dir_, name_subdir_in_aggregated_raw)
            current_dir_detailed_ = os.path.join(current_dir_, name_subdir_in_aggregated_detailed)

            val_.export(base_name_scheme=os.path.join(current_dir_raw_, postfix_scheme),
                        base_name_model=os.path.join(current_dir_raw_, postfix_model),
                        base_name_scheme_aggregated=os.path.join(current_dir_detailed_, postfix_scheme),
                        base_name_model_aggregated=os.path.join(current_dir_detailed_, postfix_model),
                        include_strip_down=include_strip_down)

        def _export_group(values_: typing.Dict[str, BaseExecutionOutput], base_name: str):
            # we take the individual pd.Series
            # and concatenate in a unique pd.DataFrame
            aggregated_scheme = []
            aggregated_model = []
            for key_, val_ in values_.items():
                # we prepend the key to the series
                key_s = pd.Series([key_], index=['Key'])
                aggregated_scheme.append(pd.concat([key_s, val_.aggregated_evaluation_scheme]))
                aggregated_model.append(pd.concat([key_s, val_.aggregated_evaluation_model]))

            aggregated_scheme = pd.DataFrame(aggregated_scheme)
            aggregated_model = pd.DataFrame(aggregated_model)

            base.export_multi(data=aggregated_scheme, base_name=f'{base_name}_{postfix_scheme}',
                              include_strip_down=include_strip_down)
            base.export_multi(data=aggregated_model, base_name=f'{base_name}_{postfix_model}',
                              include_strip_down=include_strip_down)

        def _export_aggregated_all(values_: typing.Dict[str, BaseExecutionOutput],
                                   sub_sub_dir_name_: str, super_aggregated_dir_name_: str):
            _export_group(values_=values_, base_name=super_aggregated_dir_name_)
            for key, val in self.grouped_by_dataset.items():
                _export_single(key_=key, val_=val, sub_sub_dir_name_=sub_sub_dir_name_)

        _export_aggregated_all(self.grouped_by_dataset, sub_sub_dir_name_=name_dir_aggregated_datasets,
                               super_aggregated_dir_name_=os.path.join(
                                   dir_super_aggregated, name_group_by_dataset))
        _export_aggregated_all(self.grouped_by_config, sub_sub_dir_name_=name_dir_aggregated_configs,
                               super_aggregated_dir_name_=os.path.join(dir_super_aggregated, name_group_by_config))
        _export_aggregated_all(self.grouped_by_config_group, sub_sub_dir_name_=name_dir_aggregated_configs_group,
                               super_aggregated_dir_name_=os.path.join(
                                   dir_super_aggregated, name_group_by_config_group))
        _export_aggregated_all(self.grouped_by_config_group_and_dataset,
                               sub_sub_dir_name_=name_dir_aggregated_configs_group_datasets,
                               super_aggregated_dir_name_=os.path.join(dir_super_aggregated,
                                                                       name_group_by_config_group_dataset))
