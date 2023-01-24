import abc
import dataclasses
import itertools
import typing

import joblib
import numpy as np
import numpy.typing as npt
import pandas as pd

import base
import const
from const import CHANGE_CAUSE_NO, CHANGE_CAUSE_ENV, CHANGE_CAUSE_CODE, CHANGE_CAUSE_CODE_ENV, CRITICAL_FALSE, \
    CRITICAL_TRUE, CODE_EXTENT_NO, CODE_EXTENT_MINOR, CODE_EXTENT_MAJOR, COLUMN_NAME_CRITICAL, COLUMN_NAME_CHANGE_TYPE, \
    COLUMN_NAME_CODE_EXTENT, COLUMN_NAME_CHANGED_COMP, COLUMN_NAME_CHANGED_COMP_PRIMARY, \
    COLUMN_NAME_CHANGED_COMP_SECONDARY


@dataclasses.dataclass
class GetComponentsInvolvedInChangeCode:
    components_primarily_involved: typing.List[np.ndarray]
    components_secondarily_involved: typing.List[np.ndarray]
    components_regardless_involved: typing.List[np.ndarray]

    def zip(self) -> typing.Iterable[typing.Tuple[npt.NDArray[int], npt.NDArray[int], npt.NDArray[int]]]:
        return zip(self.components_primarily_involved,
                   self.components_secondarily_involved, self.components_regardless_involved)


@dataclasses.dataclass
class DatasetGeneratorConfig(abc.ABC):
    # perc_chang: float
    name: str
    group: str
    wanted_size: int
    # this is ignored in the constructor.
    rng_: np.random.Generator = dataclasses.field(default=None)
    rng: typing.Optional[int] = dataclasses.field(default=None)
    proba_not_change_vs_change: float = dataclasses.field(default=.6)
    proba_environment_vs_code_vs_both: typing.List[float] = dataclasses.field(
        default_factory=lambda: [1 / 3 for _ in range(3)])
    proba_minor_vs_major: float = dataclasses.field(default=.75)
    # this will guide the choice of the component exhibiting critical behavior.
    proba_not_critical_vs_critical: float = dataclasses.field(default=.8)
    # zipf_env: float = dataclasses.field(default=3)
    # zipf_major: float = dataclasses.field(default=.5)
    # stddev_code_minor: float = dataclasses.field(default=.75)
    proba_one_component_env: float = dataclasses.field(default=.75)
    proba_one_component_code_minor: float = dataclasses.field(default=.7)
    proba_one_component_code_major: float = dataclasses.field(default=.55)

    n_jobs_inner: typing.Optional[int] = dataclasses.field(default=None)

    dataset_name: str = dataclasses.field(default=None)

    use_detailed_config_series: bool = dataclasses.field(default=True)

    def as_dict(self) -> dict:
        not_included = ['rng_']
        output = {}
        for attr_name in self.__annotations__.keys():
            if attr_name not in not_included:
                output[attr_name] = getattr(self, attr_name)
        return output

    @staticmethod
    def from_dict(val: dict) -> "DatasetGeneratorConfig":
        return DatasetGeneratorConfigLinearProba(**val)

    @property
    def meaningful_attributes_that_are_flat(self) -> typing.List[str]:
        return ["proba_minor_vs_major",
                "proba_not_critical_vs_critical",
                "proba_one_component_env",
                "proba_one_component_code_minor",
                "proba_one_component_code_major"]

    @property
    def meaningful_attributes(self) -> typing.List[str]:
        return ["proba_not_change_vs_change",
                "proba_environment_vs_code_vs_both",
                "proba_minor_vs_major",
                "proba_not_critical_vs_critical",
                "proba_one_component_env",
                "proba_one_component_code_minor",
                "proba_one_component_code_major"]

    def __repr__(self):
        probas = ', '.join([f'{attr_name}={getattr(self, attr_name)}' for attr_name
                            in self.meaningful_attributes])
        return f'{self.__class__.name}(name={self.name}, ' + probas

    def to_series(self, include_name_and_dataset: typing.Optional[bool] = None) -> pd.Series:
        if include_name_and_dataset is None:
            include_name_and_dataset = self.use_detailed_config_series

        # we flatten self.proba_environment_vs_code_vs_both in order to be representable as a 1d array
        flatten_main = pd.Series([self.proba_environment_vs_code_vs_both[0],
                                  self.proba_environment_vs_code_vs_both[1],
                                  self.proba_environment_vs_code_vs_both[2]
                                  ], index=['proba_environment', 'proba_code', 'proba_code_env'])

        not_change_vs_change = pd.Series([self.proba_not_change_vs_change],
                                         index=['proba_not_change_vs_change'])

        other_attr = pd.Series([getattr(self, attr_value)
                                for attr_value in self.meaningful_attributes_that_are_flat],
                               index=self.meaningful_attributes_that_are_flat)

        all_series = [not_change_vs_change, flatten_main, other_attr]

        if include_name_and_dataset:
            name_s = pd.Series([self.name, self.dataset_name], index=['Name', 'Dataset'])
            all_series = [name_s, not_change_vs_change, flatten_main, other_attr]

        return pd.concat(objs=all_series)

    def to_series_short(self) -> pd.Series:
        return pd.Series([self.name], index=[base.COLUMN_NAME_NAME])

    def __post_init__(self):
        to_check = [self.proba_not_change_vs_change,
                    self.proba_minor_vs_major, self.proba_not_critical_vs_critical,
                    self.proba_one_component_env, self.proba_one_component_code_minor,
                    self.proba_one_component_code_major]
        for single_to_check in to_check:
            if single_to_check > 1 or single_to_check < 0:
                raise ValueError('Probabilities must be <= 1.')

        if sum(self.proba_environment_vs_code_vs_both) != 1.0 and \
                sum(self.proba_environment_vs_code_vs_both) != \
                0.9999999999999989:
            raise ValueError(f'self.proba_environment_vs_code_vs_both sums to '
                             f'{sum(self.proba_environment_vs_code_vs_both)} instead of 1')

        if self.rng is None:
            self.rng_ = np.random.default_rng()
        else:
            self.rng_ = np.random.default_rng(self.rng)

    @abc.abstractmethod
    def get_probabilities_for_n_components(self, base_proba: float, n_components: int) -> np.ndarray:
        pass

    def get_n_components_involved_in_change(self, base_proba: float, n_total_components: int, n_changes: int,
                                            ) -> npt.NDArray[int]:
        """
        Retrieves the number of components involved in the change according to the probability of one component only
        involved in the change (`base_proba`).
        :param base_proba: probability that one component only is involved in [0, 1]
        :param n_total_components: total number of components in the service
        :param n_changes: number of changes to return
        :return: an array whose size matches n_changes, each cell of the array corresponds to a change, and
        each element corresponds to the number of components involved in such change.
        """
        proba = self.get_probabilities_for_n_components(base_proba=base_proba, n_components=n_total_components)
        # now we have an array of probabilities. It is time to perform
        # the random extraction among [0, n_total_components)
        # ATTENTION: proba returns an array for probabilities for n_components,
        # assume 3 components, we get [0.3, 0.3, 0.3]. But if we apply this blindly to the below
        # situation, the first element of the proba corresponds to the probabilities of choosing *no*
        # elements which is incorrect.
        # So we need to arange over 1, n_components+1
        return self.rng_.choice(np.arange(1, n_total_components + 1), p=proba, size=(n_changes,))

    def get_components_involved_in_change(self, base_proba: float, n_total_components: int, n_changes: int,
                                          ) -> typing.List[np.ndarray]:
        number_of_components = self.get_n_components_involved_in_change(base_proba=base_proba, n_changes=n_changes,
                                                                        n_total_components=n_total_components)

        # now, we randomly choose which components are involved in the change
        # to this aim, we use rng.permuted, which can permute a 2d array (contrary to permutation).
        base_to_permute = np.tile(np.arange(n_total_components), n_changes).reshape(n_changes, n_total_components)
        # now we have an array where each row contains the components to choose from.
        # now, we basically shuffle each row inside.
        permuted = self.rng_.permuted(base_to_permute, axis=1)
        components = [_ for _ in range(n_changes)]
        for i, current_permutation_to_choose_from in enumerate(permuted):
            components[i] = current_permutation_to_choose_from[:number_of_components[i]]
        return components

    def get_components_involved_in_change_code(self, *,
                                               n_total_components: int,
                                               primary_proba: float,
                                               n_changes: int,
                                               verbose: bool = False
                                               ) -> GetComponentsInvolvedInChangeCode:
        # we begin by selecting the components primarily involved in the change.
        # the length of this list equals the number of changes.
        components_primarily_involved = self.get_components_involved_in_change(base_proba=primary_proba,
                                                                               n_total_components=n_total_components,
                                                                               n_changes=n_changes)
        assert len(components_primarily_involved) == n_changes
        if verbose:
            print(f'Primary {components_primarily_involved}')

        components_secondarily_involved = []
        # each entry contains the np.ndarray of components involved in the change,
        # regardless they are primary or secondary.
        components_involved_regardless = []

        # now, for each row (i.e., list of components involved in the change)
        # we retrieve the components *not* involved in the change, and, then,
        # we selected those affected by cascading change among the latter.
        all_components = np.arange(n_total_components)
        for components_involved_in_a_change in components_primarily_involved:
            not_involved = np.setdiff1d(all_components, components_involved_in_a_change)
            # now, we extract the other components according to secondary_proba.
            # this is a two-steps process where we first extract the number
            # of components involved.
            # the probability of being chosen is uniform (default parameter of choice).
            if len(not_involved) > 0:
                n_components_involved_in_secondary_change = self.rng_.choice(not_involved)
                # and now we select the actual components
                not_involved_but_now_involved = self.rng_.permutation(not_involved)[
                                                :n_components_involved_in_secondary_change]
                components_secondarily_involved.append(not_involved_but_now_involved)
                components_involved_regardless.append(
                    np.concatenate([components_involved_in_a_change, not_involved_but_now_involved]))
            else:
                components_involved_regardless.append(components_involved_in_a_change)
                components_secondarily_involved.append(np.array([]))

        return GetComponentsInvolvedInChangeCode(
            components_primarily_involved=components_primarily_involved,
            components_secondarily_involved=components_secondarily_involved,
            components_regardless_involved=components_involved_regardless)

    def get_code_extent(self, n_changes: int):
        return self.rng_.choice([CODE_EXTENT_MINOR, CODE_EXTENT_MAJOR], size=(n_changes,),
                                p=[self.proba_minor_vs_major, 1 - self.proba_minor_vs_major])


class DatasetGeneratorConfigLinearProba(DatasetGeneratorConfig):

    def get_probabilities_for_n_components(self, base_proba: float, n_components: int) -> np.ndarray:
        """
        Compute the probabilities that more than one component is involved in the change.

        Given the probability that one component only is involved in the change (`base_proba`), the probabilities
        that `i` components are involved in the change decreases linearly as `i` grows.

        We use the following algorithm

        1. obtain the remaining part of probability that can be assigned to remaining components
        (e.g., if `base_proba=60`, then the remaining part is `40`).
        2. compute the probability that *all* remaining components are involved in the change. This is the
        smallest probability possible, and is retrieved by dividing the remaining probability by the summation
        over the number of additionally-available components (e.g., if we have 10 components in total,
        the summation is from 1 to 9)
        3. compute the probability of each number of components. We multiply the smallest probability
        by the number of available components, e.g., from 1 to 9, and then we reverse the array.
        This way, when we multiply by 1 we retrieve the smallest probability, corresponding to all components involved,
        by 2 we retrieve the probability that all remaining components - 1 are chosen, and so on.

        :param base_proba: probability that one component only is involved in [0, 1]
        :param n_components: total number of components in the service
        :return:
        """
        # first, we multiply the probability for 100 so that we can work with larger numbers
        # reducing rounding errors.
        # NO: to keep things simple, we stay with p in [0, 1]
        # base_proba_ = base_proba * 100
        base_proba_ = base_proba
        # now, we retrieve the remaining part of probabilities. I.e., base_proba indicates the probability
        # that one component is involved in the change, e.g., 60%, so we split the remaining 40%
        # among the others s.t. the probability of more components decreases linearly.
        # remaining_proba = 100 - base_proba_
        remaining_proba = 1 - base_proba_
        # this generates an array starting at 1 up to n_components - 1,
        # which we use to retrieve the actual values
        components_array = np.arange(1, n_components)
        minimum_part = remaining_proba / np.sum(components_array)
        # now we compute the probability decreasing in linear order.
        # flip reverse the order, otherwise we would have in the first position the probability corresponding
        # to all components, while in the first position we want the probability of 1 (additional) component.
        proba = np.array([minimum_part * i for i in np.nditer(np.flip(components_array))])
        # then, we return the complete probability array adding the probability of 1 as well.
        return np.hstack([[base_proba_], proba])


class DatasetGenerator:

    def __init__(self, config: DatasetGeneratorConfig, clean_df: pd.DataFrame, anomalous_df: pd.DataFrame):
        self.clean_df = clean_df
        self.anomalous_df = anomalous_df
        self.config = config
        self.components: np.ndarray = clean_df.columns.to_numpy()
        self.components_idx = np.arange(len(self.components))
        # now, for each component, we decide if it is critical or not.
        self.components_criticality = self.config.rng_.choice([CRITICAL_FALSE, CRITICAL_TRUE],
                                                              size=(len(self.components, )))
        self.dataset: pd.DataFrame = pd.DataFrame()

    @staticmethod
    def random_extract_from(target: pd.DataFrame, how_many: int,
                            rng: typing.Optional[np.random.Generator] = None,
                            columns:
                            typing.Optional[typing.Union[npt.NDArray[int], typing.List[typing.Union[int, str]]]] = None
                            ) -> pd.DataFrame:
        rng_ = rng if rng is not None else np.random.default_rng()
        # create a random permutation over the index of target,
        # where we then extract `how_many` items.
        reindex = rng_.permutation(target.index)
        df = target.iloc[reindex[:how_many]]
        if columns is None:
            return df
        else:
            # if it is a list of column indices, then we need to convert the column indices (integers)
            # to their name as string
            if (isinstance(columns, list) and isinstance(columns[0], int)) or \
                    (isinstance(columns, np.ndarray) and columns.dtype == int):
                columns = target.columns.to_numpy()[columns]
            return df[columns]

    def random_extract_with_anomalies(self, anomalous_components_idx: typing.List[np.ndarray]
                                      ) -> pd.DataFrame:
        """
        This returns a pd.DataFrame whose length matches the length of anomalous_components_idx.
        It contains n columns, with n = len(self.components).
        Each i-th row contains data points for all the components. Some features (i.e., components) are
        anomalous (according to the indices of components anomalous_components_idx[i]), while the remaining
        are non-anomalous.

        :param anomalous_components_idx:
        :return:
        """

        def __inner(list_of_components_involved_in_single_change_) -> pd.Series:
            # grab the list of components *not* involved in the change
            list_of_components_not_involved_in_single_change = np.setdiff1d(
                self.components_idx, list_of_components_involved_in_single_change_)

            anomalous_data = DatasetGenerator.random_extract_from(
                target=self.anomalous_df, columns=list_of_components_involved_in_single_change_, how_many=1)

            # now we do the same but on clean data.
            clean_data = DatasetGenerator.random_extract_from(
                target=self.clean_df, columns=list_of_components_not_involved_in_single_change, how_many=1)
            # to avoid strange behavior with concat or merge that we would have on pandas, we just stack
            # the different columns
            extracted_data = np.hstack([anomalous_data.values, clean_data.values]).flatten()

            # we need to sort the indices we are going to use
            # otherwise pd.DataFrame becomes difficult.
            idx = np.concatenate([self.components_idx[list_of_components_involved_in_single_change_],
                                  self.components_idx[list_of_components_not_involved_in_single_change]])
            sorted_idx = np.argsort(idx)

            assert len(np.intersect1d(self.components_idx[list_of_components_involved_in_single_change_],
                                      self.components_idx[list_of_components_not_involved_in_single_change])) == 0

            components = pd.Series(extracted_data[sorted_idx], index=idx[sorted_idx])

            return components

        components_for_all: typing.List[pd.Series] = joblib.Parallel(
            n_jobs=self.config.n_jobs_inner)(joblib.delayed(__inner)(list_of_components_involved_in_single_change)
                                             for list_of_components_involved_in_single_change
                                             in anomalous_components_idx)

        # print(components_for_all)
        # note that this operation should be safe because each pd.Series we are going to aggregate
        # is composed of the same indices, that is, the components indices.
        return pd.DataFrame(components_for_all)

    def generate(self, verbose: bool = False):
        """
        Creates a dataset according to the given configuration. Dataset generation works as follows.

        1.  create a dataset taking proba_not_change_vs_not_change % of self.config.wanted_size from the clean dataset.
        2.  insert at random positions the changes.
        3.  changes are generated according to the following algorithm.
            4.  select the type of change: environment with impact on behavior, code with impact on behavior,
                code without impact on the behavior.
            5.  select the component target of anomalies according to proba_not_critical_vs_critical
                6.  if code without impact on the behavior: add another "good" row taken from the clean dataset at
                    random. Decide at random the impact (minor, major).
                7.  if environment with impact on behavior: add row with involved components taken from anomalous
                    dataset, and from the clean dataset for the other components.
                8.  if code with impact on behavior: decide at random which other components are involved in the
                    cascading effect, according to the impact (minor, major).
                9.  annotated each row of the dataset according to the standard defined in v1.
        :return:
        """
        NO_CHANGE = 0
        CHANGE = 1

        # the very first thing to do is to select, for each row, if we want a change or not.
        # the first parameter is the choice (if a single number, it corresponds to np.arange ),
        # the second is the size.
        type_of_row = self.config.rng_.choice([NO_CHANGE, CHANGE], size=(self.config.wanted_size,),
                                              p=[self.config.proba_not_change_vs_change,
                                                 1 - self.config.proba_not_change_vs_change])
        # now, for each row of type NO_CHANGE, we pick from the clean dataset.
        no_changes = DatasetGenerator.random_extract_from(target=self.clean_df,
                                                          how_many=np.count_nonzero(type_of_row == NO_CHANGE))
        assert len(no_changes) == np.count_nonzero(type_of_row == NO_CHANGE)

        assert np.count_nonzero(type_of_row == NO_CHANGE) + np.count_nonzero(type_of_row == CHANGE) == len(type_of_row)

        # and the simpler thing is done.
        # now, we do another random extraction to select the type of change.
        change_type = self.config.rng_.choice([CHANGE_CAUSE_ENV, CHANGE_CAUSE_CODE, CHANGE_CAUSE_CODE_ENV],
                                              size=(np.count_nonzero(type_of_row == CHANGE),))

        idx_available_for_change = np.arange(self.config.wanted_size)[np.argwhere(type_of_row == CHANGE).flatten()]

        idx_change_no = np.argwhere(type_of_row == NO_CHANGE).flatten()
        idx_change_env = idx_available_for_change[np.argwhere(change_type == CHANGE_CAUSE_ENV)].flatten()
        idx_change_code_env = idx_available_for_change[np.argwhere(change_type == CHANGE_CAUSE_CODE_ENV)].flatten()
        idx_change_code = idx_available_for_change[np.argwhere(change_type == CHANGE_CAUSE_CODE)].flatten()

        if verbose:
            print(f'No: {idx_change_no}')
            print(f'Change type: {change_type}')
            print(f'Env: {idx_change_env}')
            print(f'Code: {idx_change_code}')
            print(f'Code env: {idx_change_code_env}')

        # now, for all the changes of type ENV,
        # we retrieve the indices of the components involved in the change.
        changed_components_env: typing.List[npt.NDArray[int]] = self.config.get_components_involved_in_change(
            base_proba=self.config.proba_one_component_env, n_total_components=len(self.components),
            n_changes=len(idx_change_env))

        # now, for each row of type CHANGE_ENV, we pick from the corresponding dataset.
        changes_env = self.random_extract_with_anomalies(anomalous_components_idx=changed_components_env)

        # now, for all the changes of type CODE and CODE_ENV, we retrieve if they are minor or major.
        # it contains the type of code change.

        code_change_type_code = self.config.get_code_extent(n_changes=len(idx_change_code))
        code_change_type_code_env = self.config.get_code_extent(n_changes=len(idx_change_code_env))

        assert len(code_change_type_code_env) == len(idx_change_code_env)
        assert len(code_change_type_code) == len(idx_change_code)

        # idx_change_code_env_minor = #idx_available_for_change[
        idx_change_code_env_minor = idx_change_code_env[
            np.argwhere(code_change_type_code_env == CODE_EXTENT_MINOR).flatten()]  # ]
        # idx_change_code_env_major = \#idx_available_for_change[
        idx_change_code_env_major = idx_change_code_env[
            np.argwhere(code_change_type_code_env == CODE_EXTENT_MAJOR).flatten()]  # ]

        idx_change_code_only_minor = idx_change_code[np.argwhere(code_change_type_code == CODE_EXTENT_MINOR).flatten()]
        idx_change_code_only_major = idx_change_code[np.argwhere(code_change_type_code == CODE_EXTENT_MAJOR).flatten()]

        if verbose:
            print(f'Code only minor: {idx_change_code_only_minor}')
            print(f'Code only major: {idx_change_code_only_major}')

        # now, we need to retrieve the indices of the components involved in the changes
        # of type CODE_ENV. Note that we are interested only in CODE_ENV because
        # CODE_ENV are situations where we need to retrieve anomalous data, while for changes of type CODE
        # we can just extract clean data from the dataset.
        # We do this retrieval of CODE_ENV separately for MINOR and MAJOR
        # because we have different probabilities for MINOR and MAJOR.
        changed_components_code_env_minor = self.config.get_components_involved_in_change_code(
            primary_proba=self.config.proba_one_component_code_minor, n_total_components=len(self.components),
            n_changes=len(idx_change_code_env_minor), verbose=verbose)
        changed_components_code_env_major = self.config.get_components_involved_in_change_code(
            primary_proba=self.config.proba_one_component_code_major, n_total_components=len(self.components),
            n_changes=len(idx_change_code_env_major), verbose=verbose)

        # now, for each row of type CHANGE_CODE_ENV of extent MINOR, we pick from the corresponding dataset.
        changes_code_env_minor = self.random_extract_with_anomalies(
            anomalous_components_idx=changed_components_code_env_minor.components_regardless_involved)
        # and we do the same for CHANGE_CODE_ENV of extent MAJOR.
        changes_code_env_major = self.random_extract_with_anomalies(
            anomalous_components_idx=changed_components_code_env_major.components_regardless_involved)

        # this captures the components changed ONLY in terms of code.
        # i.e., they do not have anomalous data in the components columns.
        # Basically it captures which components are touched during code change. So, we are going to
        # ignore secondarily involved components.
        changed_components_for_code_only_minor = self.config.get_components_involved_in_change_code(
            primary_proba=self.config.proba_one_component_code_minor, n_total_components=len(self.components),
            n_changes=len(idx_change_code_only_minor), verbose=verbose)
        changed_components_for_code_only_major = self.config.get_components_involved_in_change_code(
            primary_proba=self.config.proba_one_component_code_major, n_total_components=len(self.components),
            n_changes=len(idx_change_code_only_major), verbose=verbose)

        if verbose:
            print(f'Comp env: {changed_components_env}')
            print(f'Comp code env min: {changed_components_code_env_minor}')
            print(f'Comp code env maj: {changed_components_code_env_major}')

        if verbose:
            print(f'Code only minor comp: {changed_components_for_code_only_minor}')
            print(f'Code only major comp: {changed_components_for_code_only_major}')

        # now, for all the changes of type CODE, we pick additional clean rows, since CODE has no
        # impact on the impact on the behavior, therefore we do not anomalous data.
        clean_for_code = DatasetGenerator.random_extract_from(target=self.clean_df, how_many=len(idx_change_code))

        # now we have all the components we want.
        # so it's time to merge.
        # df_all = pd.concat(no_changes, changes_env, changes_code_env_major, changes_code_env_major)
        base_data = np.repeat(0.0, repeats=self.config.wanted_size * len(self.components)
                              ).reshape(self.config.wanted_size, len(self.components))
        df_all = pd.DataFrame(base_data, columns=self.components)
        # what is interesting is that pandas allows to assign a pd.DataFrame to another
        # pd.DataFrame at non-contiguous indices.
        # for instance:
        # df1 = pd.DataFrame([[1, 2, 3], [4, 5, 6], [7, 8, 9]], columns=list('abc'))
        # df[[0, 3, 6]] = df1 -> will split df1 among rows 0, 3, 6.
        # here, we assign the values of the behavior of the components.
        for i, (to_use_idx, corresponding_df) in enumerate([(idx_change_no, no_changes),
                                                            (idx_change_code, clean_for_code),
                                                            (idx_change_env, changes_env),
                                                            (idx_change_code_env_minor, changes_code_env_minor),
                                                            (idx_change_code_env_major, changes_code_env_major)]):
            if len(to_use_idx) > 0:
                # print()
                # print(f'{i}: idx: {to_use_idx}, values: {corresponding_df.values}')
                df_all.iloc[to_use_idx] = corresponding_df.values
                # print(f'post: {df_all}')
            # we may not have available data.
            else:
                print(f'WARNING: no generated data {i}')

        # it ain't over :)
        # we need to assign criticality.
        # we do not consider novelty: this is fine, since they are handled both by us according to the code
        # and by the state of the art according to the code.

        # now, for each change, we need to write down if it is critical or not, i.e.,
        # if it involves a critical component or not.
        # We have the list of components involved in each change.
        # let's start assigning the default.
        df_all[COLUMN_NAME_CRITICAL] = CRITICAL_FALSE

        components_touched = itertools.chain(changed_components_env,
                                             changed_components_code_env_major.components_regardless_involved,
                                             changed_components_code_env_minor.components_regardless_involved)

        # contains the indices on df where there has been a change causing components behavior to change.
        idx_of_components_with_changed_behavior = np.concatenate([idx_change_env, idx_change_code_env_major,
                                                                  idx_change_code_env_minor])

        idx_of_critical_changes = []
        for i, components_touched_in_this_row in enumerate(components_touched):
            if np.any(np.isin(components_touched_in_this_row, self.components_criticality)):
                idx_of_critical_changes.append(i)

        df_all.loc[idx_of_components_with_changed_behavior[idx_of_critical_changes], [
            COLUMN_NAME_CRITICAL]] = CRITICAL_TRUE

        # and finally, we assign the additional columns for annotation.
        df_all[COLUMN_NAME_CHANGE_TYPE] = CHANGE_CAUSE_NO
        for to_use_idx, value in [(idx_change_code, CHANGE_CAUSE_CODE), (idx_change_env, CHANGE_CAUSE_ENV),
                                  (idx_change_code_env, CHANGE_CAUSE_CODE_ENV)]:
            df_all.loc[to_use_idx, COLUMN_NAME_CHANGE_TYPE] = value

        all_code_change_type = np.concatenate([code_change_type_code, code_change_type_code_env])

        # then, we need to assign the extent of the code change.
        df_all[COLUMN_NAME_CODE_EXTENT] = CODE_EXTENT_NO
        for to_use_idx, value in [
            (np.argwhere(all_code_change_type == CODE_EXTENT_MINOR).flatten(), CODE_EXTENT_MINOR),
            (np.argwhere(all_code_change_type == CODE_EXTENT_MAJOR).flatten(), CODE_EXTENT_MAJOR)]:
            df_all.loc[to_use_idx, COLUMN_NAME_CODE_EXTENT] = value

        # one last thing to do is to annotate which components whose behavior has actually changed.
        # no option but to use json here.
        df_all[COLUMN_NAME_CHANGED_COMP] = '[]'
        for to_use_idx, involved_components in [
            (idx_change_env, changed_components_env),
            (idx_change_code_env_minor, changed_components_code_env_minor.components_regardless_involved),
            (idx_change_code_env_major, changed_components_code_env_major.components_regardless_involved)]:
            # NOTE: involved_components is a DataDFrame that should be "split" among to_use_idx.
            # since we are going to encode each row of the pd.DataFrame as a string, we
            # need to compose the array of string corresponding to the pd.DataFrame.
            df_all.loc[to_use_idx, COLUMN_NAME_CHANGED_COMP] = base.df_json(involved_components)

        # FINALLY, we need to annotate components that have changed *primarily* and *secondarily*
        df_all[COLUMN_NAME_CHANGED_COMP_PRIMARY] = '[]'
        df_all[COLUMN_NAME_CHANGED_COMP_SECONDARY] = '[]'
        # initially, we just copy column COLUMN_NAME_CHANGED_COMP for changes of type CODE
        df_all.loc[idx_change_code, COLUMN_NAME_CHANGED_COMP_PRIMARY] = df_all[COLUMN_NAME_CHANGED_COMP]
        for to_use_idx, involved_components in [(idx_change_code_env_minor, changed_components_code_env_minor),
                                                (idx_change_code_env_major, changed_components_code_env_major)]:
            df_all.loc[to_use_idx, COLUMN_NAME_CHANGED_COMP_PRIMARY] = \
                base.df_json(involved_components.components_primarily_involved)
            df_all.loc[to_use_idx, COLUMN_NAME_CHANGED_COMP_SECONDARY] = \
                base.df_json(involved_components.components_primarily_involved)

        # MORE FINALLY, we annotate which components are impacted by code change only (no behavior).
        for to_use_idx, involved_components in [(idx_change_code_only_minor, changed_components_for_code_only_minor),
                                                (idx_change_code_only_major, changed_components_for_code_only_major)
                                                ]:
            components_as_array = base.df_json(involved_components.components_primarily_involved)
            if verbose:
                print(f'IDX: {to_use_idx}: {components_as_array}')
            df_all.loc[to_use_idx, COLUMN_NAME_CHANGED_COMP] = components_as_array
            df_all.loc[to_use_idx, COLUMN_NAME_CHANGED_COMP_PRIMARY] = components_as_array

        # we finally add a very final column containing the novelty
        # which is always set to false for these experiments.
        df_all[const.COLUMN_NAME_NOVELTY] = const.NOVELTY_FALSE

        self.dataset = df_all
        self.dataset = self.dataset.reset_index(drop=True)
        return self.dataset

    def are_critical(self, components_idx: typing.List[np.ndarray]):
        # returns a boolean array.
        iterable = (np.any(self.components_criticality[list_of_comps] if len(list_of_comps) > 0 else False)
                    for list_of_comps in components_idx)
        return np.fromiter(iterable, int)
