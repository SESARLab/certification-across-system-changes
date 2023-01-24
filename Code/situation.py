import abc


import numpy as np
import pandas as pd

import const


class AbstractSituation(abc.ABC):

    @abc.abstractmethod
    def precondition(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def condition(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Takes as input an already filtered pd.DataFrame.
        :param df:
        :return:
        """
        pass

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, other):
        return str(self) == str(other)


class S0(AbstractSituation):

    def precondition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            # behavior not involved (from env or from code)
            (df[const.COLUMN_NAME_CHANGE_TYPE] != const.CHANGE_CAUSE_ENV) &
            (df[const.COLUMN_NAME_CHANGE_TYPE] != const.CHANGE_CAUSE_CODE_ENV) &
            # vulnerability not involved
            (df[const.COLUMN_NAME_CHANGE_TYPE] != const.CHANGE_CAUSE_VULN)
            ]

    def condition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            # no code changes
            (df[const.COLUMN_NAME_CHANGE_TYPE] != const.CHANGE_CAUSE_CODE) |
            # minor code changes with no impact on critical or novel components
            ((df[const.COLUMN_NAME_CODE_EXTENT] == const.CODE_EXTENT_MINOR) &
             (df[const.COLUMN_NAME_CRITICAL] == const.CRITICAL_FALSE) &
             (df[const.COLUMN_NAME_NOVELTY] == const.NOVELTY_FALSE))]


class S1(AbstractSituation):

    def precondition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            # no vulnerabilities
            (df[const.COLUMN_NAME_CHANGE_TYPE] != const.CHANGE_CAUSE_VULN) &
            # no critical or novel componentes
            (df[const.COLUMN_NAME_CRITICAL] ==
             const.CRITICAL_FALSE) & (df[const.COLUMN_NAME_NOVELTY] == const.NOVELTY_FALSE)]

    def condition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            # behavioral change from the environment
            (df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_ENV) |
            # behavioral change caused by minor code changes
            (
                    (df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE_ENV) &
                    (df[const.COLUMN_NAME_CODE_EXTENT] == const.CODE_EXTENT_MINOR)
            ) |
            # major code change with no impact on the behavior
            (
                    (df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE) &
                    (df[const.COLUMN_NAME_CODE_EXTENT] == const.CODE_EXTENT_MAJOR)
            )
            ]


class S2(AbstractSituation):

    def precondition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def condition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            # vulnerability is discovered
            (df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_VULN) |
            # behavioral change caused by major code changes
            (
                    (df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE_ENV) &
                    (df[const.COLUMN_NAME_CODE_EXTENT] == const.CODE_EXTENT_MAJOR)
            )
            ]


class S3(AbstractSituation):

    def precondition(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[
            # no vulnerabilities
            (df[const.COLUMN_NAME_CHANGE_TYPE] != const.CHANGE_CAUSE_VULN)
        ]

    def condition(self, df: pd.DataFrame) -> pd.DataFrame:
        # the condition here could be simplified, but it's not a problem
        # (e.g., th first condition can be removed by setting >= 0 in the second.
        return df[
            # behavioral change with impact on critical component
            ((df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_ENV) &
             (df[const.COLUMN_NAME_CRITICAL] == const.CRITICAL_TRUE)) |
            # code change with impact on critical component
            ((df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE) &
             (df[const.COLUMN_NAME_CRITICAL] == const.CRITICAL_TRUE)) |
            # code change with impact on novel component
            ((df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE) &
             (df[const.COLUMN_NAME_NOVELTY] == const.NOVELTY_TRUE)) |
            # code change with impact on behavior and on critical component
            ((df[const.COLUMN_NAME_CHANGE_TYPE] == const.CHANGE_CAUSE_CODE_ENV) &
             (df[const.COLUMN_NAME_CRITICAL] == const.CRITICAL_TRUE))
            ]


def applies(df: pd.DataFrame) -> pd.DataFrame:
    # the indices where to apply the situation understanding.
    # Initially equal to all the indices in df.
    filtered_indices = df.index
    # accumulated_removed_indices = df.index

    indices_removed_previously = np.array([])

    # add the column containing the situation.
    out = df.copy()
    out[const.COLUMN_NAME_SITUATION] = 'S4'

    for situation in [S0(), S1(), S2(), S3()]:
        pre_df = situation.precondition(df.iloc[filtered_indices])
        cond_df = situation.condition(pre_df)
        out.loc[cond_df.index, 'S'] = str(situation)

        indices_removed_previously = np.union1d(indices_removed_previously, cond_df.index)
        # at the next round, we operate on a dataset
        # containing all the initial points except those matched at this round (and at the previous).
        filtered_indices = np.setdiff1d(df.index, indices_removed_previously)

    return out

