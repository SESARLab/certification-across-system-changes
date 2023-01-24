import numpy as np
import pandas as pd
import pytest

import const
import situation


@pytest.mark.parametrize('in_df, expected_situations', [
    (
            pd.DataFrame(
                [
                    # start off with a bunch of s0.
                    # no change at all. We perform several combinations over the other parameters just to be sure.
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    # minor changes with no impact on behavior or critical components (novelty does not matter here)
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # from here, test cases are exactly the same as in v1/situation_test.py
                    # here we expect s0 because we have a minor code change that is non-critical and not-novel.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # here we expect s1 because we have a minor code change with impact on the behavior.
                    [const.CHANGE_CAUSE_CODE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # here we expect s1 because we have an environmental change on non-critical
                    # and non-novel components
                    [const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # here we expect s1 because we have a major change (non-critical, non-new)
                    # not impacting on the behavior
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # here we expect s2 because we have a vulnerability
                    [const.CHANGE_CAUSE_VULN, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # for a double check, we also vary code-related parameters
                    # (situations that cannot happen in reality, but it may help us in catching bugs).
                    [const.CHANGE_CAUSE_VULN, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_VULN, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_VULN, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_VULN, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_VULN, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    # here we expect s2 because we hava a behavioral change caused by a major code change
                    # (non-critical, non-new).
                    [const.CHANGE_CAUSE_CODE_ENV, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_FALSE],
                    # here we expect s3 because we have a minor change on a critical code.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                    # here we expect s3 because we have a minor change on a novel code.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    # here we expect s3 because we have a minor change on a novel and critical code.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    # here we expect s3 because we have a major change on a critical code.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                    # here we expect s3 because we have a major change on a novel code.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE, const.NOVELTY_TRUE],
                    # here we expect s3 because we have a major change on a novel and critical code.
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    # here we expect s3 because we have an environmental change on a critical component
                    # (and novel component, but this does not matter but still we check).
                    [const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    # here we expect s3 because we have an environmental change on a critical component
                    # (and non-novel component, but this does not matter but still we check).
                    [const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                    # we also do the same above conditions related to the environments but with
                    # code-change = MINOR just to check.
                    [const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_TRUE],
                    [const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE, const.NOVELTY_FALSE],
                ],
                columns=[const.COLUMN_NAME_CHANGE_TYPE,
                         const.COLUMN_NAME_CODE_EXTENT,
                         const.COLUMN_NAME_CRITICAL,
                         const.COLUMN_NAME_NOVELTY]
            ),
            pd.Series([str(s) for s in [situation.S0(), situation.S0(), situation.S0(), situation.S0(),
                                        situation.S0(), situation.S0(), situation.S0(), situation.S0(),
                                        situation.S0(), situation.S0(), situation.S0(), situation.S0(),
                                        situation.S0(),
                                        # here starts the old ones as in situation_test
                                        situation.S0(), situation.S1(), situation.S1(), situation.S1(),
                                        situation.S2(), situation.S2(), situation.S2(), situation.S2(),
                                        situation.S2(), situation.S2(), situation.S2(),
                                        situation.S3(), situation.S3(), situation.S3(), situation.S3(),
                                        situation.S3(), situation.S3(), situation.S3(), situation.S3(),
                                        situation.S3(), situation.S3(),
                                        ]], name=const.COLUMN_NAME_SITUATION)
    )
])
def test_situation(in_df: pd.DataFrame, expected_situations: pd.Series):
    applied = situation.applies(in_df)
    assert np.all(applied[const.COLUMN_NAME_SITUATION] == expected_situations)
