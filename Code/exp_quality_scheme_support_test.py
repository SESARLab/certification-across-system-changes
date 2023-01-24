import typing

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest

import const
import dataset_generator
import exp_quality_scheme_support
import situation


def inner_test_partial_and_full(got: exp_quality_scheme_support.NonePartialFullReCertificationOutput,
                                expected: exp_quality_scheme_support.NonePartialFullReCertificationOutput):
    assert np.all(got.none == expected.none)
    assert np.all(got.partial == expected.partial)
    assert np.all(got.full == expected.full)


@pytest.mark.parametrize('source, expected', [
    (
            pd.Series([str(s) for s in [
                situation.S0(),
                situation.S0(),
                situation.S0(),
                situation.S1(),
                situation.S2()
            ]]),
            exp_quality_scheme_support.NonePartialFullReCertificationOutput(
                none=np.array([0, 1, 2]),
                partial=np.array([3, 4]),
                full=np.array([])
            )
    ),
    (
            pd.Series([str(s) for s in [
                situation.S0(),  # 0
                situation.S0(),  # 1
                situation.S2(),  # 2
                situation.S2(),  # 3
                situation.S3(),  # 4
                situation.S3(),  # 5
                situation.S1(),  # 6
                situation.S0(),  # 7
                situation.S2()  # 8
            ]]),
            exp_quality_scheme_support.NonePartialFullReCertificationOutput(
                none=np.array([0, 1, 7]),
                partial=np.array([2, 3, 6, 8]),
                full=np.array([4, 5])
            )
    )
])
def test_get_partial_and_full_our(source: pd.Series,
                                  expected: exp_quality_scheme_support.NonePartialFullReCertificationOutput):
    got = exp_quality_scheme_support.get_partial_and_full_our(source=source)
    inner_test_partial_and_full(got=got, expected=expected)


@pytest.mark.parametrize('source, expected', [
    (
            pd.DataFrame(
                [
                    # no changes -> none
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_FALSE],  # 0
                    # just playing around with combinations of the above
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_TRUE],  # 1
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE],  # 2
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE],  # 3
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE],  # 4
                    [const.CHANGE_CAUSE_NO, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE],  # 5
                    # now partial
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE],  # 6
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE],  # 7
                    # now full
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE],  # 8
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE],  # 9
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE],  # 10
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_TRUE],  # 11
                    [const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE],  # 12
                ],
                columns=[const.COLUMN_NAME_CHANGE_TYPE_STOTA, const.COLUMN_NAME_CODE_EXTENT_STOTA,
                         const.COLUMN_NAME_CRITICAL_STOTA]
            ),
            exp_quality_scheme_support.NonePartialFullReCertificationOutput(
                none=np.array([0, 1, 2, 3, 4, 5]),
                partial=np.array([6, 7]),
                full=np.array([8, 9, 10, 11, 12])
            )
    )
])
def test_get_partial_and_full_stota(source: pd.DataFrame,
                                    expected: exp_quality_scheme_support.NonePartialFullReCertificationOutput):
    got = exp_quality_scheme_support.get_partial_and_full_stota(source=source)
    inner_test_partial_and_full(got=got, expected=expected)


@pytest.mark.parametrize('eval_type', [
    const.EvalType.STOTA,
    const.EvalType.OUR,
    const.EvalType.GT
])
def test_get_changes_of_type(eval_type: const.EvalType, ):
    source = pd.DataFrame(np.array([
        const.CHANGE_CAUSE_NO,  # 0
        const.CHANGE_CAUSE_NO,  # 1
        const.CHANGE_CAUSE_NO,  # 2
        const.CHANGE_CAUSE_CODE,  # 3
        const.CHANGE_CAUSE_ENV,  # 4
        const.CHANGE_CAUSE_CODE,  # 5
        const.CHANGE_CAUSE_ENV,  # 6
        const.CHANGE_CAUSE_CODE_ENV,  # 7
        const.CHANGE_CAUSE_CODE_ENV,  # 8
    ]).reshape(-1, 1), columns=[eval_type.col_name(const.COLUMN_NAME_CHANGE_TYPE)])

    expected = exp_quality_scheme_support.ChangesOfTypeOutput(
        changes_all=np.array([3, 4, 5, 6, 7, 8]),
        changes_code=np.array([3, 5]),
        changes_env=np.array([4, 6]),
        changes_code_env=np.array([7, 8])
    )

    got_changes = exp_quality_scheme_support.get_changes_of_type(eval_type=eval_type, dataset=source)
    assert np.all(got_changes.changes_all == expected.changes_all)
    assert np.all(got_changes.changes_env == expected.changes_env)
    assert np.all(got_changes.changes_code == expected.changes_code)
    assert np.all(got_changes.changes_code_env == expected.changes_code_env)


@pytest.mark.parametrize('eval_type', [
    const.EvalType.OUR,
    const.EvalType.STOTA,
    const.EvalType.GT
])
def test_get_involved_components(eval_type: const.EvalType):
    source = pd.DataFrame([
        '[1, 0, 5]',
        '[4, 3]',
        '[]',
        '[1, 2, 3, 4, 5]'
    ], columns=[eval_type.col_name(const.COLUMN_NAME_CHANGED_COMP)])

    expected = exp_quality_scheme_support.ChangedComponentsPerSituationOutput(
        s_less=[np.array([1, 0, 5]), np.array([4, 3]), np.array([]), np.array([1, 2, 3, 4, 5])]
    )

    involved_components = exp_quality_scheme_support.get_involved_components(eval_type=eval_type,
                                                                             dataset=source)
    for got_single_row, expected_single_row in zip(involved_components.s_less, expected.s_less):
        assert np.all(got_single_row == expected_single_row)


@pytest.mark.parametrize('df, expected, eval_type', [
    (
            # all correct
            pd.DataFrame(
                [
                    ['[1, 0, 5]', '[1, 0, 5]', str(situation.S1()), str(situation.S1()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_ENV],
                    ['[4, 3]', '[4, 3]', str(situation.S2()), str(situation.S2()), const.CHANGE_CAUSE_CODE,
                     const.CHANGE_CAUSE_CODE],
                    ['[]', '[]', str(situation.S0()), str(situation.S0()), const.CHANGE_CAUSE_NO,
                     const.CHANGE_CAUSE_NO],
                    ['[1, 3, 4]', '[1, 3, 4]', str(situation.S3()), str(situation.S3()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_ENV],
                    ['[4, 3]', '[4, 3]', str(situation.S2()), str(situation.S2()), const.CHANGE_CAUSE_CODE_ENV,
                     const.CHANGE_CAUSE_CODE_ENV],
                    ['[4, 3]', '[4, 3]', str(situation.S2()), str(situation.S2()), const.CHANGE_CAUSE_CODE,
                     const.CHANGE_CAUSE_CODE]
                ],
                columns=[const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGED_COMP),
                         const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGED_COMP),
                         const.EvalType.GT.col_name(const.COLUMN_NAME_SITUATION),
                         const.EvalType.OUR.col_name(const.COLUMN_NAME_SITUATION),
                         const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                         const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGE_TYPE)]
            ),
            1.0,
            const.EvalType.OUR
    ),
    (
            # all wrong
            pd.DataFrame(
                [
                    ['[]', '[2]', str(situation.S0()), str(situation.S1()), const.CHANGE_CAUSE_NO,
                     const.CHANGE_CAUSE_CODE],
                    ['[1, 0, 5]', '[2, 3]', str(situation.S2()), str(situation.S1()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_CODE],
                    ['[1, 0, 5]', '[2, 3, 4]', str(situation.S1()), str(situation.S2()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_CODE],
                    ['[1, 0, 5]', '[2, 3, 4]', str(situation.S2()), str(situation.S1()), const.CHANGE_CAUSE_CODE_ENV,
                     const.CHANGE_CAUSE_ENV],
                    ['[1, 0, 5]', '[2, 3, 4]', str(situation.S1()), str(situation.S2()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_CODE_ENV],
                    ['[1, 0, 5]', '[2, 3, 4]', str(situation.S3()), str(situation.S2()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_CODE_ENV],
                    ['[1, 0, 5]', '[2, 3, 4]', str(situation.S2()), str(situation.S3()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_CODE_ENV],
                    ['[1, 0, 5]', '[2, 3, 4]', str(situation.S2()), str(situation.S0()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_NO]
                ],
                columns=[const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGED_COMP),
                         const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGED_COMP),
                         const.EvalType.GT.col_name(const.COLUMN_NAME_SITUATION),
                         const.EvalType.OUR.col_name(const.COLUMN_NAME_SITUATION),
                         const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                         const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGE_TYPE)]
            ),
            (0.0, 0.9),
            const.EvalType.OUR
    ),
    (
            # cannot check proper values here.
            pd.DataFrame(
                [
                    ['[1, 0, 5]', '[1, 0, 5]', str(situation.S1()), str(situation.S1()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE],
                    ['[4, 3]', '[4, 3]', str(situation.S2()), str(situation.S2()), const.CHANGE_CAUSE_CODE,
                     const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE],
                    ['[]', '[]', str(situation.S0()), str(situation.S0()), const.CHANGE_CAUSE_NO,
                     const.CHANGE_CAUSE_NO, const.CODE_EXTENT_NO, const.CRITICAL_FALSE],
                    ['[1, 3, 4]', '[1, 3, 4]', str(situation.S3()), str(situation.S3()), const.CHANGE_CAUSE_ENV,
                     const.CHANGE_CAUSE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_TRUE],
                    ['[4, 3]', '[4, 3]', str(situation.S2()), str(situation.S2()), const.CHANGE_CAUSE_CODE_ENV,
                     const.CHANGE_CAUSE_CODE_ENV, const.CODE_EXTENT_MINOR, const.CRITICAL_FALSE],
                    ['[4, 3]', '[4, 3]', str(situation.S2()), str(situation.S2()), const.CHANGE_CAUSE_CODE,
                     const.CHANGE_CAUSE_CODE, const.CODE_EXTENT_MAJOR, const.CRITICAL_FALSE]
                ],
                columns=[const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGED_COMP),
                         const.EvalType.STOTA.col_name(const.COLUMN_NAME_CHANGED_COMP),
                         const.EvalType.GT.col_name(const.COLUMN_NAME_SITUATION),
                         const.EvalType.STOTA.col_name(const.COLUMN_NAME_SITUATION),
                         const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                         const.EvalType.STOTA.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                         const.EvalType.STOTA.col_name(const.COLUMN_NAME_CODE_EXTENT),
                         const.EvalType.STOTA.col_name(const.COLUMN_NAME_CRITICAL)]
            ),
            None,
            const.EvalType.STOTA
    )
])
def test_evaluate_single(df: pd.DataFrame,
                         expected: typing.Optional[typing.Union[float, pd.Series, typing.Tuple[float, float]]],
                         eval_type: const.EvalType
                         ):
    result = exp_quality_scheme_support.evaluate_single(eval_type=eval_type, dataset=df)
    if expected is not None:
        expected_extended = expected
        if isinstance(expected, float):
            expected_extended = np.repeat(expected, repeats=len(result))
        if not isinstance(expected, tuple):
            assert np.all(result == expected_extended)
        else:
            expected_lower = np.repeat(expected[0], repeats=len(result))
            expected_higher = np.repeat(expected[1], repeats=len(result))
            assert np.all((result >= expected_lower) & (result < expected_higher))


def get_generator_with_fixed_criticality(components_criticality: npt.NDArray[bool]
                                         ) -> dataset_generator.DatasetGenerator:
    generator = dataset_generator.DatasetGenerator(config=dataset_generator.DatasetGeneratorConfigLinearProba(
        group='P1.X', name='P1.1', wanted_size=100,
    ), clean_df=pd.DataFrame(), anomalous_df=pd.DataFrame())
    # hard-code which components are critical according to the input.
    generator.components_criticality = components_criticality
    return generator


def compare_df_verbose(result_df: pd.DataFrame, expected_df: pd.DataFrame):
    # if columns are not in the same order, it is difficult.
    col_order = result_df.columns
    expected_df = expected_df.loc[:, col_order]

    for col in result_df.columns:
        assert np.all(result_df[col] == expected_df[col]), col


@pytest.mark.parametrize('df, components_criticality, column_detected_change, changed_components, expected', [
    (
            pd.DataFrame([
                # 1
                [const.CHANGE_CAUSE_NO, const.CHANGE_CAUSE_NO, const.CRITICAL_FALSE, const.CODE_EXTENT_NO, '[]',
                 const.NOVELTY_FALSE],
                # 1
                [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_FALSE, const.CODE_EXTENT_MINOR,
                 '[1]', const.NOVELTY_FALSE],
                # -1
                [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_FALSE,
                 const.CODE_EXTENT_MINOR, '[1]', const.NOVELTY_FALSE],
                # 1
                [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_FALSE, const.CODE_EXTENT_MAJOR,
                 '[1]', const.NOVELTY_FALSE],
                # -1
                [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_FALSE,
                 const.CODE_EXTENT_MAJOR, '[1]', const.NOVELTY_FALSE],
                # 1
                [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_TRUE, const.CODE_EXTENT_MINOR,
                 '[2]', const.NOVELTY_FALSE],
                # -1
                [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_TRUE, const.CODE_EXTENT_MINOR,
                 '[2]', const.NOVELTY_FALSE],
                # 1
                [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_TRUE, const.CODE_EXTENT_MAJOR,
                 '[2]', const.NOVELTY_FALSE],
                # -1
                [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_TRUE, const.CODE_EXTENT_MAJOR,
                 '[2]', const.NOVELTY_FALSE],
                # -1
                [const.CHANGE_CAUSE_ENV, const.CHANGE_CAUSE_ENV, const.CRITICAL_TRUE, const.CODE_EXTENT_NO, '[2]',
                 const.NOVELTY_FALSE],
                # -1
                [const.CHANGE_CAUSE_ENV, const.CHANGE_CAUSE_ENV, const.CRITICAL_FALSE, const.CODE_EXTENT_NO, '[1]',
                 const.NOVELTY_FALSE],
            ], columns=[
                const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                const.EvalType.GT.col_name(const.COLUMN_NAME_CRITICAL),
                const.EvalType.GT.col_name(const.COLUMN_NAME_CODE_EXTENT),
                const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGED_COMP),
                const.EvalType.GT.col_name(const.COLUMN_NAME_NOVELTY),
            ]),
            # comp0 and comp1 are not critical, comp2 is
            np.array([False, False, True]),
            np.array([1, 1, -1, 1, -1, 1, -1, 1, -1, -1, -1]),
            [
                np.array([]),
                # 1
                np.array([]),
                # -1
                np.array([1]),
                # 1
                np.array([]),
                # -1
                np.array([1]),
                # 1
                np.array([]),
                # -1
                np.array([2]),
                # 1
                np.array([]),
                # -1
                np.array([2]),
                # -1
                np.array([2]),
                # -1
                np.array([1])],
            exp_quality_scheme_support.ApplyOurOutput(
                any_changes_idx=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
                code_idx=np.array([1, 3, 5, 7]),
                code_env_idx=np.array([2, 4, 6, 8]),
                env_idx=np.array([9, 10]),
                code_and_code_env_idx=np.array([1, 2, 3, 4, 5, 6, 7, 8]),
                df=pd.DataFrame([
                    [const.CHANGE_CAUSE_NO, const.CHANGE_CAUSE_NO, const.CRITICAL_FALSE, const.CRITICAL_FALSE,
                     const.CODE_EXTENT_NO, const.CODE_EXTENT_NO,
                     '[]', '[]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_FALSE, const.CRITICAL_FALSE,
                     const.CODE_EXTENT_MINOR, const.CODE_EXTENT_MINOR,
                     '[1]', '[1]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_FALSE,
                     const.CRITICAL_FALSE,
                     const.CODE_EXTENT_MINOR, const.CODE_EXTENT_MINOR,
                     '[1]', '[1]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_FALSE, const.CRITICAL_FALSE,
                     const.CODE_EXTENT_MAJOR, const.CODE_EXTENT_MAJOR,
                     '[1]', '[1]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_FALSE,
                     const.CRITICAL_FALSE,
                     const.CODE_EXTENT_MAJOR, const.CODE_EXTENT_MAJOR,
                     '[1]', '[1]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_TRUE, const.CRITICAL_TRUE,
                     const.CODE_EXTENT_MINOR, const.CODE_EXTENT_MINOR,
                     '[2]', '[2]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_TRUE, const.CRITICAL_TRUE,
                     const.CODE_EXTENT_MINOR, const.CODE_EXTENT_MINOR,
                     '[2]', '[2]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE, const.CHANGE_CAUSE_CODE, const.CRITICAL_TRUE, const.CRITICAL_TRUE,
                     const.CODE_EXTENT_MAJOR, const.CODE_EXTENT_MAJOR,
                     '[2]', '[2]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_CODE_ENV, const.CHANGE_CAUSE_CODE_ENV, const.CRITICAL_TRUE, const.CRITICAL_TRUE,
                     const.CODE_EXTENT_MAJOR, const.CODE_EXTENT_MAJOR,
                     '[2]', '[2]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_ENV, const.CHANGE_CAUSE_ENV, const.CRITICAL_TRUE, const.CRITICAL_TRUE,
                     const.CODE_EXTENT_NO, const.CODE_EXTENT_NO,
                     '[2]', '[2]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                    [const.CHANGE_CAUSE_ENV, const.CHANGE_CAUSE_ENV, const.CRITICAL_FALSE, const.CRITICAL_FALSE,
                     const.CODE_EXTENT_NO, const.CODE_EXTENT_NO,
                     '[1]', '[1]', const.NOVELTY_FALSE, const.NOVELTY_FALSE],
                ], columns=[
                    const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                    const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGE_TYPE),
                    const.EvalType.GT.col_name(const.COLUMN_NAME_CRITICAL),
                    const.EvalType.OUR.col_name(const.COLUMN_NAME_CRITICAL),
                    const.EvalType.GT.col_name(const.COLUMN_NAME_CODE_EXTENT),
                    const.EvalType.OUR.col_name(const.COLUMN_NAME_CODE_EXTENT),
                    const.EvalType.GT.col_name(const.COLUMN_NAME_CHANGED_COMP),
                    const.EvalType.OUR.col_name(const.COLUMN_NAME_CHANGED_COMP),
                    const.EvalType.GT.col_name(const.COLUMN_NAME_NOVELTY),
                    const.EvalType.OUR.col_name(const.COLUMN_NAME_NOVELTY),
                ])
            )
    )
])
def test_apply_our(df: pd.DataFrame, components_criticality: npt.NDArray[int],
                   column_detected_change: np.ndarray,
                   changed_components: typing.List[np.ndarray],
                   expected: exp_quality_scheme_support.ApplyOurOutput):
    generator = get_generator_with_fixed_criticality(components_criticality=components_criticality)

    result = exp_quality_scheme_support.apply_our(dataset=df, generator=generator, y_pred=column_detected_change,
                                                  changed_components=changed_components)
    assert np.all(result.any_changes_idx == expected.any_changes_idx)
    assert np.all(result.code_and_code_env_idx == expected.code_and_code_env_idx)
    assert np.all(result.code_idx == expected.code_idx)
    assert np.all(result.code_env_idx == expected.code_env_idx)

    compare_df_verbose(result_df=result.df, expected_df=expected.df)


@pytest.mark.parametrize('df, expected, components_criticality', [
    (
        pd.DataFrame(
            [
                [const.CHANGE_CAUSE_CODE, '[1, 2, 3]', const.CODE_EXTENT_MINOR, const.NOVELTY_FALSE],
                [const.CHANGE_CAUSE_CODE, '[1, 2]', const.CODE_EXTENT_MAJOR, const.NOVELTY_FALSE],
                [const.CHANGE_CAUSE_NO, '[]', const.CODE_EXTENT_NO, const.NOVELTY_FALSE]
            ], columns=[
                const.COLUMN_NAME_CHANGE_TYPE, const.COLUMN_NAME_CHANGED_COMP_PRIMARY, const.COLUMN_NAME_CODE_EXTENT,
                const.COLUMN_NAME_NOVELTY
            ]
        ),
        pd.DataFrame(
            [
                [const.CHANGE_CAUSE_CODE, '[1, 2, 3]', const.CHANGE_CAUSE_CODE, '[1, 2, 3]', const.CRITICAL_TRUE,
                 const.CODE_EXTENT_MINOR, const.NOVELTY_FALSE, const.CODE_EXTENT_MINOR, const.NOVELTY_FALSE],
                [const.CHANGE_CAUSE_CODE, '[1, 2]', const.CHANGE_CAUSE_CODE, '[1, 2]', const.CRITICAL_FALSE,
                 const.CODE_EXTENT_MAJOR, const.NOVELTY_FALSE, const.CODE_EXTENT_MAJOR, const.NOVELTY_FALSE],
                [const.CHANGE_CAUSE_NO, '[]', const.CHANGE_CAUSE_NO, '[]', const.CRITICAL_FALSE, const.CODE_EXTENT_NO,
                 const.NOVELTY_FALSE, const.CODE_EXTENT_NO, const.NOVELTY_FALSE]
            ], columns=[
                const.COLUMN_NAME_CHANGE_TYPE, const.COLUMN_NAME_CHANGED_COMP_PRIMARY,
                const.COLUMN_NAME_CHANGE_TYPE_STOTA, const.COLUMN_NAME_CHANGED_COMP_STOTA,
                const.COLUMN_NAME_CRITICAL_STOTA,
                const.COLUMN_NAME_CODE_EXTENT_STOTA, const.COLUMN_NAME_NOVELTY_STOTA,
                const.COLUMN_NAME_CODE_EXTENT, const.COLUMN_NAME_NOVELTY
            ]
        ),
        np.array([False, False, False, True])
    )
])
def test_apply_stota(df: pd.DataFrame, expected: pd.DataFrame, components_criticality: npt.NDArray[int]):
    generator = get_generator_with_fixed_criticality(components_criticality=components_criticality)
    result = exp_quality_scheme_support.apply_state_of_the_art(dataset=df, generator=generator)

    compare_df_verbose(result_df=result, expected_df=expected)
