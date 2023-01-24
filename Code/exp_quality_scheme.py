import itertools
import typing

import joblib
import numpy as np
import pandas as pd

import base
import const
import dataset_generator
import situation
import exp_quality_scheme_support


class DatasetGeneratorForTrainingAndModel:

    @staticmethod
    def from_files(config: base.DatasetGeneratorForTrainingConfig,
                   generator_clazz: typing.Type[base.AbstractDatasetGeneratorForTraining]
                   ) -> "DatasetGeneratorForTrainingAndModel":
        clean_df, anomalous_df = config.read_files()
        return DatasetGeneratorForTrainingAndModel(config=config, clean_df=clean_df, anomalous_df=anomalous_df,
                                                   generator_clazz=generator_clazz)

    def __init__(self, config: base.DatasetGeneratorForTrainingConfig,
                 clean_df: pd.DataFrame, anomalous_df: pd.DataFrame,
                 generator_clazz: typing.Type[base.AbstractDatasetGeneratorForTraining]):
        self.config = config
        self.generator = generator_clazz(config=self.config,
                                         clean_df=clean_df, anomalous_df=anomalous_df)
        self.model: base.AbstractAnomalyModel = None
        self.model_metrics_0 = pd.Series()

    @property
    def dataset(self) -> pd.DataFrame:
        return self.generator.dataset

    def fit(self) -> "DatasetGeneratorForTrainingAndModel":
        if self.model is None:
            self.generator.generate()

            self.model, X_test, y_test = self.generator.get_and_fit_model()
            self.model_metrics_0 = self.model.evaluate(X_test=X_test, y_test=y_test)
        return self


class ExperimentQualityScheme:

    def __init__(self, config: exp_quality_scheme_support.QualitySchemeConfig,
                 dataset_generator_and_model: DatasetGeneratorForTrainingAndModel):
        self.config = config
        self.aggregated_results: exp_quality_scheme_support.SingleConfigMultiExecutionOutput = None
        self.dtg = dataset_generator_and_model

    def run_single_execution_single_config(self) -> exp_quality_scheme_support.SingleExecutionOutput:
        if self.dtg.model is None:
            raise ValueError('self.model not trained yet!')
        dg = dataset_generator.DatasetGenerator(config=self.config.dataset_config, clean_df=self.dtg.generator.clean_df,
                                                anomalous_df=self.dtg.generator.anomalous_df)
        # time to have fun.
        # let's generate a dataset.
        dataset = dg.generate()
        # now, we the real goal of this function is to create a new dataset whose shape
        # is actually the same of the dataset used by situation in function applies,
        # so basically we are going to add another column
        # containing the changes detected according to our scheme.

        # the first step is to detect the environmental changes.
        # so, we extract the portion of the dataset important to us.

        columns_related_to_components = [col for col in dataset.columns if isinstance(col, str)
                                         and col.startswith(const.COMP_PREFIX)]

        y_true = np.ones(len(dataset))
        y_true[np.where((dataset[const.COLUMN_NAME_CHANGE_TYPE].values == const.CHANGE_CAUSE_ENV) |
                        (dataset[const.COLUMN_NAME_CHANGE_TYPE].values == const.CHANGE_CAUSE_CODE_ENV))[0]] = -1

        y_pred, changed_components = self.dtg.model.predict_and_get_components(
            dataset[columns_related_to_components].values)

        # now we re-evaluate the performance of the model.
        model_metrics = self.dtg.model.evaluate(y_test=y_true, y_pred=y_pred)

        applied_our = exp_quality_scheme_support.apply_our(dataset=dataset, generator=dg, y_pred=y_pred,
                                                           changed_components=changed_components)
        dataset = exp_quality_scheme_support.apply_state_of_the_art(dataset=applied_our.df, generator=dg)

        # now, we get the situation according to the different schemes.
        for prefix in [const.EvalType.GT, const.EvalType.OUR, const.EvalType.STOTA]:
            subset_columns = prefix.get_pertinent_columns(dataset.columns)
            subset_df = dataset[subset_columns]
            subset_columns_no_prefix = prefix.remove_prefixes(subset_columns)
            subset_df.columns = subset_columns_no_prefix
            situation_column = situation.applies(subset_df)[const.COLUMN_NAME_SITUATION]
            dataset[prefix.col_name(const.COLUMN_NAME_SITUATION)] = situation_column

        return exp_quality_scheme_support.SingleExecutionOutput.from_double(
            dataset=dataset,
            evaluation_our=exp_quality_scheme_support.evaluate_single(const.EvalType.OUR, dataset=dataset),
            evaluation_stota=exp_quality_scheme_support.evaluate_single(const.EvalType.STOTA, dataset=dataset),
            evaluation_model=model_metrics, config=self.config.dataset_config)

    def run_multi_execution_on_single_config(self) -> "ExperimentQualityScheme":

        results: typing.List[exp_quality_scheme_support.SingleExecutionOutput] = joblib.Parallel(
            n_jobs=self.config.max_n_jobs)(joblib.delayed(self.run_single_execution_single_config)()
                                           for _ in range(self.config.n_run_for_each))
        # now, we perform some aggregations, as usual.
        self.aggregated_results = exp_quality_scheme_support.SingleConfigMultiExecutionOutput.from_individual(
            results, training_dataset=self.dtg.dataset)
        return self

    @staticmethod
    def run_multi_config(dataset_training: DatasetGeneratorForTrainingAndModel,
                         quality_config: typing.Sequence[exp_quality_scheme_support.QualitySchemeConfig]
                         ) -> typing.Sequence["ExperimentQualityScheme"]:
        # first of all, we generate the training dataset and train our models
        dataset_training.fit()

        experiments = [ExperimentQualityScheme(config=config, dataset_generator_and_model=dataset_training)
                       for config in quality_config]

        experiments = [exp.run_multi_execution_on_single_config() for exp in experiments]
        return experiments


def aggregate_all_results(experiments: typing.Iterable[ExperimentQualityScheme]
                          ) -> exp_quality_scheme_support.OverallResult:

    # here we perform the very last steps.

    # we use this function to group over a set of experiments
    def group_over_a_set_of_experiments(list_of_exps_: typing.Iterable[ExperimentQualityScheme],
                                        series_to_append_: pd.Series) -> exp_quality_scheme_support.BaseExecutionOutput:
        evaluation_schemes_list_ = []
        evaluation_models_list_ = []

        for single_exp_ in list_of_exps_:
            evaluation_schemes_list_.append(single_exp_.aggregated_results.aggregated_evaluation_scheme)
            evaluation_models_list_.append(single_exp_.aggregated_results.aggregated_evaluation_model)

        # we also append the dataset name to this result.
        return exp_quality_scheme_support.BaseExecutionOutput.from_raw_lists(
            evaluation_scheme_list=evaluation_schemes_list_, evaluation_model_list=evaluation_models_list_
        ).append_series(series_to_append_)

    # 1. group per dataset (i.e., target system) and retrieve average per-dataset.
    def group_by_dataset(experiment: ExperimentQualityScheme) -> str:
        return experiment.config.dataset_config.dataset_name
    grouped = itertools.groupby(sorted(experiments, key=group_by_dataset), key=group_by_dataset)

    results_grouped_by_datasets = {}

    for dataset_name, list_of_exps in grouped:
        dataset_name: str = dataset_name
        list_of_exps: typing.Iterable[ExperimentQualityScheme] = list_of_exps
        # here we have *all* the execution run on this dataset.
        # so, we perform the overall summary.

        series_s = pd.Series([dataset_name], index=['Dataset'])

        results_grouped_by_datasets[dataset_name] = group_over_a_set_of_experiments(list_of_exps_=list_of_exps,
                                                                                    series_to_append_=series_s)

    # 2. group per config group, and dataset name
    def group_by_config_group(experiment: ExperimentQualityScheme) -> str:
        return experiment.config.dataset_config.group

    grouped = itertools.groupby(sorted(experiments, key=group_by_config_group), key=group_by_config_group)

    results_grouped_by_config_group = {}
    results_grouped_by_config_group_and_dataset = {}

    for config_group_name, list_of_exps in grouped:
        config_group_name: str = config_group_name
        list_of_exps: typing.List[ExperimentQualityScheme] = list(list_of_exps)

        series_s = pd.Series([config_group_name], index=['ConfigGroup'])
        results_grouped_by_config_group[config_group_name] = group_over_a_set_of_experiments(list_of_exps_=list_of_exps,
                                                                                             series_to_append_=series_s)

        # we also aggregated per dataset on each config_group.
        additionally_grouped = itertools.groupby(sorted(list_of_exps, key=group_by_dataset), key=group_by_dataset)
        for dataset_name, sublist_of_exp in additionally_grouped:
            dataset_name: str = dataset_name
            sublist_of_exp: typing.Iterable[ExperimentQualityScheme] = sublist_of_exp

            series_s = pd.Series([config_group_name, dataset_name], index=['ConfigGroup', 'Dataset'])
            results_grouped_by_config_group_and_dataset[f'{config_group_name}_{dataset_name}'] = \
                group_over_a_set_of_experiments(list_of_exps_=sublist_of_exp, series_to_append_=series_s)

    # 3. group per individual config.
    def group_by_individual_config(experiment: ExperimentQualityScheme) -> str:
        return experiment.config.dataset_config.name

    grouped = itertools.groupby(sorted(experiments, key=group_by_individual_config), key=group_by_individual_config)

    results_grouped_by_config = {}

    for config_name, list_of_exps in grouped:
        config_name: str = config_name
        list_of_exps: typing.Iterable[ExperimentQualityScheme] = list_of_exps

        series_s = pd.Series([config_name], index=['Config'])
        results_grouped_by_config[config_name] = group_over_a_set_of_experiments(list_of_exps_=list_of_exps,
                                                                                 series_to_append_=series_s)

    return exp_quality_scheme_support.OverallResult(
        grouped_by_dataset=results_grouped_by_datasets,
        grouped_by_config=results_grouped_by_config,
        grouped_by_config_group=results_grouped_by_config_group,
        grouped_by_config_group_and_dataset=results_grouped_by_config_group_and_dataset,
        results_of_each=[(exp.config, exp.aggregated_results) for exp in experiments]
    )
