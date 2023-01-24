# import os, sys
# sys.path.append(os.getcwd())

import dataclasses
import itertools
import json
import typing

import base
import exp_quality_scheme
import exp_quality_scheme_support


@dataclasses.dataclass
class OverallConfig:
    dataset_training: base.DatasetGeneratorForTrainingConfig
    exp_configs: typing.List[exp_quality_scheme_support.QualitySchemeConfig]

    def as_dict(self) -> dict:
        dataset_training = self.dataset_training.as_dict()
        exp_configs = [exp_config.as_dict() for exp_config in self.exp_configs]
        return {
            'dataset_training': dataset_training,
            'exp_configs': exp_configs
        }

    @staticmethod
    def from_dict(val: dict) -> "OverallConfig":
        dataset_training = base.DatasetGeneratorForTrainingConfig.from_dict(val['dataset_training'])
        exp_configs = [exp_quality_scheme_support.QualitySchemeConfig.from_dict(single_val)
                       for single_val in val['exp_configs']]
        return OverallConfig(
            dataset_training=dataset_training, exp_configs=exp_configs
        )


def main_func(config_file_name: str, output_directory: typing.Optional[str] = None, include_strip_down: bool = False):

    # we read the config
    with open(config_file_name) as config_file:
        config_parsed = json.load(config_file)

    if not isinstance(config_parsed, list):
        raise ValueError(f'Malformed config. Wanted type: list, got: {type(config_parsed)}')

    configs = [OverallConfig.from_dict(data) for data in config_parsed]

    all_exps = []

    for single_config in configs:

        # create the dataset for training
        print(f'{single_config.dataset_training.directory}: creating dataset')
        dataset_training_generator = exp_quality_scheme.DatasetGeneratorForTrainingAndModel.from_files(
            config=single_config.dataset_training, generator_clazz=base.DatasetGeneratorForTrainingEnsemble)
        print(f'{single_config.dataset_training.directory}: training')
        dataset_training_generator.fit()
        print(f'{single_config.dataset_training.directory}: training: ok')

        print(f'{single_config.dataset_training.directory}: running experiment')
        exps = exp_quality_scheme.ExperimentQualityScheme.run_multi_config(dataset_training=dataset_training_generator,
                                                                           quality_config=single_config.exp_configs)
        print(f'{single_config.dataset_training.directory}: running experiment: ok')
        all_exps.append(exps)

    print('merging results')
    merged = exp_quality_scheme.aggregate_all_results(list(itertools.chain(*all_exps)))

    print(f'exporting results at {output_directory}')
    if output_directory is not None:
        merged.export(base_directory=output_directory, include_strip_down=include_strip_down)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-file-name', type=str, required=True)
    parser.add_argument('--output-directory', type=str, required=True)
    parser.add_argument('--include-strip-down', type=bool, default=True)

    args = parser.parse_args()

    main_func(config_file_name=args.config_file_name, output_directory=args.output_directory,
              include_strip_down=args.include_strip_down)
