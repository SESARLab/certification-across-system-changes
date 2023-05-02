# Continuous Certification of Non-Functional Properties Across System Changes

> Existing certification schemes implement continuous verification techniques aimed to prove  non-functional (e.g., security) properties of software systems over time. These schemes provide different re-certification techniques for managing the certificate life cycle, though their strong assumptions make them ineffective against modern distributed systems. Re-certification techniques are in fact built on static system models, which do not properly represent the system evolution and on static detection of system changes that result in an inaccurate planning of re-certification activities. In this paper, we propose a continuous certification scheme that departs from a static certificate life cycle management and provides a dynamic approach built on a machine learning-based modeling of the system behavior. It reduces the amount of unnecessary re-certification by monitoring and detecting changes on the system behavior. The quality and performance of the proposed scheme is experimentally evaluated using a publicly-available dataset built on three composite (micro)services in literature.

This repository contains the source code, input dataset, and detailed results of our experimental evaluation.

<!-- vscode-markdown-toc -->
* 1. [Overview](#Overview)
* 2. [Organization](#Organization)
	* 2.1. [Details: Code Organization](#Details:CodeOrganization)
	* 2.2. [Details: Input](#Details:Input)
	* 2.3. [Details: Output](#Details:Output)
* 3. [Example Execution](#ExampleExecution)
* 4. [Appendix of the Paper](#AppendixPaper)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

##  1. <a name='Overview'></a>Overview

The code is written in Python 3 and tested in an MacOS environment (`virtualenv`) with Python 3.10; dependencies are listed in [requirements.txt](requirements.txt).

The aim of our experimental evaluation is to compare our scheme with a scheme representing the state of the art, covering all the scenarios as described in our paper. For this reason, our experimental data are based on the dataset available at [https://doi.org/10.13012/B2IDB-6738796_V1](https://doi.org/10.13012/B2IDB-6738796_V1). It measures the response time of a set of microservices along a given execution path in normal and anomalous conditions. Specifically, the creators of the dataset executed three well-known distributed systems in the literature with and without injecting anomalies (for details, see the corresponding paper: [https://www.usenix.org/conference/osdi20/presentation/qiu](https://www.usenix.org/conference/osdi20/presentation/qiu)).

We defined our experimental settings extracting normal and anomalous data from the above data, to generate a dataset including environmental changes and code changes with and without impact on the behavior. Each data point of the dataset is also annotated with additional information (e.g., presence of critical components affected by the change).
We then apply our scheme and the state of the art scheme on the generated dataset.

Each row of the table below represents an experimental settings driving the generation of our datasets. The process of dataset generation and schemes application have been repeated 10 times for each distributed system and experimental settings.

##  2. <a name='Organization'></a>Organization

The repository is organized in the following directories:

- [Code](code): contains the Python code to run the experiments.
- [Input](Input): contains the raw data (response times) of the microservices, taken from [https://doi.org/10.13012/B2IDB-6738796_V1](https://doi.org/10.13012/B2IDB-6738796_V1).
- [Output](Output): contains the results of our experiments, including generated datasets, with results aggregated at different levels.
- [input.json](input.json): contains the actual input to run the code and reproduce our results.

###  2.1. <a name='Details:CodeOrganization'></a>Details: Code Organization

The code consists of the following files:

- [Code/base.py](Code/base.py): contains utility classes and functions used in the rest of the code, including the code to train the system model based on isolation forest.
- [Code/const.py](Code/const.py): contains constants.
- [Code/dataset_generator.py](Code/dataset_generator.py): contains the code to generate the experimental dataset starting from data in the provided input directory.
- [Code/entrypoint.py](Code/entrypoint.py): contains the main entrypoint
- [Code/exp_quality_scheme.py](Code/exp_quality_scheme.py): contains the code running the *core* function applying our scheme and the state of the art on a set of experimental settings and evaluating the results
- [Code/exp_quality_scheme_support.py](Code/exp_quality_scheme_support.py): contains support code, including code to export data
- [Code/situation.py](Code/situation.py): contains the code selecting the *scenario* that applies in each detected change

Each file has its own set of tests executable with `pytest`.

###  2.2. <a name='Details:Input'></a>Details: Input

The code requires two inputs: initial data from [https://doi.org/10.13012/B2IDB-6738796_V1](https://doi.org/10.13012/B2IDB-6738796_V1), and experimental settings as reported in the table above.

**Initial data**: the code works on the data as is, as long as each distributed system has its own directory.
**Experimental settings**: they need to be provided as a json file; this file contains the different probabilities as well as path to input data. This repository contains the input data we used during our experiments, therefore paths to be adjusted properly.

###  2.3. <a name='Details:Output'></a>Details: Output

The code produces aggregated and detailed results in two formats: excel and csv. The two formats are *always* generated, that is, there is no option to choose the desired format.

During execution, the code requires the base directory where output data should be placed. With the hope that they may be useful, we include **all our data**: our results as well as generated datasets.

In particular, we generate two types of datasets:

- one dataset for each distributed system used to train system model based on isolation forest, to detect anomalies later. Basically, it is merge of the individual csv files in [https://doi.org/10.13012/B2IDB-6738796_V1](https://doi.org/10.13012/B2IDB-6738796_V1); located at directory [Output/DatasetsTraining](Output/DatasetsTraining).
- 10 datasets for each distributed system and experimental setting, where we apply our scheme as well as the state of the art; located at [Output/Detailed/*system-name*/*config-name*/*N_dataset*](Output/Detailed/MS/P1.1/0_dataset.csv), with *N* in *[0, 9]*.

More in detail, the generated output is contained in the following directories.

- [Output/Aggregated_Configs](Output/Aggregated_Configs): contains data aggregated over the three distributed systems, reporting results for each experimental setting.
- [Output/Aggregated_ConfigsGroup](Output/Aggregated_ConfigsGroup): contains data aggregated over the groups of experimental settings (e.g., *P1* is a group, *P2* is a group, and so on) reporting separated results for each distributed system.
- [Output/Aggregated_ConfigsGroup_Datasets](Output/Aggregated_ConfigsGroup_Datasets): contains data aggregated over the groups of experimental settings (e.g., *P1* is a group, *P2* is a group, and so on) and each distributed system.
- [Output/Aggregated_Datasets](Output/Aggregated_Datasets): contains data aggregated over each distributed system.
- [Output/Aggregated_Datasets](Output/Aggregated_Super): contains data aggregated at the highest level possible.
- [Output/DatasetsTraining](Output/DatasetsTraining): contains the dataset generated to train the system model based on isolation forest.
- [Output/Detailed](Output/Detailed): contains detailed non-aggregated results.

Individual files have a pretty self-explanatory names, those containing `scheme` contains data evaluating the two certification schemes, those containing `model` contains data evaluating the system model.

**Data in our paper are created mostly from [Output/Aggregated_Datasets/by_config_scheme_stripped.xlsx](Output/Aggregated_Datasets/by_config_scheme_stripped.xlsx) and[Output/Aggregated_Datasets/by_dataset_model_stripped.xlsx](Output/Aggregated_Datasets/by_dataset_model_stripped.xlsx) for what concerns system model**.

**Note**: our experiments have been executed with option `--include-strip-down True`, this means that for each output file there exists a *stripped down version* where only columns including averaged data are reported. This also means that some files are basically empty because this filter removes all data. We nevertheless decided to include these files, since we run our experiments including this option.

##  3. <a name='ExampleExecution'></a>Example Execution

To reproduce our results you need to:

- modify [input.json](input.json) according to the path where you placed directory [Input](Input)
- run the following command, replacing `path-to-input.json` and `path-to-output` with the desired paths.

```bash
python entrypoint.py \
	--config-file-name path-to-inputv.json \
	--output-directory path-to-output \
	--include-strip-down True
```

Execution is parallelized and experiments should last some minutes.

## 4. <a name="AppendixPaper"></a> Appendix of the Paper

The file [appendix.pdf](appendix.pdf) contains the appendix of the corresponding paper, including quality evaluation of the machine learning-based system model.