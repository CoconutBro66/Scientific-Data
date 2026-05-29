# Technical Validation Code

This repository provides the custom code used for the technical validation experiments reported in the manuscript.

The repository contains two code directories and one environment configuration file.

## Repository structure

| File or directory | Description |
|---|---|
| `TAMA.yaml` | Conda environment file used to create the runtime environment for the validation experiments. |
| `01_Single_Task/` | Code for independent single-task fine-tuning experiments. This directory contains scripts for target detection, fine-grained abuse type classification, and phrase-level span localization. |
| `02_Shared-encoder multi-task/` | Code for the shared-encoder multi-task baseline used in the technical validation experiments. |

## Install dependencies

Create a conda environment using the provided environment file.

```bash
conda env create --file=abuse.yaml
conda activate abuse
```

## Baseline model

The baseline model used in the validation experiments is `multilingual-e5-base`.

The experiments include two baseline settings.

| Setting | Description |
|---|---|
| Single-task baseline | Each task is trained independently using the scripts in `01_Single_Task/`. |
| Shared-encoder multi-task baseline | The three task layers are trained under a shared multilingual encoder setting using the scripts in `02_Shared-encoder multi-task/`. |

## Experimental parameters

The experimental parameters used for the validation experiments are defined in the corresponding code files. These include the baseline model configuration, training settings, evaluation settings, and task-specific parameters.

## Running the experiments

Run each script from the corresponding directory after preparing the environment and setting the paths expected by the scripts.

Example commands for the single-task baseline:

```bash
python t1.py
python t2.py
python t3.py
```

Example command for the shared-encoder multi-task baseline:

```bash
python MTL_Base.py
```

## Notes

This repository is intended to support reproduction of the technical validation experiments. The code runtime environment, baseline model settings, and commands needed to run the validation experiments are described in this README and in the corresponding scripts.



