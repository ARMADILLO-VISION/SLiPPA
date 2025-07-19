# SLiPPA Code Repository

**S**egmentation of the **Li**ver with **P**ost-**P**rocessing **A**pplied (SLiPPA) is the codebase used in the MIUA 2025 conference paper: [Intraoperative Liver Segmentation Through Deep Learning and Mask Post-processing in Laparoscopic Liver Surgery](https://doi.org/10.1007/978-3-031-98694-9_15).

**Author's note**: This public codebase is the result of a large scale refactoring of the code used in the conference pape for increased value to others; the code has not yet been tested since the refactoring. It has been released such that code can be viewed alongside the published article from its release. Code may require changes to have working functionality on your machine, but the improved quality of the code should make this an easier task.

## Installation
This repository contains an Anaconda environment (`environment.yml`) containing the required dependencies to use this code. Please ensure Anaconda is installed before proceeding. To create an environment, whilst in the repository directory, run the following command:
```bash
conda env create --name slippa --file=environment.yml
```
This will create an environment called `slippa` on your system with the required dependencies.

## Datasets
This codebase has provided compatibility with two datasets: L3D (available from [this repository](https://github.com/PJLallen/D2GPLand)) and P2ILF (not currently publicly available). A preparation script is provided to extract and organise the data in the way the codebase expects:
```bash
python prepare.py --path=[PATH TO DATASET ZIP] --dataset=L3D_train
```
The available dataset options are: `L3D_train`, `L3D_val`, `L3D_test`, `P2ILF_train`, and `P2ILF_test`. Please provide the path to the ZIP file that has been downloaded without any extractions or alterations applied.

The preparation script automatically separates the P2ILF training data into a training set and a validation set, where patients 1 and 2 make the validation set.

## Inference
Random inferences from a dataset can be made using the following command:
```bash
python predict.py --arch=[unet, unetpp, unet3p, resunetpp, deeplabv3p] --weights=[PATH TO PTH CHECKPOINT] --dataset=[L3D_train, L3D_val, L3D_test, P2ILF_train, P2ILF_test]
```
Dataset options are the same as the above preparation script with the addition of `P2ILF_val` (patients 1 and 2 are not present in `P2ILF_train`). Options for the `model` argument are: `unet`, `unetpp`, `unet3p`, `resunetpp`, and `deeplabv3p`.

## Training
```bash
python train.py --arch=[unet, unetpp, unet3p, resunetpp, deeplabv3p] --model=[PATH TO PTH CHECKPOINT]
```

Optional settings include `--batch`, `--lr`, `--coeffs`, `--patience`, `--no_pretrain`, `--no_ft`, `--batch_ft`, and `--lr_ft`.

## Evaluation
```bash
python evaluate.py --arch=[unet, unetpp, unet3p, resunetpp, deeplabv3p] --path=[PATH TO PTH CHECKPOINT]
```

This will output a CSV file with the performance of the model against the P2ILF test set.

**Note:** In the paper, the François symmetric distance evaluation code was used from [here](https://github.com/sharib-vision/P2ILF), there may be differences between implementations. All other metrics have identical code.

## Citation
If this code is used in any work, please cite the following:
```bibtex
@InProceedings{10.1007/978-3-031-98694-9_15,
author="Borgars, James
and Raja, Jibran
and Ramakrishnan, Abhinav
and Abbas, Abdul Karim
and Gallagher, Aodhan
and Mohamad Shahir, Ahmad Najmi
and Vraimakis, Theodora
and Ali, Sharib",
editor="Ali, Sharib
and Hogg, David C.
and Peckham, Michelle",
title="Intraoperative Segmentation Through Deep Learning and Mask Post-processing in Laparoscopic Liver Surgery",
booktitle="Medical Image Understanding and Analysis",
year="2026",
publisher="Springer Nature Switzerland",
address="Cham",
pages="204--218",
abstract="Laparoscopic liver surgery is a popular surgical approach due to its capabilities of minimising trauma, complications, and recovery times. The use of a laparoscope allows for developments in the field of machine-assisted surgery due to the availability of intraoperative imagery. Accurate landmark detection of the liver using laparoscopic footage is a dependency to many developments, such as 3D-2D registration. In this paper, we present experimental results measuring the suitability of popular segmentation models, and their compatibility with different loss functions when handling intraoperative images; we also present a pipeline in training models for this segmentation task, including a novel step of applying post-processing techniques to maximise accuracy. Our results are evaluated using precision, Dice similarity coefficient, and a symmetric distance metric. Our results show that through the use of our proposed pipeline, models retain their ability to generalise, and can lead to noticeably improved accuracy both quantitatively and qualitatively. We demonstrate the feasibility of utilising post-processing to improve predictions. Finally, possible future directions in this field following from our results are discussed. The code from this research has been made available and can be accessed here: https://github.com/ARMADILLO-VISION/SLiPPA.",
isbn="978-3-031-98694-9"
}
```
