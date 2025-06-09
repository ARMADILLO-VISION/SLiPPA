# SLiPPA Code Repository

**S**egmentation of the **Li**ver with **P**ost-**P**rocessing **A**pplied (SLiPPA) is the codebase used in the MIUA 2025 conference paper: [Intraoperative Liver Segmentation through Deep Learning and Mask Post-processing in Laparoscopic Liver Surgery](https://google.co.uk).

## Installation
This repository contains an Anaconda environment (`environment.yml`) containing the required dependencies to use this code. Please ensure Anaconda is installed before proceeding. To create an environment, whilst in the repository directory, run the following command:
```bash
conda env create --name slippa --file=environment.yml
```
This will create an environment called `slippa` on your system with the required dependencies.

## Datasets
This codebase has provided compatibility with two datasets: L3D (available from [this repository]()) and P2ILF (not currently publicly available). A preparation script is provided to extract and organise the data in the way the codebase expects:
```bash
python prepare.py --path=[PATH TO DATASET ZIP] --dataset=L3D_train
```
The available dataset options are: `L3D_train`, `L3D_val`, `L3D_test`, `P2ILF_train`, and `P2ILF_test`. Please provide the path to the ZIP file that has been downloaded without any extractions or alterations applied.

The preparation script automatically separates the P2ILF training data into a training set and a validation set, where patients 1 and 2 make the validation set.

## Inference
Random inferences from a dataset can be made using the following command:
```bash
python predict.py --model=[unet, unetpp, unet3p, resunetpp, deeplabv3p] --weights=[PATH TO PTH CHECKPOINT] --dataset=[L3D_train, L3D_val, L3D_test, P2ILF_train, P2ILF_test]
```
Dataset options are the same as the above preparation script with the addition of `P2ILF_val` (patients 1 and 2 are not present in `P2ILF_train`). Options for the `model` argument are: `unet`, `unetpp`, `unet3p`, `resunetpp`, and `deeplabv3p`.

To make a single inference on a specific image, the following command is used:
```bash
python predict.py --model=[unet, unetpp, unet3p, resunetpp, deeplabv3p] --weights=[PATH TO PTH CHECKPOINT] --image=[PATH TO IMAGE]
```
## Training

## Citation
If this code is used in any work, please cite the following:
```bibtex
@inproceedings{Borgars2025,
    author      =   "James Borgars and Jibran Raja and Abhinav Ramakrishnan and
                    Abdul Karim Abbas and Ahmad Najmi Mohamad Shahir and Aodhan Gallagher and
                    Theodora Vraimakis and Sharib Ali",
    editor      =   "TBC",
    title       =   "Intraoperative Segmentation through Deep Learning and Mask Post-processing in
                    Laparoscopic Liver Surgery",
    booktitle   =   "Medical Imaging Understanding and Analysis",
    year        =   "2025",
    publisher   =   "Springer Nature Switzerland",
    address     =   "Cham",
    pages       =   "TBC",
    ibsn        =   "TBC",
    doi         =   "TBC",
    abstract    =   "TBC"
}
```
