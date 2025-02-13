## Motivation

The dataset was created as a standardized collection of biomedical images to benchmark machine learning models on lightweight 2D and 3D image classification tasks. It is designed to support research and education in biomedical image analysis, computer vision, and machine learning. This dataset is NOT intended for clinical use. 

MedMNIST v2 was created by researchers from Shanghai Jiao Tong University, Boston College, RWTH Aachen University, Fudan University, Shanghai General Hospital, and Harvard University. It was supported by the National Science Foundation of China (U20B200011, 61976137) and Grant YG2021ZD18 from Shanghai Jiao Tong University Medical Engineering Cross Research.

However, data used in this specific project are exploited according to expectations of the competition Tensor Reloaded: MedMNIST Multi-Task Challenge, organized by Tensor Reloaded, a research group at the Faculty of Computer Science, Iasi, Romania, part of Alexandru Ioan Cuza University (see README.md). Organizers shares sources:  https://medmnist.com/ and Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023 (which is a source of information about dataset).  
 
## Composition

The original dataset consists of standardized 2D and 3D biomedical images (e.g., X-rays, CT scans, ultrasound, electron microscopy images) with corresponding classification labels.

There is no reported missing data; all images are pre-processed and labeled.

The dataset does not contain legally protected or patient-identifiable data. All source datasets are either under a Creative Commons (CC) License or developed by the authors with permission for redistribution.

The organisers of the competition require to utlize chosen data: 
PathMNIST: Colon pathology images (9 classes), DermaMNIST: Dermatoscope images (7 classes), OCTMNIST: Retinal OCT images (4 classes), PneumoniaMNIST: Chest X-rays for pneumonia detection (2 classes), RetinaMNIST: Fundus camera images (ordinal regression, 5 classes), BreastMNIST: Breast ultrasound images (2 classes),BloodMNIST: Blood cell microscope images (8 classes), TissueMNIST: Kidney cortex microscope images (8 classes), OrganAMNIST: Abdominal CT images (11 classes), OrganCMNIST: Abdominal CT images (11 classes),OrganSMNIST: Abdominal CT images (11 classes) (see: README.md, details: https://medmnist.com/)

## Collection process

The images were obtained from various publicly available biomedical image datasets. Some were pre-existing datasets, while others were curated specifically for MedMNIST. Orginisers expect use datasets that are included MedMNIST2D, according to https://medmnist.com/ (excluding ChestMNIST).

Dataset used for the competition purpose follows the official train-validation-test splits from the source datasets.

We need to be aware, it includes images from various sources collected over multiple years before being compiled into MedMNIST v2.


## Preprocessing/cleaning/labelling

Only the preprocessed data is included in MedMNIST v2. That means that images were resized to 28×28 pixels (2D) (or 28×28×28 voxels (3D)). MedMNIST also offer a larger-size version (MedMNIST+). The organisers of the competition provide on Kaggle the lowest-size version, which was used in this project.

The original raw images are available in the source datasets.

## Uses

Potential Uses:
 - Benchmarking deep learning models for biomedical image classification.
 - Educational purposes, such as teaching medical imaging and machine learning concepts.
 - AutoML research, including hyperparameter tuning and model selection.

The small image resolution (28×28) may lead to a loss of fine-grained medical details, making it unsuitable for clinical applications.
Moreover, some datasets contain imbalanced classes, which could affect model generalization.

Not for clinical diagnosis—substantial downscaling of images limits diagnostic accuracy.

## Distribution

The dataset is publicly available at https://medmnist.com/.
The organiser of the competition provide data set on https://www.kaggle.com/competitions/tensor-reloaded-multi-task-med-mnist/ (used in this project).

The dataset follows Creative Commons (CC) Licenses, with some sub-datasets under CC-BY-NC (requiring non-commercial use). Authors obtained redistribution permissions from dataset creators.

## Maintenance

The dataset is maintained by the original research team, led by Bingbing Ni (Shanghai Jiao Tong University). Updates and support are provided through the MedMNIST GitHub repository.

## Sources
Jiancheng Yang, Rui Shi, Donglai Wei, Zequan Liu, Lin Zhao, Bilian Ke, Hanspeter Pfister, Bingbing Ni. Yang, Jiancheng, et al. "MedMNIST v2-A large-scale lightweight benchmark for 2D and 3D biomedical image classification." Scientific Data, 2023
