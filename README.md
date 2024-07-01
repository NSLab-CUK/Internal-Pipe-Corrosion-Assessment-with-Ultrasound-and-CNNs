# Internal Pipe Corrosion Assessment with Ultrasound and CNNs

This study introduces a dual-mode methodology for quantifying pipe corrosion by employing ultrasound technology in conjunction with convolutional neural networks (CNN).

## Overview

This project focuses on assessing internal pipe corrosion using ultrasound and Convolutional Neural Networks (CNNs). It includes components for data handling, model development, preprocessing, and utility functions.

#### Components

-   **Data Handling**:

    -   `data.py`: Functions for loading datasets.

-   **Model Development**:

    -   `model/`: Directory containing various CNN architectures:
        -   `AlexNet.py`
        -   `DenseNet.py`
        -   `EfficientNet.py`
        -   `InceptionNet.py`
        -   `ResNet.py`
        -   `VGG.py`
        -   `__pycache__/`: Cached Python bytecode files.

-   **Preprocessing**:

    -   `preprocess.py`: Implementation code for data preprocessing.
    -   `preprocess_fn.py`: Functions required for preprocessing.

-   **Execution Scripts**:
    -   `run_experiment.py`: Main execution script.
    -   `training.py`: Contains classes and functions related to training.
    -   `utils.py`: Collection of utility functions.
-   **Output and Results**:
    -   `confusion_matrix.zip`, `output.hwp`, `plot.zip`, `vgg_loss_plot.zip`: Output files and archives related to results and plots.
    -   `draw_acc-lossplot.py`: Script for plotting accuracy and loss.

#### Usage

-   Run `run_experiment.py` to execute the main functionalities of the project.
-   The `model/` directory contains implementations of different CNN architectures.

This README provides a structured overview of the project, highlighting key components and their functionalities for assessing internal pipe corrosion using ultrasound and CNNs.
