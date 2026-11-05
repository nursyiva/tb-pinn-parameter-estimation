# TB Compartmental Model Using Fourier Feature-Based PINNs

## Overview

This repository provides the Python implementation of a Fourier Feature-Based Physics-Informed Neural Network (PINN) for inverse epidemiological parameter estimation and hidden-state reconstruction in a tuberculosis (TB) compartmental model incorporating reinfection and drug resistance.

The proposed framework integrates epidemiological observations with the governing ordinary differential equations (ODEs) to estimate unknown model parameters and reconstruct unobserved epidemiological compartments.

## Research Objectives

- Estimate unknown epidemiological parameters in a TB compartmental model.
- Reconstruct hidden epidemiological states using physics-informed learning.
- Incorporate Fourier feature mapping to improve the representation of temporal dynamics.
- Evaluate the reconstruction performance of PINNs against numerical ODE solutions.

## Model Description

The model consists of eight epidemiological compartments:

- Susceptible population (\(S\))
- Drug-sensitive exposed population (\(E_s\))
- Drug-sensitive infectious population (\(I_s\))
- Drug-sensitive treated population (\(T_s\))
- Recovered population (\(R\))
- Drug-resistant exposed population (\(E_r\))
- Drug-resistant infectious population (\(I_r\))
- Drug-resistant treated population (\(T_r\))

Weekly observations are available for four compartments: \(I_s\), \(I_r\), \(T_s\), and \(T_r\). The remaining compartments are reconstructed as hidden states.

## Methodology

The implementation includes:

1. TB compartmental ODE model.
2. Fourier feature mapping.
3. Physics-Informed Neural Network architecture.
4. Data and physics-informed loss functions.
5. Inverse epidemiological parameter estimation.
6. Hidden-state reconstruction.
7. Quantitative performance evaluation using RMSE, MAE, and MAPE.



## Requirements

The implementation uses Python and TensorFlow. The required packages are listed in `requirements.txt`.

## How to Run

1. Clone this repository.
2. Install the required Python packages.
3. Prepare the epidemiological dataset according to the data instructions.
4. Open the notebook in Jupyter Notebook or Google Colab.
5. Run the notebook sequentially.

## Reproducibility

The notebook documents the model configuration, parameter estimation procedure, training process, and evaluation metrics. Users should ensure that the required input dataset is available before running the experiments.

## Citation

If you use this code in your research, please cite the associated research article once its publication details become available.

## Author

Research implementation accompanying a study on Fourier Feature-Based Physics-Informed Neural Networks for tuberculosis parameter estimation and hidden-state reconstruction.
