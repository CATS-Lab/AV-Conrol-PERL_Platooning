# Online Adaptive Platoon Control for Connected and Automated Vehicles via Physical Enhanced Residual Learning

## Introduction

This repo provides the source code and data for the following paper:

P. Zhang, H. Zhou, H. Huang, H. Shi, K. Long and X. Li, "Online Adaptive Platoon Control for Connected and Automated Vehicles via Physical Enhanced Residual Learning".


This paper introduces a physically enhanced residual learning (PERL) framework for connected and automated vehicle (CAV) platoon control, addressing the dynamics and unpredictability inherent to platoon systems. The framework combines a physical model of vehicle platoons with data-driven online learning methods, enhancing centralized platoon control and emphasizing multi-objective collaborative optimization. The residual controller, based on neural network (NN) learning, enriches the prior knowledge of the physical model and corrects residuals caused by vehicle dynamics. 

## Usage

### Code

The project is developed by Python 3. Please ensure you have a Python 3 environment set up.

The code related to our data processing and algorithm are:

- **trajectory_generation.py** - Code used to generate reference trajectories through IDM model or the OpenACC dataset.
- **PERL_simulation.py** - The main function calls the PERL algorithm.

To use this repo, run the Python script `PERL_simulation.py`. As you proceed through each Python script, always verify the paths for both the input and output files. This ensures that everything runs smoothly.

### Data

Folder 'data' contains some sample trajectories with a brief readme.

## Developers

Developer - Peng Zhang, Hang Zhou (pzhang257@wisc.edu, hzhou364@wisc.edu).

If you have any questions, please feel free to contact CATS Lab in UW-Madison. We're here to help!
