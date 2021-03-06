# The Price of Local Fairness in Multistage Selection

This repository contains an implementation of multistage fair selection algorithm and all the experiments conducted in the paper: https://arxiv.org/abs/1906.06613

## Project Structure

* The implementation of the algorithm can be found in *optimizer/linear3.py*.
* Data preprocessing method is given in *dataLoad.py*
* The code for numerical simulation of 2-stage and 3-stage selection is implemented in *fair-simulation-2-stage.ipynb* and *fair-simulation-3-stage.ipynb* respectively.
* The code to generate figures can be found in *figures.ipynb*.

## Software requirements

* python (>=3)
* numpy 
* pandas 
* pulp
* matplotlib
* jupyter-notebook
