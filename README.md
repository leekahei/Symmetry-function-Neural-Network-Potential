# Symmetry-function-Neural-Network-Potential
This is a neural network potential using symmetry function and another structure information (e.g. configuration type) as an input. 

I made this project after I studied a course about machine learning for molecular physics. 

When the prediction accross different sizes and configuration, the potential energy prediction is not very accurate. Therefore, the original idea of this project is simply adding the information of the configuration and size in the descriptor to improve the performance.

The first idea is using the pairwise distance, force, structure information as the descriptor, but the performance is not good. Therefore, I use the symmetry function instead of pairwise distance to predict energy per atom and reference a paper https://doi.org/10.1063/1.3553717. 

Scikit-learn and ase are used in this project.

```
conda create -n SNNP python=3.8
conda activate SNNP

pip install --upgrade ase

conda install -c conda-forge scikit-learn
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib

```
