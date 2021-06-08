# Exploring Assembly of Globular Multi-Subunit Complexes Using Deterministic Steric-Free Vectorized Simulator #
*Author: Spencer Loggia*

## Installation ##

The easiest way to install the simulator is to clone this repo and then build an environment containing all dependencies using the provided `base_requirements.txt` file. In order to do this you will need to have an up to date version of the anaconda package manager (https://www.anaconda.com/products/individual#Downloads). 

-first clone this repository into the desired directory on your system. `git clone git@github.com:mjohn218/multi_assembly.git`
-navigate to the `steric_free_simulator` directory and run `conda create --name <env> --file base_requirements.txt` where `<env>` is the desired name of your new environment. *NOTE: This requirements file only includes dependencies available from conda or pip. For any application involving rosetta, for example estimating free energies from pdb structures, you will need to also install pyrosetta to your environment from http://www.pyrosetta.org/*
-now run `conda activate <env>` in order to use the new environment. 
-you may now use the included modules.

## Documentation ##
Detailed functionality and documentation can be found in the Jupyter Notebooks located in the `docs` directory. 
You can start the jupyter server by activating the conda environment and then running `jupyter notebook`. This should open a browser window
showing the current directory. You can then open the `docs` folder and then any of the notebooks therewithin.

## Results ##
Similar to the above, some initial results are contained in notebooks located in the `results` folder. 
