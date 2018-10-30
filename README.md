# Welcome to the machine learning support package
The purpose of this project is to provide a convenient set of tools to speed up development of machine learning algorithms.

## Setup
This project runs on python 3.6 using Keras with Tensorflow backend. The guide below is created for windows users, Linux/Mac users might have different commands.

1. Install Anaconda: https://www.anaconda.com/download/
2. Open the conda prompt.
3. From the root of this folder write "conda env create -f environment.yml" to install all the required packages.
    For more information regarding conda package managment please see: https://conda.io/docs/user-guide/tasks/manage-environments.html
4. Write "conda activate mlsp" to activate the newly created environment.
    For Linux/Mac use "source activate mlsp"


### Optional
The environment automatically installs the tensorflow-gpu package. This allows for Keras to automatically utilize any compatible GPU on the system. 
Please follow the following guidelines from TensorFlow for how to install CUDA drivers for compatible nVIDIA cards: 
https://www.tensorflow.org/install/gpu

#### NOTE: 
- CUDA Toolkit 10 does NOT work with tensorflow. Please follow the guidelines strictly to avoid issues.
- Your "normal" graphics drivers may become outdated by installing CUDA. Reinstalling the latest graphics drivers after CUDA install is recommended and should not affect tensorflow.

## Notes on machine learning
Here are some things to keep in mind when working with machine learning in python.
### NumPy
Machine learning algorithms use linear algebra computations in order to parallelize the computations. and NumPy introduce new N-dimensional array objects (implemented in C code in the backend) with extended linera algerbra functionality.
Whenever you'r working with arrays or matricies, use numpy 

### For-loops
Just don't if it can be avoided. For-loops in python are very slow and if used in code that run frequently it may grind your code to a standstill. For-loops are still perfectly fine to use for infrequent code pieces (such as iterating over different hyperparameters). 

