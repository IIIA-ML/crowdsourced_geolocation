# Crowdsourced Geolocation
Simulation for geolocalization using crowdsourcing.

This repository contains the code used in the research article:

R. Ballester, Y. Labeyrie, M. O. Mulayim, J. L. Fernandez-Marquez, and J. Cerquides, “Mathematical and Computational Models for Crowdsourced Geolocation,” in _Frontiers in Artificial Intelligence and Applications, Vol 375: Artificial Intelligence Research and Development_, I. Sanz, R. Ros, and J. Nin, Eds. IOS Press, 2023, pp. 301–310. [doi: 10.3233/FAIA230699](https://doi.org/10.3233/FAIA230699)

## Before every run
First, make sure you update the local repo with the remote changes. 
Moreover, please ensure you also have the `make` and `g++` packages installed. You can do this by running the following commands (for Debian-based systems such as Ubuntu):

```bash
sudo apt-get update
sudo apt-get install make g++
```

Then, run the initialization script that will create (if necessary) and activate the virtual environment, and will install missing dependencies.

```bash
git pull
source bin/init-local.sh
jupyter-notebook
```

Alternative to the Jupyter Notebook, you can start the JupyterLab: 

```bash
jupyter lab
```

## To run the experiments from the COGNITIVE SCIENCE RESEARCH article
Move to the CSR branch and follow the following instructions.

### Constant Variance Experiments
In order to perform experiments with constant variances and reproduce the figures presented in the article, one must run the notebook called `Constant_Variance_Experiments.ipynb` found inside the `nb` folder. 
Once the necessary libraries have been imported (cell 1) and the different functions have been executed (`compare()` in cell 2 and `run_and_plot()` in cell 3) the user can create an experiment by choosing the number of points, the distribution from which these points are sampled, the number of annotators, the distribution from which the annotators' errors (i.e. the sigmas) are sampled and the number of annotations per point (i.e. the redundancy value). These arguments are passed in the `save_experiment_setup()` function to create a pickle file with the experiment setup. These parameters are set in the notebook in order to reproduce Fig.1 and Fig.2 of the paper.
Next, the user can select the methods to run within the saved setup experiments. If the user passes a dictionary with the selected methods and the different experiment setup files to the `run_and_plot()` function it obtains two plots: one comparing the execution time of each method within each experiment and one comparing the performance of each method within each experiment.
These cells are already prepared to plot Fig. 1, 2, 8, 9, 3, and 10, in this particular order.

Next, in order to reproduce Fig. 4 and Fig. 11 of the article one needs to first execute the `necessary_budget_continuous()` function (cell 9). This function reads a pickle file with an experiment setup and a dictionary containing the selected methods. Cells 10-11 plot Fig. 4 and Fig. 11 respectively by passing the previously saved setup files.

Finally, the `greediness_impact()` function reads a setup file and an array of floats to plot the impact on the K-Shot method performance depending on its greediness value. Executing cells 12-13, Fig. 12 is obtained.

At the end of the notebook, one can find a few more cells in order to simply run a given experiment and obtain the plain predicted values for the points and the annotators' errors. In this way, it might be easier to perform other analyses or plot other graphs.

### Spatial Variance Experiments
In order to perform experiments with continuous variances and reproduce the figures presented in the article, one must run the notebook called `Spatial_Variance_Experiments.ipynb` found inside the `nb` folder. 

Analogously to the constant variance notebook, in the parameters section the user can change the number of points, number of annotators, and redundancy values to use in the experiments, as well as the distribution of the points (cell 2).

Later, in cell 3 the variance profiles generated for each annotator are shown.
The following cells (4-7) plot a comparison of all methods' performance, corresponding to Fig. 15. Depending on the experiment's parameters this execution can take several minutes.

The subsequent cells (8-11) plot the predicted error for each annotator and method and compare them with their true values, corresponding to Fig. 14 and Fig. 17 of the article.

Finally, analogously to the constant variance experiments, the last two cells of the notebook are used to plot the impact on the Spatial K-Sot method performance depending on its greediness value (Fig. 16).


## To run the experiments from the CCIA article
Move to the master branch and follow the following instructions.

### Constant Variance Experiments
In order to perform experiments with constant variances and reproduce the figures presented in the article, one must run the notebook called `Constant_Variance_Experiments.ipynb` found inside the `nb` folder. 
Once the necessary libraries have been imported (cell 1) and the different functions have been executed (`compare()` in cell 2 and `run_and_plot()` in cell 3) the user can create an experiment by choosing the number of points, the distribution from which these points are sampled, the number of annotators, the distribution from which the annotators' errors (i.e. the sigmas) are sampled and the number of annotations per point (i.e. the redundancy value). These arguments are passed in the `save_experiment_setup()` function to create a pickle file with the experiment setup. These parameters are set in the notebook in order to reproduce Fig.1 of the paper.
Next, the user can select the methods to run within the saved setup experiments. If the user passes a dictionary with the selected methods and the different experiment setup files to the `run_and_plot()` function it obtains a plot comparing the performance of each method within each experiment.
These cells are already prepared to plot Fig. 1.

Next, in order to reproduce Fig. 2 of the article one needs to first execute the `necessary_budget_continuous()` function (cell 6). This function reads a pickle file with an experiment setup and a dictionary containing the selected methods. Cell 7 plots Fig. 2 of the article by passing the previously saved setup files.

At the end of the notebook, one can find a few more cells in order to simply run a given experiment and obtain the plain predicted values for the points and the annotators' errors. In this way, it might be easier to perform other analyses or plot other graphs.


### Spatial Variance Experiments
In order to perform experiments with continuous variances and reproduce the figures presented in the article, one must run the notebook called `Spatial_Variance_Experiments.ipynb` found inside the `nb` folder. 

Analogously to the constant variance notebook, in the parameters section the user can change the number of points, number of annotators, and redundancy values to use in the experiments, as well as the distribution of the points (cell 2).

Later, in cell 3 the variance profiles generated for each annotator are shown.
The following cells (4-7) plot a comparison of all methods' performance, corresponding to Fig. 3 (left). Depending on the experiment's parameters this execution can take several minutes.

Finally, the subsequent cells (8-10) plot the predicted error for each annotator and method and compare them with their true values, corresponding to Fig. 3 (right) of the article.

## Citation

If you find our software useful for your research, kindly consider citing the associated article as follows:

```bibtex
@incollection{Ballester_et_al_2023,
author = {Ballester, Rocco and Yanis, Labeyrie and Mulayim, Mehmet Oguz and Fernandez-Marquez, Jose Luis and Cerquides, Jesus},
booktitle = {Frontiers in Artificial Intelligence and Applications},
doi = {10.3233/FAIA230699},
editor = {Sanz, Ismael and Ros, Raquel and Nin, Jordi},
keywords = {social media,disaster response,machine learning,geolocation,crowdsourcing},
pages = {301--310},
publisher = {IOS Press},
title = {Mathematical and Computational Models for Crowdsourced Geolocation},
url = {https://ebooks.iospress.nl/volumearticle/64978},
volume = {375},
year = {2023}
}
```


