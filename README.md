# Crowdsourced Geolocation
Simulation for geolocalization using crowdsourcing.

The code for the research article: 

R. Ballester, Y. Labeyrie, M. O. Mulayim, J. L. Fernandez-Marquez, and J. Cerquides, “Mathematical and Computational Models for Crowdsourced Geolocation,” in _Frontiers in Artificial Intelligence and Applications, Vol 375: Artificial Intelligence Research and Development_, I. Sanz, R. Ros, and J. Nin, Eds. IOS Press, 2023, pp. 301–310. [doi: 10.3233/FAIA230699](https://doi.org/10.3233/FAIA230699)


## Before every run

First, make sure you update the local repo with the remote changes. 
Then, run the initialization script that will create (if necessary) and activate the virtual environment 
and install missing dependencies.

```bash
git pull
source bin/init-local.sh
jupyter-notebook
```

Alternative to the Jupyter Notebook, you can start the JupyterLab: 

```bash
jupyter lab
```

## To run the experiments
Inside the `nb` folder there are two `jupyter notebook` files, one regarding the constant variance experiments and the other regarding the spatial variance experiments.

### Constant variance
In order to create an experiment setup for the constant variance experiments run the `save_experiment_setup` function as follows:
```
save_experiment_setup([100000,50,3,'uniform','uniform'])
```
This will create a pickle file containing 100000 points, 50 annotators, a redundancy of 3, specifying a uniform distribution for the points and a uniform distribution for the sigmas (variances).
In order to reproduce Figure 1 of the paper, you need to use the `run_and_plot` function as follows:

```
run_and_plot(['np_1000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl','np_10000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl',
'np_100000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl'])
```
This will create the comparison plot between 1000, 10000, 100000 points setups. Make sure before doing so to have created the different setup files using the `save_experiment_setup` function.

In order to reproduce Figure 2 of the paper, run the `necessary_budget_continuous` function in the following way:
```
necessary_budget_continuous(['np_1000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl'])
```
This will plot the graph using the experimental setup contained in the above setup file.

To run an experiment and obtain the predicted point positions and sigmas use the `run_exp_from_setup` function in the following way:
```
setup1=load_experiment_setup('np_10000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl')
test=run_exp_from_setup(setup1,{"10shot":KShot(greediness=0.1)})
```
This function takes an experiment setup array and a model as argument and returns a tuple composed of: _true_positions, true_sigmas, predicted_positions, predicted_sigmas_.

To plot a grap showing the impact of "greediness" parameter in terms of model error, use the function `greediness_impact` in the following way:
```
greedyness_impact("np_10000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl",[k/10 for k in range(10,20)])
```
This function takes a setup file name and an array of greediness value as arguments and plots the corresponding graph.

### Spatial variance
In order to reproduce the experiments using non constant variances you need to execute all cells from top to bottom of the `Spatial_Variance_Experiments` notebook.
In the parameters section one can change the number of points, number of annotators and redundancy values to use in the experiments, as well as the distribution of the points. Next, the variance profiles generated for each annotator are shown. The following graph, under the Mean error comparison section, is the left-hand one from Figure 3 of the paper. Later, in section Variance profiles' learning comparison, one can find how well the different methods learn the variance profiles. Among other graphs, one can find the right-hand plot from Figure 3 of the paper. Finally, in the last section, a greediness analysis plot is shown.

## Citation

If you find our software useful for your research, kindly consider citing the associated article as follows:

```bibtex
@incollection{Ballester2023,
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


