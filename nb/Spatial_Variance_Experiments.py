# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Spatial variance

# ## Imports

# +
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform
import seaborn as sn
import pandas as pd
import cmdstanpy

from crowdgeoloc.one_d import FunctionNormalAnnotatorPopulation, SimpleNormalAnnotatorPopulation
from crowdgeoloc.experiment import ActiveAnnotationContest, mean_location_norm_error, mean_sigma_norm_error
from crowdgeoloc.fixed import OneShotDirect,OneShotBayesian, OneShotConservative, OneShotConservative2, OneShotIterative, \
                                OneShotMean, KShot
from crowdgeoloc.spatial import OneShotSpatialBayesian, KShotSpatial
# -

# ## Parameters

# +
n_points = 10000 #number of points
n_annotators = 10 #number of annotators
redundancy = 5 #redundancy value

point_distribution = uniform() #points distribution
# -

# ## Visualization of variance profiles of the population

# +
annotator_population = FunctionNormalAnnotatorPopulation()
points = point_distribution.rvs(n_points)
ann_set = annotator_population.sample(n_annotators)

allx = np.arange(1000) / 1000.
for i in range(n_annotators):
    plt.plot(allx, ann_set.annotators[i].sigma(allx), 'b')
    plt.title(f"Annotator{i}'s variance profile")
    plt.xlabel("Location")
    plt.ylabel("Value")
    plt.show()
# -

# ## Mean error comparison: Performance of the different methods

models = {"mean": OneShotMean(),
          "direct":OneShotDirect(), 
          "iterative":OneShotIterative(), 
          "conservative":OneShotConservative(),
          "conservative2":OneShotConservative2(),
          #"FixedBayesian": OneShotBayesian(),
          "KShot_1.1":KShot(greediness=1.1),
          "KShot_2":KShot(greediness=2.),
          "SpatialOneShot": OneShotSpatialBayesian(),
          "SpatialKShot_1.1": KShotSpatial(greediness=1.1),
          "SpatialKShot_2": KShotSpatial(greediness=2.)
         }
metrics = {"mean_location_norm_error":mean_location_norm_error}


def compare(models, exp, metrics, repeats):
    results = []
    for model_name, m in models.items():
        print(model_name)
        result = {"model": model_name}
        for i in range(repeats):
            exp.reset()
            this_run = m.run(exp)
            result["iteration"] = i
            for metric_name, metric in metrics.items():
                result["metric"] = metric_name
                result["value"] = metric(exp, this_run)
                result["sigmas"] = this_run['sigmas']
                results.append(result.copy())
    return results


exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)
iterations = 20 #number of repetitions to ensure statistical significance 
results2 = compare(models, exp, metrics, iterations)
df2=pd.DataFrame(results2)

ax = sn.boxplot(data=df2[df2["metric"]=="mean_location_norm_error"], x="model", y="value")
plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
plt.show()

# ## Variance profiles' learning comparison

# +
sigmas_dict = {}
for model in models:
        sigmas_dict[model] = []

for ann in range(n_annotators):

    for model in models:
        if model in ['SpatialOneShot', 'SpatialKShot_2']:
            mean_sigma_functions = 0
            for i in range(iterations):
                mean_sigma_functions += df2[df2['model']==model].sigmas.values[i][ann]
            sigma_ann = mean_sigma_functions/iterations
            sigmas_dict[model].append(sigma_ann)
        
            
        else:
            sigma_ann = df2[df2['model']==model].sigmas.values.mean()[ann]
        
            sigmas_dict[model].append(sigma_ann)

# +
allx = np.arange(1000) / 1000.
sigmas_plot_df = pd.DataFrame(sigmas_dict)

for ann in range(n_annotators):
    plt.plot(allx, ann_set.annotators[ann].sigma(allx), 'b', label='True')
    plt.plot(allx, sigmas_plot_df.SpatialOneShot[ann], label = 'SpatialOneShot')
    plt.plot(allx, sigmas_plot_df.SpatialKShot_2[ann], label = 'SpatialKShot')
    plt.legend(bbox_to_anchor=(1.1,1.05))
    plt.show()
# -

plt.plot(allx, ann_set.annotators[7].sigma(allx), 'black', label='True', alpha=1)
plt.plot(allx, sigmas_plot_df.SpatialOneShot[7], label = 'SpatialOneShot', color ='blue', alpha = 0.8)
plt.plot(allx, sigmas_plot_df.SpatialKShot_2[7], label = 'SpatialKShot_2', color='red', alpha=0.7)
plt.plot(allx, [sigmas_plot_df.conservative2[7]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':', color='green')
plt.legend(bbox_to_anchor=(0.28,0.68), fontsize=12)
plt.ylim(0,0.1)
# plt.legend(loc='upper center', bbox_to_anchor=(0.5,-0.15), fancybox=True, shadow=True, ncol=4)
# plt.xlabel("Position")
# plt.ylabel("Sigma value")
plt.rc('xtick', labelsize=13)  
plt.rc('ytick', labelsize=13)
plt.gcf().set_size_inches(5,5)
plt.show()


