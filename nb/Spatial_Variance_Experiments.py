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
#     display_name: Python 3 (ipykernel)
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

# +
# cmdstanpy.install_cmdstan()
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

# +
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(10,5))
# fig.suptitle('Sharing x per column, y per row')
ax1.plot(allx, ann_set.annotators[0].sigma(allx), label='True', alpha = 0.5, color='black')
ax1.plot(allx, sigmas_plot_df.SpatialOneShot[0], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax1.plot(allx, [sigmas_plot_df.direct[0]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax1.plot(allx, [sigmas_plot_df.iterative[0]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax1.plot(allx, [sigmas_plot_df.conservative[0]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax1.plot(allx, [sigmas_plot_df.conservative2[0]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax1.plot(allx, [sigmas_plot_df.KShot_2[0]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax1.plot(allx, sigmas_plot_df.SpatialKShot_2[0], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax2.plot(allx, ann_set.annotators[1].sigma(allx), label='True', alpha = 0.5, color='black')
ax2.plot(allx, sigmas_plot_df.SpatialOneShot[1], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax2.plot(allx, [sigmas_plot_df.direct[1]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax2.plot(allx, [sigmas_plot_df.iterative[1]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax2.plot(allx, [sigmas_plot_df.conservative[1]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax2.plot(allx, [sigmas_plot_df.conservative2[1]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax2.plot(allx, [sigmas_plot_df.KShot_2[1]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax2.plot(allx, sigmas_plot_df.SpatialKShot_2[1], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax3.plot(allx, ann_set.annotators[2].sigma(allx), label='True', alpha = 0.5, color='black')
ax3.plot(allx, sigmas_plot_df.SpatialOneShot[2], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax3.plot(allx, [sigmas_plot_df.direct[2]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax3.plot(allx, [sigmas_plot_df.iterative[2]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax3.plot(allx, [sigmas_plot_df.conservative[2]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax3.plot(allx, [sigmas_plot_df.conservative2[2]]*len(allx), label = 'Conservative2', alpha = 0.9, ls=':')
ax3.plot(allx, [sigmas_plot_df.KShot_2[2]]*len(allx), label = 'KShot_2', alpha = 0.5, ls=':')
ax3.plot(allx, sigmas_plot_df.SpatialKShot_2[2], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax4.plot(allx, ann_set.annotators[3].sigma(allx), label='True', alpha = 0.5, color='black')
ax4.plot(allx, sigmas_plot_df.SpatialOneShot[3], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax4.plot(allx, [sigmas_plot_df.direct[3]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax4.plot(allx, [sigmas_plot_df.iterative[3]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax4.plot(allx, [sigmas_plot_df.conservative[3]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax4.plot(allx, [sigmas_plot_df.conservative2[3]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax4.plot(allx, [sigmas_plot_df.KShot_2[3]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax4.plot(allx, sigmas_plot_df.SpatialKShot_2[3], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax5.plot(allx, ann_set.annotators[4].sigma(allx), label='True', alpha = 0.5, color='black')
ax5.plot(allx, sigmas_plot_df.SpatialOneShot[4], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax5.plot(allx, [sigmas_plot_df.direct[4]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax5.plot(allx, [sigmas_plot_df.iterative[4]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax5.plot(allx, [sigmas_plot_df.conservative[4]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax5.plot(allx, [sigmas_plot_df.conservative2[4]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax5.plot(allx, [sigmas_plot_df.KShot_2[4]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax5.plot(allx, sigmas_plot_df.SpatialKShot_2[4], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax6.plot(allx, ann_set.annotators[5].sigma(allx), label='True', alpha = 0.5, color='black')
ax6.plot(allx, sigmas_plot_df.SpatialOneShot[5], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax6.plot(allx, [sigmas_plot_df.direct[5]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax6.plot(allx, [sigmas_plot_df.iterative[5]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax6.plot(allx, [sigmas_plot_df.conservative[5]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax6.plot(allx, [sigmas_plot_df.conservative2[5]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax6.plot(allx, [sigmas_plot_df.KShot_2[5]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax6.plot(allx, sigmas_plot_df.SpatialKShot_2[5], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

for ax in fig.get_axes():
    ax.label_outer()
    ax.set_ylim([0,0.16])
    
plt.legend(loc='upper center', bbox_to_anchor=(-0.8,-0.2), fancybox=True, shadow=True, ncol=6, prop={'size': 10.5})


# +
fig, ((ax7, ax8), (ax9, ax10)) = plt.subplots(2, 2, figsize=(10,5))
# fig.suptitle('Sharing x per column, y per row')
ax7.plot(allx, ann_set.annotators[6].sigma(allx), label='True', alpha = 0.5, color = 'black')
ax7.plot(allx, sigmas_plot_df.SpatialOneShot[6], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax7.plot(allx, [sigmas_plot_df.direct[6]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax7.plot(allx, [sigmas_plot_df.iterative[6]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax7.plot(allx, [sigmas_plot_df.conservative[6]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax7.plot(allx, [sigmas_plot_df.conservative2[6]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax7.plot(allx, [sigmas_plot_df.KShot_2[6]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax7.plot(allx, sigmas_plot_df.SpatialKShot_2[6], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax8.plot(allx, ann_set.annotators[7].sigma(allx), 'black', label='True', alpha = 0.5)
ax8.plot(allx, sigmas_plot_df.SpatialOneShot[7], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax8.plot(allx, [sigmas_plot_df.direct[7]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax8.plot(allx, [sigmas_plot_df.iterative[7]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax8.plot(allx, [sigmas_plot_df.conservative[7]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax8.plot(allx, [sigmas_plot_df.conservative2[7]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax8.plot(allx, [sigmas_plot_df.KShot_2[7]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax8.plot(allx, sigmas_plot_df.SpatialKShot_2[7], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax9.plot(allx, ann_set.annotators[8].sigma(allx), 'black', label='True', alpha = 0.5)
ax9.plot(allx, sigmas_plot_df.SpatialOneShot[8], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax9.plot(allx, [sigmas_plot_df.direct[8]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax9.plot(allx, [sigmas_plot_df.iterative[8]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax9.plot(allx, [sigmas_plot_df.conservative[8]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax9.plot(allx, [sigmas_plot_df.conservative2[8]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax9.plot(allx, [sigmas_plot_df.KShot_2[8]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax9.plot(allx, sigmas_plot_df.SpatialKShot_2[8], label = 'SpatialKShot', alpha = 0.7, color = 'blue')

ax10.plot(allx, ann_set.annotators[9].sigma(allx), 'black', label='True', alpha = 0.5)
ax10.plot(allx, sigmas_plot_df.SpatialOneShot[9], label = 'SpatialOneShot', alpha = 0.8, color = 'red')
ax10.plot(allx, [sigmas_plot_df.direct[9]]*len(allx), label = 'Direct', alpha = 0.7, ls=':')
# ax10.plot(allx, [sigmas_plot_df.iterative[9]]*len(allx), label = 'Iterative', alpha = 0.7, ls=':')
ax10.plot(allx, [sigmas_plot_df.conservative[9]]*len(allx), label = 'Conservative', alpha = 0.7, ls=':')
# ax10.plot(allx, [sigmas_plot_df.conservative2[9]]*len(allx), label = 'Conservative2', alpha = 0.7, ls=':')
ax10.plot(allx, [sigmas_plot_df.KShot_2[9]]*len(allx), label = 'KShot_2', alpha = 0.7, ls=':')
ax10.plot(allx, sigmas_plot_df.SpatialKShot_2[9], label = 'SpatialKShot', alpha = 0.7, color = 'blue')


for ax in fig.get_axes():
    ax.label_outer()
    ax.set_ylim([0,0.16])
    
plt.legend(loc='upper center', bbox_to_anchor=(-0.2,-0.2), fancybox=True, shadow=True, ncol=6, prop={'size': 10.5})  
# -
# ## Greediness comparison


# +
n_points = 1000 #number of points
n_annotators = 10  #number of annotators
redundancy = 5  #redundancy value

point_distribution = uniform()  #points distribution
## Visualization of variance profiles of the population
annotator_population = FunctionNormalAnnotatorPopulation()
points = point_distribution.rvs(n_points)
ann_set = annotator_population.sample(n_annotators)
# -

exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)

models = {#"mean": OneShotMean(),
#           "direct":OneShotDirect(), 
#           #"iterative":OneShotIterative(), 
#           "conservative2":OneShotConservative2(),
#           #"conservative":OneShotConservative(),
#           "10shot":KShot(),
#           "SpatialOneShot":OneShotSpatialBayesian(),
          "G1.0": KShotSpatial(greediness=1.0),
          "G1.1": KShotSpatial(greediness=1.1),
          "G1.2": KShotSpatial(greediness=1.2),
          "G1.3": KShotSpatial(greediness=1.3),
          "G1.4": KShotSpatial(greediness=1.4),
          "G1.5": KShotSpatial(greediness=1.5),
          "G1.6": KShotSpatial(greediness=1.6),
          "G1.7": KShotSpatial(greediness=1.7),
          "G1.8": KShotSpatial(greediness=1.8),
          "G1.9": KShotSpatial(greediness=1.9),
          "G2.0": KShotSpatial(greediness=2.0)
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
                results.append(result.copy())
            # print('REPEAT NUM:', i)
    return results


results3 = compare(models, exp, metrics, 20)
df3 = pd.DataFrame(results3)

ax = sn.boxplot(data=df3[df3["metric"]=="mean_location_norm_error"], x="model", y="value")
plt.xlabel("greediness")
plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
plt.show()


