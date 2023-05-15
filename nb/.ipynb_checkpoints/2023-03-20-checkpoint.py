# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Implementation of KShot with spatial-dependent sigmas

# + tags=[]
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform
import seaborn as sn
import pandas as pd

from crowdgeoloc.one_d import FunctionNormalAnnotatorPopulation, SimpleNormalAnnotatorPopulation
from crowdgeoloc.experiment import ActiveAnnotationContest, mean_location_norm_error, mean_sigma_norm_error
# from crowdgeoloc.active_Rocco import OneShotDirect, OneShotConservative, OneShotConservative2, OneShotIterative, \
#     OneShotMean, KShot, OneShotBayesian, OneShotSpatialBayesian, KShotSpatial
from crowdgeoloc.spatial import OneShotSpatialBayesian, KShotSpatial
# -

# ## KShot

# + tags=[]
n_points = 10000
n_annotators = 10
redundancy = 5

# + tags=[]
annotator_population = FunctionNormalAnnotatorPopulation()
point_distribution = uniform()
points = point_distribution.rvs(n_points)
ann_set = annotator_population.sample(n_annotators)

exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)

# + tags=[]
allx = np.arange(1000) / 1000.
for i in range(n_annotators):
    print(np.mean(ann_set.annotators[i].sigma(allx)))
    plt.plot(allx, ann_set.annotators[i].sigma(allx), 'b')
    plt.show()
# -



# + tags=[]
exp.reset()
results = KShotSpatial().run(exp)

# + tags=[]
allx = np.arange(1000) / 1000.
for i in range(n_annotators):
    print(np.mean(ann_set.annotators[i].sigma(allx)))
    plt.plot(allx, ann_set.annotators[i].sigma(allx), 'b')
    plt.plot(allx, results['sigmas'][i], 'g')
    plt.show()
# -


plt.plot(points, results['locations'], 'bo')

from sklearn.metrics import mean_squared_error
mean_squared_error(points, results['locations'])

exp.annotations_per_individual

mean_location_norm_error(exp, results)



exp.annotations_per_individual

exp.annotations_per_individual



# ### OneShot

exp.reset()
results2 = OneShotSpatialBayesian().run(exp)

allx = np.arange(1000) / 1000.
for i in range(n_annotators):
    print(np.mean(ann_set.annotators[i].sigma(allx)))
    plt.plot(allx, ann_set.annotators[i].sigma(allx), 'b')
    plt.plot(allx, results2['sigmas'][i], 'r')
    plt.plot(allx, results['sigmas'][i], 'g')
    plt.show()

plt.plot(points, results2['locations'], 'bo')

from sklearn.metrics import mean_squared_error
mean_squared_error(points, results2['locations'])

exp.annotations_per_individual

mean_location_norm_error(exp, results2)









# # Comparison

models = {#"mean": OneShotMean(),
#           "direct":OneShotDirect(), 
#           #"iterative":OneShotIterative(), 
#           "conservative2":OneShotConservative2(),
#           #"conservative":OneShotConservative(),
#           "10shot":KShot(),
          "SpatialOneShot":OneShotSpatialBayesian(),
          "SpatialKShot": KShotSpatial()
         }
metrics = {"mean_location_norm_error":mean_location_norm_error}


# +

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
        print('REPEAT NUM:', i)
    return results
    


# +
n_points = 10000
n_annotators = 10
redundancy = 5
annotator_population = FunctionNormalAnnotatorPopulation()
point_distribution = uniform()
points = point_distribution.rvs(n_points)
ann_set = annotator_population.sample(n_annotators)

exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)

results = compare(models, exp, metrics, 10)
df = pd.DataFrame(results)





# -

sn.boxplot(data=df[df["metric"]=="mean_location_norm_error"], x="model", y="value")
plt.show()

plot_models = [
          "SpatialOneShot",
          "SpatialKShot"]
le = df[df["metric"]=="mean_location_norm_error"]
for n in plot_models:
    le1 = le[le["model"]==n]
    sn.kdeplot(le1["value"])
plt.legend(plot_models)




