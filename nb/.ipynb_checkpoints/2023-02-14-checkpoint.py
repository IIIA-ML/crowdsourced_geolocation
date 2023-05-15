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
from crowdgeoloc.active_Rocco import OneShotDirect, OneShotConservative, OneShotConservative2, OneShotIterative, \
    OneShotMean, KShot, OneShotBayesian, OneShotSpatialBayesian, KShotSpatial
# -

# ## KShot

# + tags=[]
####### pruebas implementando el KShotSpatial

# + tags=[]
n_points = 10000
n_annotators = 10
redundancy = 10

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

# + tags=[]
exp.reset()
test = KShotSpatial().run(exp)
# -

print(f'Number of iterations: {test[5]}')
print(f'Annotations per individual: {test[6]}')

allx = np.arange(1000) / 1000.
for i in range(n_annotators):
    print(np.mean(ann_set.annotators[i].sigma(allx)))
    plt.plot(allx, ann_set.annotators[i].sigma(allx), 'b')
    plt.plot(allx, test[3][i], 'r')
    plt.plot(allx, test[4][i], 'g')
    plt.show()

plt.plot(points, test[7]['locations'], 'bo')
plt.plot(np.linspace(0,1,1000), np.linspace(0,1,1000), 'r')
plt.show()

results = test[7]
mean_location_norm_error(exp, results)
