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

# +
# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

from crowdgeoloc.one_d import sample_tokyo_latitudes, SimpleNormalAnnotatorPopulation
from crowdgeoloc.experiment import ActiveAnnotationContest, mean_location_norm_error, mean_sigma_norm_error, \
 load_experiment_setup, save_experiment_setup
from crowdgeoloc.fixed import OneShotDirect, OneShotConservative, OneShotConservative2, OneShotIterative, \
                                OneShotMean, KShot

import pandas as pd
import random
import re
import time
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, uniform
import seaborn as sn


# -

# ## CONSTANT VARIANCE PROFILE

def compare(methods, exp, metrics, repeats):
    '''
    function to compare methods by running the experiments the number of repeats given and evaluate them using metrics provided
    '''
    results = []
    duration_list=[]
    for model_name, m in methods.items():
        #print(model_name)
        result = {"model": model_name}
        start_time = time.time()
        for i in range(repeats):
            exp.reset()
            this_run = m.run(exp)
            result["iteration"] = i
            for metric_name, metric in metrics.items():
                result["metric"] = metric_name
                result["value"] = metric(exp, this_run)
                results.append(result.copy())
        end_time=time.time()
        duration=end_time-start_time
        duration_list.append(duration)
    return results,duration_list


def run_and_plot(list_setup_file, methods):
    '''
    takes a list of setup files as arguments and plot the metrics bar chart for each of the setups, return the list of durations of the different methods 
    '''
    
    list_df=[]
    list_duration=[]
    for element in list_setup_file:
        print("Computing", element)        
        name_list=list(methods.keys())
        metrics = {"mean_location_norm_error":mean_location_norm_error,
          "mean_sigma_norm_error":mean_sigma_norm_error}
        
        # Reads data from pickle file
        params=load_experiment_setup(element)
        random.seed(params[7])
        n_points, n_annotators, redundancy=(params[0],params[1],params[2])
        points, sigmas = (params[5], params[6])
        
        annotator_population = SimpleNormalAnnotatorPopulation(uniform(scale=0.1))
        
        ann_set = annotator_population.sample(n_annotators) 
        for k,elem in enumerate(ann_set.annotators):
            elem._sigma=sigmas[k]
       
        
        exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)
        results, durations = compare(methods, exp, metrics, 100)
      
        list_duration.append(durations)
        df=pd.DataFrame(results)
        list_df.append(df)
        #print(results)
        
    # Create metric box plot
    fig, axs = plt.subplots(1,len(list_setup_file), figsize=(10, 5),sharey=True)
    for i, elem in enumerate(list_df):
        sn.boxplot(data=elem[elem["metric"]=="mean_location_norm_error"], x="model", y="value", ax=axs[i])
        integers = re.findall(r'\d+',list_setup_file[i])
        int_tuple = tuple(int(k) for k in integers)
        axs[i].set_title(str(int_tuple))
        axs[i].set(xlabel="Model", ylabel="Mean location norm error") 
    fig.autofmt_xdate(rotation=45)

# +
# Create setup files for different number of points
number_of_points = [1000, 10000, 100000]
number_of_annotators = 50
redundancy = 3
points_distribution = 'uniform'
sigma_distribution = 'uniform'

for n_points in number_of_points:
    save_experiment_setup([n_points, number_of_annotators, redundancy, points_distribution, sigma_distribution])

# +
# Plots metric comparison (Fig.1)
# Chooses the methods
methods = {"mean": OneShotMean(),
          "direct":OneShotDirect(), 
          "iterative":OneShotIterative(),
          "conserv":OneShotConservative(),
          "conserv2":OneShotConservative2(),
          # "10shot-1.1":KShot(greediness=1.1),
          "10shot-1.4":KShot(greediness=1.4)
         }

# Compare the methods within the setup files
run_and_plot(['np_1000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl','np_10000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl','np_100000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl'], methods)


# -

def necessary_budget_continuous(setup_file, methods):
    '''
    takes a str: setup file as argument and compare the methods based on this setup to find how much redundany is needed 
    to achieve a same value of error for each model, then plot the results
    '''
    params=load_experiment_setup(setup_file) #open setup file
    #np.random.seed(params[7])
    
    name_list=list(methods.keys())
    metrics = {"mean_location_norm_error":mean_location_norm_error}
    #"mean_sigma_norm_error":mean_sigma_norm_error}
    
    n_points,n_annotators,redundancy=(params[0],params[1],params[2]) #choice of general parameters
    sig_distr=params[3] #choice of the sigma distrib
    if sig_distr=='uniform':
        annotator_population = SimpleNormalAnnotatorPopulation(uniform(scale=0.1))
    if sig_distr=='beta':
        annotator_population = SimpleNormalAnnotatorPopulation()
    
    points=params[5]
    sigmas = params[6]
    ann_set = annotator_population.sample(n_annotators)
    for k,elem in enumerate(ann_set.annotators): #we put the setup sigmas
        elem._sigma=sigmas[k]
    

    model_curves=[]
    for model_name, m in methods.items():
        print(f"Computing {model_name} method")
        error_list=[]
        for redundancy in range(2,40):
            results=[]
            exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)
            repeats=60
            result = {"model": model_name}
            for i in range(repeats):
                exp.reset()
                this_run = m.run(exp)
                result["iteration"] = i
                for metric_name, metric in metrics.items():
                    result["metric"] = metric_name
                    result["value"] = metric(exp, this_run)
                    results.append(result.copy())
                
        
            metrics_value=np.mean(np.array([results[i]['value'] for i in range(0,len(results),2)]))
            error_list.append(metrics_value)
     
        model_curves.append([[k for k in range(2,40)],np.array([k for k in error_list])])
        
            
    fig=plt.figure()
    ax = fig.add_subplot(111)
    for k,elem in enumerate(model_curves):
        if name_list[k]=="10shot-1.1" or name_list[k]=="10shot-1.4" :
            ax.plot(elem[1],elem[0],label=name_list[k])
        else:
            ax.plot(elem[1],elem[0],label=name_list[k],linestyle="dashed")
        
   
    ax.set_xlabel("error values") 
    
    
    #inversed axis
    ax.spines['left'].set_visible(False)
    ax.yaxis.set_ticks_position('right')
    ax.yaxis.set_label_position("right")
    ax.set_ylabel("redundancy") 

    plt.legend()
    plt.xscale('log')
    plt.show()

# +
# Plots metric error depending on the redundancy value (Fig. 2)
# Chooses the methods
methods = {"mean": OneShotMean(),
          "direct":OneShotDirect(), 
          "iterative":OneShotIterative(),
          "conservative":OneShotConservative(),
          "conservative2":OneShotConservative2(),
          "10shot-1.1":KShot(greediness=1.1),
          "10shot-1.4":KShot(greediness=1.4)
         }

# Compares the methods within the setup file
necessary_budget_continuous('np_1000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl', methods)
# -





# +
# To obtain plain predictions of a given experiment
# -

def tok(nb_points):
    '''function to create the tokyo distribution array'''
    tab=sample_tokyo_latitudes(nb_points)
    tab=(tab-np.min(tab))/(np.max(tab)-np.min(tab))
    return(tab)


def runexp(setup_file, methods):

    '''
    takes a str: setup file name as argument and compare the methods based on this setup
    '''

    params=load_experiment_setup(setup_file) #open setup file
    np.random.seed(params[7])

    metrics = {"mean_location_norm_error":mean_location_norm_error,
          "mean_sigma_norm_error":mean_sigma_norm_error}

    n_points,n_annotators,redundancy=(params[0],params[1],params[2]) #choice of general parameters
    sig_distr=params[3] #choice of the sigma distrib
    if sig_distr=='uniform':
        annotator_population = SimpleNormalAnnotatorPopulation(uniform(scale=0.1))
    if sig_distr=='beta':
        annotator_population = SimpleNormalAnnotatorPopulation()

    point_distr=params[4] #choice of point distrib

    if point_distr=='uniform':
        point_distribution = uniform()
        points = point_distribution.rvs(n_points)
    else:
        points =  tok(n_points)

    list_tru_sig=[]
    ann_set = annotator_population.sample(n_annotators)
    list_true_sig=[ann_set.annotators[k]._sigma for k in range(len(ann_set.annotators))]

    exp = ActiveAnnotationContest(points, ann_set, max_total_annotations=n_points*redundancy)
    list_sigm_pred=[]
    list_point_pred=[]
    for model_name, m in methods.items():
        print(model_name)
        result = {"model": model_name}
        exp.reset()
        this_run = m.run(exp)
        locations=this_run["locations"]
        sigmas=this_run["sigmas"]
        list_sigm_pred.append(sigmas)
        list_point_pred.append(locations)
    return points,list_true_sig,list_point_pred,list_sigm_pred


# +
methods = {#"mean": OneShotMean(),
          "direct":OneShotDirect(), 
          #"iterative":OneShotIterative(), 
          "conservative2":OneShotConservative2(),
          #"conservative":OneShotConservative(),
          "10shot":KShot(greediness=1.2)
         }

true_points, true_errors, pred_points, pred_errors = runexp('np_10000_na_50_rd_3_sd_uniform_pd_uniform_setup.pkl', methods)
# -

print(f"Predicted points: {pred_points}")
print("--------------------------------")
print(f"Predicted errors: {pred_errors}")
