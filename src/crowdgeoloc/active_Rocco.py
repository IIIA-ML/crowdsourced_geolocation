import numpy as np
import cmdstanpy as cmd

from .one_d import random_assignment
from .experiment import ActiveAnnotationContest
from .cmdstan import resource_filename


class ActiveAnnotationMethod:
    def run(self, exp: ActiveAnnotationContest):
        return None #points


def point_average(t_A: np.ndarray, ann: np.ndarray):
    _ann = ann.copy()
    _t_A = t_A.copy()
    _ndx = np.argsort(_t_A)
    tasks, _pos, g_count = np.unique(_t_A[_ndx],
                                   return_index=True,
                                   return_counts=True)
    g_sum = np.add.reduceat(_ann[_ndx], _pos, axis=0)
    g_mean = g_sum / g_count
    return tasks, g_mean


def point_averages_and_sigmas(t, w, ann):
    # Compute means
    _ann = ann.copy()
    _t = t.copy()
    _ndx = np.argsort(_t)
    tasks, _pos, g_count = np.unique(_t[_ndx],
                                   return_index=True,
                                   return_counts=True)
    g_sum = np.add.reduceat(_ann[_ndx], _pos, axis=0)
    g_mean = g_sum / g_count

    # Compute square error of each annotation with respect to the mean of that point

    inv_tasks = np.ones(np.max(tasks)+1, dtype=int)*(-1)
    inv_tasks[tasks] = np.arange(len(tasks))
    sq_errors = (_ann - g_mean[inv_tasks[t]])**2

    # Compute the variances

    _ndxa = np.argsort(w)
    # print(_ndx)
    workers, _posa, g_counta = np.unique(w[_ndxa],
                                      return_index=True,
                                      return_counts=True)
    # print(_pos)
    # print(g_count)
    g_suma = np.add.reduceat(sq_errors[_ndxa], _posa, axis=0)
    g_meana = g_suma / (g_counta)

    return tasks, g_mean, workers, np.sqrt(g_meana)


def imeans_and_sigmas(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    weights = sigmas ** -2
    weights_per_ann = weights[w]
    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                             return_index=True)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    imeans = nums / denoms

    # Compute square error of each annotation with respect to the mean of that point

    inv_tasks = np.ones(np.max(tasks) + 1, dtype=int) * (-1)
    inv_tasks[tasks] = np.arange(len(tasks))
    sq_errors = (ann - imeans[inv_tasks[t]]) ** 2

    # Compute the variances

    ndx_w = np.argsort(w)
    # print(_ndx)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                                        return_index=True,
                                        return_counts=True)
    sum_w = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0)
    sample_variances = sum_w / (count_w)

    return tasks, imeans, workers, np.sqrt(sample_variances)


def make_conservative(weights, max_dif=50):
    max_weight = np.max(weights)
    min_weight = np.min(weights)
    q = max_weight / min_weight
    if q > max_dif:
        #print("q:", q)
        a = np.log(weights) - np.log(min_weight)
        b = np.log(q)
        #print("Before:", weights)
        weights = np.exp((a/b)*np.log(max_dif))*min_weight
        #print("After:", weights)
        max_weight = np.max(weights)
        min_weight = np.min(weights)
        q = max_weight / min_weight
        #print("q after:", q)
    return weights


class OverconfidenceException(Exception):
    pass


def imeans_and_sigmas2(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    print(sigmas)
    weights = sigmas ** -2
    if (np.max(weights) / np.min(weights) > 50):
        raise OverconfidenceException()
    #weights = make_conservative(weights)
    weights_per_ann = weights[w]
    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                             return_index=True)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    imeans = nums / denoms

    # Compute square error of each annotation with respect to the mean of that point

    inv_tasks = np.ones(np.max(tasks) + 1, dtype=int) * (-1)
    inv_tasks[tasks] = np.arange(len(tasks))
    sq_errors = (ann - imeans[inv_tasks[t]]) ** 2

    # Compute the variances

    ndx_w = np.argsort(w)
    # print(_ndx)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                                        return_index=True,
                                        return_counts=True)
    sum_w = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0)
    sample_variances = sum_w / (count_w)

    return tasks, imeans, workers, np.sqrt(sample_variances)


def mean_averaging(T, t, w, ann):
    # Compute the means
    _id, mean = point_average(t, ann)
    v = np.ones(T) * 0.5
    v[_id] = mean
    return v


def direct_weights(T, W, t, w, ann):
    tasks, means, workers, st_devs = point_averages_and_sigmas(t, w, ann)
    v = np.ones(T) * 0.5
    v[tasks] = means
    sigmas = np.ones(W) * (-1)
    sigmas[workers] = st_devs
    sigmas[sigmas < 0] = np.max(sigmas)

    return sigmas


def conservative_means_and_sigmas(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    weights = sigmas ** -2
    weights_per_ann = weights[w]

    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                                return_index=True)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)

    nums_per_ann = nums[t] - (weights_per_ann*ann)
    denoms_per_ann = denoms[t] - weights_per_ann
    means_per_ann = nums_per_ann / denoms_per_ann
    sq_errors = (ann - means_per_ann) ** 2

    ndx_w = np.argsort(w)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                            return_index=True,
                            return_counts=True)
    sample_variances = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0) / (count_w - 1)
    sigmas = np.sqrt(sample_variances)
    print(sigmas)
    return means_per_ann, workers, sigmas


def conservative_means_and_sigmas2(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)

    #print("sigmas:", sigmas)
    weights = sigmas ** -2
    #print("weights:", weights/np.sum(weights))
    if (np.max(weights) / np.min(weights) > 50):
        raise OverconfidenceException()
    weights_per_ann = weights[w]
    best_worker = np.argmax(weights)
    best_worker_self_weight = np.partition(weights, -2)[-2] # Assign the second largest weight
    best_worker_annotations = (w == best_worker)

    ndx_t = np.argsort(t)
    tasks, pos_t = np.unique(t[ndx_t],
                                return_index=True)
    nums = np.add.reduceat((weights_per_ann * ann)[ndx_t], pos_t, axis=0)
    denoms = np.add.reduceat(weights_per_ann[ndx_t], pos_t, axis=0)

    nums_per_ann = nums[t]
    denoms_per_ann = denoms[t]

    nums_per_ann[best_worker_annotations] -= (weights_per_ann[best_worker_annotations] * ann[best_worker_annotations])
    nums_per_ann[best_worker_annotations] += (best_worker_self_weight * ann[best_worker_annotations])
    denoms_per_ann[best_worker_annotations] -= weights_per_ann[best_worker_annotations]
    denoms_per_ann[best_worker_annotations] += best_worker_self_weight

    means_per_ann = nums_per_ann / denoms_per_ann
    sq_errors = (ann - means_per_ann) ** 2

    ndx_w = np.argsort(w)
    workers, pos_w, count_w = np.unique(w[ndx_w],
                            return_index=True,
                            return_counts=True)
    sample_variances = np.add.reduceat(sq_errors[ndx_w], pos_w, axis=0) / (count_w)

    return means_per_ann, workers, np.sqrt(sample_variances)


def imean_averaging(t, w, ann, sigmas=None):
    if sigmas is None:
        sigmas = np.ones(np.max(w) + 1)
    weights = sigmas ** -2
    weights_per_ann = weights[w]
    #print(weights_per_ann[:5])
    #print(w[:5])
    _ndx = np.argsort(t)
    tasks, _pos = np.unique(t[_ndx],
                                return_index=True)
    denoms = np.add.reduceat(weights_per_ann[_ndx], _pos, axis=0)
    #print(denoms[:5])
    nums = np.add.reduceat((weights_per_ann * ann)[_ndx], _pos, axis=0)
    #print(nums[:5])
    return tasks, nums/denoms


class OneShotMean(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        tasks, means = imean_averaging(t, w, ann)
        v = np.ones(T) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": np.ones(W)}


class OneShotDirect(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        sigmas = direct_weights(T, W, t, w, ann)
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(T) * 0.5
        v[tasks] = means
        return {"locations": v, "sigmas": sigmas}


class OneShotIterative(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        sigmas = np.ones(W)
        means = np.zeros(T)
        eps = 1e-4
        difference = 1e99
        reached_overconfidence = False
        while (difference > eps) and not reached_overconfidence:
            try:
                old_sigmas = sigmas.copy()
                tasks, imeans, workers, partial_sigmas = imeans_and_sigmas2(t, w, ann, sigmas)
                sigmas[workers] = partial_sigmas
                means[tasks] = np.nan_to_num(imeans, nan=0.5)
                difference = np.sum(np.abs(old_sigmas-sigmas))
            except OverconfidenceException:
                reached_overconfidence = True
        return {"locations": means, "sigmas": sigmas}


class OneShotBayesian(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        # sigmas = direct_weights(T, W, t, w, ann)

        n_annotators = exp.n_annotators
        n_points = exp.n_points

        d = {"w": n_annotators,
             "a": len(t),
             "t": n_points,
             "t_A": t + 1,
             "w_A": w + 1,
             "ann": ann
             }
        inits = {"sigma": [np.ones(n_annotators)]}

        model = cmd.CmdStanModel(stan_file=resource_filename('normal.2.stan'))
        s = model.sample(data=d, inits=inits, show_console=False)

        rec_sigmas_sample = s.stan_variable("sigmas")
        stan_sigmas = []
        for annotator in range(n_annotators):
            stan_sigmas.append(np.median(rec_sigmas_sample[:, annotator]))

        tasks, means = imean_averaging(t, w, ann, np.array(stan_sigmas))

        v = np.ones(np.max(t)+1) * 0.5
        v[tasks] = means

        rec_points_sample = s.stan_variable("mu")
        stan_points = []
        for p in range(n_points):
            stan_points.append(np.median(rec_points_sample[:, p]))

        return {"locations": v, "sigmas": stan_sigmas, "stan points": stan_points}


class OneShotConservative(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        T = np.max(t) + 1
        W = np.max(w) + 1
        sigmas = np.ones(W)
        means = np.zeros(T)
        eps = 1e-4
        difference = 1e99
        while difference > eps:
            old_sigmas = sigmas.copy()
            means_per_ann, workers, partial_sigmas = conservative_means_and_sigmas(t, w, ann, sigmas)
            sigmas[workers] = partial_sigmas
            difference = np.sum(np.abs(old_sigmas-sigmas))
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(T) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": sigmas}


def compute_sigmas_conservative2(t, w, ann):
    sigmas = np.ones(np.max(w) + 1)
    eps = 1e-4
    difference = 1e99
    overconfidence_reached = False
    while (difference > eps) and not overconfidence_reached:
        old_sigmas = sigmas.copy()
        try:
            means_per_ann, workers, partial_sigmas = conservative_means_and_sigmas2(t, w, ann, sigmas)
            sigmas[workers] = partial_sigmas
            difference = np.sum(np.abs(old_sigmas - sigmas))
        except OverconfidenceException:
            overconfidence_reached = True
    return sigmas


class OneShotConservative2(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)
        # _, counts = np.unique(w,return_counts=True)
        # print(counts)
        sigmas = compute_sigmas_conservative2(t, w, ann)
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(np.max(t) + 1) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": sigmas}


def max_redundancy(exp):
    max_total_annotations = min(exp.max_annotations_per_individual * exp.n_points, exp.max_total_annotations)
    k = max_total_annotations // exp.n_points
    return k


def random_annotation(exp, batch_start=0, batch_size=None, k=None):
    if batch_size is None:
        batch_size = exp.n_points
    if k is None:
        k = max_redundancy(exp)
    t, w = random_assignment(batch_size, exp.n_annotators, k)
    t += batch_start
    ann = exp.batch_request(t, w)
    return t, w, ann


def softmax_stable(x):
    return (np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum())


def sigma_assignment(batch_size, sigmas, k, greedyness = 0.1, batch_start=0):
    n_annotators = len(sigmas)
    p = softmax_stable(greedyness * (sigmas ** (-2)))
    # print("p=", p, np.sum(p))
    # pm = softmax_stable(-greedyness * (sigmas ** (-2)))
    # print("pm=", pm, np.sum(pm))
    t_A = np.zeros(batch_size * k, dtype=int)
    w_A = np.zeros(batch_size * k, dtype=int)
    selected_annotators_indexes = np.argsort(p[None, :] * np.random.rand(batch_size, n_annotators), axis=1)[:, -k:]
    # print("sel = ", selected_annotators_indexes)
    total_annotations = 0
    for j in range(n_annotators):
        j_point_indices = np.argwhere(selected_annotators_indexes == j)[:, 0]
        end_annotations = total_annotations + len(j_point_indices)
        t_A[total_annotations:end_annotations] = j_point_indices
        w_A[total_annotations:end_annotations] = j
        total_annotations = end_annotations
    # print("w=", w_A)
    return t_A+batch_start, w_A


def sigma_annotation(exp, sigmas, batch_start=0, batch_size=None, k=None, greedyness=0.01):
    if batch_size is None:
        batch_size = exp.n_points
    if k is None:
        k = max_redundancy(exp)
    t, w = sigma_assignment(batch_size, sigmas, k, greedyness=greedyness, batch_start=batch_start)
    ann = exp.batch_request(t, w)
    return t, w, ann


class KShot(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        n_batches = 10
        batch_start = 0
        batch_size = exp.n_points // n_batches
        t, w, ann = random_annotation(exp, batch_start, batch_size)
        print("anns:", t, w, ann)
        sigmas = compute_sigmas_conservative2(t, w, ann)
        print("sigmas", sigmas)
        while (batch_start + batch_size) < exp.n_points:
            batch_start = batch_start + batch_size
            batch_size = min(batch_size, exp.n_points - batch_start)
            t_, w_, ann_ = sigma_annotation(exp, sigmas, batch_start, batch_size)
            print("anns2:", t_, w_, ann_)
            t = np.concatenate((t, t_))
            w = np.concatenate((w, w_))
            ann = np.concatenate((ann, ann_))
            sigmas = compute_sigmas_conservative2(t, w, ann)
            print("sigmas2", sigmas)
        _, counts = np.unique(t, return_counts=True)
        print("c/w:", counts)
        print(np.sum(counts))
        _, counts = np.unique(w, return_counts=True)
        print("c/w:", counts)
        print(np.sum(counts))
        tasks, means = imean_averaging(t, w, ann, sigmas)
        v = np.ones(np.max(t) + 1) * 0.5
        v[tasks] = np.nan_to_num(means, nan=0.5)
        return {"locations": v, "sigmas": sigmas}


def learn_variance_profiles(t, w, ann, l=15):
    n_annotators = np.max(w) + 1
    # print(n_annotators)
    n_points = np.max(t) + 1
    # print(n_points)
    d = {"w": n_annotators,
         "a": len(ann),
         "t": n_points,
         "t_A": t + 1,
         "w_A": w + 1,
         "ann": ann,
         "l": l
         }
    inits = {"y_grid": np.ones((n_annotators, l)) * 0.1}
    gp = cmd.CmdStanModel(stan_file=resource_filename('gp-learn-variances-ma.stan'))
    s = gp.optimize(data=d, inits=inits, show_console=True, iter=10000, algorithm='lbfgs', tol_rel_grad=10000.)
    kappa = s.stan_variable("kappa")
    y_grid = s.stan_variable("y_grid")
    return kappa, y_grid


# def compute_mean_sigmas(kappa, y_grid, l=15):
#     n_annotators = y_grid.shape[0]
#     gridpoints = np.arange(l) / (l - 1.)
#     allx = np.arange(1000) / 1000.
#     functionsigmas_learned = []
#     for i in range(n_annotators):
#         functionsigmas_learned.append(compute_functionsigmas_aux(allx, kappa, gridpoints, y_grid[i]))
#
#     # 3. Se calcula la sigma media para cada anotador
#     mean_sigmas = []
#     for i in range(n_annotators):
#         mean_sigmas.append(np.mean(functionsigmas_learned[i]))
#     return mean_sigmas
def compute_mean_sigmas(n_annotators, sigma_functions):
    mean_sigmas = []
    for i in range(n_annotators):
        mean_sigmas.append(np.mean(sigma_functions[i]))
    return np.array(mean_sigmas)


def position_based_round(exp, tasks, positions, annotators_per_task, functionlearned_sigmas=None):
    # Miramos las sigmas-funciones en cada punto para volver a hacer un request:
    N = len(tasks)
    t3 = np.zeros(N, dtype=int)
    w3 = np.zeros(N, dtype=int)
    ann3 = np.zeros(N)

    for i, p in enumerate(positions):
        # sigma_at_point = np.array([annotator.sigma([p])[0] for annotator in exp.annotator_set.annotators])
        if p < 0.:
            p = 0.
        elif p >= 1.:
            p = 0.999 #we should treat this better... maybe we could assert it when annotating
        sigma_at_point = []
        index_aux = int(p*1000)
        for annotator in range(exp.n_annotators):
            sigma_at_point.append(functionlearned_sigmas[annotator][index_aux])

        for ann_index in annotators_per_task[tasks[i]]:
            sigma_at_point[ann_index] = 1e2
        # print("sigma=", sigma_at_point)

        t3[i:i + 1], w3[i:i + 1], ann3[i:i + 1] = sigma_annotation(exp, np.array(sigma_at_point),
                                                                   batch_start=tasks[i],
                                                                   batch_size=1, k=1, greedyness=100)
                                            #podemos tener problemas si otorgamos una probabilidad pequeña al anotador que ha anotado anteriormente

        # t_aux, w_aux = sigma_assignment(1, np.array(sigma_point), 1, batch_start=t2[i])
        # print(t_aux)
        # t_aux = np.array([t2[i]])
        # print(t_aux)
        # ann_aux = exp.batch_request(t_aux, w_aux)
        # print(p,t_aux,w_aux)
        # t3.append(t_aux[0])
        # w3.append(w_aux[0])
        # ann3.append(ann_aux[0])
    return t3, w3, ann3


def compute_annotators_per_task(t, w):
    from collections import defaultdict
    annotators_per_task = defaultdict(set)
    for i, w in enumerate(w):
        annotators_per_task[t[i]].add(w)
    return annotators_per_task


def diagnose_errors(t, w, ann):
    annotators_per_task = compute_annotators_per_task(t, w)
    for t in annotators_per_task.values():
        if len(t) < 2:
            print("There is a task with less than two annotators")
            raise Exception("Task with less than two annotators")
    print("All tasks in the dataset contain at least two annotators. GREAT!")


class KShotSpatial(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        # 1. 10% of the points are annotated randomly
        initial_percentage = 0.1
        initial_batch_size = int(exp.n_points*initial_percentage)
        t1, w1, ann1 = random_annotation(exp, 0, initial_batch_size)

        # 2. We learn the sigma functions of all annotators
        kappa1, y_grid1 = learn_variance_profiles(t1, w1, ann1)
        functionsigmas_learned1 = compute_functionsigmas(kappa1, y_grid1)

        # 3. We compute the mean sigma for each annotator
        mean_sigmas = compute_mean_sigmas(len(functionsigmas_learned1), functionsigmas_learned1)

        # 4. The other 90% of the points are annotated just once taking into account the mean sigmas
        second_batch_size = exp.n_points - initial_batch_size
        second_batch_start = initial_batch_size
        t2, w2, ann2 = sigma_annotation(exp, mean_sigmas, batch_start=second_batch_start,
                                        batch_size=second_batch_size, k=1) #should greediness follow a certain criteria?

        last_t, last_w, last_consensus = t2, w2, ann2
        last_sigmafunctions = functionsigmas_learned1
        t_historic, w_historic, ann_historic = t2, w2, ann2

        iter = 0
        while iter < (exp.max_total_annotations/exp.n_points - 1):
            # 5. The same 90% of points are annotated just once using the learned sigma values at the previous points
            annotators_per_task = compute_annotators_per_task(t_historic, w_historic)
            t3, w3, ann3 = position_based_round(exp, last_t, last_consensus, annotators_per_task, last_sigmafunctions)
            t_historic = np.concatenate((t_historic, t3))
            w_historic = np.concatenate((w_historic, w3))
            ann_historic = np.concatenate((ann_historic, ann3))

            # 6. We recompute the sigma profiles
            t = np.concatenate((t1, t_historic))
            w = np.concatenate((w1, w_historic))
            ann = np.concatenate((ann1, ann_historic))
            kappa2, y_grid2 = learn_variance_profiles(t, w, ann)
            functionsigmas_learned2 = compute_functionsigmas(kappa2, y_grid2)
            last_sigmafunctions = functionsigmas_learned2

            # 7. We perform an intelligent mean between the last two annotations taking into account the new sigmas
            tasks, means = spatial_imean_averaging(t_historic, w_historic, ann_historic, last_sigmafunctions)
            last_consensus = means
            last_t = tasks
            iter += 1

        initial_tasks, initial_means = spatial_imean_averaging(t1, w1, ann1, last_sigmafunctions)
        locations = np.concatenate((initial_means, last_consensus))

        return {'locations': locations, 'sigmas': last_sigmafunctions}


def spatial_imean_averaging(t, w, ann, sigma_functions):

    tasks = []
    means = []
    for i, task in enumerate(np.unique(t)):
        sigmas_at_point = []
        for j, annotator in enumerate(w[t == task]):
            annotation_aux = ann[t == task][j]
            if annotation_aux < 0.:
                annotation_aux = 0.
            elif annotation_aux >= 1.:
                annotation_aux = 0.999
            index_aux = int(annotation_aux * 1000)
            sigmas_at_point.append(sigma_functions[annotator][index_aux])

        tasks.append(
            imean_averaging(t[t == task], np.arange(j + 1), ann[t == task], sigmas=np.array(sigmas_at_point))[0][0])
        means.append(
            imean_averaging(t[t == task], np.arange(j + 1), ann[t == task], sigmas=np.array(sigmas_at_point))[1][0])

    return tasks, means


def compute_functionsigmas(kappa, y_grid, l=15):

    gridpoints = np.arange(l) / (l - 1.)
    allx = np.arange(1000) / 1000.
    n_annotators = y_grid.shape[0]

    functionsigmas_learned = []
    for i in range(n_annotators):
        functionsigmas_learned.append(compute_functionsigmas_aux(allx, kappa, gridpoints, y_grid[i]))

    return functionsigmas_learned


def compute_functionsigmas_aux(t, kappa, x, y):
    r = len(x)
    tr = np.repeat(t, r)
    tr.shape = (len(t), r)
    d = (tr - x) * (tr - x)
    ed = np.exp(-kappa * d)
    s = np.sum(ed, axis=1)
    res = np.dot(ed, y)/s
    return res


class OneShotSpatialBayesian(ActiveAnnotationMethod):
    def run(self, exp: ActiveAnnotationContest):
        t, w, ann = random_annotation(exp)

        n_annotators = exp.n_annotators
        n_points = exp.n_points

        allx = np.arange(1000) / 1000.
        l = 15

        d = {"w": n_annotators,
             "a": len(ann),
             "t": n_points,
             "t_A": t + 1,
             "w_A": w + 1,
             "ann": ann,
             "l": l
             }
        inits = {"y_grid": np.ones((n_annotators, l)) * 0.1}

        gp = cmd.CmdStanModel(stan_file=resource_filename('gp-learn-variances-ma.stan'))
        s = gp.optimize(data=d, inits=inits, show_console=True, iter=1000, algorithm='lbfgs', tol_rel_grad=10000.)

        kappa = s.stan_variable("kappa")
        y_grid = s.stan_variable("y_grid")
        v = s.stan_variable("x")

        gridpoints = np.arange(l) / (l - 1.)
        functionsigmas_learned = []
        for i in range(n_annotators):
            functionsigmas_learned.append(compute_functionsigmas_aux(allx, kappa, gridpoints, y_grid[i]))

        return {"locations": v, "sigmas": functionsigmas_learned}
