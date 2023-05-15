data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations

  array[a] int<lower=1,upper=t> t_A; // the item the n-th annotation belongs to
  array[a] int<lower=1,upper=w> w_A; // the annotator which produced the n-th annotation
  array[a] real ann;              // the annotation

  int<lower=2> l;                    // number of approximation points per annotator

  //real shape, location, scale;
}

transformed data {
    real kappa_ = 5.*(l-1.)*(l-1.);
    array[l] real<lower=0> x_grid;
    for (l_ in 1:l) {
        x_grid[l_] = (l_-1.)/(l-1.);
    }

    array[t] int counts; // Number of times each task is annotated
    array[t] real mean;  // Rough estimate of the position of the points using the average
    
    // Compute the means
    
    for (t_ in 1:t) {
        mean[t_] = 0;
        counts[t_] = 0;
    }
    for (a_ in 1:a) {
        counts[t_A[a_]] += 1;
        mean[t_A[a_]] += ann[a_];
    }
    for (t_ in 1:t) {
        mean[t_] = mean[t_] / counts[t_];
    }
    // Next, we will order the data by task to have an efficient access later on
    array[t+1] int start_task_index; // For each of the tasks, this points to the index of the first annotation that contains this task
   
    array[a] real ordered_ann; // All the annotations reordered by task
    array[a] int ordered_w;    // The workers reorderd by task
    array [a] int ordered_to_unordered;
    array [a] int unordered_to_ordered;
   
    array[t] int tmp_counts; // Number of annotations per task as we compute the mapping (this is a temporary .
    
    start_task_index[1] = 1;
    for (t_ in 1:t) {
        tmp_counts[t_] = 0;
        start_task_index[t_ + 1] = counts[t_] + start_task_index[t_];
    }
    for (a_ in 1:a) {
        int t_ = t_A[a_];
        int index = start_task_index[t_] + tmp_counts[t_];
        ordered_ann[index] = ann[a_];
        ordered_w[index] = w_A[a_];
        ordered_to_unordered[index] = a_;
        unordered_to_ordered[a_] = index;
        tmp_counts[t_] += 1;
    }
}

parameters {
   array[w,l] real<lower=0> y_grid;
}

transformed parameters {

}

model {
    array[a] real ordered_x_a; // Contains the intelligent mean taking into account all the annotations of this task except this one.


    // Compute the sigmas for each annotation

    array [a] real sigmas;
    array [a] real inv_sigmas;
    array [a] real ordered_sigmas;
    array[a] real ordered_sum;
    array[a] real ordered_denom;
    array[a] real ordered_x_a_;
    array[l] real d;
    real d_s;

    for (a_ in 1:a) {
        int t_ = t_A[a_];
        int w_ = w_A[a_];

        // Compute weights
        d_s = 0;
        for (l_ in 1:l) {
            d[l_] = mean[t_] - x_grid[l_];
            d[l_] *= d[l_];
            d[l_] = exp(-kappa_ * d[l_]);
            d_s += d[l_];
        }
        if (d_s==0) {
            print("d_s:", d_s);
        }
        sigmas[a_] = 0;
        for (l_ in 1:l) {
            sigmas[a_] += y_grid[w_,l_] * (d[l_] / d_s);
        }
        inv_sigmas[a_] = 1. / (sigmas[a_] * sigmas[a_]);
        ordered_sum[a_] = 0;
        ordered_denom[a_] = 0;
    }

    real v;
    int t_;
    int w_;
    for (a_ in 1:a) {
        t_ = t_A[a_];
        w_ = w_A[a_];
        v = inv_sigmas[a_] * ann[a_];
        for (o_a_ in start_task_index[t_]:start_task_index[t_]+counts[t_]-1) {
            if (ordered_w[o_a_] != w_) {
                ordered_sum[o_a_] += v;
                ordered_denom[o_a_] += inv_sigmas[a_];
            }
        }
    }

    for (o_a_ in 1:a) {
        if (ordered_denom[o_a_]==0) {
            print("Error, division por 0:", o_a_);
        }
        ordered_x_a_[o_a_] = ordered_sum[o_a_] / ordered_denom[o_a_];
    }



    for (a_ in 1:a) {
        t_ = t_A[a_];
        ann[a_] ~ normal(ordered_x_a_[unordered_to_ordered[a_]], sigmas[a_]);
    }
    //for (j in 1:w) {
        //for (c in 1:l) {
             //y_grid[j,c] ~ normal(def_std_dev_mean, def_std_dev_std_dev)
        //}
    //}
}

generated quantities {
    array[t] real x;
    array [a] real sigmas;
    array [a] real inv_sigmas;
    array [a] real ordered_sigmas;
    array[t] real x_sum;
    array[t] real x_denom;
    array[l] real d;
    real d_s;

    for (a_ in 1:a) {
        int t_ = t_A[a_];
        int w_ = w_A[a_];

        // Compute weights
        d_s = 0;
        for (l_ in 1:l) {
            d[l_] = mean[t_] - x_grid[l_];
            d[l_] *= d[l_];
            d[l_] = exp(-kappa_ * d[l_]);
            d_s += d[l_];
        }
        sigmas[a_] = 0;
        for (l_ in 1:l) {
            sigmas[a_] += y_grid[w_,l_] * (d[l_] / d_s);
        }
        inv_sigmas[a_] = 1. / (sigmas[a_] * sigmas[a_]);
    }

    for (t_ in 1:t) {
        x_sum[t_] = 0;
        x_denom[t_] = 0;
    }

    real v;
    int w_;
    for (a_ in 1:a) {
        int t_ = t_A[a_];
        w_ = w_A[a_];
        v = inv_sigmas[a_] * ann[a_];
        x_sum[t_] += v;
        x_denom[t_] += inv_sigmas[a_];
    }
    for (t_ in 1:t) {
        x[t_] = x_sum[t_] / x_denom[t_];
    }
    real kappa = kappa_;
}
