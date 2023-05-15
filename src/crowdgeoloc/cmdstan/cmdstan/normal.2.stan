data {
  int<lower=1> w; //number of workers
  int<lower=1> t; //number of tasks
  int<lower=1> a; //number of annotations

  array[a] int<lower=1,upper=t> t_A; // the item the n-th annotation belongs to
  array[a] int<lower=1,upper=w> w_A; // the annotator which produced the n-th annotation
  array[a] real ann;              // the annotation

  
}


parameters {
   array[w] real<lower=0> sigmas;
   array[t] real mu;
}


model {
    

    for (a_ in 1:a) {
        ann[a_] ~ normal(mu[t_A[a_]], sigmas[w_A[a_]]);
    }
}

