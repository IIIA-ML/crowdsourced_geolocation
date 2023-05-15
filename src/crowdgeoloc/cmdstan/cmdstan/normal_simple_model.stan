data {
  int<lower=0> N;
  int<lower=2> L;
  int<lower=2> P;
  matrix[N,L] x;
  array[P-1] real x_limit;
  real priorfit_alpha;
  real priorfit_beta;
  
}

parameters {
  array[L,P] real<lower=0> sigma;
  array[N] real mu;
}

model {
  for (l in 1:L){
    for (p in 1:P){
        sigma[l,p] ~ lognormal(priorfit_alpha, priorfit_beta);
    }

    for (n in 1:N){
      if (mu[n] < x_limit[1])
        x[n,l] ~ normal(mu[n], sigma[l,1]);
      else if (mu[n] < x_limit[2])
        x[n,l] ~ normal(mu[n], sigma[l,2]);
      else
        x[n,l] ~ normal(mu[n], sigma[l,3]);
    
    }
    /*
    for (n in 1:N){
      int point_checked;
      for (p in 1:(P-1)){
        point_checked = 0;
        if (mu[n] < x_limit[p]){
          x[n,l] ~ normal(mu[n], sigma[l,p]);
          point_checked = 1;
          break;
        }
      }
      if (point_checked == 0){
        x[n,l] ~ normal(mu[n], sigma[l,P]);
      }
      */
    //}
  }
  
}