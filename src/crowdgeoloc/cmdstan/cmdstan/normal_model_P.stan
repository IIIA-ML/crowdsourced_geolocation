data {
  int<lower=0> N;
  int<lower=2> L;
  int<lower=2> P;
  matrix[N,L] x;
  array[P-1] real x_limit;
  real gamma_alpha;
  real gamma_beta;
  
}

parameters {
  array[L,P] real<lower=0> sigma;
  array[N] real mu;
}

model {
  for (l in 1:L){
    sigma[l,1] ~ gamma(gamma_alpha, gamma_beta);
    sigma[l,2] ~ gamma(gamma_alpha, gamma_beta);
    for (n in 1:N){
      if (mu[n] < x_limit[1])
        x[n,l] ~ normal(mu[n], sigma[l,1]);
      else
        x[n,l] ~ normal(mu[n], sigma[l,2]);
    
    }
  }
  
}