data {
  int<lower=0> N; //number of points
  int<lower=2> L; //number of annotators
  matirx[N,L] x

}
parameters {
  array[L] real<lower=0> sigma;
  array[N] real mu;
}
model {
  for (l in 1:L)
      for (n in 1:N)
          x[n,l] ~ normal(mu[n], sigma[l]);
}