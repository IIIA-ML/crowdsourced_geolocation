data {
  int<lower=0> N;
  int<lower=2> L;
  matrix[N,L] marked;
  matrix[N,L] x;
}
parameters {
  array[L] real<lower=0> sigma;
  array[N] real mu;
}
model {
  for (l in 1:L)
      for (n in 1:N)
          if (marked[n,l] == 1)
              x[n,l] ~ normal(mu[n], sigma[l]);
}