data {
  int<lower=0> N; //number of points
  int<lower=2> L; //number of annotators
  int<lower=1> R; //redundancy i.e. number of reports for each point
  matrix[N,R] reports; //reported annotations for each point
  matrix[N,R] anns_index; //annotators which have reported each point 

}
parameters {
  array[L] real<lower=0> sigma;
  array[N] real mu;
}
model {
  for (n in 1:N)
      for (r in 1:R)
          reports[n,r] ~ normal(mu[n], sigma[anns_index[n,r]]);
}