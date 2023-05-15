data {
  int<lower=1> Nk; // Number of points known
  array[Nk] real xk; // Positions of the points known
  array[Nk] real yk; // Values of the points known
  int<lower=1> Nu; // Number of points unknown
  array[Nu] real xu; // Positions of the points unknown
  real alpha; // Controls the spread of the functions generated by the GP. The smaller the alpha, the smaller the range in y's.
  real rho;   // Controls the variability of the functions generated by the GP. The smaller the rho, the larger the variability.
  real sigma; // Controls the randomness of the functions. The larger the sigma, the less continuous the generated functions.
  real mu_0;  // The average values of the function.
}

transformed data {
  int<lower=1> N = Nk + Nu;
  array[N] real x;
  for (nk in 1:Nk) {
    x[nk] = xk[nk];
  }
  for (nu in 1:Nu) {
    x[Nk + nu] = xu[nu];
  }
  
  vector[N] mu = rep_vector(mu_0, N);
  
  real delta = 1e-9;
  //matrix[N, N] L;
  matrix[N, N] K = cov_exp_quad(x, alpha, rho) + diag_matrix(rep_vector(delta , N));

  //L = cholesky_decompose(K);
}

parameters {
    vector[N] log_y;
}

model {
    log_y ~ multi_normal(mu, K);
    //if (iteration()%50==0) { print("log_y",log_y); }
    yk ~ normal(exp(log_y[1:Nk]), sigma);
}