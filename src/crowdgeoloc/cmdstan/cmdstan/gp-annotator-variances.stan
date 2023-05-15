data {
  int<lower=0> N; // Number of points
  int<lower=2> L; // Number of annotators
  matrix[L,N] y; // y[i,j] is the position reported for point j by annotator i
  real alf
  //real gamma_alpha;
  //real gamma_beta;
}

parameters {
  array[N] real x; // x[i] is the position of the i-th point
}

transformed data {
  //matrix[N, N] L_M;
  matrix[N, N] K_M = cov_exp_quad(x, 1.0, 0.1);
  vector[N] mu = rep_vector(0, N);
  for (n in 1:N) {
    K_M[n, n] = K_M[n, n]+1e-9;
  }
  //L_M = cholesky_decompose(K_M);
}

model {
    //vector[N] eta;
    for (l in 1:L) {
        eta ~ std_normal();
        y[l] ~ multi_normal(x, K_M);
    }
}