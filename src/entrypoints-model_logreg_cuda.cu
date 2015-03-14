#include <R.h>
#include <Rdefines.h>

#include <cuda.h>
#include <cublas_v2.h>

#include "cudaLogReg.h"
#include "model_logreg_cuda.h"
#include "entrypoints-model_logreg_cuda.h"

SEXP get_condprob_logreg_cuda(SEXP X, SEXP W, SEXP normalize, SEXP log_domain) {
  // initializing protection counter
  int nprot = 0;
  //  Getting pointers to underlying variables
  int * normalize_ptr = LOGICAL(normalize);
  int * log_domain_ptr = LOGICAL(log_domain);
  // Getting pointers to underlying C arrays.
  // Host pointers
  double * X_ptr = (double *) REAL(X);  // N X M
  double * W_ptr = (double *) REAL(W);  // K X M
  // getting arrays underlying dimension
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_W = INTEGER(GET_DIM(W));
  // Getting dimensions

  int N = dim_X[0], M = dim_X[1], K = dim_W[0];
  // assigining for class conditional probabilities Y. is  K X N
  SEXP Y; PROTECT(Y = allocMatrix(REALSXP, K, N)); nprot++;
  memset(REAL(Y), 0.0, K * N * sizeof(double));  // Initializing Y
  double * Y_ptr = (double *)REAL(Y);  // Getting pointers to Y

  // device pointers
  double * dev_Y_ptr;
  double * dev_W_ptr;
  double * dev_X_ptr;

  // Initializing cublas handle
  cublasHandle_t handle;

  // allocating memory on the device
  CUDA_CALL(cudaMalloc((void **) &dev_X_ptr, N * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_W_ptr, K * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_Y_ptr, K * N * sizeof(double)));

  // Creating handle
  CUBLAS_CALL(cublasCreate(&handle));

  // Copying matrices to device
  CUBLAS_CALL(cublasSetMatrix(N, M, sizeof(double), X_ptr, N, dev_X_ptr, N));
  CUBLAS_CALL(cublasSetMatrix(K, M, sizeof(double), W_ptr, K, dev_W_ptr, K));

  // Setting condtional probabilities
  _set_condprob_logreg_cuda(handle, dev_X_ptr, dev_W_ptr, dev_Y_ptr, M, N, K,
                            *normalize_ptr, *log_domain_ptr);

  // Copying to host
  CUBLAS_CALL(cublasGetMatrix(K, N, sizeof(double), dev_Y_ptr, K, Y_ptr, K));

  cudaFree(dev_X_ptr);
  cudaFree(dev_W_ptr);
  cudaFree(dev_Y_ptr);
  cublasDestroy(handle);

  UNPROTECT(nprot);

  return Y;
}

// Cross entropy (minus log-likelihood)
SEXP get_cost_logreg_cuda(SEXP X, SEXP W, SEXP T, SEXP decay) {
  // initializing protection counter
  int nprot = 0;
  // Assigning SEXP Variable for cross-entropy
  SEXP cost; PROTECT(cost = allocVector(REALSXP, 1)); nprot++;
  // Initializing cross entropy
  memset(REAL(cost), 0.0, sizeof(double));
  // Getting pointers to underlying arrays
  double * X_ptr = REAL(X); // N X K
  double * dev_X_ptr;
  double * W_ptr = REAL(W); // K X M
  double * dev_W_ptr;
  double * T_ptr = REAL(T); // K X N
  double * dev_T_ptr;
  double * dev_log_Y_ptr;
  double * cost_ptr = REAL(cost);
  double * decay_ptr = REAL(decay);

  // getting arrays underlying dimension
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_W = INTEGER(GET_DIM(W));
  // Getting dimensions
  int N = dim_X[0], M = dim_X[1], K = dim_W[0];

  // Allocating and transfering to device
  // Initializing cublas handle
  cublasHandle_t handle;
  // Creating handle
  CUBLAS_CALL(cublasCreate(&handle));
  // allocating memory on the device
  CUDA_CALL(cudaMalloc((void **) &dev_X_ptr, N * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_W_ptr, K * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_T_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_log_Y_ptr, K * N * sizeof(double)));

  // Copying matrices to device
  CUBLAS_CALL(cublasSetMatrix(N, M, sizeof(double), X_ptr, N, dev_X_ptr, N));
  CUBLAS_CALL(cublasSetMatrix(K, M, sizeof(double), W_ptr, K, dev_W_ptr, K));
  CUBLAS_CALL(cublasSetMatrix(K, N, sizeof(double), T_ptr, K, dev_T_ptr, K));

  // Computing cost on the device but returning on hist
  _set_cost_logreg_cuda(handle, dev_X_ptr, dev_W_ptr, dev_T_ptr, dev_log_Y_ptr,
                        M, N, K, decay_ptr, cost_ptr);

  // Cleaning up
  cudaFree(dev_X_ptr);
  cudaFree(dev_W_ptr);
  cudaFree(dev_T_ptr);
  cudaFree(dev_log_Y_ptr);
  cublasDestroy(handle);

  UNPROTECT(nprot);

  return cost;
}

SEXP get_grad_logreg_cuda(SEXP X, SEXP W, SEXP T, SEXP decay) {
  // Initializing protection counte
  int nprot = 0;
  // Getting pointers to underlying C arrays.
  double * X_ptr = REAL(X); // N X M
  double * W_ptr = REAL(W); // K X M
  double * T_ptr = REAL(T); // K X N
  double * decay_ptr = REAL(decay);

  // getting arrays underlying dimension */
  int * dim_X = INTEGER(GET_DIM(X));
  int * dim_W = INTEGER(GET_DIM(W));

  int M = dim_X[1], N = dim_X[0], K = dim_W[0];

  double * dev_W_ptr;
  double * dev_X_ptr;
  double * dev_T_ptr;
  double * dev_Y_ptr;
  double * dev_Y_minus_T_ptr;

  // Defining and allocating return SEXP
  SEXP grad; PROTECT(grad = allocMatrix(REALSXP, K, M)); nprot++;
  double * grad_ptr = REAL(grad);
  double * dev_grad_ptr;
  memset(grad_ptr, 0.0, sizeof(double) * K * M);

  // Allocating and transfering to device
  // Initializing cublas handle
  cublasHandle_t handle;
  // Creating handle
  CUBLAS_CALL(cublasCreate(&handle));

  // allocating memory on the device
  CUDA_CALL(cudaMalloc((void **) &dev_X_ptr, N * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_W_ptr, K * M * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_T_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_Y_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_Y_minus_T_ptr, K * N * sizeof(double)));
  CUDA_CALL(cudaMalloc((void **) &dev_grad_ptr, K * M * sizeof(double)));

  // Copying matrices to device
  CUBLAS_CALL(cublasSetMatrix(N, M, sizeof(double), X_ptr, N, dev_X_ptr, N));
  CUBLAS_CALL(cublasSetMatrix(K, M, sizeof(double), W_ptr, K, dev_W_ptr, K));
  CUBLAS_CALL(cublasSetMatrix(K, N, sizeof(double), T_ptr, K, dev_T_ptr, K));

  // Computing gradient
  _set_grad_logreg_cuda(handle, dev_X_ptr, dev_W_ptr, dev_T_ptr, dev_Y_ptr,
                        dev_Y_minus_T_ptr, M, N, K, decay_ptr, dev_grad_ptr);

  // Transferring data to host
  // Copying to host
  CUBLAS_CALL(cublasGetMatrix(K, M, sizeof(double), dev_grad_ptr,
                              K, grad_ptr, K));

  // freeing allocated memory
  cudaFree(dev_X_ptr);
  cudaFree(dev_W_ptr);
  cudaFree(dev_T_ptr);
  cudaFree(dev_Y_ptr);
  cudaFree(dev_Y_minus_T_ptr);
  cudaFree(dev_grad_ptr);
  cublasDestroy(handle);

  UNPROTECT(nprot);

  return grad;
}
