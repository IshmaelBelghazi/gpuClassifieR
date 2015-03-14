#ifndef MODEL_LOGREG_CUDA
#define MODEL_LOGREG_CUDA
#include <cublas_v2.h>

void _set_condprob_logreg_cuda(cublasHandle_t handle,
                               const double * __restrict__ dev_X,
                               const double * __restrict__ dev_W,
                               double * __restrict__ dev_Y,
                               const int M, const int N, const int K,
                               const int normalize, const int log_domain);
// dev_log_Y is preallocated to avoid expensive and repeated cudaMalloc calls
void _set_cost_logreg_cuda(cublasHandle_t handle,
                           const double * __restrict__ dev_X,
                           const double * __restrict__ dev_W,
                           const double * __restrict__ dev_T,
                           double * __restrict__ dev_log_Y,
                           const int M, const int N, const int K,
                           const double * decay,
                           double * cost);
// dev_Y and dev_Y_minus_T are preallocated to avoid

void _set_grad_logreg_cuda(cublasHandle_t handle,
                           const double * __restrict__ dev_X,
                           const double * __restrict__ dev_W,
                           const double * __restrict__ dev_T,
                           double * __restrict__ dev_Y,
                           double * __restrict__ dev_Y_minus_T,
                           const int M, const int N, const int K,
                           const double * decay,
                           double * dev_grad);
#endif // MODEL_LOGREG_CUDA
