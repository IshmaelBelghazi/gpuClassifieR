#ifndef ENTRYPOINTS_TRAINSGD_CUDA
#define ENTRYPOINTS_TRAINSGD_CUDA

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif

SEXP train_gd_cuda(SEXP X, SEXP W, SEXP T, SEXP decay, SEXP step_size,
                   SEXP max_iter, SEXP verbose, SEXP tol);
#ifdef	__cplusplus
}
#endif

#endif
