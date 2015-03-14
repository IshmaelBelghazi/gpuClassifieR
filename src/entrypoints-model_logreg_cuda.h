#ifndef ENTRYPOINTSMODEL_LOGREG_CUDA
#define ENTRYPOINTSMODEL_LOGREG_CUDA

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif

SEXP get_condprob_logreg_cuda(SEXP X, SEXP W, SEXP normalize, SEXP log_domain);
SEXP get_cost_logreg_cuda(SEXP X, SEXP W, SEXP T, SEXP decay);
SEXP get_grad_logreg_cuda(SEXP X, SEXP W, SEXP T, SEXP decay);

#ifdef	__cplusplus
}
#endif

#endif // ENTRYPOINTSMODEL_LOGREG_CUDA
