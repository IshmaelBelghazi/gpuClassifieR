#ifndef ENTRYPOINTS_TRAINSGD_C
#define ENTRYPOINTS_TRAINSGD_C

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif
SEXP train_gd_(SEXP X, SEXP W, SEXP T, SEXP decay, SEXP step_size,
               SEXP max_iter, SEXP verbose, SEXP tol);
#ifdef	__cplusplus
}
#endif

#endif
