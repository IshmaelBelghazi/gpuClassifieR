#ifndef ENTRYPOINSMODEL_LOGREG_C
#define ENTRYPOINSMODEL_LOGREG_C

#include <Rdefines.h>

#ifdef	__cplusplus
extern "C" {
#endif
SEXP get_condprob_logreg_(SEXP X, SEXP W, SEXP normalize, SEXP log_domain);
SEXP get_cost_logreg_(SEXP X, SEXP W, SEXP T, SEXP decay);
SEXP get_grad_logreg_(SEXP X, SEXP W, SEXP T, SEXP decay);
#ifdef	__cplusplus
}
#endif

#endif
