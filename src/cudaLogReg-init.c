#include <R.h>
#include <Rdefines.h>
#include <Rinterface.h>

#include <R_ext/Rdynload.h>
#include <R_ext/Visibility.h>

#include "cudaLogReg.h"
#include "entrypoints-model_logreg_C.h"
#include "entrypoints-trainer_gd_C.h"

#include "entrypoints-model_logreg_cuda.h"
#include "entrypoints-trainer_gd_cuda.h"

// Registering .Call entry points. We use native registration to lower call
// overhead as much as possible.
#define CALLDEF(name, n) {#name, (DL_FUNC) &name, n} // '#' is the preprocessor
// tringizing operator
static const R_CallMethodDef callMethods[] = {
  // C functions
  CALLDEF(get_condprob_logreg_, 4),
  CALLDEF(get_cost_logreg_, 4),
  CALLDEF(get_grad_logreg_, 4),
  CALLDEF(train_gd_, 8),
  // Cuda functions
  CALLDEF(get_condprob_logreg_cuda, 4),
  CALLDEF(get_cost_logreg_cuda, 4),
  CALLDEF(get_grad_logreg_cuda, 4),
  CALLDEF(train_gd_cuda, 8),
  {NULL, NULL, 0}
};

// Controlling visibility with 'attribute_visible'. Note that the mechanism is
// not avaiable on windows. See: ./cudaLogReg-win.def.
void attribute_visible R_init_cudaLogReg(DllInfo * info) {
  R_registerRoutines(info, NULL, callMethods, NULL, NULL);
  // R_useDynamicSymbols(info, FALSE);
  // R_forceSymbols(info, TRUE);
}
