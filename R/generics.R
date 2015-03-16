## * Generics

## ** Weights getter

coef.model.spec <- function(object, ...) object$weights

## ** Gradient getter

##' Model gradient accessor generic
##'
##' @param object object of class model.spec
##' @param feats Matrix of features (N X M)
##' @param targets matrix of one hot encoded target (N X K)
##' @param decay L2 regularisation decay
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @param ... Pass through parameters
##' @return gradient (M X K)
##' @author Mohamed Ishmael Diwan Belghazi
get_grad <- function(object, feats, targets, decay=NULL, backend="R", ...){

    UseMethod("get_grad")
}

get_grad.default <-  function(object, feats, targets, decay=NULL, backend="R", ...) {
    stop("unknown object class")
}
##' @describeIn get_grad Model gradient accessor
get_grad.model.spec <- function(object, feats, targets, decay=NULL, backend="R", ...) {
    object$grad_fun(feats, coef(object), targets, decay, backend)
}

## ** Cost getter

##' Model cost accessor generic
##'
##' @param object object of class model.spec
##' @param feats Matrix of features (N X M)
##' @param targets matrix of one hot encoded target (N X K)
##' @param decay L2 regularisation decay
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @param ... Pass through parameters
##' @return cost (M X K)
##' @author Mohamed Ishmael Diwan Belghazi
get_cost <- function(object, feats, targets, decay=NULL, backend="R", ...){

    UseMethod("get_cost")
}

get_cost.default <-  function(object, feats, targets, decay=NULL, backend="R", ...) {
    stop("unknown object class")
}
##' @describeIn get_cost Model cost accessor
get_cost.model.spec <- function(object, feats, targets, decay=NULL, backend="R", ...) {
    object$cost_fun(feats, coef(object), targets, decay, backend)
}
## ** Class conditional probabilities getter

##' Class conditional probabilities getter generic
##'
##'
##' @param object Object of class model.spec
##' @param feats features (N X M)
##' @param normalize normalize probabilities
##' @param log_domain return log probabilities
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @param ... Pass through parameters
##' @return Class conditional probabilities
##' @author Mohamed Ishmael Diwan Belghazi
get_prob <- function(object, feats, normalize=TRUE,
                     log_domain=FALSE, backend="R", ...) {
    UseMethod("get_prob")
}

get_prob.default <- function(object, feats, normalize=TRUE,
                     log_domain=FALSE, backend="R", ...) {
    stop("unknown object class")
}

##' @describeIn get_prob Class conditional probabilities getter
get_prob.model.spec <- function(object, feats, normalize=TRUE,
                                log_domain=FALSE, backend="R", ...) {
    object$cond_prob_fun(feats, coef(object), normalize=normalize,
                         log_domain=log_domain, backend=backend)
}

## ** Predict
predict.model.spec <- function(object, newfeats, backend="R", ...) {

    if (NCOL(newfeats) != object$n_feats)
        stop("dimension mismatch")

    condprob <- get_prob(object, newfeats, normalize=TRUE, log_domain=FALSE,
                          backend=backend)
    .predict_class(condprob)
}

## ** Misclassification rate

##' Get model misclassification rate generic
##'
##' @param object Object of class model.spec
##' @param feats features to predict
##' @param targets Real targets
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @return Misclassification rate
##' @author Mohamed Ishmael Diwan Belghazi
get_error <- function(object, feats, targets, backend="R") UseMethod("get_error")

get_error.default <- function(object, feats, targets, backend="R") {
    stop("Unknown object class")
}
##' @describeIn get_error Get model miscclassification rate
get_error.model.spec <- function(object, feats, targets, backend="R") {

    .get_error(predict(object, feats, backend), targets, backend)

}
## * Train
##' Gradient Descent trainer generic
##'
##' @param object model.spec object
##' @param feats features (N X M)
##' @param targets targets (N X K)
##' @param decay L2 regularisation coefficient
##' @param step_size gradient descent step size
##' @param max_iter Maximum iteration
##' @param verbose prints output
##' @param tol assume convergence when norm of gradient is smaller than tol
##' @param backend Computation backend ("R", "C", or "CUDA")
##' @param ... any other passthrough parameters
##' @return model.spec object
##' @author Mohamed Ishmael Diwan Belghazi
train <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                  verbose=FALSE, tol=1e-6, backend="R", ...) {
    UseMethod("train")
}

train.default <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                          verbose=FALSE, tol=1e-6, backend="R", ...) {
    stop("unkown object class")
}
##' @describeIn train Gradient descent trainer
train.model.spec <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                             verbose=FALSE, tol=1e-6, backend="R", ...) {

    .train_gd(object, feats, targets, decay=decay, step_size=step_size, max_iter=max_iter,
              verbose=verbose, tol=tol, backend=backend, ...)
}
