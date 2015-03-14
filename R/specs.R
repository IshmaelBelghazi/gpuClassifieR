## * Logistic regression specification class

##' Logistic regression model specification
##'
##' @title Logistic regression model specification
##' @param weights_init initial weights.
##' @return model.spec object
##' @author Mohamed Ishmael Diwan Belghazi
LogReg <- function(weights_init=NULL) {

    if(is.null(weights_init))
        stop("initial weights specification is mandatory to set problem dimensions")

    n_classes <- NCOL(weights_init)
    n_feats <- NROW(weights_init)
    ## structure and return
    return(structure(
        list(
            weights=weights_init,
            n_classes=n_classes,
            n_feats=n_feats,
            cond_prob_fun=get_condprob_logreg,
            cost_fun=get_cost_logreg,
            grad_fun=get_grad_logreg,
            final_grad=NULL,
            final_iter=NULL
        ),
        class=c("model.spec")
    ))
}

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
