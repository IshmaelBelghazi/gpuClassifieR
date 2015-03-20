## * Generics

## ** Weights getter
##' @export
coef.model.spec <- function(object, ...) object$weights

## ** Gradient getter

##' Computes model gradient
##'
##' @param object Linear classifier specification object.
##' @param feats Numeric Matrix of features. Follows the usual convention of having
##' one example per row. For a model with M features and dataset with N
##' examples the matrix should be N \eqn{\times} K
##' @param targets Numeric matrix of one hot encoded target. Follows the usual
##' convention of having one target per row. For a model with K classes and a
##' dataset with N examples the matrix should be N \eqn{\times} K
##' @param decay Numeric scalar. Tikhonov regularization coefficient (weight decay). Should be
##' a non-negative real number.
##' @param backend Computation back-end ('R', 'C', or 'CUDA')
##' @param ... other arguments passed to specific methods
##' @return Numeric Matrix of gradients. One for each class. Gradient are arrayed in
##' columns. For a model with M features and K classes the matrix should be M
##' \eqn{\times} K
##' @export
##' @author Mohamed Ishmael Diwan Belghazi
##' @examples
##'
##' # Generate random initial weights
##' w_init <- matrix(rnorm(784 * 10), 784, 10)
##' # construct model
##' linear_classifier <- Classifier(weights_init=w_init)
##' # Fetch training variables
##' feats <- mini_mnist$train$images
##' targets <- mini_mnist$train$labels
##' # Set decay coefficient
##' decay <- 0.01
##' # compute gradient at the training set using the three back-ends
##' gradient_R <- get_grad(linear_classifier, feats, targets, decay, 'R')
##' gradient_C <- get_grad(linear_classifier, feats, targets, decay, 'C')
get_grad <- function(object, feats, targets, decay=NULL, backend='R', ...){

    UseMethod('get_grad')
}
##' @export
get_grad.default <- function(object, feats, targets, decay=NULL, backend='R', ...) {
    stop('unknown object class')
}
##' @describeIn get_grad Computes model gradient for linear classifier
##' specification objects
##' @export
get_grad.model.spec <- function(object, feats, targets, decay=NULL, backend='R', ...) {
    object$grad_fun(feats, coef(object), targets, decay, backend)
}

## ** Cost getters

##' Computes model cost
##'
##' @return cost Numeric scalar. model cost
##' @inheritParams get_grad
##' @author Mohamed Ishmael Diwan Belghazi
##' @export
##' @examples
##'
##' # Generate random initial weights
##' w_init <- matrix(rnorm(784 * 10), 784, 10)
##' # construct model
##' linear_classifier <- Classifier(weights_init=w_init)
##' # Fetch training variables
##' feats <- mini_mnist$train$images
##' targets <- mini_mnist$train$labels
##' # Set decay coefficient
##' decay <- 0.01
##' # compute cost of the training set using the three back-ends
##' cost_R <- get_cost(linear_classifier, feats, targets, decay, 'R')
##' cost_C <- get_cost(linear_classifier, feats, targets, decay, 'C')
get_cost <- function(object, feats, targets, decay=NULL, backend='R', ...){
    UseMethod('get_cost')
}
##' @export
get_cost.default <-  function(object, feats, targets,
                              decay=NULL, backend='R', ...) {
    stop('unknown object class')
}
##' @describeIn get_cost Computers model cost for linear classification
##' specification objects
##' @export
get_cost.model.spec <- function(object, feats, targets,
                                decay=NULL, backend='R', ...) {
    object$cost_fun(feats, coef(object), targets, decay, backend)
}
## ** Class conditional probabilities getter

##' Computes model class conditional probabilities
##'
##'
##' @param normalize Logical scalar. If TRUE return normalized probabilities
##' @param log_domain Logical scalar. If TRUE returns log probabilities.
##' @return Numeric Matrix. Class conditional probabilities matrix. Each row corresponds to the
##' probabilities of one example. For model with K classes and a dataset of N
##' examples the returned matrix should be N \eqn{\times} K
##' @author Mohamed Ishmael Diwan Belghazi
##' @inheritParams get_grad
##' @export
##' @examples
##'
##' # Generate random initial weights
##' w_init <- matrix(rnorm(784 * 10), 784, 10)
##' # construct model
##' linear_classifier <- Classifier(weights_init=w_init)
##' # Fetch training variables
##' feats <- mini_mnist$train$images
##' targets <- mini_mnist$train$labels
##' # Set decay coefficient
##' decay <- 0.01
##' # compute log probabilities of the training set using the three back-ends
##' log_prob_R <- get_prob(linear_classifier, feats, TRUE, TRUE, 'R')
##' log_prob_C <- get_prob(linear_classifier, feats, TRUE, TRUE, 'C')
get_prob <- function(object, feats, normalize=TRUE,
                     log_domain=FALSE, backend='R', ...) {
    UseMethod('get_prob')
}
##' @export
get_prob.default <- function(object, feats, normalize=TRUE,
                     log_domain=FALSE, backend='R', ...) {
    stop('unknown object class')
}

##' @describeIn get_prob Computes model class conditional probabilities for
##' linear classifier specification objects
##' @export
get_prob.model.spec <- function(object, feats, normalize=TRUE,
                                log_domain=FALSE, backend='R', ...) {
    object$cond_prob_fun(feats, coef(object), normalize=normalize,
                         log_domain=log_domain, backend=backend)
}

## ** Predict
##' @export
predict.model.spec <- function(object, newfeats, backend='R', ...) {

    if (NCOL(newfeats) != object$n_feats)
        stop('dimension mismatch')

    condprob <- get_prob(object, newfeats, normalize=TRUE, log_domain=FALSE,
                          backend=backend)
    .predict_class(condprob)
}

## ** Misclassification rate

##' Computes model misclassification rate
##'
##' @param object Linear classifier specification object.
##' @param feats Numeric matrix. Features to predict. Dimensionality
##' should be consistent with that of the model.
##' @param targets Numeric matrix. TRUE targets to compare prediction against.
##' Dimensionality
##' should be consistent with that of the model. Should always be one-hot encoded.
##' @param backend Computation back-end ('R', 'C', or 'CUDA')
##' @return Numeric scalar. Misclassification rate. Percentage of wrong predictions.
##' @author Mohamed Ishmael Diwan Belghazi
##' @export
get_error <- function(object, feats, targets, backend='R') UseMethod('get_error')
##' @export
get_error.default <- function(object, feats, targets, backend='R') {
    stop('Unknown object class')
}
##' @describeIn get_error Computes model misclassification rate for linear
##' classifier model object
get_error.model.spec <- function(object, feats, targets, backend='R') {

    .get_error(predict(object, feats, backend), targets, backend)

}
## * Train

##' Gradient Descent trainer
##'
##' @param step_size Numeric scalar. Initial, gradient descent step size. If
##' the current cost is worst that the previous, model parameters will not be
##' update and the step_size will be divided by 2. If the current cost not
##' the worse than the previous, parameters will be updated and the step_size
##' will be multiplied by 1.1 .
##' @param max_iter Numeric Scalar. Maximum number of iterations allowed. Inf
##' will keep the training going until the number of the current gradient
##' Frobenius norm is less than tol.
##' @param verbose Logical scalar. If TRUE, iteration number, current cost and
##' step_size will be printed to the standard output.
##' @param tol Numeric scalar. Assumes convergence when the gradient Frobenius
##' norm is less than tol.
##' @return model.spec object
##' @author Mohamed Ishmael Diwan Belghazi
##' @inheritParams get_grad
##' @export
##' @examples
##'
##' # Train model on single example of MIST and a single iteration
##' # Generate random initial weights
##' w_init <- matrix(rnorm(784 * 10), 784, 10)
##' # construct model
##' linear_classifier <- Classifier(weights_init=w_init)
##' # Fetch training variables
##' feats <- mini_mnist$train$images[1, , drop=FALSE]
##' targets <- mini_mnist$train$labels[1, , drop=FALSE]
##' # Specifying training parameters
##' step_size <- 0.01
##' decay <- 0.0001
##' max_iter <- 1
##' tol <- 1e-6
##' verbose <- FALSE
##' # Train model one a single example using the three back-ends
##' linear_classifier_R <- train(linear_classifier, feats, targets, decay,
##' step_size, max_iter, verbose, tol, backend='R')
##' linear_classifier_C <- train(linear_classifier, feats, targets, decay,
##' step_size, max_iter, verbose, tol, backend='C')
train <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                  verbose=FALSE, tol=1e-6, backend='R', ...) {
    UseMethod('train')
}
##' @export
train.default <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                          verbose=FALSE, tol=1e-6, backend='R', ...) {
    stop('unkown object class')
}
##' @describeIn train Gradient Descent trainer for linear classifier specification model objects.
##' @export
train.model.spec <- function(object, feats, targets, decay=NULL, step_size=NULL, max_iter=NULL,
                             verbose=FALSE, tol=1e-6, backend='R', ...) {

    .train_gd(object, feats, targets, decay=decay, step_size=step_size, max_iter=max_iter,
              verbose=verbose, tol=tol, backend=backend, ...)
}
