## ----, echo=FALSE, results="hide"----------------------------------------
devtools::load_all(".")

## ----, results="hide"----------------------------------------------------
data(mini_mnist)
train_feats <- mini_mnist$train$images
train_targets <- mini_mnist$train$labels
test_feats <- mini_mnist$test$images
test_targets <- mini_mnist$test$labels
M <- NCOL(train_feats)  ## Number of features
K <- NCOL(train_targets)  ## Number of targets.

## ----, results="hide"----------------------------------------------------
w_init <- mat.or.vec(M, K)
models <- list(R=w_init, C=w_init, CUDA=w_init)
models <- lapply(models, Classifier)

## ----, results="hide"----------------------------------------------------
step_size <- 0.01  ## Initial step size
decay <- 0.0  ## Tikhonov regularization coefficients
max_iter <- 10000  ## Maximum number of iterations
verbose <- FALSE  ## TRUE to monitor the evoluation of the step size, cost and
## iterations
tol <- 1e-6  ## The training will consider to have converged if the Frobenius
## norm of the gradient is less than tol

## ----, results='hide'----------------------------------------------------
models <- mapply(function(X, Y) train(X, train_feats,
                                      train_targets, decay,
                                      step_size, max_iter,
                                      verbose, tol, Y), X=models, Y=names(models),
                 SIMPLIFY=FALSE)

## ----grad, echo=FALSE----------------------------------------------------
knitr::kable(as.data.frame(lapply(models, function(X) norm(X$final_grad, "F"))), caption="Gradients Euclidean Norm")

## ----is, echo=FALSE, results='asis'--------------------------------------
knitr::kable(as.data.frame(lapply(X=models, function(X) get_error(X, train_feats, train_targets))), caption="In Sample Misclassification Rate")

## ----oos, echo=FALSE, results='asis'-------------------------------------
knitr::kable(as.data.frame(lapply(models, function(X) get_error(X, test_feats, test_targets))), caption="Out of sample Misclassification Rate")

## ----, results='hide'----------------------------------------------------
benchmark_fun <- function(n_sample, feats, targets) {
    w_init <- mat.or.vec(NCOL(feats), NCOL(targets))
    models <- lapply(list(R=w_init, C=w_init, CUDA=w_init), Classifier)
    time <- mapply(function(X, Y) system.time(train(X, feats[1:n_sample,, drop=FALSE],
                                                    targets[1:n_sample,, drop=FALSE],
                                                    0.0, 0.01, 1000,
                                                    FALSE, -1, Y))[['elapsed']]
                 , X=models, Y=names(models),
                   SIMPLIFY=FALSE)

}

## ----benchplot, echo=FALSE, results="hide", fig.width=14, fig.height=8----
feats <- rbind(train_feats, test_feats)
targets <- rbind(train_targets, test_targets)

## Plotting
sample_points <- c(1, seq(100, NROW(feats), 100))
times <- t(sapply(X=sample_points, function(X) benchmark_fun(X, feats, targets)))
times <- cbind(stack(as.data.frame(times)), rep(sample_points, 3))
colnames(times) <- c("time", "backend", "size")
require(ggplot2)
time_plot <- ggplot(times, aes(x=size, y=time, group=backend, colour=backend)) + geom_line() + geom_point()
time_plot <- time_plot + xlab("sample size") + ylab("Elapsed time in seconds") + ggtitle("gpuClassifieR implementations benchmark")
plot(time_plot)  



