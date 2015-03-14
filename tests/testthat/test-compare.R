context("Implementations consistency")
############################
## Setting test variables ##
############################
## Defining dimensions
N <- 5
M <- 3
K <- 2
## Defining Matrices
weights <- matrix(rnorm(M * K), nrow=M)
feats <- matrix(rnorm(N * M), nrow=N)
targets <- matrix(rmultinom(N,
                      size = 1,
                      (function(feats) feats/sum(feats))(runif(K))),
            nrow = N, byrow = TRUE)
storage.mode(targets) <- "double"

tol <- 1e-6
decay <- runif(1, min=tol, max=1)
##----------------------------------------
test_that("Conditional probabilties R/C/CUDA", {
    normalize <- TRUE
    log_domain <- as.logical(rbinom(1, 1, 0.5))

    expect_equal(get_condprob_logreg(feats, weights, normalize, log_domain, backend="R"),
                 get_condprob_logreg(feats, weights, normalize, log_domain, backend="C"),
                 tolerance=tol,
                 scale=1)
    expect_equal(get_condprob_logreg(feats, weights, normalize, log_domain, backend="R"),
                 get_condprob_logreg(feats, weights, normalize, log_domain, backend="CUDA"),
                 tolerance=tol,
                 scale=1)
})

##---------------------------------------
test_that("Cross-entropies R/C/CUDA", {
    expect_equal(get_cost_logreg(feats, weights, targets, decay, backend="R"),
                 get_cost_logreg(feats, weights, targets, decay, backend="C"),
                 tolerance=tol,
                 scale=1)
    expect_equal(get_cost_logreg(feats, weights, targets, decay, backend="R"),
                 get_cost_logreg(feats, weights, targets, decay, backend="CUDA"),
                 tolerance=tol,
                 scale=1)
})
##---------------------------------------
test_that("gradient R/C/CUDA", {
    expect_equal(get_grad_logreg(feats, weights, targets, decay, backend="R"),
                 get_grad_logreg(feats, weights, targets, decay, backend="C"),
                 tolerance=tol,
                 scale=1)
    expect_equal(get_grad_logreg(feats, weights, targets, decay, backend="R"),
                 get_grad_logreg(feats, weights, targets, decay, backend="CUDA"),
                 tolerance=tol,
                 scale=1)
})
