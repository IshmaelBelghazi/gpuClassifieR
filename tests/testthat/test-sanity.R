context("Basic sanity checks")
N <- 2
K <- 3
M <- 4
targets <- matrix(c(1, 0, 0, 0, 0, 1),
                  nrow=2, byrow=TRUE)
weights <- matrix(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                  nrow = 4, byrow = TRUE)
feats <- matrix(c(13, 14, 15, 16, 17, 18, 19, 20),
            nrow = 2, byrow = TRUE)
feats_zero <- mat.or.vec(N, M)
weights_zero <- mat.or.vec(M, K)
decay <- 0.0

activation <- feats %*% weights
cond_prob <- t(apply(activation, 1, function(feats) feats - (logsumexp_R(feats))))
cond_prob <- exp(cond_prob)
cost <- -mean(targets * log(cond_prob)) + 0.5 * decay * sum(weights^2)
grad <- t(feats) %*% (cond_prob - targets) + decay * weights
tol <- 1e-6
## --------------------------------------------------------------
test_that("Conditional probability R backend sanity checks", {
    expect_true(all(get_condprob_logreg(feats, weights, backend="R") >= -tol))
    expect_equal(rowSums(get_condprob_logreg(feats, weights,
                                             backend="R")), rep(1.0, N),
                 tolerance=tol, scale=1)
    expect_equal(get_condprob_logreg(feats_zero, weights_zero,
                                     backend="R"),
                 matrix(rep(1/K, N * K), nrow=N),
                 tolerance=tol,
                 scale=1)
})
## --------------------------------------------------------------
test_that("Conditional probability C backend sanity checks", {
    expect_true(all(get_condprob_logreg(feats, weights, backend="C") >= -tol))
    expect_equal(rowSums(get_condprob_logreg(feats, weights,
                                             backend="C")), rep(1.0, N),
                 tolerance=tol, scale=1)
    expect_equal(get_condprob_logreg(feats_zero, weights_zero,
                                     backend="C"),
                 matrix(rep(1/K, N * K), nrow=N),
                 tolerance=tol,
                 scale=1)
})
## ---------------------------------------------------------------
test_that("Conditional probability CUDA backend sanity checks", {
    expect_true(all(get_condprob_logreg(feats, weights, backend="CUDA") >= -tol))
    expect_equal(rowSums(get_condprob_logreg(feats, weights,
                                             backend="CUDA")), rep(1.0, N),
                 tolerance=tol, scale=1)
    expect_equal(get_condprob_logreg(feats_zero, weights_zero,
                                     backend="CUDA"),
                 matrix(rep(1/K, N * K), nrow=N),
                 tolerance=tol,
                 scale=1)
})

## ---------------------------------------------------------------
test_that("Cost function R backend sanity checks", {
    expect_true(get_cost_logreg(feats, weights, targets,
                                decay, backend="R") >= 0, TRUE)
    expect_equal(get_cost_logreg(feats, weights, targets,
                                 decay, backend="R"), cost,
                 tolerance=tol, scale=1)
})
## ---------------------------------------------------------------
test_that("Cost function C backend sanity checks", {
    expect_true(get_cost_logreg(feats, weights, targets,
                                decay, backend="C") >= 0, TRUE)
    expect_equal(get_cost_logreg(feats, weights, targets,
                                 decay, backend="C"), cost,
                 tolerance=tol, scale=1)
})
## ----------------------------------------------------------------
test_that("Cost function CUDA backend sanity checks", {
    expect_true(get_cost_logreg(feats, weights, targets,
                                decay, backend="CUDA") >= 0, TRUE)
    expect_equal(get_cost_logreg(feats, weights, targets,
                                 decay, backend="CUDA"), cost,
                 tolerance=tol, scale=1)
})
## -----------------------------------------------------------------
test_that("Gradient function R backend sanity checks", {
    expect_equal(get_grad_logreg(feats_zero, weights, targets,
                                 decay, backend="R"), mat.or.vec(M, K),
                 tolerance=tol,
                 scale=1)
    expect_equal(get_grad_logreg(feats, weights, targets,
                                 decay, backend="R"), grad,
                 tolerance=tol,
                 scale=1)
})
## -------------------------------------------------------------------
test_that("Gradient function C backend sanity checks", {
    expect_equal(get_grad_logreg(feats_zero, weights, targets,
                                 decay, backend="C"), mat.or.vec(M, K),
                 tolerance=tol,
                 scale=1)
    expect_equal(get_grad_logreg(feats, weights, targets,
                                 decay, backend="C"), grad,
                 tolerance=tol,
                 scale=1)
})
## ---------------------------------------------------------------------
test_that("Gradient function CUDA backend sanity checks", {
    expect_equal(get_grad_logreg(feats_zero, weights, targets,
                                 decay, backend="CUDA"), mat.or.vec(M, K),
                 tolerance=tol,
                 scale=1)
    expect_equal(get_grad_logreg(feats, weights, targets,
                                 decay, backend="CUDA"), grad,
                 tolerance=tol,
                 scale=1)
})
