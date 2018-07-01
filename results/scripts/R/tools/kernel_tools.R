library (pracma)

# Function which imitats the behaivor of 'ones' in Matlab. This
# function is copied form the R package 'Matrix'.
ones <- function (n, m = n) {
    stopifnot (is.numeric(n), length (n) == 1,
               is.numeric(m), length (m) == 1)
    n <- floor (n)
    m <- floor (m)
    if ((n <= 0) || (m <= 0))
        return (matrix (1, nrow = 0, ncol = 0))
    else
        return (matrix (1, nrow = n, ncol = m))
}

#' Build the kernel matrix between two sets of feature vectors.
#'
#' @param X1 Matrix of size n1 x d
#' @param X2 Matrix of size n2 x d (default \code{X2 = NULL} --> \code{X2 = X1})
#' @param kernel.opt List of parameters to specify the kernel and its
#'   parameters. (default \code{list (type = "linear")})
#' @return Kernel-matrix of size n1 x n2
build_kernel <- function (X1, X2 = NULL, kernel.opt = list (type = "linear")) {
    stopifnot (is.list (kernel.opt))
    stopifnot ("type" %in% names (kernel.opt))
    stopifnot (kernel.opt$type %in% c("linear",
                                      "cosine",
                                      "tanimoto",
                                      "gaussian",
                                      "tanimoto_gaussian"))

    if (is.null (X2))
        X2 <- X1

    stopifnot (is.matrix (X1), is.matrix (X2))

    if (any (dim (X1) == 0) || any (dim (X2) == 0))
        return (matrix (1, ncol = 0, nrow = 0))

    # Feature dimensions of the two sets must match
    stopifnot (ncol (X1) == ncol (X2))

    if      (kernel.opt$type == "linear") {
        KX <- X1 %*% t(X2)
    }

    else if (kernel.opt$type == "cosine") {
        D1 <- sqrt (rowSums (X1 * X1))
        D2 <- sqrt (rowSums (X2 * X2))

        D1[D1 == 0] <- 1
        D2[D2 == 0] <- 1

        X1 <- X1 / matrix (D1, nrow = nrow (X1), ncol = ncol (X1))
        X2 <- X2 / matrix (D2, nrow = nrow (X2), ncol = ncol (X2))

        KX <- X1 %*% t(X2)
    }

    else if (kernel.opt$type == "tanimoto") {
        n1 <- nrow (X1)
        n2 <- nrow (X2)

        D1 <- rowSums (X1 * X1) %*% ones (1, n2)
        D2 <- ones (n1, 1) %*% rowSums (X2 * X2)

        P <- X1 %*% t(X2)

        KX <- P / (D1 + D2 - P)

        # If A and B are empty than we define the similarity as 1. See
        # wikipedia.
        D1_and_D2_empty <- (D1 == 0) & (D2 == 0)
        KX[D1_and_D2_empty] <- 1
    }

    else if (kernel.opt$type == "gaussian") {
        stopifnot ("gamma" %in% names (kernel.opt))

        n1 <- nrow (X1) 
        n2 <- nrow (X2)

        D1 <- rowSums (X1 * X1)
        D2 <- rowSums (X2 * X2)

        Dis <- D1 %*% ones (1, n2) + ones (n1, 1) %*% D2 - 2 * X1 %*% t(X2)
        KX  <- exp (- kernel.opt$gamma * Dis)
    }

    else if (kernel.opt$type == "tanimoto_gaussian") {
        stopifnot ("gamma" %in% names (kernel.opt))

        n1 <- nrow (X1)
        n2 <- nrow (X2)

        D1 <- rowSums (X1 * X1)
        D2 <- rowSums (X2 * X2)

        D1[D1 == 0] <- 1
        D2[D2 == 0] <- 1

        P  <- X1 %*% t(X2)
        P1 <- X1 %*% t(X1)
        P2 <- X2 %*% t(X2)

        K  <- P  / (D1 %*% ones (1, n2) + ones (n1, 1) %*% D2 -  P)
        K1 <- P1 / (D1 %*% ones (1, n1) + ones (n1, 1) %*% D1 - P1)
        K2 <- P2 / (D2 %*% ones (1, n2) + ones (n2, 1) %*% D2 - P2)

        Dis <- diag (K1) %*% ones (1, n2) + ones (n1, 1) %*% diag (K2) - 2 * K
        KX  <- exp (- kernel.opt$gamma * Dis)
    }

    return (KX)
}

#' Finds the gamma parameter for the Gaussian-kernel maximizing the
#' entropy of the kernel-matrix.
#'
#' @param K n x n kernel matrix
#' @return gamma maximizing the entropy of the of Gaussian-kernel based
#'   on the given kernel-matrix.
select_gamma_entropy <- function (K) {
    stopifnot (is.matrix (K))
    stopifnot (nrow (K) == ncol (K))
    # stopifnot (all.equal(diag (K), rep(1, nrow (K))))

    n <- nrow (K)
    D <- diag (K)

    Kdist <- D %*% ones (1, n) + ones (n, 1) %*% D - 2 * K
    gamma <- (n * (n - 1)) / (2 * sum (Kdist[upper.tri (Kdist)]))

    return (gamma)
}

#' Centering and normalization of a kernel-matrix
#'
#' @param K n x n kernel-matrix
#' @param train_set indices of the columns and rows belonging to the
#'   training set
#' @param test_set indicies of the columns and rows belonging to the test
#'   set
#' @param center boolean indicating whether the output matrix should be
#'   centered
#' @param normalize boolean indicating whether the output matrix should
#'   be normalized
#' @return List containing the following elements:
#' \itemize{
#'   \item KX_train: \code{KX[train_set, train_set]} centered and normalized as specified.
#'   \item KX_train_test: \code{KX[train_set, test_set]} centered and normalized as specified.
#' }
transform_kernel_matrix <- function (K, train_set, test_set,
                                       center = TRUE, normalize = TRUE) {
    stopifnot (is.matrix (K))
    stopifnot (isSymmetric.matrix (K))

    # How should train_set == NULL be handled?

    K_train      <- K[train_set, train_set, drop = FALSE]
    K_test       <- K[ test_set,  test_set, drop = FALSE]
    K_train_test <- K[train_set,  test_set, drop = FALSE]

    if (center) {
        mean_K_train      <- colMeans (K_train)
        mean_K_train_test <- colMeans (K_train_test)

        K_train      <- center (K_train, mean_K_train)
        K_test       <- center (K_test, mean_K_train,
                                matrix (mean_K_train_test, ncol = 1),
                                mean_K_train_test)
        K_train_test <- center (K_train_test, mean_K_train,
                                matrix (mean_K_train, ncol = 1),
                                mean_K_train_test)
    }
    if (normalize) {
        D_K_train    <- diag (K_train)
        K_train      <- K_train / sqrt (D_K_train %*% t(D_K_train))
        K_train_test <- K_train_test / sqrt (D_K_train %*% t(diag (K_test)))
    }

    return (list (K_train = K_train, K_train_test = K_train_test))
}

#' Centering kernel-matrix without explicit feature space relative to the
#' a given mean.
#'
#' @param K n1 x n2 kernel-matrix
#' @param mean_K Mean for the centering.
#' @param mean_K_rows Row-means
#' @param mean_K_cols Col-means
#' @return Centered kernel-matrix
center <- function (K, mean_K = colMeans (K), mean_K_rows = NULL, mean_K_cols = NULL) {
    n1 <- nrow (K)
    n2 <- ncol (K)

    if (is.null (mean_K_rows) || is.null (mean_K_cols)) {
        mean_K_rows <- matrix (mean_K, ncol = 1)
        mean_K_cols <- matrix (mean_K, nrow = 1)
    }

    K_c <- K - matrix (mean_K_cols, nrow = n1, ncol = length (mean_K_cols), byrow = TRUE) -
        matrix (mean_K_rows, nrow = length (mean_K_rows), ncol = n2) +
        mean (mean_K)

    return (K_c)
}

normalize_kernel <- function (K) {
    D_K <- diag (K)
    return (K / sqrt (D_K %*% t(D_K)))
}

#' Function to combine kernel matrices using MKL.
#'
#' @param K_list List of length k containing the kernels to combine.
#' @param train_set Indices of the columns and rows belonging to the
#'   training set
#' @param test_set Indicies of the columns and rows belonging to the test
#'   set
#' @param weights Vector of the length k containing the weights for each
#'   kernel
#' @param center Boolean indicating whether the output matrix should be
#'   centered
#' @param normalize Boolean indicating whether the output matrix should
#'   be normalized
#' @return List containing the combined kernels
#' \itemize{
#'   \item K_train: \code{K[train_set, train_set]}
#'   \item K_train_test: \code{K[train_set, test_set]}
#' }
mkl_combine_train_test <- function (K_list, train_set, test_set,
                                    weights = NULL,
                                    center = TRUE, normalize = TRUE) {
    stopifnot (is.list (K_list))

    n_k <- length (K_list)
    stopifnot (n_k > 0)
    if (is.null (weights))
        weights <- rep (1 / n_k, n_k)

    n_train <- length (train_set)
    n_test  <- length (test_set)

    K_mkl_train      <- matrix (0, nrow = n_train, ncol = n_train)
    K_mkl_train_test <- matrix (0, nrow = n_train, ncol = n_test)

    for (k_idx in 1:n_k) {
        K_list_k_nc <- transform_kernel_matrix (K_list[[k_idx]], train_set, test_set,
                                                center, normalize)
        K_mkl_train      <- K_mkl_train      + weights[k_idx] * K_list_k_nc$K_train
        K_mkl_train_test <- K_mkl_train_test + weights[k_idx] * K_list_k_nc$K_train_test
    }

    return (list (K_train = K_mkl_train, K_train_test = K_mkl_train_test))
}
