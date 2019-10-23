#' Forward-backward algorithm
#'
#' Forward-backward algorithm using the scaling technique.
#' That's more stable (and maybe even faster) than the method with the logarithm.
#' Warning: this function overwrites the lliks matrix.
#' @param clzz binary matrix of histone marks
#' @param emissionPars matrix with emission probabilities for each datapoint and each state.
#' @param initP matrix of initial probabilities: each column corresponds to a sequence
#' @param transitionProbs transition matrix (rows are previous state, columns are next state)
#' Columns are datapoints and rows are states.
#' @param seqlens named list containing the length of chromosomes/contigs.
#' @param updateEmission update emission vector with length is same as the number of states, this vector contains boolean values (TRUE or FALSE)
#' @param updateTransition update emission matrix with same dimensions as emission probability matrix containing boolean value (TRUE or FALSE)
#' @param maxiter number of iterations.
#' @param nthreads number of threads used. Sequences of observations are
#' processed independently by different threads (if \code{length(seqlens) > 1}).
#' @return a list with the following arguments:
#'    \item{post}{posterior probability of being in a certain state for a certain datapoint.
#'     Same matrix used as input argument.}
#'    \item{tot_llik}{total log-likelihood of the data given the hmm model.}
#'    \item{new_trans}{update for the transition probabilities (it is already normalized).}
#'    \item{epar}{estimated parameters}
#' @export
EPI.genefinder <- function(
  clzz,
  emissionPars,
  initP = NULL,
  transitionProbs,
  seqlens = nrow(clzz),
  updateEmission = NULL,
  updateTransition = NULL,
  maxiter = 10,
  tol = 1e-4,
  nthreads = 1
){

  nStates <- ncol(emissionPars)
  nloci <- nrow(clzz);
  if (is.null(updateEmission)){
    updateEmission <- rep(TRUE, nStates)
  }
  if (is.null(updateTransition)){
    updateTransition <- matrix(TRUE, nStates, nStates)
  }



  ## map unique observation vectors to positions
  index <- index(clzz)



  ## allocate memory for the posteriors
  posteriors <- matrix(0, nrow = nStates, ncol = nloci)
  if (is.null(initP)){
    initP <- matrix(1 / nStates, nrow = nStates, ncol = length(seqlens))
  }
  trans = transitionProbs
  epar = emissionPars
  converged = FALSE

  ## baum-welch
  for (iter in 1:maxiter){
    ## generate lookup table for the emission probabilities
    emissionProb <- emission_table(index, epar)

    res <- forward_backward_from_index(initP, trans, index$map, emissionProb,
              seqlens, posteriors, nthreads)

    new_loglik <- res$tot_llik
    new_trans <- res$new_trans
    new_initP <- res$new_initP
    if (iter > 1){
      if (abs(new_loglik - loglik) < tol){
        converged <- TRUE
      } else if (new_loglik < loglik){
        warning(paste0("decrease in log-likelihood at iteration ", iter))
      }
    }
    # check if the user pressed CTRL-C
    #checkInterrupt()
    cat("Iteration:", iter, "log-likelihood:", new_loglik, "\n", sep="\t")


    # update emission parameter
    new_epar <- emission_update(index, posteriors, seqlens, nthreads)
    for (i in 1:nStates){
      if (updateEmission[i] == TRUE){
        epar[, i] = new_epar[, i]
      }
    }

    # update transition probababilities
    new_trans2 = new_trans
    new_trans = new_trans / rowSums(new_trans)
    trans[updateTransition] <- new_trans[updateTransition]
    ##cat(iter, " rowsums: ", rowSums(trans), "\n")
    ##cat(iter, " colsums: ", colSums(trans), "\n")

    # update log likelihood
    loglik <- new_loglik

    # update initial probabilities
    initP <- new_initP
    if (converged == TRUE) break
  }
  emissionProb <- emission_table(index, epar)
  V = viterbi_from_index(initP, trans, index$map, emissionProb, seqlens)




  list(initP = initP, trans = trans, trans2 = new_trans2, epar = epar, loglik = loglik, viterbi = V, post = posteriors)
}
