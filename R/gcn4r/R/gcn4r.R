library(reticulate)

#' Install GCN4R Python Package.
#'
#' This function installs the gcn4r package from PyPI and places it in local or conda environment.
#'
#' @param custom.path Location/name of pip package.
#' @param pip Whether to use PyPI.
#' @param conda.env Name of conda environment to save to.
#' @export
install_gcn4r <- function(custom.path='git+https://github.com/jlevy44/GCN4R',pip=F, conda.env='gcn4r') {
  reticulate:::py_install(custom.path,pip=pip,envname=conda.env)
}

#' Create conda environment with specified name.
#'
#' @param conda.env Name of conda environment.
#' @export
create.conda <- function(conda.env='gcn4r'){
  reticulate:::create_conda(conda.env)
}

#' Search for existing conda environment.
#'
#' @param conda.env Name of conda environment.
#' @export
detect.conda <- function(conda.env='gcn4r'){
  reticulate:::py_install_method_detect(conda.env)
}

#' Source python from user specified path.
#'
#' @param python.exec Python executable.
#' @export
source.python <- function(python.exec='/usr/local/bin/python3'){
  reticulate:::use_python(python.exec)
}

#' Import gcn4r package after sourcing python.
#'
#' @export
import_gcn4r <- function() {
  gcn4r<-reticulate:::import('gcn4r')
}

