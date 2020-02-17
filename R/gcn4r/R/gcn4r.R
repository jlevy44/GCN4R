library(reticulate)
library(statnet)
library(latentnet)
library(wfg)
library(igraph)
library(data.table)



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
source.python <- function(python.exec='/anaconda2/envs/gcn4r/bin/python'){
  reticulate:::use_python(python.exec)
}

#' Import gcn4r package after sourcing python.
#'
#' @export
import_gcn4r <- function() {
  GCN4R<-reticulate:::import('gcn4r')
}

train_model<- function (inputs_dir='inputs',
                        learning_rate=1e-4,
                        n_epochs=300L,
                        encoder_base='GCNConv',
                        n_hidden=30L,
                        n_layers=2L,
                        discriminator_layers=c(20,20),
                        ae_type='GAE',
                        bias=T,
                        attention_heads=1L,
                        decoder_type='inner',
                        model_save_loc='saved_model.pkl',
                        predictions_save_path='predictions.pkl',
                        predict=F,
                        lambda_kl=1e-3,
                        lambda_adv=1e-3,
                        lambda_cluster=1e-3,
                        epoch_cluster=301L,
                        K=10L,
                        Niter=10L,
                        sparse_matrix='A.npz',
                        feature_matrix='X.npy',
                        custom_dataset='none',
                        val_ratio=0.05,
                        test_ratio=0.1,
                        random_seed=42L,
                        task='clustering') {
  results<-GCN4R$api$train_model_(inputs_dir,
                         learning_rate,
                         n_epochs,
                         encoder_base,
                         n_hidden,
                         n_layers,
                         discriminator_layers,
                         ae_type,
                         bias,
                         attention_heads,
                         decoder_type,
                         model_save_loc,
                         predictions_save_path,
                         predict,
                         lambda_kl,
                         lambda_adv,
                         lambda_cluster,
                         epoch_cluster,
                         K,
                         Niter,
                         sparse_matrix,
                         feature_matrix,
                         custom_dataset,
                         val_ratio,
                         test_ratio,
                         random_seed,
                         task)
  reticulate::py_run_string("import sys; sys.stdout.flush()")
  if (predict) {
    return(results)
  }
}

visualize <- function(predictions_save_path='predictions.pkl',
                      use_predicted_graph=F,
                      pos_threshold=0.5,
                      layout='latent',
                      color_clusters=F,
                      axes_off=F,
                      output_fname='network_plot.html',
                      size=4L) {
  GCN4R$api$visualize_(predictions_save_path,
                       use_predicted_graph,
                       pos_threshold,
                       layout,
                       color_clusters,
                       axes_off,
                       output_fname,
                       size)
}



# R functions

build_network <- function (reldata, nodecov, directed=T) {
  pnet <- network(reldata,directed=directed,matrixtype="adjacency",
                  vertex.attr=nodecov,
                  vertex.attrnames=colnames(nodecov))
  return(pnet)
}

plot_network <- function (net,fname,mode="fruchtermanreingold") {
  png(fname)
  plot(net,mode=mode,displaylabels=T)
  dev.off()
}

plot_model <- function (model,fname) {
  png(fname)
  statnet:::plot.ergm(model)#latentnet:::plot.ergmm(model,displaylabels=T)
  dev.off()
}

load_reldata <- function (network.txt) {
  reldata <- scan(network.txt)
  nr=sqrt(length(reldata))
  reldata=matrix(reldata,ncol=nr,nrow=nr,byrow=T)
  return(reldata)
}

load_covariate_data <- function (covariate.dat, nr) {
  nodecov <- scan(covariate.dat)
  nodecov <- matrix(nodecov,nrow=nr,byrow=T)
  return(nodecov)
}

calculate_nr <- function (reldata) {sqrt(length(reldata))}

sim.network <- function (model) {simulate(model)}

net2mat <- function (G) {
  G=as.matrix.network(G)
  return(G)
}
# add ergm, ergmm, mple

plot.net <- function(net,group) {
  V(net)$size <- 7
  V(net)$color <- group
  plot(net, vertex.label='')
}

sim.and.plot <- function(nv=c(32, 32, 32, 32),
                         p.in=c(0.452, 0.452, 0.452, 0.452),
                         p.out=0.021, p.del=0) {
  net.simu <- network.simu(nv=nv, p.in=p.in, p.out=p.out, p.del=p.del)
  net <- net.simu$net
  group <- net.simu$group
  plot.net(net,group)

}

to.igraph <- function(A,X){
  return(graph_from_data_frame(A, directed = TRUE, vertices = X))
}

to.networkx <- function(A) {
  return(GCN4R$api$nx$from_edgelist(A))
}

run.tests <- function(K=3L) {
  GCN4R<-import_gcn4r()
  # run GCN model
  train_model(custom_dataset = 'lawyer', random_seed = 42L, lambda_adv = 0L, lambda_cluster = 1e-4, epoch_cluster=150L, K=K, lambda_kl=0L, learning_rate = 1e-3, task='clustering')
  results<-train_model(custom_dataset = 'lawyer', random_seed = 42L, lambda_adv = 0L, lambda_cluster = 1e-4, epoch_cluster=150L, K=K, lambda_kl=0L, learning_rate = 1e-3, task='clustering',predict=T)

  # create synthetic graph
  A.adj<-matrix(as.integer(results$A>results$threshold),nrow=nrow(results$A))
  A<-get.edgelist(graph.adjacency(A.adj))
  X<-setDT(as.data.frame(results$X), keep.rownames = TRUE)[]
  net<-to.igraph(A,X)
  plot.net(net,results$cl)

  # create real graph, plot found clusters
  G<-matrix(as.vector(results$G$edge_index$numpy()), nc = 2, byrow = TRUE)+1
  #G<-as.matrix(data.frame(G[c(T,F)],G[c(F,T)]))
  net<-to.igraph(G,X)
  plot.net(net,results$cl)

  # run louvain, plot found communities
  coms<-GCN4R$api$cdlib$algorithms$louvain(to.networkx(G))
  coms.dict<-coms$to_node_community_map()
  coms<-unlist(lapply(1:nrow(A.adj),function(i){coms.dict[i]}))
  plot.net(net,coms)

  return(results)

}

