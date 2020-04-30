library(reticulate)
library(statnet)
library(latentnet)
library(wfg)
library(igraph)
library(data.table)
library(pheatmap)

####################### IMPORT #######################

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

####################### LOAD DATA #######################

generate.net.list <- function (adj.csv,cov.csv) {
  A<-read.csv(adj.csv)#[,-1]
  X<-read.csv(cov.csv)#[,-1]
  return(list(A=A,X=X))
}

####################### VISUALIZE DATA #######################

to.igraph <- function(A,X){
  return(graph_from_data_frame(A, directed = TRUE, vertices = X))
}

plot.net <- function(net,group=NULL) {
  V(net)$size <- 7
  if (!is.null(group)){
    V(net)$color <- group
  }
  plot(net, vertex.label='')
}

visualize.net<- function(net.list, covar=NULL) {
  net<-to.igraph(net.list$A,net.list$X)
  plot.net(net,covar)
}

####################### SET PARAMETERS #######################

generate_default_parameters <- function() {
  parameters<-list(learning_rate=1e-4,
                   n_epochs=300L,
                   encoder_base='GCNConv',
                   n_hidden=30L,
                   n_layers=2L,
                   discriminator_layers=c(20L,20L),
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
                   lambda_recon=1.,
                   lambda_pred=0.,
                   epoch_cluster=301L,
                   kl_warmup=20L,
                   K=10L,
                   Niter=10L,
                   sparse_matrix='A.npz',
                   feature_matrix='X.npy',
                   custom_dataset='none',
                   val_ratio=0.05,
                   test_ratio=0.1,
                   task='clustering',
                   use_mincut=F,
                   kmeans_use_probs=F,
                   prediction_column=-1L)
  return(parameters)

}

update.parameters <- function(parameters, new.parameters=list()) {
  return(modifyList(parameters, new.parameters))
}

####################### TRAIN MODEL #######################

flush.stdout <- function(){
  reticulate::py_run_string("import sys; sys.stdout.flush()")
}

add.graph.info <- function(parameters,net.list,prediction_column=-1L){
  parameters$sparse_matrix<-as.matrix(net.list$A[,-1])
  parameters$feature_matrix<-as.matrix(net.list$X[,-1])
  return(parameters)
}

cluster.model.fit<- function (parameters, net.list){
  parameters$task<-"clustering"
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
}

classify.model.fit<- function (parameters, net.list, prediction_column=-1L){
  parameters$task<-"classification"
  parameters<-add.graph.info(parameters,net.list,prediction_column)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
}

regression.model.fit<- function (parameters, net.list, prediction_column=-1L){
  parameters$task<-"regression"
  parameters<-add.graph.info(parameters,net.list,prediction_column)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
}

link.prediction.model.fit<- function (parameters, net.list){
  parameters$task<-"link_prediction"
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
}

link.prediction.model.fit<- function (parameters, net.list){
  parameters$task<-"generation"
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
}

####################### SUMMARIZE MODEL (RUN PREDICTION) #######################

return.results<- function(parameters, net.list, prediction_column=-1L, task="clustering") {
  parameters$task<-task
  parameters$predict<-T
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
}

extract.clusters<- function(results) {
  return(results$cl)
}

extract.graphs<-function(results) {
  A.adj<-matrix(as.integer(results$A>results$threshold),nrow=nrow(results$A))
  A<-get.edgelist(graph.adjacency(A.adj))
  X<-setDT(as.data.frame(results$X), keep.rownames = TRUE)[]
  net1<-to.igraph(A,X)
  G<-matrix(as.vector(results$G$edge_index$numpy()), nc = 2, byrow = TRUE)+1
  net2<-to.igraph(G,X)
  graphs<-list(A.pred=net1,A.true=net2)
  return(graphs)
}

####################### VISUALIZE RESULTS #######################

# maybe pca plot of embeddings



####################### OLD #######################

train_model<- function (learning_rate=1e-4,
                        n_epochs=300L,
                        encoder_base='GCNConv',
                        n_hidden=30L,
                        n_layers=2L,
                        discriminator_layers=c(20L,20L),
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
                        task='clustering',
                        use_mincut=F) {
  results<-GCN4R$api$train_model_(learning_rate,
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
                         task,
                         use_mincut,
                         initialize_spectral = F)
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


sim.and.plot <- function(nv=c(32, 32, 32, 32),
                         p.in=c(0.452, 0.452, 0.452, 0.452),
                         p.out=0.021, p.del=0,
                         seed=42L) {
  set.seed(seed)
  net.simu <- network.simu(nv=nv, p.in=p.in, p.out=p.out, p.del=p.del)
  net <- net.simu$net
  group <- net.simu$group
  plot.net(net,group)
  return(net.simu)

}






to.networkx <- function(A) {
  return(GCN4R$api$nx$from_edgelist(A))
}

run.tests <- function(K=4L, use_mincut=T) {
  GCN4R<-import_gcn4r()
  # run GCN model
  train_model(custom_dataset = 'lawyer', random_seed = 42L, lambda_cluster = 1., ae_type="ARGA", lambda_adv = 1e-3, epoch_cluster=200L, n_epoch=800L, K=K, lambda_kl=0L, learning_rate = 1e-3, task='clustering', use_mincut=use_mincut, encoder_base="GATConv")
  results<-train_model(custom_dataset = 'lawyer', random_seed = 42L, lambda_cluster = 1e-2, ae_type="ARGA", lambda_adv = 1e-3, epoch_cluster=200L, n_epoch=800L, K=K, lambda_kl=0L, learning_rate = 1e-3, task='clustering', use_mincut=use_mincut, predict=T, encoder_base = "GATConv")

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
# add soft cluster assignments from mincut
run.tests2 <- function(K=4L, use_mincut=T) {
  GCN4R<-import_gcn4r()
  # run GCN model
  net<-sim.and.plot()
  G<-as_adjacency_matrix(net$net,sparse=F)
  train_model(custom_dataset = 'none', sparse_matrix = G, feature_matrix=NULL, random_seed = 43L, ae_type="ARGA", lambda_adv = 1e-3, lambda_cluster = 1., epoch_cluster=50L, K=K, lambda_kl=0L, learning_rate = 1e-2, task='clustering',use_mincut=use_mincut, n_epochs=400L)
  results<-train_model(custom_dataset = 'none', sparse_matrix = G, feature_matrix=NULL, random_seed = 42L, ae_type="ARGA", lambda_adv = 1e-3, lambda_cluster = 1e-2, epoch_cluster=150L, K=K, lambda_kl=0L, learning_rate = 1e-2, task='clustering',predict=T,use_mincut=use_mincut)
  # results=train_model(custom_dataset = 'none', sparse_matrix = G, feature_matrix=NULL, random_seed = 42L, lambda_adv = 0L, lambda_cluster = 1e-3, epoch_cluster=150L, K=K, lambda_kl=0L, learning_rate = 1e-3, task='clustering',use_mincut=use_mincut, n_epochs=800L,predict=T)
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

}

