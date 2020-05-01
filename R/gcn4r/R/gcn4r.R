.onLoad <- function(libname, pkgname){
  library(reticulate)
  library(statnet)
  library(latentnet)
  library(wfg)
  library(igraph)
  library(data.table)
  library(pheatmap)
  library(plotly)
  library(htmlwidgets)
  library(magick)
  library(Matrix)
}

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

to.igraph <- function(A,X,directed=T){
  return(graph_from_data_frame(A, directed = directed, vertices = X))
}

plot.net <- function(net,group=NULL,title="") {
  V(net)$size <- 7
  if (!is.null(group)){
    V(net)$color <- group
  }
  plot(net, vertex.label='', main=title)
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
  class(parameters)<-"parameters"
  return(parameters)

}

update.parameters <- function(parameters, new.parameters=list()) {
  return(modifyList(parameters, new.parameters))
}

####################### TRAIN MODEL #######################

flush.stdout <- function(){
  reticulate::py_run_string("import sys; sys.stdout.flush()")
}

return.results<- function(parameters, net.list, prediction_column=-1L, task="clustering") {
  parameters$task<-task
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
}

add.graph.info <- function(parameters,net.list,prediction_column=-1L){
  parameters$sparse_matrix<-as.matrix(net.list$A[,-1])
  parameters$feature_matrix<-as.matrix(net.list$X[,-1])
  return(parameters)
}

build.model.class <- function(parameters,net.list,class.name,prediction_column=-1L){
  fit.model<-list(parameters=parameters,
                  results=return.results(parameters,net.list,prediction_column=prediction_column,task=parameters$task))
  fit.model<-structure(fit.model,class=c("gnn.model",class.name))
  return(fit.model)
}

cluster.model.fit<- function (parameters, net.list){
  parameters$task<-"clustering"
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
  parameters$predict<-T
  fit.model<-build.model.class(parameters,net.list,"gnn.cluster.model")
  return(fit.model)
}

classify.model.fit<- function (parameters, net.list, prediction_column=-1L){
  parameters$task<-"classification"
  parameters<-add.graph.info(parameters,net.list,prediction_column)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
  fit.model<-build.model.class(parameters,net.list,"gnn.classify.model",prediction_column)
  return(fit.model)
}

regression.model.fit<- function (parameters, net.list, prediction_column=-1L){
  parameters$task<-"regression"
  parameters<-add.graph.info(parameters,net.list,prediction_column)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
  fit.model<-build.model.class(parameters,net.list,"gnn.regress.model",prediction_column)
  return(fit.model)
}

link.prediction.model.fit<- function (parameters, net.list){
  parameters$task<-"link_prediction"
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
  fit.model<-build.model.class(parameters,net.list,"gnn.link.model")
  return(fit.model)
}

generative.model.fit<- function (parameters, net.list){
  parameters$task<-"generation"
  parameters<-add.graph.info(parameters,net.list)
  do.call(GCN4R$api$train_model_, parameters)
  flush.stdout()
  fit.model<-build.model.class(parameters,net.list,"gnn.generative.model")
  return(fit.model)
}

####################### SUMMARIZE MODEL (RUN PREDICTION) #######################

summary.gnn.cluster.model<- function(gnn.model){
  cl<-extract.clusters(gnn.model)
  z<-extract.embeddings(gnn.model)
  cluster.breakdown<-table(cl)
  print(paste("Extracted",toString(length(cluster.breakdown)),"clusters:"))
  print(cluster.breakdown)
  print(paste("Extracted low dimensional embeddings of shape",toString(nrow(z)),toString(ncol(z))))
  print("Clustering Vector:")
  print(cl)
  print("Fuzzy Cluster Assignment Matrix:")
  print(gnn.model$results$s)
  print("Generated and Real Networks:")
  print(extract.graphs(gnn.model))
}

summary.gnn.generative.model<-function(gnn.model){
  print("Not Implemented")
}

summary.gnn.class.model<-function(gnn.model){
  print("Not Implemented")
}

summary.gnn.regress.model<-function(gnn.model){
  print("Not Implemented")
}

summary.gnn.link.model<-function(gnn.model){
  print("Not Implemented")
}

extract.results<- function(gnn.model) {
  return(gnn.model$results)
}

extract.clusters<- function(gnn.model) {
  return(gnn.model$results$cl)
}

extract.embeddings<- function(gnn.model) {
  return(gnn.model$results$z)
}

extract.parameters<- function(gnn.model) {
  return(gnn.model$parameters)
}

extract.graphs<-function(gnn.model,directed=T) {
  results<-extract.results(gnn.model)
  A.adj<-matrix(as.integer(results$A>results$threshold),nrow=nrow(results$A))
  A<-get.edgelist(graph.adjacency(A.adj))
  X<-setDT(as.data.frame(results$X), keep.rownames = TRUE)[]
  net1<-to.igraph(A,X,directed=directed)
  G<-matrix(as.vector(results$G$edge_index$numpy()), nc = 2, byrow = TRUE)+1
  net2<-to.igraph(G,X,directed=directed)
  graphs<-list(A.pred=net1,A.true=net2)
  return(graphs)
}

####################### VISUALIZE RESULTS #######################

plot.nets<-function(gnn.model,cl){
  graphs<-extract.graphs(gnn.model)
  plot.net(graphs$A.pred,cl,title="Predicted Network")
  plot.net(graphs$A.true,cl,title="Original Network")
}

# maybe pca plot of embeddings
plot.gnn.cluster.model <- function(gnn.model) {
  cl<-extract.clusters(gnn.model)
  plot.nets(gnn.model,cl)
}

plot.gnn.classify.model <- function(gnn.model) {
  cl<-gnn.model$results$y
  plot.nets(gnn.model,cl)
}

plot.gnn.regress.model <- function(gnn.model) {
  cl<-gnn.model$results$y
  plot.nets(gnn.model,cl)
}

plot.gnn.link.model <- function(gnn.model) {
  cl<-NULL
  plot.nets(gnn.model,cl)
}

plot.gnn.generative.model <- function(gnn.model) {
  cl<-NULL
  plot.nets(gnn.model,cl)
}

####################### INTERPRET RESULTS #######################

visualize.attention<-function(gnn.model,weight.scaling.factor=20.){
  parameters<-extract.parameters(gnn.model)
  parameters$mode<-"attention"
  attribution<-do.call(GCN4R$api$interpret_model, parameters)
  cl<-NULL
  if (class(gnn.model)[2]=='gnn.cluster.model'){
    cl<-extract.clusters(gnn.model)
  } else if (class(gnn.model)[2] %in% c('gnn.classify.model','gnn.regress.model')) {
    cl<-gnn.model$results$y
  }

  flush.stdout()
  weight_matrices<-GCN4R$interpret$return_attention_weights(attribution,T)
  # net.orig<-#extract.graphs(gnn.model,directed=F)$A.true
  c_scale <- colorRamp(c('grey','red'))

  for (i in 1:length(weight_matrices)) {

    weight_matrix<-as.matrix(weight_matrices[[i]])
    weight_matrix<-(weight_matrix+t(weight_matrix))/2
    weight_matrices[[i]]<-weight_matrix
    net.true<-graph_from_adjacency_matrix(weight_matrix, mode="upper", weighted=TRUE)
    # E(net.true)$edge.color<-weight/max(weight)*weight.scaling.factor
    E(net.true)$edge.curved=0.
    V(net.true)$color <- cl
    net.true<-simplify(net.true)
    V(net.true)$size <- 5
    # <-weight
    weight<-E(net.true)$weight
    E(net.true)$width<-weight/max(weight)*weight.scaling.factor
    E(net.true)$color = apply(c_scale(weight/max(weight)), 1, function(x) rgb(x[1]/255,x[2]/255,x[3]/255) )
    l <- layout_with_fr(net.true)
    l <- norm_coords(l, ymin=-1, ymax=1, xmin=-1, xmax=1)
    plot(net.true, layout=l*1., rescale=F, edge.curved=0., vertex.label="", main="")

  }
  return(weight_matrices)
}

interpret.predictors<-function(gnn.model,interpretation.mode="integrated_gradients"){
  parameters<-extract.parameters(gnn.model)
  parameters$mode<-"captum"
  attributions<-do.call(GCN4R$api$interpret_model, parameters)
  classes<-names(attributions)
  classes<-classes[classes!="cluster_assignments"]
  attr.list<-GCN4R$interpret$plot_attribution(attributions,F,T)

  for (i in 1:length(classes)){
    attribution<-as.data.frame(attr.list$attributions[i])
    row_annot<-data.frame(cluster=as.factor(attr.list$cl))
    col_annot<-data.frame(importance=attr.list$feature_importances)
    rownames(attribution)<-1:nrow(attribution)
    colnames(attribution)<-1:ncol(attribution)

    rownames(col_annot) <- colnames(attribution)
    colnames(col_annot) <- c("feature")#colnames(attribution)
    rownames(row_annot) <- rownames(attribution)
    colnames(row_annot) <- c("cluster")

    pheatmap(attribution,annotation_col=col_annot,annotation_row=row_annot)
  }
  return(attr.list)
}

####################### OLD/DEPRECATED #######################

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

