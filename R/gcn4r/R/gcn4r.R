.onLoad <- function(libname, pkgname){
  library(reticulate)
  library(statnet)
  library(sna)
  library(latentnet)
  library(wfg)
  library(igraph)
  library(data.table)
  library(pheatmap)
  library(plotly)
  library(htmlwidgets)
  library(magick)
  library(Matrix)
  library(gridExtra)
  library(ggplot2)
  library(data.table)
  library(gganimate)
  library(ggnetwork)
  library(png)
  library(gifski)
  library(centiserve)
  library(intergraph)
  library(rdist)
  library(stats)
  library(linkcomm)
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

generate.net.list <- function(adj.csv,cov.csv) {
  A<-read.csv(adj.csv)#[,-1]
  X<-read.csv(cov.csv)#[,-1]
  return(list(A=A,X=X))
}

igraph2net.list <- function(net) {
  A<-as.matrix(as_adjacency_matrix(net))
  X<-as.data.frame(vertex_attr(net))
  return(list(A=A,X=X))
}

####################### VISUALIZE DATA #######################

net.list.to.igraph<-function(net.list){
  return(graph_from_data_frame(net.list$A, directed = T, vertices = net.list$X))
}

to.igraph <- function(A,X,directed=T){
  return(graph_from_data_frame(A, directed = directed, vertices = X))
}

plot.net <- function(net,group=NULL,title="",layout=NULL) {
  V(net)$size <- 7
  if (!is.null(group)){
    V(net)$color <- group
  }
  if (is.null(layout)){
    layout<-layout_with_fr(net)
  }
  plot(net, vertex.label='', main=title, layout=layout)
}

visualize.net<- function(net.list, covar=NULL, layout=NULL) {
  net<-to.igraph(net.list$A,net.list$X)
  plot.net(net,covar,layout=layout)
}

visualize.net2<- function(net.list, covar=NULL, layout=NULL) {
  vis.weighted.graph(as.matrix(net.list$A),cl=net.list$X[,covar], cscale.colors=c("black","grey"))
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
                   prediction_column=-1L,
                   animation_save_file="")
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
  output<-py_capture_output(res<-do.call(GCN4R$api$train_model_, parameters))
  return(list(output=output,res=res))
}

add.graph.info <- function(parameters,net.list,prediction_column=-1L){
  parameters$sparse_matrix<-as.matrix(net.list$A)#[,-1]
  parameters$feature_matrix<-as.matrix(net.list$X)#[,-1]
  return(parameters)
}

build.model.class <- function(parameters,net.list,class.name,prediction_column=-1L, loss.log=NULL){
  results.returned<-return.results(parameters,net.list,prediction_column=prediction_column,task=parameters$task)
  fit.model<-list(parameters=parameters,
                  results=results.returned$res,
                  loss.log=loss.log,
                  diagnostic.text=results.returned$output)
  fit.model<-structure(fit.model,class=c("gnn.model",class.name))
  return(fit.model)
}

cluster.model.fit<- function (parameters, net.list, verbose=F){
  parameters$task<-"clustering"
  parameters<-add.graph.info(parameters,net.list)
  log<-py_capture_output(loss.log<-do.call(GCN4R$api$train_model_, parameters))
  if (verbose){
    import_builtins()$print(log)
  }
  parameters$predict<-T
  fit.model<-build.model.class(parameters,net.list,"gnn.cluster.model",loss.log=loss.log)
  return(fit.model)
}

classify.model.fit<- function (parameters, net.list, prediction_column=-1L, verbose=F){
  parameters$task<-"classification"
  parameters<-add.graph.info(parameters,net.list,prediction_column)
  log<-py_capture_output(loss.log<-do.call(GCN4R$api$train_model_, parameters))
  if (verbose){
    import_builtins()$print(log)
  }
  parameters$predict<-T
  fit.model<-build.model.class(parameters,net.list,"gnn.classify.model",prediction_column,loss.log=loss.log)
  return(fit.model)
}

regression.model.fit<- function (parameters, net.list, prediction_column=-1L, verbose=F){
  parameters$task<-"regression"
  parameters<-add.graph.info(parameters,net.list,prediction_column)
  log<-py_capture_output(loss.log<-do.call(GCN4R$api$train_model_, parameters))
  if (verbose){
    import_builtins()$print(log)
  }
  parameters$predict<-T
  fit.model<-build.model.class(parameters,net.list,"gnn.regress.model",prediction_column,loss.log=loss.log)
  return(fit.model)
}

link.prediction.model.fit<- function (parameters, net.list, verbose=F){
  parameters$task<-"link_prediction"
  parameters<-add.graph.info(parameters,net.list)
  log<-py_capture_output(loss.log<-do.call(GCN4R$api$train_model_, parameters))
  if (verbose){
    import_builtins()$print(log)
  }
  parameters$predict<-T
  fit.model<-build.model.class(parameters,net.list,"gnn.link.model",loss.log=loss.log)
  return(fit.model)
}

generative.model.fit<- function (parameters, net.list, verbose=F){
  parameters$task<-"generation"
  parameters<-add.graph.info(parameters,net.list)
  log<-py_capture_output(loss.log<-do.call(GCN4R$api$train_model_, parameters))
  if (verbose){
    import_builtins()$print(log)
  }
  parameters$predict<-T
  fit.model<-build.model.class(parameters,net.list,"gnn.generative.model",loss.log=loss.log)
  return(fit.model)
}

####################### SUMMARIZE MODEL (RUN PREDICTION) #######################

summary.gnn.cluster.model<- function(gnn.model, additional.info=F){
  import_builtins()$print(gnn.model$diagnostic.text)
  if (additional.info){
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
}

summary.gnn.generative.model<-function(gnn.model){
  import_builtins()$print(gnn.model$diagnostic.text)
}

summary.gnn.classify.model<-function(gnn.model){
  import_builtins()$print(gnn.model$diagnostic.text)
}

summary.gnn.regress.model<-function(gnn.model){
  import_builtins()$print(gnn.model$diagnostic.text)
}

summary.gnn.link.model<-function(gnn.model){
  import_builtins()$print(gnn.model$diagnostic.text)
}

extract.results<- function(gnn.model) {
  return(gnn.model$results)
}

extract.features<-function(gnn.model) {
  return(colnames(gnn.model$parameters$feature_matrix))
}

extract.performance<- function(gnn.model) {
  return(gnn.model$results$performance)
}

extract.clusters<- function(gnn.model) {
  return(gnn.model$results$cl)
}

extract.prediction<- function(gnn.model) {
  return(as.matrix(gnn.model$results$y$numpy()))
}

extract.embeddings<- function(gnn.model) {
  return(gnn.model$results$z)
}

make.layout<- function(z) {
  set.seed(42)
  return(prcomp(z, scale = T)$x[,c(1,2)])
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

plot.nets<-function(gnn.model,cl,plots=c(1,2),layout=NULL){
  graphs<-extract.graphs(gnn.model)
  if (1 %in% plots){
    plot.net(graphs$A.pred,cl,title="Predicted Network",layout=layout)
  }
  if (2 %in% plots){
    plot.net(graphs$A.true,cl,title="Original Network",layout=layout)
  }
}

# maybe pca plot of embeddings
plot.gnn.cluster.model <- function(gnn.model,latent=F,...) {
  cl<-extract.clusters(gnn.model)
  layout<-NULL
  if (latent){
    z<-extract.embeddings(gnn.model)
    layout<-make.layout(z)
  }
  plot.nets(gnn.model,cl,layout=layout,...)
}

plot.gnn.classify.model <- function(gnn.model,latent=F,...) {
  cl<-gnn.model$results$y
  layout<-NULL
  if (latent){
    z<-extract.embeddings(gnn.model)
    layout<-make.layout(z)
  }
  plot.nets(gnn.model,cl,layout=layout,...)
}

plot.gnn.regress.model <- function(gnn.model,latent=F,...) {
  cl<-gnn.model$results$y
  layout<-NULL
  if (latent){
    z<-extract.embeddings(gnn.model)
    layout<-make.layout(z)
  }
  plot.nets(gnn.model,cl,layout=layout,...)
}

plot.gnn.link.model <- function(gnn.model,latent=F,...) {
  cl<-NULL
  layout<-NULL
  if (latent){
    z<-extract.embeddings(gnn.model)
    layout<-make.layout(z)
  }
  plot.nets(gnn.model,cl,layout=layout,...)
}

plot.gnn.generative.model <- function(gnn.model,latent=F,...) {
  cl<-NULL
  layout<-NULL
  if (latent){
    z<-extract.embeddings(gnn.model)
    layout<-make.layout(z)
  }
  plot.nets(gnn.model,cl,layout=layout,...)
}

plot.diagnostics <- function(gnn.model){
  loss.log<-py_to_r(gnn.model$loss.log$loss_log)
  loss.fns<-names(loss.log)
  loss.fns<-loss.fns[loss.fns!="epoch"]
  melt(loss.log,id.vars="epoch")
  ggplot(data=melt(loss.log,id.vars="epoch"), aes(x=epoch, y=value)) +
    geom_line() +
    facet_grid(rows = vars(variable), scales = "free")
}

####################### SIMULATION #######################

simulate.networks <- function(gnn.model,nsim=100) {
  results<-extract.results(gnn.model)
  threshold<-results$threshold
  X<-NULL
  parameters<-gnn.model$parameters
  parameters$task<-"generation"
  sim.graphs<-list()
  embeddings<-list()
  Adj<-list()
  cl<-list()
  for (i in 1:nsim) {
    parameters$random_seed<-i
    invisible(py_capture_output(res<-do.call(GCN4R$api$train_model_, parameters)))
    A<-res$A
    embeddings[[i]]<-res$z
    Adj[[i]]<-A
    A.adj<-matrix(as.integer(A>threshold),nrow=nrow(A))
    A<-get.edgelist(graph.adjacency(A.adj))
    sim.graphs[[i]]<-to.igraph(A,X)
    cl[[i]]<-res$cl
  }
  sim.graphs<-list(networks=sim.graphs,
                   embeddings=embeddings,
                   cl=cl,
                   Adj=Adj
                   )
  return(sim.graphs)
}

####################### INTERPRET RESULTS #######################

calc.centrality.measure<-function(net,measure="none",norm=T){
  measures<-c("laplacian","clusterrank","e_cent","comm","strength")
  norm.centrality.measure<-0
  if (measure=="none"){
    norm.centrality.measure<-0
  } else if (measure%in%measures) {
    if (measure=="laplacian"){
      norm.centrality.measure<-laplacian(net)
    } else if (measure=="clusterrank") {
      norm.centrality.measure<-clusterrank(net)
    } else if (measure=="e_cent") {
      norm.centrality.measure<-eigen_centrality(net)$vector
    } else if (measure=='comm'){
      norm.centrality.measure<-communitycent(net)
    } else if (measure=="strength"){
      norm.centrality.measure<-strength(net)
    }
    print(norm.centrality.measure)
    norm.centrality.measure[is.nan(norm.centrality.measure)]=0
    if (norm){
      norm.centrality.measure=abs(norm.centrality.measure)
      norm.centrality.measure<-norm.centrality.measure/max(norm.centrality.measure)
    }
  }
  return(norm.centrality.measure)
}

weight.matrix.to.net<-function(weight_matrix,threshold=NULL){
  weight_matrix<-(weight_matrix+t(weight_matrix))/2
  if (!is.null(threshold)){
    weight_matrix[(weight_matrix<threshold)]<-0.
  }
  net.true<-graph_from_adjacency_matrix(weight_matrix, mode="upper", weighted=TRUE)
  return(net.true)
}

vis.weighted.graph<-function(weight_matrix=NULL, cl=0, weight.scaling.factor=2, cscale.colors=c("grey","red"), threshold=NULL, important.nodes=NULL, floor.size=5, ceil.size=10, centrality.measure="none", net.input=NULL, node.weight=NULL, layout=NULL) {
  set.seed(42)
  c_scale <- colorRamp(cscale.colors)
  if (!is.null(net.input)) {
    net.true<-net.input
  } else {
    net.true<-weight.matrix.to.net(weight_matrix,threshold)
  }
  E(net.true)$edge.curved=0.
  V(net.true)$color <- cl
  net.true<-simplify(net.true) %>%
    set_vertex_attr("shape",value="circle")
  if (!is.null(important.nodes)){
    net.true<-net.true %>%
      set_vertex_attr("shape",value="square", index=important.nodes)
  }
  if (!is.null(threshold)){
    isolated.nodes <- which(degree(net.true)==0)
    net.true <- delete.vertices(net.true, isolated.nodes)
  }
  if (!is.null(node.weight)){
    norm.centrality.measure<-node.weight/max(node.weight)
  } else {
    norm.centrality.measure<-calc.centrality.measure(net.true,centrality.measure)
  }
  V(net.true)$size <- floor.size+(ceil.size-floor.size)*norm.centrality.measure
  if (!is.null(E(net.true)$weight)){
    weight<-E(net.true)$weight
    E(net.true)$width<-weight/max(weight)*weight.scaling.factor
    E(net.true)$color = apply(c_scale(weight/max(weight)), 1, function(x) rgb(x[1]/255,x[2]/255,x[3]/255) )
  }
  if (!is.null(layout)){
    l <- layout

    if (!is.null(threshold)){
      print(isolated.nodes)
      l<-l[-isolated.nodes,]
    }
  } else {
    l <- layout_with_fr(net.true)
  }
  l <- norm_coords(l, ymin=-1, ymax=1, xmin=-1, xmax=1)
  plot(net.true, layout=l*1., rescale=F, edge.curved=0., vertex.label="", main="")
  return(weight_matrix)
}

visualize.attention<-function(gnn.model,weight.scaling.factor=20.,latent=F,plot=T,...){
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

  layout<-NULL
  if (latent){
    z<-extract.embeddings(gnn.model)
    layout<-make.layout(z)
  }

  for (i in 1:length(weight_matrices)) {
    weight_matrix<-as.matrix(weight_matrices[[i]])
    if (plot){
      weight_matrices[[i]]<-vis.weighted.graph(weight_matrix,cl,weight.scaling.factor,layout=layout,...)
    } else {
      weight_matrices[[i]]<-weight_matrix
    }
  }
  class(weight_matrices)<-"attention"
  return(weight_matrices)
}

interpret.predictors<-function(gnn.model,interpretation.mode="integrated_gradients",plot=T){
  parameters<-extract.parameters(gnn.model)
  parameters$mode<-"captum"
  py_capture_output(attributions<-do.call(GCN4R$api$interpret_model, parameters))
  classes<-names(attributions)
  classes<-classes[classes!="cluster_assignments"]
  attr.list<-GCN4R$interpret$plot_attribution(attributions,F,T)

  if (plot){
    for (i in 1:length(classes)){
      attribution<-as.data.frame(attr.list$attributions[i])
      row_annot<-data.frame(cluster=as.factor(attr.list$cl))
      col_annot<-data.frame(importance=attr.list$feature_importances)
      rownames(attribution)<-1:nrow(attribution)
      colnames(attribution)<-colnames(gnn.model$parameters$feature_matrix)#1:ncol(attribution)

      rownames(col_annot) <- colnames(attribution)
      colnames(col_annot) <- c("feature")#colnames(attribution)
      rownames(row_annot) <- rownames(attribution)
      colnames(row_annot) <- c("cluster")

      pheatmap(attribution,annotation_col=col_annot,annotation_row=row_annot)
    }
  }

  class(attr.list)<-"gradient"
  return(attr.list)
}

extract.motifs<-function(gnn.model,node_idx=NULL){
  nx<-import('networkx')
  py <- import_builtins()
  parameters<-extract.parameters(gnn.model)
  parameters$mode<-"gnn_explainer"
  nodes<-lapply(as.integer(V(extract.graphs(gnn.model)$A.true)$name)-1,function(x){py$int(x)})
  if (!is.null(node_idx)){
    parameters$node_idx<-node_idx
  }
  py_capture_output(attributions<-do.call(GCN4R$api$interpret_model, parameters))
  node_idx<-names(attributions)
  for (i in node_idx){
    G.tmp<-attributions[[i]]$explain_graph
    G<-nx$OrderedGraph()
    G$add_nodes_from(nodes)
    G$add_edges_from(G.tmp$edges(data=T))
    G<-as.matrix(nx$convert_matrix$to_numpy_matrix(G,weight='att'))
    attributions[[i]]<-list(feature_scores=attributions[[i]]$node_feat,G=G)
  }
  class(attributions)<-"motif"
  return(attributions)
}

get.motif<-function(attributions,i){
  return(attributions[[as.character(i)]]$G)
}

get.feat.importance.scores<-function(attributions,i){
  return(attributions[[i]]$feature_scores)
}

build.importance.matrix<-function(attributions,cl=NULL,attr.col.names=NULL){
  N<-length(attributions)
  feature.scores<-c()
  for (attribution in attributions){
    feature.scores<-c(feature.scores,attribution$feature_scores)
  }
  attribution<-as.data.frame(matrix(feature.scores,nrow=N))
  rownames(attribution)<-names(attributions)
  colnames(attribution)<-1:ncol(attribution)
  if (!is.null(attr.col.names)){
    colnames(attribution)<-attr.col.names
  }
  row_annot=NULL
  if (!is.null(cl) & length(cl)==N){
    row_annot<-data.frame(cluster=as.factor(cl))



    rownames(row_annot) <- rownames(attribution)
    colnames(row_annot) <- c("cluster")

  }

  pheatmap(attribution,annotation_row=row_annot)
  return(attribution)
}

vis.motif<-function(attributions,i,cl, weight.scaling.factor=2, cscale.colors=c("grey","red"), threshold=NULL, other.idx=NULL, ...){
  motif<-get.motif(attributions,i)
  important.nodes<-c(i+1L)
  if (!is.null(other.idx)) {
    for (i in other.idx){
      motif<-motif+get.motif(attributions,i)
      important.nodes<-c(important.nodes,i+1L)
    }
    motif<-motif/(length(other.idx)+1)
  }
  weight.matrix<-vis.weighted.graph(motif, cl, weight.scaling.factor, cscale.colors, threshold, important.nodes = important.nodes,...)
}

cluster.performance.node.importance<-function(cluster.model){
  sklearn_metrics<-import("sklearn.metrics")
  cl<-extract.clusters(cluster.model)
  scores<-c()
  for (i in 1:nrow(cluster.model$parameters$sparse_matrix)){
    parameters.nodes<-update.parameters(cluster.model$parameters,
                                        list(sparse_matrix=cluster.model$parameters$sparse_matrix[-i,-i],
                                             feature_matrix=cluster.model$parameters$feature_matrix[-i,]))

    py_capture_output(res<-do.call(GCN4R$api$train_model_, parameters.nodes))
    scores<-c(scores,sklearn_metrics$v_measure_score(cl[-i],res$cl))
  }
  return(1-scores)
}

non.cluster.performance.node.importance<-function(gnn.model){
  base.performance<-extract.performance(gnn.model)
  scores<-c()
  for (i in 1:nrow(gnn.model$parameters$sparse_matrix)){
    parameters.nodes<-update.parameters(gnn.model$parameters,
                                        list(sparse_matrix=gnn.model$parameters$sparse_matrix[-i,-i],
                                             feature_matrix=gnn.model$parameters$feature_matrix[-i,]))

    py_capture_output(res<-do.call(GCN4R$api$train_model_, parameters.nodes))
    scores<-c(scores,-(base.performance-res$performance)/base.performance)
  }
  return(scores)
}

performance.node.importance<-function(gnn.model){
  if (gnn.model$parameters$task=="clustering"){
    return(cluster.performance.node.importance(gnn.model))
  } else {
    return(non.cluster.performance.node.importance(gnn.model))
  }
}

node.importance.attention<-function(attention.list, layers.idx=NULL){
  if (is.null(layers.idx)){
    layers.idx<-1:length(attention.list)
  }
  attention<-attention.list[[layers.idx[1]]]
  if (length(layers.idx)>1){
    for (i in layers.idx[2:length(layers.idx)]){
      attention<-attention+attention.list[[layers.idx[i]]]
    }
  }
  centrality.measure<-calc.centrality.measure(weight.matrix.to.net(attention),measure="strength",norm=F)
  return(centrality.measure)
}

node.importance.gradient<-function(attributions){
  return(apply(Reduce(function(x,y){abs(x)+abs(y)},attributions$attributions),1,mean)/length(attributions$attributions))
}

node.importance.motif<-function(motif.graphs,threshold=NULL){
  return(unlist(lapply(1:length(motif.graphs),function(i){calc.centrality.measure(weight.matrix.to.net(get.motif(motif.graphs,i-1),threshold=threshold),measure="strength",norm=F)[i]})))
}

plot.node.importance<-function(gnn.model,importance.type="performance",layers.idx=c(1),motif.threshold=NULL,relate.cluster.meas=F,...){
  if (importance.type=="attention"){
    attention.matrices<-visualize.attention(gnn.model, plot=F)
    node.importance<-node.importance.attention(attention.matrices,layers.idx = layers.idx)
  } else if (importance.type=="performance"){
    node.importance<-performance.node.importance(gnn.model)
  } else if (importance.type=="gradient"){
    attribution<-interpret.predictors(gnn.model,plot=F)
    node.importance<-node.importance.gradient(attribution)
  } else if (importance.type=="motif") {
    motif.graphs<-extract.motifs(gnn.model)
    node.importance<-node.importance.motif(motif.graphs,threshold=motif.threshold)
  } else {
    print("Please specify importance.type attention|performance|gradient|motif")
    node.importance<-0.
  }
  net<-extract.graphs(cluster.model)$A.true
  vis.weighted.graph(net.input=net, node.weight=node.importance, ...)
  if (relate.cluster.meas){
    relate.clustering.measures(net,node.importance)
  }
  return(node.importance)
}

####################### MISC #######################

animate.plot<-function(animate.model,gif_file="animate.gif", delay=0.2, res = 92){
  pandas<-import('pandas')
  py <- import_builtins()
  animation<-pandas$read_pickle(animate.model$parameters$animation_save_file)
  animation.layouts<-data.frame(prcomp(animation[,-ncol(animation)], scale = F)$x[,c(1,2)])
  animation.layouts$epoch<-animation$epoch
  cl<-extract.clusters(animate.model)
  net<-extract.graphs(animate.model)$A.true
  V(net)$color <- cl
  make_plot<-function(){
    lapply(0:max(animation.layouts$epoch), function(i){
      n<-ggnetwork(net,layout=as.matrix(animation.layouts[animation.layouts$epoch==i,1:2]))
      n$color<-as.character(n$color)
      p<-ggplot(n, aes(x = x, y = y, xend = xend, yend = yend)) +
        geom_edges(aes(), color = "grey50") +
        geom_nodes(aes(color = color)) +
        theme_blank()
      print(p)
    })
  }
  gif_file <- save_gif(make_plot(), gif_file = gif_file, width = 800, height = 800, res = res)
  utils::browseURL(gif_file)
}

relate.clustering.measures<-function(net,node.importance){
  plot(betweenness(net),node.importance)
  plot(eigen_centrality(net)$vector,node.importance)
  plot(transitivity(net,"local"),node.importance)
  print(summary(lm(betweenness(net)~node.importance)))
  print(summary(lm(eigen_centrality(net)$vector~node.importance)))
  print(summary(lm(transitivity(net,"local")~node.importance)))
}

ergm.network<-function(formula,gnn.model,add.dist=T,distance.metric="euclidean", use_ergmm=FALSE, simulate=F, pseudo=F, ...){
  net.list<-list(A=as.data.frame(gnn.model$parameters$sparse_matrix),
                 X=as.data.frame(gnn.model$parameters$feature_matrix))
  net.list$X$cl<-as.factor(extract.clusters(gnn.model))
  net<-graph_from_adjacency_matrix(as.matrix(net.list$A))
  dist<-pdist(extract.embeddings(gnn.model), metric = distance.metric, p = 2)
  if (simulate){
    sim<-simulate.networks(gnn.model,nsim=1)
    net<-sim$networks[[1]]
    dist<-pdist(sim$embeddings[[1]], metric = distance.metric, p = 2)
    net.list$X$cl<-sim$cl[[1]]
  }
  for (nm in colnames(net.list$X)){
    col<-net.list$X[,nm]
    net<-net %>% set_vertex_attr(name=nm, index = V(net), value=col)
  }
  net<-asNetwork(net)

  if (add.dist){
    formula<-as.formula(paste(deparse(formula),"edgecov(dist)",sep="+"))
  } else {
    formula<-as.formula(deparse(formula))
  }
  if (pseudo & !use_ergmm){
    return(ergmMPLE(formula,output="fit",...))
  }
  else if (use_ergmm) {
    return(ergmm(formula,...))
  }
  else {
    return(ergm(formula,...))
  }
}

to.networkx <- function(A) {
  return(GCN4R$api$nx$from_edgelist(A))
}

run.louvain<-function(net){
  A<-as_edgelist(net)
  coms<-GCN4R$api$cdlib$algorithms$louvain(to.networkx(A))
  coms.dict<-coms$to_node_community_map()
  coms<-unlist(lapply(1:nrow(A),function(i){coms.dict[i]}))
  plot.net(net,coms)
  return(coms)
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

