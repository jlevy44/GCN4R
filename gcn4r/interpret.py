import torch
from captum.attr import IntegratedGradients,Saliency,DeepLift,InputXGradient,GuidedBackprop,Deconvolution
import matplotlib, seaborn as sns, numpy as np
matplotlib.rcParams['figure.dpi']=300
sns.set(style='white')
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from gcn4r.models import ExplainerModel
from torch_geometric.data import Data
try:
	from torch_geometric.utils import k_hop_subgraph, to_networkx
except:
	pass
try:
	from torch_geometric.utils import erdos_renyi_graph, negative_sampling, dropout_adj, sort_edge_index
except:
	pass

def captum_interpret_graph(G, model, use_mincut, target=0, method='integrated_gradients'):
	# assert use_mincut , "Interpretations only work for min-cut pooling for now"
	x,edge_index=G.x,G.edge_index
	# print(x.shape,edge_index.shape)
	if model.encoder.prediction_task:
		output_key='y'
	else:
		output_key='s'
	def custom_forward(x,edge_index):
		# print(x.shape,edge_index.shape)
		return torch.cat([model.encoder(x[i], edge_index[i])[output_key] for i in range(x.shape[0])],0)
	# custom_forward=(lambda x,edge_index:)
	interpretation_method=dict(integrated_gradients=IntegratedGradients,
							   saliency=Saliency,
							   deeplift=DeepLift,
							   input_x_gradient=InputXGradient)
							   # guided_backprop=GuidedBackprop,
							   # deconvolution=Deconvolution
	assert method in list(interpretation_method.keys())
	interpretation_method=interpretation_method[method]
	ig = interpretation_method(custom_forward)
	attr = ig.attribute(x.unsqueeze(0), additional_forward_args=(edge_index.unsqueeze(0)), target=target)
	return attr

def return_attention_scores(G, model):
	x,edge_index=G.x,G.edge_index
	outputs=model.encoder(x,edge_index)
	attention_scores=[]
	for conv in model.encoder.convs:
		attention_scores.append(conv.attention_coefs)
	return attention_scores

def attn2graph(attribute,i, symmetrize=False):
	# print(attribute)
	edge=attribute[i]['edge_index'].T
	# if edge.shape[1]!=2:
	#     edge=edge.T
	N=edge.max()+1
	attn_scores=attribute[i]['coef'].detach().numpy()
	weights=attn_scores.flatten()
	edge_list=(edge[:,0].flatten(), edge[:,1].flatten())
	G=csr_matrix((weights, edge_list), shape=(N, N))
	if symmetrize:
		G=(G+G.T)/2.
	weights=G.data
	return G, weights, edge_list

def visualize_attention_map(attribute,i,clustermap=False, symmetrize=False,y=None):
	G,_,_=attn2graph(attribute,i,symmetrize)
	plt.figure(figsize=(7,7))
	if clustermap:
		sns.clustermap(G.todense())
	else:
		sns.heatmap(G.todense())
	fig=plt.figure(figsize=(18,18))
	G_new=nx.Graph(G)
	pos = nx.spring_layout(G_new, seed=1)
	# print(G.data)
	#cmap=sns.cubehelix_palette(rot=.3,reverse=True,as_cmap=True)
	cmap=plt.get_cmap('coolwarm')
	nx.draw_networkx(G_new, pos=pos, node_size=50, cmap=plt.get_cmap('coolwarm'),
					 node_color=[cmap(c) for c in (y-y.min())/(y.max()-y.min())], edge_color=[cmap(w) for w in G.data/G.data.max()*2],#'k',
					 arrows=False, width=G.data*25, style='solid', with_labels=False)
	sns.despine()

def return_attention_weights(attention_coefs,symmetrize=False):
	# print(attention_coefs)
	return [attn2graph(attention_coefs,i,symmetrize)[0].todense() for i in range(len(attention_coefs))]

def plot_attention(attention_coefs,y):
	for i in range(len(attention_coefs)):
		visualize_attention_map(attention_coefs,i,False,True,y=y)

def plot_attribution(attribute,standard_scale=False,minmax_scale=False,str_idx=False):
	feature_importances=np.stack([np.abs(attribute[k][0].detach().numpy()) for k in attribute if k!='cluster_assignments']).mean(0).mean(0)#.sum(0).sum(0)
	feature_importances/=feature_importances.max()/2
	attributions=[]
	cl=attribute['cluster_assignments'].detach().numpy().argmax(1)
	for k in attribute:
		if k!='cluster_assignments':
			attribution=attribute[k][0].detach().numpy()
			if minmax_scale:
				attribution=MinMaxScaler().fit_transform(attribution)
			attributions.append(attribution)
			colors=sns.color_palette("cubehelix", len(np.unique(cl))+1)
			row_colors=[colors[c] for c in cl]
			cmap = sns.cubehelix_palette(dark=0.0,light=1.,rot=5, as_cmap=True)
			sns.clustermap(attribution,standard_scale=standard_scale,col_colors=[cmap(ft) for ft in feature_importances],row_colors=row_colors)
	return dict(attributions=attributions,feature_importances=feature_importances,cl=cl)

def create_explainer(model=None, epochs=100, lr=0.01):
	from torch_geometric.nn.models.gnn_explainer import GNNExplainer
	return GNNExplainer(model, epochs=epochs, lr=lr)

def explainer2graph(i,explainer,edge_mask,edge_index,threshold,y):
	assert edge_mask.size(0) == edge_index.size(1)
	subset, edge_index, _, hard_edge_mask = k_hop_subgraph(
		int(i), explainer.__num_hops__(), edge_index, relabel_nodes=True,
		num_nodes=None, flow=explainer.__flow__())

	edge_mask = edge_mask[hard_edge_mask]

	if threshold is not None:
		edge_mask = (edge_mask >= threshold).to(torch.float)

	if y is None:
		y = torch.zeros(edge_index.max().item() + 1,
						device=edge_index.device)
	else:
		y = y[subset].to(torch.float) / y.max().item()

	data = Data(edge_index=edge_index, att=edge_mask, y=y,
				num_nodes=y.size(0)).to('cpu')
	G = to_networkx(data, node_attrs=['y'], edge_attrs=['att'])
	mapping = {k: i for k, i in enumerate(subset.tolist())}
	G = nx.relabel_nodes(G, mapping)
	return G

def explain_nodes(G, model, task, y, node_idx=10, epochs=100, lr=0.01, threshold=None):
	assert task in ['clustering','classification']
	if task=='clustering':
		key='s'
	else:
		key='y'
	x=G.x
	edge_index=G.edge_index
	if node_idx == None:
		node_idx = np.arange(x.shape[0]).tolist()
	elif isinstance(node_idx,int):
		node_idx = [node_idx]
	node_idx=map(int,node_idx)
	model=ExplainerModel(model,key)
	explainer=create_explainer(model,epochs,lr)
	subgraphs={}
	for i in node_idx:
		try:
			node_feat_mask, edge_mask = explainer.explain_node(i, x, edge_index)
		except:
			node_feat_mask, edge_mask = torch.zeros(x.shape[1]).float(), torch.zeros(edge_index.shape[1]).float()
		subgraphs[i]=dict(node_feat=node_feat_mask.numpy(),
						  explain_graph=explainer2graph(i,explainer,edge_mask,edge_index,threshold,y))
	return subgraphs

def perturb_graph(G, perturb="none", erdos_flip_p=0.5):
	assert perturb in ['none','erdos','flip']
	x,edge_index=G.x,G.edge_index
	if perturb=='erdos':
		edge_index=erdos_renyi_graph(num_nodes=x.shape[0],p=erdos_flip_p)
	elif perturb=='flip':
		edge_index_pos=dropout_adj(edge_index,p=erdos_flip_p,num_nodes=x.shape[0])[0]
		edge_index_neg=negative_sampling(edge_index,num_nodes=x.shape[0],num_neg_samples=int(round(erdos_flip_p*x.shape[0])))
		edge_index=sort_edge_index(torch.hstack([edge_index_pos,edge_index_neg]))
	G.edge_index=edge_index
	return G
