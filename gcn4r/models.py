import torch, numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.models.autoencoder import GAE, VGAE, ARGA, ARGVA
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv
from gcn4r.cluster import KMeansLayer
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from collections import defaultdict
import copy
from torch_geometric.utils import softmax

class LatentSpaceDecoder(torch.nn.Module):
	def __init__(self,n_latent,bias=True):
		super(LatentSpaceDecoder,self).__init__()
		self.W=nn.Linear(n_latent,n_latent,bias=bias)

	def forward(self, z, edge_index, sigmoid=True):
		value = (z[edge_index[0]] * self.W(z[edge_index[1]])).sum(dim=1)
		return torch.sigmoid(value) if sigmoid else value

	def forward_all(self, z, sigmoid=True):
		adj = torch.matmul(z, self.W(z).t())
		return torch.sigmoid(adj) if sigmoid else adj

class LatentDistanceDecoder(torch.nn.Module):
	def __init__(self,n_latent,bias=True,p=2.0,no_space=True):
		super(LatentDistanceDecoder,self).__init__()
		self.p=p
		self.d=nn.PairwiseDistance(p=p)
		self.W=nn.Linear(n_latent,n_latent,bias=bias) if not no_space else torch.eye(n_latent)

	def forward(self, z, edge_index, sigmoid=True):
		value = self.d(z[edge_index[0]],self.W(z[edge_index[1]]))
		return torch.exp(-value) if sigmoid else value

	def forward_all(self, z, sigmoid=True):
		dist = torch.cdist(z,self.W(z),p=self.p)
		return torch.exp(-dist) if sigmoid else dist

class ExplainerModel(nn.Module):
	def __init__(self, model, key='s'):
		super(ExplainerModel, self).__init__()
		self.model=model
		self.key=key

	def forward(self, x, edge_index):
		z=self.model.encoder(x,edge_index)[self.key]
		return F.log_softmax(z, dim=1)

class GATConvInterpret(GATConv):
	def __init__(self, in_channels, out_channels, heads=1, concat=True,
				 negative_slope=0.2, dropout=0, bias=True, **kwarg):
		super(GATConvInterpret,self).__init__(in_channels, out_channels, heads, concat,
				 negative_slope, dropout, bias, **kwarg)
		self.attention_coefs=dict()#defaultdict(list)

	def forward(self,x,edge_index,**kwarg):
		x,(edge_index, alpha)=super().forward(x,edge_index,return_attention_weights=True)
		self.attention_coefs['coef']=alpha
		self.attention_coefs['edge_index']=edge_index
		return x
	# super(GATConvInterpret,self).__init__()
	# 	self.model=GATConv(in_channels, out_channels, heads, concat,
	# 			 negative_slope, dropout, bias, **kwarg)
	# 	self.attention_coefs=dict()#defaultdict(list)
	#
	# def reset_parameters(self):
	# 	self.model.reset_parameters()
	#
	# def forward(self,x,edge_index):
	# 	x,(edge_index, alpha)=self.model(x,edge_index,return_attention_weights=True)
	# 	self.attention_coefs['coef']=alpha
	# 	self.attention_coefs['edge_index']=edge_index
	# 	return x

class Encoder(nn.Module):
	def __init__(self, n_input, n_hidden, n_layers, conv_operator, variational=False, adversarial=False, bias=True, use_mincut=False, K=10, Niter=10, n_classes=-1):
		super(Encoder, self).__init__()
		self.convs=nn.ModuleList()
		conv_layer=conv_operator(n_input,n_hidden,bias=bias)
		conv_layer.reset_parameters()#glorot(conv_layer.weight)
		self.convs.append(conv_layer)
		for i in range(n_layers):
			conv_layer=conv_operator(n_hidden,n_hidden,bias=bias)
			conv_layer.reset_parameters()#glorot(conv_layer.weight)
			self.convs.append(conv_layer)
		self.adversarial=adversarial
		self.variational=variational
		self.use_mincut=use_mincut
		self.prediction_task=(n_classes>0)
		if self.use_mincut:
			self.pool1 = nn.Linear(n_hidden, K)
			# n_hidden=K
		else:
			self.kmeans = KMeansLayer(K=K,Niter=Niter)
		if self.prediction_task:
			self.classification_layer=nn.Linear(n_hidden,n_classes)
		if self.variational:
			# self.convs=self.convs[:-1]
			conv_mu=conv_operator(n_hidden,n_hidden,bias=bias)
			conv_logvar=conv_operator(n_hidden,n_hidden,bias=bias)
			conv_mu.reset_parameters()#glorot(conv_mu.weight)
			conv_logvar.reset_parameters()#glorot(conv_logvar.weight)
			self.conv_mu=conv_mu
			self.conv_logvar=conv_logvar
		self.n_hidden=n_hidden
		self.activate_kmeans=False
		self.relu=nn.ReLU()

	def toggle_kmeans(self):
		if not self.use_mincut:
			self.activate_kmeans=bool((int(self.activate_kmeans)+1)%2)

	def reparametrize(self, mu, logvar):
		if self.training:
			return mu + torch.randn_like(logvar) * torch.exp(logvar)
		else:
			return mu

	def forward(self, x, edge_index):
		z=x
		for conv in self.convs[:-1]:
			z=self.relu(conv(z,edge_index))
		# if not self.variational:
		z=self.convs[-1](z,edge_index)
		if self.use_mincut:
			z_p, mask = to_dense_batch(z, None)
			adj = to_dense_adj(edge_index, None)
			s = self.pool1(z)
			# print(s.shape)
			# print(np.bincount(s.detach().argmax(1).numpy().flatten()))
			_, adj, mc1, o1 = dense_mincut_pool(z_p, adj, s, mask)
		output=dict()
		if self.variational:
			output['mu'],output['logvar']=self.conv_mu(z,edge_index),self.conv_logvar(z,edge_index)
			output['z']=self.reparametrize(output['mu'],output['logvar'])
			# output=[self.conv_mu(z,edge_index), self.conv_logvar(z,edge_index)]
		else:
			output['z']=z
			# output=[z]
		if self.prediction_task:
			output['y']=self.classification_layer(z)
		if self.use_mincut:
			output['s']=s
			output['mc1']=mc1
			output['o1']=o1
			# output.extend([s, mc1, o1])
		elif self.activate_kmeans:
			s=self.kmeans(z)
			output['s']=s
			# output.extend([s])
		return output

class Discriminator(nn.Module):
	def __init__(self, n_input, hidden_layers=[30,20]):
		super(Discriminator, self).__init__()
		hidden_layers=[n_input]+hidden_layers+[1]
		self.fcs=[]
		for (i,layer) in enumerate(hidden_layers[:-1]):
			layer=nn.Linear(hidden_layers[i],hidden_layers[i+1])
			torch.nn.init.xavier_uniform(layer.weight)
			self.fcs.append(layer)
		for i in range(len(self.fcs)-1):
			self.fcs[i]=nn.Sequential(self.fcs[i], nn.ReLU())
		self.fcs[-1]=nn.Sequential(self.fcs[-1],nn.Sigmoid())
		self.fcs=nn.Sequential(*self.fcs)

	def forward(self, z):
		return self.fcs(z)

def get_model(encoder_base='GCNConv',
				n_input=30,
				n_hidden=30,
				n_layers=2,
				discriminator_layers=[20,20],
				ae_type='GAE',
				bias=True,
				attention_heads=1,
				decoder_type='inner',
				use_mincut=False,
				K=10,
				Niter=10,
				interpret=False,
				n_classes=-1,
				decoder="InnerProduct"):
	conv_operators=dict(GraphConv=GraphConv,
						GCNConv=GCNConv,
						GATConv=GATConv,
						SAGEConv=SAGEConv,
						GATConvInterpret=GATConvInterpret)
	ae_types=dict(GAE=GAE,
					VGAE=VGAE,
					ARGA=ARGA,
					ARGVA=ARGVA)
	decoder=dict(LatentSpace=LatentSpaceDecoder(n_latent=n_hidden,bias=bias),
					LatentDistance=LatentSpaceDecoder(n_latent=n_hidden,bias=bias,p=2.0),
					InnerProduct=None)[decoder]
	assert encoder_base in (list(conv_operators.keys())+([] if not interpret else ['GATConvInterpret']))
	assert attention_heads == 1
	assert ae_type in list(ae_types.keys())
	assert decoder_type in ['inner']
	conv_operator = conv_operators[encoder_base]
	ae_model = ae_types[ae_type]
	encoder=Encoder(n_input, n_hidden, n_layers, conv_operator, (ae_type in ['VGAE','ARGVA']), (ae_type in ['ARGA','ARGVA']), bias, use_mincut=use_mincut, K=K, Niter=Niter, n_classes=n_classes)
	model_inputs=dict(encoder=encoder)
	if ae_type in ['ARGA','ARGVA']:
		model_inputs['discriminator']=Discriminator(encoder.n_hidden,discriminator_layers)
	model_inputs['decoder']=decoder
	model=ae_model(**model_inputs)
	return model
