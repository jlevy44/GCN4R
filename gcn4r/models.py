import torch, numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.models.autoencoder import GAE, VGAE, ARGA, ARGVA
from torch_geometric.nn import GCNConv, GATConv, GraphConv, SAGEConv
from torch_geometric.nn.inits import glorot
from torch_geometric.nn import DenseGraphConv, dense_mincut_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj
from collections import defaultdict
import copy
from torch_geometric.utils import softmax

class GATConvInterpret(GATConv):
	def __init__(self, in_channels, out_channels, heads=1, concat=True,
				 negative_slope=0.2, dropout=0, bias=True, **kwarg):
		super(GATConvInterpret,self).__init__(in_channels, out_channels, heads, concat,
				 negative_slope, dropout, bias, **kwarg)
		self.attention_coefs=dict()#defaultdict(list)

	def message(self, edge_index_i, x_i, x_j, size_i):
		# Compute attention coefficients.
		x_j = x_j.view(-1, self.heads, self.out_channels)
		if x_i is None:
			alpha = (x_j * self.att[:, :, self.out_channels:]).sum(dim=-1)
		else:
			x_i = x_i.view(-1, self.heads, self.out_channels)
			alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)

		alpha = F.leaky_relu(alpha, self.negative_slope)
		alpha = softmax(alpha, edge_index_i, size_i)
		print(alpha.shape)
		# self.attention_coefs['edge_index_i']=edge_index_i
		self.attention_coefs['coef']=alpha

		# Sample attention coefficients stochastically.
		alpha = F.dropout(alpha, p=self.dropout, training=self.training)

		return x_j * alpha.view(-1, self.heads, 1)

class Encoder(nn.Module):
	def __init__(self, n_input, n_hidden, n_layers, conv_operator, variational=False, adversarial=False, bias=True, use_mincut=False, K=10):
		super(Encoder, self).__init__()
		self.convs=nn.ModuleList()
		conv_layer=conv_operator(n_input,n_hidden,bias=bias)
		glorot(conv_layer.weight)
		self.convs.append(conv_layer)
		for i in range(n_layers):
			conv_layer=conv_operator(n_hidden,n_hidden,bias=bias)
			glorot(conv_layer.weight)
			self.convs.append(conv_layer)
		self.adversarial=adversarial
		self.variational=variational
		self.use_mincut=use_mincut
		if self.use_mincut:
			self.pool1 = nn.Linear(n_hidden, K)
			# n_hidden=K

		if self.variational:
			self.convs=self.convs[:-1]
			conv_mu=conv_operator(n_hidden,n_hidden,bias=bias)
			conv_logvar=conv_operator(n_hidden,n_hidden,bias=bias)
			glorot(conv_mu.weight)
			glorot(conv_logvar.weight)
			self.conv_mu=conv_mu
			self.conv_logvar=conv_logvar
		self.n_hidden=n_hidden

	def forward(self, x, edge_index):
		z=x
		for conv in self.convs[:-1]:
			z=F.relu(conv(z,edge_index))
		if not self.variational:
			z=self.convs[-1](z,edge_index)
		if self.use_mincut:
			z_p, mask = to_dense_batch(z, None)
			adj = to_dense_adj(edge_index, None)
			s = self.pool1(z)
			# print(s.shape)
			# print(np.bincount(s.detach().argmax(1).numpy().flatten()))
			_, adj, mc1, o1 = dense_mincut_pool(z_p, adj, s, mask)
		if self.variational:
			output=[self.conv_mu(z,edge_index), self.conv_logvar(z,edge_index)]
		else:
			output=[z]
		if self.use_mincut:
			output.extend([s, mc1, o1])
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
				interpret=False):
	conv_operators=dict(GraphConv=GraphConv,
						GCNConv=GCNConv,
						GATConv=GATConv,
						SAGEConv=SAGEConv,
						GATConvInterpret=GATConvInterpret)
	ae_types=dict(GAE=GAE,
					VGAE=VGAE,
					ARGA=ARGA,
					ARGVA=ARGVA)
	assert encoder_base in (list(conv_operators.keys())+([] if not interpret else ['GATConvInterpret']))
	assert attention_heads == 1
	assert ae_type in list(ae_types.keys())
	assert decoder_type in ['inner']
	conv_operator = conv_operators[encoder_base]
	ae_model = ae_types[ae_type]
	encoder=Encoder(n_input, n_hidden, n_layers, conv_operator, (ae_type in ['VGAE','ARGVA']), (ae_type in ['ARGA','ARGVA']), bias, use_mincut=use_mincut, K=K)
	model_inputs=dict(encoder=encoder)
	if ae_type in ['ARGA','ARGVA']:
		model_inputs['discriminator']=Discriminator(encoder.n_hidden,discriminator_layers)
	model=ae_model(**model_inputs)
	return model
