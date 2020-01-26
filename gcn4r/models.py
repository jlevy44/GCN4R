import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn.models.autoencoder import GAE, VGAE, ARGA, ARGVA
from torch_geometric.nn import GCNConv, GATConv


def get_model(encoder_base='GCNConv',
				n_input=30,
				n_hidden=30,
				n_layers=2,
				discriminator_layers=[20,20],
				ae_type='GAE',
				bias=True,
				attention_heads=1,
				decoder_type='inner'):
	conv_operators=dict(GraphConv=GraphConv,
						GCNConv=GCNConv,
						GATConv=GATConv,
						SAGEConv=SAGEConv)
	ae_types=dict(GAE=GAE,
					VGAE=VGAE,
					ARGA=ARGA,
					ARGVA=ARGVA)
	assert encoder_base in list(conv_operators.keys())
	assert attention_heads == 1
	assert ae_type in list(ae_types.keys())
	assert decoder_type in ['inner']
	conv_operator = conv_operators[encoder_base]
	ae_model = ae_types[ae_type]
	model_inputs=dict(encoder=Encoder(n_input, n_hidden, n_layers, conv_operator, (ae_type in ['VGAE','ARGVA']), (ae_type in ['ARGA','ARGVA']), bias)))
	if ae_type in ['ARGA','ARGVA']:
		model_inputs['discriminator']=Discriminator(n_hidden,discriminator_layers)
	model=ae_model(**model_inputs)
	return model


class Encoder(nn.Module):
	def __init__(self, n_input, n_hidden, n_layers, conv_operator, variational=False, adversarial=False, bias=True):
		super(Encoder, self).__init__()
		self.convs=[]
		self.convs.append(conv_operator(n_input,n_hidden,bias=bias))
		for i in range(self.n_layers):
			self.convs.append(conv_operator(n_hidden,n_hidden,bias=bias))
		self.variational=variational
		if self.variational:
			self.convs=self.convs[:-1]
			self.conv_mu=conv_operator(n_hidden,n_hidden,bias=bias)
			self.conv_logvar=conv_operator(n_hidden,n_hidden,bias=bias)

	def forward(self, x, edge_index):
		z=x
		for conv in self.convs[:-1]:
			z=F.relu(conv(z,edge_index))
		if self.variational:
			return self.conv_mu(z,edge_index), self.conv_logvar(z,edge_index)
		else:
			return self.convs[-1](z,edge_index)

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
