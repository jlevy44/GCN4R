import torch
from torch_geometric.data import Data
import sys, os
import numpy as np, pandas as pd
import random
import gcn4r
from gcn4r.models import get_model
from gcn4r.model_trainer import ModelTrainer
from torch_geometric.utils.convert import from_scipy_sparse_matrix, to_networkx
import scipy.sparse as sps
import plotly.graph_objs as go
import plotly.offline as py
import pandas as pd, numpy as np
import networkx as nx
from networkx import kamada_kawai_layout, spring_layout
from sklearn.decomposition import PCA
import seaborn as sns
import cdlib
sns.set()

GCN4R_PATH = os.path.join(os.path.dirname(gcn4r.__file__), "data")
DATA = dict(physician=dict(A=os.path.join(GCN4R_PATH,'A_physician.csv'),
							X=os.path.join(GCN4R_PATH,'X_physician.csv')),
			lawyer=dict(A=os.path.join(GCN4R_PATH,'A_lawyer.npz'),
						X=os.path.join(GCN4R_PATH,'X_lawyer.npy')))

SEED=42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class PlotlyPlot:
	"""Creates plotly html plots."""
	def __init__(self):
		self.plots=[]

	def add_plot(self, t_data_df, G=None, color_col='color', name_col='name', xyz_cols=['x','y','z'], size=2, opacity=1.0, custom_colors=[]):
		"""Adds plotting data to be plotted.

		Parameters
		----------
		t_data_df:dataframe
			3-D transformed dataframe.
		G:nx.Graph
			Networkx graph.
		color_col:str
			Column to use to color points.
		name_col:str
			Column to use to name points.
		xyz_cols:list
			3 columns that denote x,y,z coords.
		size:int
			Marker size.
		opacity:float
			Marker opacity.
		custom_colors:list
			Custom colors to supply.
		"""
		plots = []
		x,y,z=tuple(xyz_cols)
		if t_data_df[color_col].dtype == np.float64:
			plots.append(
				go.Scatter3d(x=t_data_df[x], y=t_data_df[y],
							 z=t_data_df[z],
							 name='', mode='markers',
							 marker=dict(color=t_data_df[color_col], size=size, opacity=opacity, colorscale='Viridis',
							 colorbar=dict(title='Colorbar')), text=t_data_df[color_col] if name_col not in list(t_data_df) else t_data_df[name_col]))
		else:
			colors = t_data_df[color_col].unique()
			c = sns.color_palette('hls', len(colors))
			c = np.array(['rgb({})'.format(','.join(((np.array(c_i)*255).astype(int).astype(str).tolist()))) for c_i in c])#c = ['hsl(' + str(h) + ',50%' + ',50%)' for h in np.linspace(0, 360, len(colors) + 2)]
			if custom_colors:
				c = custom_colors
			color_dict = {name: c[i] for i,name in enumerate(sorted(colors))}

			for name,col in color_dict.items():
				plots.append(
					go.Scatter3d(x=t_data_df[x][t_data_df[color_col]==name], y=t_data_df[y][t_data_df[color_col]==name],
								 z=t_data_df[z][t_data_df[color_col]==name],
								 name=str(name), mode='markers',
								 marker=dict(color=col, size=size, opacity=opacity), text=t_data_df.index[t_data_df[color_col]==name] if 'name' not in list(t_data_df) else t_data_df[name_col][t_data_df[color_col]==name]))
		if G is not None:
			#pos = nx.spring_layout(G,dim=3,iterations=0,pos={i: tuple(t_data.loc[i,['x','y','z']]) for i in range(len(t_data))})
			Xed, Yed, Zed = [], [], []
			for edge in G.edges():
				if edge[0] in t_data_df.index.values and edge[1] in t_data_df.index.values:
					Xed += [t_data_df.loc[edge[0],x], t_data_df.loc[edge[1],x], None]
					Yed += [t_data_df.loc[edge[0],y], t_data_df.loc[edge[1],y], None]
					Zed += [t_data_df.loc[edge[0],z], t_data_df.loc[edge[1],z], None]
			plots.append(go.Scatter3d(x=Xed,
					  y=Yed,
					  z=Zed,
					  mode='lines',
					  line=go.scatter3d.Line(color='rgb(210,210,210)', width=2),
					  hoverinfo='none'
					  ))
		self.plots.extend(plots)

	def plot(self, output_fname, axes_off=False):
		"""Plot embedding of patches to html file.

		Parameters
		----------
		output_fname:str
			Output html file.
		axes_off:bool
			Remove axes.

		"""
		if axes_off:
			fig = go.Figure(data=self.plots,layout=go.Layout(scene=dict(xaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				yaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False),
				zaxis=dict(title='',autorange=True,showgrid=False,zeroline=False,showline=False,ticks='',showticklabels=False))))
		else:
			fig = go.Figure(data=self.plots)
		py.plot(fig, filename=output_fname, auto_open=False)


def train_model_(#inputs_dir,
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
				random_seed=42,
				task='link_prediction',
				use_mincut=False,
				initialize_spectral=True
				):

	assert custom_dataset in ['lawyer', 'physician', 'none']
	assert task in ['link_prediction', 'generation', 'clustering', 'embedding']
	torch.manual_seed(random_seed)
	np.random.seed(random_seed)
	random.seed(random_seed)

	if custom_dataset != 'none':
		sparse_matrix=DATA[custom_dataset]['A']
		feature_matrix=DATA[custom_dataset]['X']

	if isinstance(sparse_matrix,str) and os.path.exists(sparse_matrix) and sparse_matrix.split('.')[-1] in ['npz','csv']:
		if sparse_matrix.endswith('.csv'):
			sparse_matrix=sps.csr_matrix(pd.read_csv(sparse_matrix).values)
		else:
			sparse_matrix=sps.load_npz(sparse_matrix)
	elif not sps.issparse(sparse_matrix):
		sparse_matrix=sps.csr_matrix(sparse_matrix)

	print(sparse_matrix.shape)

	if isinstance(feature_matrix,str) and os.path.exists(feature_matrix) and feature_matrix.split('.')[-1] in ['npy','csv']:
		if feature_matrix.endswith('.csv'):
			X=pd.read_csv(feature_matrix).values.astype(float)
		else:
			X=np.load(feature_matrix,allow_pickle=True).astype(float)
	elif isinstance(feature_matrix,type(None)):
		if initialize_spectral:
			from sklearn.manifold import SpectralEmbedding
			X=SpectralEmbedding(n_components=3,affinity="precomputed",random_state=42).fit_transform(sparse_matrix)
		else:
			X=np.ones(sparse_matrix.shape[0],dtype=float)[:,np.newaxis]#np.eye(sparse_matrix.shape[0])*sparse_matrix.sum(axis=1)#modify#np.ones(sparse_matrix.shape[0],dtype=float)[:,np.newaxis]
	else:
		X=feature_matrix

	X=torch.FloatTensor(X)

	n_input = X.shape[1]

	edge_index,edge_attr=from_scipy_sparse_matrix(sparse_matrix)

	G=Data(X,edge_index,edge_attr)

	G.num_nodes = X.shape[0]

	lambdas=dict(cluster=lambda_cluster,
				adv=lambda_adv,
				kl=lambda_kl,
				recon=1.)

	model=get_model(encoder_base,
					n_input,
					n_hidden,
					n_layers,
					discriminator_layers,
					ae_type,
					bias,
					attention_heads,
					decoder_type,
					use_mincut,
					K)

	if task in 'link_prediction':
		G=model.split_edges(G, val_ratio=val_ratio, test_ratio=test_ratio)

	if torch.cuda.is_available():
		model=model.cuda()

	optimizer_opts=dict(name='adam',
						lr=learning_rate,
						weight_decay=1e-4)

	scheduler_opts=dict(scheduler='warm_restarts',
						lr_scheduler_decay=0.5,
						T_max=10,
						eta_min=5e-8,
						T_mult=2)

	trainer=ModelTrainer(model,
						n_epochs,
						optimizer_opts,
						scheduler_opts,
						epoch_cluster=epoch_cluster,
						K=K,
						Niter=Niter,
						lambdas=lambdas,
						task=task,
						use_mincut=use_mincut)

	if not predict:

		trainer.fit(G)

		torch.save(trainer.model.state_dict(),model_save_loc)

	else:

		trainer.model.load_state_dict(torch.load(model_save_loc))

		_,z,cl,c,A,threshold=trainer.predict(G)

		G=Data(X,edge_index,edge_attr)

		output=dict(G=G,z=z,cl=cl,c=c,A=A,X=X.detach().cpu().numpy(),threshold=threshold)

		torch.save(output,predictions_save_path)

		return output

def visualize_(predictions_save_path,
				use_predicted_graph,
				pos_threshold,
				layout,
				color_clusters,
				axes_off,
				output_fname,
				size):
	assert layout in ['spring','latent']
	pred=torch.load(predictions_save_path)
	G=to_networkx(pred['G']) if not use_predicted_graph else nx.from_scipy_sparse_matrix(sps.csr_matrix(pred['A']>=pos_threshold))
	t_data=pd.DataFrame((spring_layout(G,dim=3).values() if layout=='spring' else PCA(n_components=3,random_state=SEED).fit_transform(pred['z'])),columns=['x','y','z'])
	t_data['color']=pred['cl'] if color_clusters else 1
	pp=PlotlyPlot()
	pp.add_plot(t_data, G, size=size)
	pp.plot(output_fname=output_fname,axes_off=axes_off)
