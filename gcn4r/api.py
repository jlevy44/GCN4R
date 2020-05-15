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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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

def print_parameters(learning_rate,
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
		initialize_spectral):
	print(dict(learning_rate=learning_rate,
			n_epochs=n_epochs,
			encoder_base=encoder_base,
			n_hidden=n_hidden,
			n_layers=n_layers,
			discriminator_layers=discriminator_layers,
			ae_type=ae_type,
			bias=bias,
			attention_heads=attention_heads,
			decoder_type=decoder_type,
			model_save_loc=model_save_loc,
			predictions_save_path=predictions_save_path,
			predict=predict,
			lambda_kl=lambda_kl,
			lambda_adv=lambda_adv,
			lambda_cluster=lambda_cluster,
			epoch_cluster=epoch_cluster,
			K=K,
			Niter=Niter,
			sparse_matrix=sparse_matrix,
			feature_matrix=feature_matrix,
			custom_dataset=custom_dataset,
			val_ratio=val_ratio,
			test_ratio=test_ratio,
			random_seed=random_seed,
			task=task,
			use_mincut=use_mincut,
			initialize_spectral=initialize_spectral))

def get_data_model(custom_dataset,
					task,
					random_seed,
					sparse_matrix,
					feature_matrix,
					initialize_spectral,
					encoder_base,
					n_hidden,
					n_layers,
					discriminator_layers,
					ae_type,
					bias,
					attention_heads,
					decoder_type,
					use_mincut,
					K,
					Niter,
					val_ratio,
					test_ratio,
					interpret=False,
					prediction_column=-1
					):
	assert custom_dataset in ['lawyer', 'physician', 'none']
	assert task in ['link_prediction', 'generation', 'clustering', 'embedding', 'classification', 'regression']
	print("Random Seed:",random_seed)
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

	# print(sparse_matrix.shape)

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

	y=None
	idx_df=None
	label_encoder=None
	n_classes=-1
	if task in ['classification','regression']:
		X=pd.DataFrame(X)
		# print(X)
		assert prediction_column>=0 #in X.columns
		prediction_column=X.columns.values[prediction_column]
		y=X.pop(prediction_column).values.flatten()
		X=X.values
		# print(X,y)
		idx_df=pd.DataFrame(dict(idx=np.arange(len(y)),y=y))
		idx_df_train,idx_df_test=train_test_split(idx_df,test_size=test_ratio,stratify=idx_df['y'] if task=='classification' else None, random_state=random_seed)
		idx_df_train,idx_df_val=train_test_split(idx_df_train,test_size=val_ratio,stratify=idx_df_train['y'] if task=='classification' else None, random_state=random_seed)
		idx_df_train['set']='train'
		idx_df_val['set']='val'
		idx_df_test['set']='test'
		idx_df=pd.concat([idx_df_train,idx_df_val,idx_df_test])
		if task=='classification':
			n_classes=idx_df['y'].nunique()
			label_encoder=LabelEncoder()
			y=torch.tensor(label_encoder.fit_transform(y)).long()
		else:
			n_classes=1
			y=torch.FloatTensor(y)

	X=torch.FloatTensor(X)

	n_input = X.shape[1]

	edge_index,edge_attr=from_scipy_sparse_matrix(sparse_matrix)

	G=Data(X,edge_index,edge_attr,y=y,idx_df=idx_df)

	G.num_nodes = X.shape[0]

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
					K,
					Niter,
					interpret,
					n_classes)

	if task == 'link_prediction':
		G=model.split_edges(G, val_ratio=val_ratio, test_ratio=test_ratio)

	if torch.cuda.is_available():
		model=model.cuda()
	return G,model,X,edge_index,edge_attr

def train_model_(#inputs_dir,
				learning_rate=1e-4,
				n_epochs=300,
				encoder_base="GCNConv",
				n_hidden=30,
				n_layers=2,
				discriminator_layers=[20,20],
				ae_type="GAE",
				bias=True,
				attention_heads=1,
				decoder_type='inner',
				model_save_loc='saved_model.pkl',
				predictions_save_path='predictions.pkl',
				predict=False,
				lambda_kl=1e-3,
				lambda_adv=1e-3,
				lambda_cluster=1e-3,
				lambda_recon=1.,
				lambda_pred=0.,
				epoch_cluster=301,
				kl_warmup=20,
				K=10,
				Niter=10,
				sparse_matrix='A.npz',
				feature_matrix='X.npz',
				custom_dataset=None,
				val_ratio=0.05,
				test_ratio=0.1,
				random_seed=42,
				task='link_prediction',
				use_mincut=False,
				initialize_spectral=True,
				kmeans_use_probs=False,
				prediction_column=-1,
				animation_save_file='',
				**kwargs
				):

	optimizer_opts=dict(name='adam',
						lr=learning_rate,
						weight_decay=1e-4)

	scheduler_opts=dict(scheduler='warm_restarts',
						lr_scheduler_decay=0.5,
						T_max=10,
						eta_min=5e-8,
						T_mult=2)

	lambdas=dict(cluster=lambda_cluster,
				adv=lambda_adv,
				kl=lambda_kl,
				recon=lambda_recon,
				pred=lambda_pred)

	G,model,X,edge_index,edge_attr=get_data_model(custom_dataset,
													task,
													random_seed,
													sparse_matrix,
													feature_matrix,
													initialize_spectral,
													encoder_base,
													n_hidden,
													n_layers,
													discriminator_layers,
													ae_type,
													bias,
													attention_heads,
													decoder_type,
													use_mincut,
													K,
													Niter,
													val_ratio,
													test_ratio,
													prediction_column=prediction_column
													)


	trainer=ModelTrainer(model,
						n_epochs,
						optimizer_opts,
						scheduler_opts,
						epoch_cluster=epoch_cluster,
						kl_warmup=kl_warmup,
						K=K,
						Niter=Niter,
						lambdas=lambdas,
						task=task,
						use_mincut=use_mincut,
						kmeans_use_probs=kmeans_use_probs,
						return_animation=(True if animation_save_file else False))

	if not predict:

		trainer.fit(G)

		torch.save(trainer.model.state_dict(),model_save_loc)

		if trainer.return_animation:
			from functools import reduce
			pd.DataFrame(np.stack(trainer.Z),index=reduce(lambda x,y: x+y, [[i]*X.shape[0] for i in range(n_epochs)])).to_pickle(animation_save_file)

		return trainer.loss_log

	else:

		trainer.model.load_state_dict(torch.load(model_save_loc))

		_,z,cl,c,A,threshold,s,y=trainer.predict(G)

		G=Data(X,edge_index,edge_attr)

		output=dict(G=G,z=z,cl=cl,c=c,A=A,X=X.detach().cpu().numpy(),threshold=threshold,s=s,y=y)

		torch.save(output,predictions_save_path)

		return output

def interpret_model(custom_dataset='none',
					task='clustering',
					random_seed=42,
					sparse_matrix='A.npz',
					feature_matrix='X.npy',
					initialize_spectral=False,
					encoder_base='GCNConv',
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
					val_ratio=0.05,
					test_ratio=0.1,
					model_save_loc='saved_model.pkl',
					prediction_column=-1,
					mode='captum',
					method='integrated_gradients',
					epochs=100,
					lr=0.01,
					node_idx=None,
					threshold=None,
					**kwargs):
	from gcn4r.interpret import captum_interpret_graph, return_attention_scores
	assert mode in ['captum','attention','gnn_explainer']
	if mode=='attention':
		assert encoder_base=='GATConv'
		encoder_base='GATConvInterpret'
	elif mode=='gnn_explainer':
		assert encoder_base=='GCNConv'
	G,model,X,edge_index,edge_attr=get_data_model(custom_dataset,
													task,
													random_seed,
													sparse_matrix,
													feature_matrix,
													initialize_spectral,
													encoder_base,
													n_hidden,
													n_layers,
													discriminator_layers,
													ae_type,
													bias,
													attention_heads,
													decoder_type,
													use_mincut,
													K,
													Niter,
													val_ratio,
													test_ratio,
													prediction_column=prediction_column
													)
	model.load_state_dict(torch.load(model_save_loc))
	model.train(False)
	model.encoder.toggle_kmeans()
	attr_results={}
	if model.encoder.prediction_task:
		output_key='y'
	else:
		output_key='s'
	if mode in ['captum','gnn_explainer']:
		attr_results['cluster_assignments']=model.encoder(G.x, G.edge_index)[output_key]
	if mode=='captum':
		# add y from prediction
		# print(attr_results['cluster_assignments'])
		for i in range(K):
			attr_results[i]=captum_interpret_graph(G, model, use_mincut, target=i, method=method)
	elif mode=='attention':
		attr_results=return_attention_scores(G, model)
	else:
		try:
			from gcn4r.interpret import explain_nodes
		except:
			print("Please reinstall torch_geometric via the following command: pip uninstall torch-geometric && pip install git+https://github.com/rusty1s/pytorch_geometric.git")
			exit()
		assert use_mincut
		attr_results=explain_nodes(G, model, task, attr_results['cluster_assignments'], node_idx=node_idx, epochs=epochs, lr=lr, threshold=threshold)
	return attr_results

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
