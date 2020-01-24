import fire
import torch
from torch_geometric.data import Data
import sys, os
import numpy as np, pandas as pd
from models import generate_model, ModelTrainer
from torch_geometric.utils.convert import from_scipy_sparse_matrix

def train_model(inputs_dir='inputs',
				learning_rate=1e-4,
				n_epochs=300,
				encoder_base='GCNConv',
				n_input=30,
				n_hidden=30,
				n_layers=2,
				discriminator_layers=[20,20],
				ae_type='GAE',
				bias=True,
				attention_heads=1,
				decoder_type='inner',
				model_save_loc='saved_model.pkl',
				predictions_save_path='predictions.pkl',
				predict_set='test',
				lambda_kl=1e-3,
				lambda_adv=1e-3,
				lambda_cluster=1e-3,
				epoch_cluster=301,
				K=10,
				Niter=10,
				sparse_matrix='A.npz',
				feature_matrix='X.npy'
				):

	if isinstance(sparse_matrix,str):
		sparse_matrix=sps.load_npz(sparse_matrix)

	if isinstance(feature_matrix,str):
		X=np.load(feature_matrix)
	else:
		X=feature_matrix

	edge_index,edge_attr=from_scipy_sparse_matrix(sparse_matrix)

	G=Data(X,edge_index,edge_attr)

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
					decoder_type)

	G=model.split_edges(G, val_ratio=0.05, test_ratio=0.1)

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
						loss_fn='ce',
						epoch_cluster=epoch_cluster,
						K=K,
						Niter=Niter,
						lambdas=lambdas)

	if not predict:

		trainer.fit(G)

		torch.save(trainer.model.state_dict(),model_save_loc)

	else:

		trainer.model.load_state_dict(torch.load(model_save_loc))

		G,z,cl,c,A=trainer.predict(G)

		torch.save(dict(G=G,
						z=z,
						cl=cl,
						c=c,
						A=A),predictions_save_path)


if __name__=='__main__':
	fire.Fire(train_model)
