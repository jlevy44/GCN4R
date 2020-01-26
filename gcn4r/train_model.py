import fire
import torch
from torch_geometric.data import Data
import sys, os
import numpy as np, pandas as pd
import gcn4r
from gcn4r.models import get_model
from gcn4r.model_trainer import ModelTrainer
from torch_geometric.utils.convert import from_scipy_sparse_matrix
import scipy.sparse as sps

SEED=42
GCN4R_PATH = os.path.join(os.path.dirname(gcn4r.__file__), "data")
DATA = dict(physician=dict(A=os.path.join(GCN4R_PATH,'A_physician.csv'),
							X=os.path.join(GCN4R_PATH,'X_physician.csv')),
			lawyer=dict(A=os.path.join(GCN4R_PATH,'A_lawyer.npy'),
						X=os.path.join(GCN4R_PATH,'X_lawyer.npz')))


def train_model(inputs_dir='inputs',
				learning_rate=1e-4,
				n_epochs=300,
				encoder_base='GCNConv',
				n_hidden=30,
				n_layers=2,
				discriminator_layers=[20,20],
				ae_type='GAE',
				bias=True,
				attention_heads=1,
				decoder_type='inner',
				model_save_loc='saved_model.pkl',
				predictions_save_path='predictions.pkl',
				predict=False,
				lambda_kl=1e-3,
				lambda_adv=1e-3,
				lambda_cluster=1e-3,
				epoch_cluster=301,
				K=10,
				Niter=10,
				sparse_matrix='A.npz',
				feature_matrix='X.npy',
				custom_dataset='none'
				):

	assert custom_dataset in ['lawyer', 'physician', 'none']

	if custom_dataset != 'none':
		sparse_matrix=DATA[custom_dataset]['A']
		feature_matrix=DATA[custom_dataset]['X']

	if isinstance(sparse_matrix,str) and os.path.exists(sparse_matrix) and feature_matrix.split('.')[-1] in ['npz','csv']:
		if sparse_matrix.endswith('.csv'):
			sparse_matrix=sps.csr_matrix(pd.read_csv(sparse_matrix).values)
		else:
			sparse_matrix=sps.load_npz(sparse_matrix)

	if isinstance(feature_matrix,str) and os.path.exists(feature_matrix) and feature_matrix.split('.')[-1] in ['npy','csv']:
		if feature_matrix.endswith('.csv'):
			X=pd.read_csv(feature_matrix).values
		else:
			X=np.load(feature_matrix)
	else:
		X=feature_matrix

	n_input = X.shape[1]

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

	np.random.seed(SEED)

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

		output=dict(G=G,z=z,cl=cl,c=c,A=A)

		torch.save(output,predictions_save_path)

		return output

def main():
	fire.Fire(train_model)

if __name__=='__main__':
	main()
