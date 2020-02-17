import fire, os
import gcn4r
from gcn4r.api import *

class Commands(object):
	def __init__(self):
		pass
	# TODO: ADD https://danielegrattarola.github.io/posts/2019-07-25/mincut-pooling.html
	def train_model(self,
					# inputs_dir='inputs',
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
					custom_dataset='none',
					val_ratio=0.05,
					test_ratio=0.1,
					task='clustering'
					):

		train_model_(#inputs_dir,
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
						task=task
						)

	def visualize(self,
					predictions_save_path='predictions.pkl',
					use_predicted_graph=False,
					pos_threshold=0.5,
					layout='latent',
					color_clusters=False,
					axes_off=False,
					output_fname='network_plot.html',
					size=4):
		visualize_(predictions_save_path,
						use_predicted_graph,
						pos_threshold,
						layout,
						color_clusters,
						axes_off,
						output_fname,
						size)

def main():
	fire.Fire(Commands)

if __name__=='__main__':
	main()
