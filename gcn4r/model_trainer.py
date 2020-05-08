import torch
import torch.nn as nn
import copy
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, confusion_matrix
from gcn4r.schedulers import Scheduler
from gcn4r.cluster import *
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import random
import pysnooper
from collections import defaultdict
sns.set(style='white')


class ModelTrainer:
	"""Trainer for the neural network model that wraps it into a scikit-learn like interface.

	Parameters
	----------
	model:nn.Module
		Deep learning pytorch model.
	n_epoch:int
		Number training epochs.
	optimizer_opts:dict
		Options for optimizer.
	scheduler_opts:dict
		Options for learning rate scheduler.
	loss_fn:str
		String to call a particular loss function for model.
	reduction:str
		Mean or sum reduction of loss.
	num_train_batches:int
		Number of training batches for epoch.
	"""
	def __init__(self, model,
						n_epoch=300,
						optimizer_opts=dict(name='adam',lr=1e-3,weight_decay=1e-4),
						scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2),
						loss_fn='ce',
						reduction='mean',
						num_train_batches=None,
						opt_level='O1',
						epoch_cluster=301,
						kl_warmup=20,
						K=10,
						Niter=10,
						lambdas=dict(),
						task='link_prediction',
						use_mincut=False,
						print_clusters=True,
						kmeans_use_probs=False):

		self.model = model
		self.kmeans_use_probs=kmeans_use_probs
		optimizers = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}
		loss_functions = {'bce':nn.BCEWithLogitsLoss(reduction=reduction), 'ce':nn.CrossEntropyLoss(reduction=reduction), 'mse':nn.MSELoss(reduction=reduction), 'nll':nn.NLLLoss(reduction=reduction)}
		if 'name' not in list(optimizer_opts.keys()):
			optimizer_opts['name']='adam'
		self.optimizer = optimizers[optimizer_opts.pop('name')](self.model.parameters(),**optimizer_opts)
		self.scheduler = Scheduler(optimizer=self.optimizer,opts=scheduler_opts)
		self.n_epoch = n_epoch
		self.task = task
		if self.task=='regression':
			loss_fn='mse'
		elif self.task=='classification':
			loss_fn='ce'
		self.loss_fn = loss_functions[loss_fn]
		self.loss_fn_name = loss_fn
		self.bce=(self.loss_fn_name=='bce')
		self.sigmoid = nn.Sigmoid()
		self.original_loss_fn = copy.deepcopy(loss_functions[loss_fn])
		self.num_train_batches = num_train_batches
		self.val_loss_fn = copy.deepcopy(loss_functions[loss_fn])
		self.centroids=None
		self.epoch_cluster=epoch_cluster
		self.add_cluster_loss=False
		self.K=K
		self.Niter=Niter
		self.cluster_loss_fn = ClusteringLoss(self.K, self.Niter,self.kmeans_use_probs)
		self.lambdas=lambdas
		self.use_mincut=use_mincut
		self.print_clusters=print_clusters
		self.add_kl=False
		self.kl_warmup=kl_warmup


	def establish_clusters(self, x, edge_index):
		z=self.model.encoder(x, edge_index)['z']
		cl,centroids=KMeans(torch.FloatTensor(z).cuda() if torch.cuda.is_available() else torch.FloatTensor(z),self.K,self.Niter)
		if self.print_clusters:
			print(' '.join(np.bincount(cl.numpy()).astype(str)))
		self.centroids=torch.tensor(centroids,dtype=torch.float)
		if torch.cuda.is_available():
			self.centroids=self.centroids.cuda()
		return self.centroids

	def prediction_loss(self,y_pred,y_true):
		return self.loss_fn(y_pred,y_true)

	# @pysnooper.snoop()
	def calc_loss(self, x, edge_index, val_edge_index=None, y=None, idx_df=None, loss_log=False):
		output=self.model.encoder(x, edge_index)
		if not self.use_mincut:
			z = output['z']
		else:
			z, s, mc1, o1 = output['z'],output['s'],output['mc1'],output['o1']#self.model.encode(x, edge_index)
			if self.print_clusters:
				print(' '.join(np.bincount(s.argmax(1).numpy()).astype(str)))
		# print(z.shape)
		if not isinstance(val_edge_index, type(None)):
			edge_index=val_edge_index
		losses=dict(cluster=0.,
					adv=0.,
					kl=0.,
					recon=0.,
					pred=0.)
		losses['recon'] = self.model.recon_loss(z, edge_index)
		if self.model.encoder.variational:
			losses['kl'] = self.model.kl_loss(output['mu'],output['logvar'])
		if self.model.encoder.adversarial:
			losses['adv'] = self.model.discriminator_loss(z)
		if self.model.encoder.prediction_task:
			idx=idx_df['idx'].values
			losses['pred'] = self.prediction_loss(output['y'][idx],y[idx])
		if self.add_cluster_loss:
			losses['cluster'] = (self.cluster_loss_fn(z,self.centroids)[0] if not self.use_mincut else mc1+o1)
		loss = losses['recon']
		for k in ['adv','kl','recon','cluster','pred']:
			loss+=self.lambdas.get(k,0.)*losses[k]
			if loss_log:
				self.loss_log[k].append(losses[k].item() if isinstance(losses[k],torch.FloatTensor) else 0. )
		# if self.model.encoder.variational:
		#     loss = loss + (1 / data.num_nodes) * model.kl_loss()
		return loss #self.loss_fn(y_pred, y_true)

	def calc_val_loss(self, x, edge_index):
		return self.calc_loss(x, edge_index)

	def reset_loss_fn(self):
		self.loss_fn = self.original_loss_fn

	def calc_best_confusion(self, y_pred, y_true):
		"""Calculate confusion matrix on validation set for classification/segmentation tasks, optimize threshold where positive.

		Parameters
		----------
		y_pred:array
			Predictions.
		y_true:array
			Ground truth.

		Returns
		-------
		float
			Optimized threshold to use on test set.
		dataframe
			Confusion matrix.

		"""
		fpr, tpr, thresholds = roc_curve(y_true, y_pred)
		threshold=thresholds[np.argmin(np.sum((np.array([0,1])-np.vstack((fpr, tpr)).T)**2,axis=1)**.5)]
		y_pred = (y_pred>threshold).astype(int)
		print(classification_report(y_true, y_pred))
		return threshold, pd.DataFrame(confusion_matrix(y_true,y_pred),index=['F','T'],columns=['-','+']).iloc[::-1,::-1].T

	def loss_backward(self,loss):
		"""Backprop using mixed precision for added speed boost.

		Parameters
		----------
		loss:loss
			Torch loss calculated.

		"""
		loss.backward()

	def train_loop(self, epoch, G):
		"""One training epoch, calculate predictions, loss, backpropagate.

		Parameters
		----------
		epoch:int
			Current epoch.
		train_dataloader:DataLoader
			Training data.

		Returns
		-------
		float
			Training loss for epoch

		"""
		self.model.train(True)
		starttime=time.time()
		if self.task == 'link_prediction':
			x,edge_index=G.x,G.train_pos_edge_index
		else:
			x,edge_index=G.x,G.edge_index
		if torch.cuda.is_available():
			x,edge_index = x.cuda(),edge_index.cuda()
		loss = self.calc_loss(x,edge_index,None,G.y,G.idx_df.loc[G.idx_df['set']=='train'] if self.model.encoder.prediction_task else None)
		train_loss=loss.item()
		self.optimizer.zero_grad()
		self.loss_backward(loss)
		self.optimizer.step()
		torch.cuda.empty_cache()
		endtime=time.time()
		print("Epoch {} Time:{}, Train Loss:{}".format(epoch,round(endtime-starttime,3),train_loss))
		self.scheduler.step()
		return train_loss

	def val_loop(self, epoch, G, print_val_confusion=False, save_predictions=False):
		"""Calculate loss over validation set.

		Parameters
		----------
		epoch:int
			Current epoch.
		val_dataloader:DataLoader
			Validation iterator.
		print_val_confusion:bool
			Calculate confusion matrix and plot.
		save_predictions:int
			Print validation results.

		Returns
		-------
		float
			Validation loss for epoch.
		"""
		self.model.train(False)
		if self.task == 'link_prediction':
			x,edge_index,val_edge_index=G.x,G.train_pos_edge_index,G.val_pos_edge_index
		else:
			x,edge_index,val_edge_index=G.x,G.edge_index,G.edge_index

		if torch.cuda.is_available():
			x,edge_index = x.cuda(),edge_index.cuda()

		loss = self.calc_loss(x,edge_index,val_edge_index,G.y,G.idx_df.loc[G.idx_df['set']=='val'] if self.model.encoder.prediction_task else None,loss_log=True) # .view(-1,1)
		val_loss=loss.item()
		print("Epoch {} Val Loss:{}".format(epoch,val_loss))

		return val_loss

	def test_loop(self, G):
		"""Calculate final predictions on loss.

		Parameters
		----------
		test_dataloader:DataLoader
			Test dataset.

		Returns
		-------
		array
			Predictions or embeddings.
		"""
		self.model.train((False if self.task!='generation' else True))
		with torch.no_grad():
			if self.task == 'link_prediction':
				x,edge_index,test_pos_edge_index,test_neg_edge_index=G.x,G.train_pos_edge_index,G.test_pos_edge_index,G.test_neg_edge_index
			else:
				num_nodes = G.num_nodes
				row, col = G.edge_index
				neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
				neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
				neg_adj_mask[row, col] = 0
				neg_row, neg_col = neg_adj_mask.nonzero().t()
				perm = random.sample(range(neg_row.size(0)),
									 G.edge_index.shape[1])
				perm = torch.tensor(perm)
				perm = perm.to(torch.long)
				neg_row, neg_col = neg_row[perm], neg_col[perm]

				x,edge_index,test_pos_edge_index,test_neg_edge_index = G.x,G.edge_index,G.edge_index,torch.stack([neg_row, neg_col], dim=0)


			if torch.cuda.is_available():
				x,edge_index = x.cuda(),edge_index.cuda()

			self.model.encoder.toggle_kmeans()
			output = self.model.encoder(x, edge_index)#[0]
			self.model.encoder.toggle_kmeans()
			z,s=output['z'],output['s']
			if not self.model.encoder.prediction_task:
				y=0.
			else:
				y=output['y']
			if not self.use_mincut:
				cl,c=KMeans(z, K=self.K, Niter=self.Niter, verbose=False)
				cl,c=cl.numpy(),c.numpy()
			else:
				cl,c=s.argmax(1).numpy(),0.

			A=self.model.decoder.forward_all(z).numpy()

			y_pred=np.hstack((self.model.decode(z,test_pos_edge_index).numpy().flatten(),self.model.decode(z,test_neg_edge_index).numpy().flatten()))
			y_test=np.hstack((np.ones(test_pos_edge_index.shape[1]),np.zeros(test_neg_edge_index.shape[1])))

			threshold,confusion=self.calc_best_confusion(y_pred, y_test)

			print("AUC={}, threshold={}".format(roc_auc_score(y_test,y_pred),threshold))

			print(confusion)

			z=z.numpy()

		return G,z,cl,c,A,threshold,s,y

	def fit(self, G, verbose=False, print_every=10, save_model=True, plot_training_curves=False, plot_save_file=None, print_val_confusion=True, save_val_predictions=True):
		"""Fits the segmentation or classification model to the patches, saving the model with the lowest validation score.

		Parameters
		----------
		train_dataloader:DataLoader
			Training dataset.
		verbose:bool
			Print training and validation loss?
		print_every:int
			Number of epochs until print?
		save_model:bool
			Whether to save model when reaching lowest validation loss.
		plot_training_curves:bool
			Plot training curves over epochs.
		plot_save_file:str
			File to save training curves.
		print_val_confusion:bool
			Print validation confusion matrix.
		save_val_predictions:bool
			Print validation results.

		Returns
		-------
		self
			Trainer.
		float
			Minimum val loss.
		int
			Best validation epoch with lowest loss.

		"""
		# choose model with best f1
		self.train_losses = []
		self.val_losses = []
		self.epochs = []
		self.loss_log = defaultdict(list)
		for epoch in range(self.n_epoch):
			if epoch >= self.epoch_cluster:
				self.add_cluster_loss=True
				if not self.use_mincut:
					self.centroids=self.establish_clusters(G.x, (G.train_pos_edge_index if self.task=='link_prediction' else G.edge_index))
			if epoch >= (self.epoch_cluster+self.kl_warmup):
				self.add_kl=True
			start_time=time.time()
			train_loss = self.train_loop(epoch,G)
			current_time=time.time()
			train_time=current_time-start_time
			self.train_losses.append(train_loss)
			val_loss = self.val_loop(epoch,G, print_val_confusion=False, save_predictions=False)
			val_time=time.time()-current_time
			self.loss_log['epoch'].append(epoch)
			self.loss_log['val_loss'].append(val_loss)
			if self.add_cluster_loss and self.add_kl:
				self.val_losses.append(val_loss)
			if False and verbose and not (epoch % print_every):
				if plot_training_curves:
					self.plot_train_val_curves(plot_save_file)
				print("Epoch {}: Train Loss {}, Val Loss {}, Train Time {}, Val Time {}".format(epoch,train_loss,val_loss,train_time,val_time))
			if self.add_cluster_loss and self.add_kl and val_loss <= min(self.val_losses) and save_model:
				print("New Best Model at Epoch {}".format(epoch))
				min_val_loss = val_loss
				best_epoch = epoch
				best_model = copy.deepcopy(self.model.state_dict())
		if save_model:
			self.model.load_state_dict(best_model)
		self.loss_log=dict(loss_log=pd.DataFrame(dict(self.loss_log)),best_epoch=best_epoch)
		return self, min_val_loss, best_epoch

	def plot_train_val_curves(self, save_file=None):
		"""Plots training and validation curves.

		Parameters
		----------
		save_file:str
			File to save to.

		"""
		plt.figure()
		sns.lineplot('epoch','value',hue='variable',
					 data=pd.DataFrame(np.vstack((np.arange(len(self.train_losses)),self.train_losses,self.val_losses)).T,
									   columns=['epoch','train','val']).melt(id_vars=['epoch'],value_vars=['train','val']))
		if save_file is not None:
			plt.savefig(save_file, dpi=300)

	def predict(self, test_dataloader):
		"""Make classification segmentation predictions on testing data.

		Parameters
		----------
		test_dataloader:DataLoader
			Test data.

		Returns
		-------
		array
			Predictions.

		"""
		y_pred = self.test_loop(test_dataloader)
		return y_pred

	def fit_predict(self, G):
		"""Fit model to training data and make classification segmentation predictions on testing data.

		Parameters
		----------
		train_dataloader:DataLoader
			Train data.
		test_dataloader:DataLoader
			Test data.

		Returns
		-------
		array
			Predictions.

		"""
		return self.fit(G)[0].predict(G)

	def return_model(self):
		"""Returns pytorch model.
		"""
		return self.model
