import torch
import torch.nn as nn
import copy
from sklearn.metrics import classification_report, roc_curve
from gcn4r.schedulers import Scheduler
from gcn4r.cluster import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white')


class ModelTrainer:
	"""Trainer for the neural network model that wraps it into a scikit-learn like interface.

	Parameters
	----------
	model:nn.Module
		Deep learning pytorch model.
	n_epoch:int
		Number training epochs.
	validation_dataloader:DataLoader
		Dataloader of validation dataset.
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
						validation_dataloader=None,
						optimizer_opts=dict(name='adam',lr=1e-3,weight_decay=1e-4),
						scheduler_opts=dict(scheduler='warm_restarts',lr_scheduler_decay=0.5,T_max=10,eta_min=5e-8,T_mult=2),
						loss_fn='ce',
						reduction='mean',
						num_train_batches=None,
						opt_level='O1',
						epoch_cluster=301,
						K=10,
						Niter=10,
						lambdas=dict()):

		self.model = model
		optimizers = {'adam':torch.optim.Adam, 'sgd':torch.optim.SGD}
		loss_functions = {'bce':nn.BCEWithLogitsLoss(reduction=reduction), 'ce':nn.CrossEntropyLoss(reduction=reduction), 'mse':nn.MSELoss(reduction=reduction), 'nll':nn.NLLLoss(reduction=reduction)}
		if 'name' not in list(optimizer_opts.keys()):
			optimizer_opts['name']='adam'
		self.optimizer = optimizers[optimizer_opts.pop('name')](self.model.parameters(),**optimizer_opts)
		self.scheduler = Scheduler(optimizer=self.optimizer,opts=scheduler_opts)
		self.n_epoch = n_epoch
		self.validation_dataloader = validation_dataloader
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
		self.cluster_loss_fn = ClusteringLoss(self.K, self.Niter)
		self.lambdas=lambdas

	def establish_clusters(self, x, edge_index):
		z=self.model.encode(x, edge_index)
		self.centroids=torch.tensor(KMeans(torch.FloatTensor(z).cuda() if torch.cuda.is_available() else torch.FloatTensor(z),self.K,self.Niter)[1],dtype=torch.float)
		if torch.cuda.is_available():
			self.centroids=self.centroids.cuda()
		return self.centroids

	def calc_loss(self, x, edge_index, val_edge_index=None):
		z = self.model.encode(x, edge_index)
		if val_edge_index:
			edge_index=val_edge_index
		losses=dict(cluster=0.,
					adv=0.,
					kl=0.,
					recon=0.))
		losses['recon'] = self.model.recon_loss(z, edge_index)
		if self.model.encoder.variational:
			losses['kl'] = self.model.kl_loss()
		if self.model.encoder.adversarial:
			losses['adv'] = self.model.discriminator_loss(z)
		if self.add_cluster_loss:
			losses['cluster'] = self.cluster_loss_fn(z,self.centroids)
		loss = losses['recon']
		for k in ['adv','kl','recon']:
			loss+=self.lambdas[k]*losses[k]
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
		x,edge_index=G.x,G.train_pos_edge_index

		if torch.cuda.is_available():
			x,edge_index = x.cuda(),edge_index.cuda()

		loss = self.calc_loss(x,edge_index)
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
		x,edge_index,val_edge_index=G.x,G.train_pos_edge_index,G.val_pos_edge_index

		if torch.cuda.is_available():
			x,edge_index = x.cuda(),edge_index.cuda()

		loss = self.calc_loss(x,edge_index,val_edge_index) # .view(-1,1)
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

		x,edge_index,test_edge_index=G.x,G.train_pos_edge_index,G.test_pos_edge_index

		if torch.cuda.is_available():
			x,edge_index = x.cuda(),edge_index.cuda()

		z = self.model.encode(x, edge_index)

		cl,c=KMeans(z, K=self.K, Niter=self.Kiter, verbose=False)

		A=self.model.decode(z)

		return G,z,cl,c,A

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
		for epoch in range(self.n_epoch):
			if epoch >= self.epoch_cluster:
				self.add_cluster_loss=True
				self.centroids=self.establish_clusters(G.x, G.train_pos_edge_index)
			start_time=time.time()
			train_loss = self.train_loop(epoch,G)
			current_time=time.time()
			train_time=current_time-start_time
			self.train_losses.append(train_loss)
			val_loss = self.val_loop(epoch,G, print_val_confusion=False, save_predictions=False)
			val_time=time.time()-current_time
			self.val_losses.append(val_loss)
			if False and verbose and not (epoch % print_every):
				if plot_training_curves:
					self.plot_train_val_curves(plot_save_file)
				print("Epoch {}: Train Loss {}, Val Loss {}, Train Time {}, Val Time {}".format(epoch,train_loss,val_loss,train_time,val_time))
			if val_loss <= min(self.val_losses) and save_model:
				min_val_loss = val_loss
				best_epoch = epoch
				best_model = copy.deepcopy(self.model)
		if save_model:
			self.model = best_model
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
