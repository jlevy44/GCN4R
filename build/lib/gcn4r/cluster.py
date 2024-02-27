import torch
import torch.nn as nn
from torch_cluster.nearest import nearest
import time
# import pysnooper

def KMeans(x, K=10, Niter=10, verbose=True):
	# https://github.com/jeanfeydy/geomloss/blob/60785fdbb6e9e8f2689d881e3fc027887a2ac4e4/geomloss/examples/sinkhorn_multiscale/plot_optimal_transport_cluster.py
	N, D = x.shape
	start = time.time()
	perm = torch.randperm(N)
	idx = perm[:K]
	c = x[idx, :].clone()

	for i in range(Niter):
		cl  = nearest(x,c)
		Ncl = torch.bincount(cl, minlength=K).type(torch.float)
		for d in range(D):
			# print(torch.bincount(cl, weights=x[:, d], minlength=K))
			c[:, d] = torch.bincount(cl, weights=x[:, d], minlength=K) / Ncl
		c[torch.isnan(c)]=0
	end = time.time()
	return cl, c

class KMeansLayer(nn.Module):
	def __init__(self,K=10,Niter=10):
		super(KMeansLayer,self).__init__()
		self.K=K
		self.Niter=Niter

	@staticmethod
	def calculate_distances(x, centroids):
		assert x.size(1) == centroids.size(1), "Dimension mismatch"
		return ((x[:, None]-centroids[None])**2).sum(2)

	@staticmethod
	def calc_probs(distances):
		return 1.-KMeansLayer.normalize_distance(distances)

	@staticmethod
	def normalize_distance(distances):
		return distances/distances.sum(1,keepdim=True)[0]

	@staticmethod
	def calculate_loss(encode_output=None, centroids=None, use_probs=False):
		return sq_loss_clusters(encode_output,centroids,use_probs)

	def forward(self, x):
		_,centroids=KMeans(x.detach(), K=self.K, Niter=self.Niter, verbose=False)
		distances=KMeansLayer.calculate_distances(x,centroids)
		probs=KMeansLayer.calc_probs(distances)
		return probs

# https://discuss.pytorch.org/t/k-means-loss-calculation/22041/7
def sq_loss_clusters(encode_output=None, centroids=None, use_probs=False):
	distances=KMeansLayer.calculate_distances(encode_output, centroids)
	if use_probs:
		distances=KMeansLayer.normalize_distance(distances)
	return distances.min(1)[0].mean()

# @pysnooper.snoop()
def clustering_loss(Z,K=10,Niter=10,centroids=None,use_probs=False):
	Z2=Z.detach()
	if isinstance(centroids,type(None)):
		cl,centroids=KMeans(Z2, K=K, Niter=Niter, verbose=False)
	else:
		cl=nearest(Z2,centroids)
	loss=KMeansLayer.calculate_loss(Z,centroids, use_probs=use_probs)
	return loss,cl

class ClusteringLoss(nn.Module): # maybe do at start every epoch,
	def __init__(self,K=10,Niter=10,kmeans_use_probs=False):
		super(ClusteringLoss,self).__init__()
		self.K=K
		self.Niter=Niter
		self.kmeans_use_probs=kmeans_use_probs

	def forward(self, Z, centroids=None):
		return clustering_loss(Z,self.K,self.Niter, centroids, use_probs=self.kmeans_use_probs)
