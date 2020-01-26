import torch
from torch_cluster.knn import nearest
import time

def KMeans(x, K=10, Niter=10, verbose=True):
	# https://github.com/jeanfeydy/geomloss/blob/60785fdbb6e9e8f2689d881e3fc027887a2ac4e4/geomloss/examples/sinkhorn_multiscale/plot_optimal_transport_cluster.py
	N, D = x.shape
	start = time.time()
	perm = torch.randperm(N)
	idx = perm[:K]
	c = x[idx, :].clone()

	for i in range(Niter):
		cl  = nearest(x,c)
		Ncl = torch.bincount(cl).type(torch.float)
		for d in range(D):
			c[:, d] = torch.bincount(cl, weights=x[:, d]) / Ncl
		c[torch.isnan(c)]=0
	end = time.time()
	return cl, c

# https://discuss.pytorch.org/t/k-means-loss-calculation/22041/7
def sq_loss_clusters(encode_output, centroids):
	assert encode_output.size(1) == centroids.size(1), "Dimension mismatch"
	return ((encode_output[:, None]-centroids[None])**2).sum(2).min(1)[0].mean()

def clustering_loss(Z,K=10,Niter=10,centroids=None):
	Z2=Z.detach()
	if isinstance(centroids,type(None)):
		cl,centroids=KMeans(Z2, K=K, Niter=Niter, verbose=False)
	else:
		cl=nearest(Z2,centroids)
	loss=sq_loss_clusters(Z,centroids)
	return loss,cl

class ClusteringLoss(nn.Module): # maybe do at start every epoch,
	def __init__(self,K=10,Niter=10):
		super(ClusteringLoss,self).__init__()
		self.K=K
		self.Niter=Niter

	def forward(self, Z, centroids=None):
		return clustering_loss(Z,self.K,self.Niter, centroids)
