import torch
from captum.attr import IntegratedGradients,Saliency,DeepLift,InputXGradient,GuidedBackprop,Deconvolution


def captum_interpret_graph(G, model, use_mincut, target=0, method='integrated_gradients'):
	# assert use_mincut , "Interpretations only work for min-cut pooling for now"
	x,edge_index=G.x,G.edge_index
	print(x.shape,edge_index.shape)
	if model.encoder.prediction_task:
		output_key='y'
	else:
		output_key='s'
	def custom_forward(x,edge_index):
		print(x.shape,edge_index.shape)
		return torch.cat([model.encoder(x[i], edge_index[i])[output_key] for i in range(x.shape[0])],0)
	# custom_forward=(lambda x,edge_index:)
	interpretation_method=dict(integrated_gradients=IntegratedGradients,
							   saliency=Saliency,
							   deeplift=DeepLift,
							   input_x_gradient=InputXGradient)
							   # guided_backprop=GuidedBackprop,
							   # deconvolution=Deconvolution
	assert method in list(interpretation_method.keys())
	interpretation_method=interpretation_method[method]
	ig = interpretation_method(custom_forward)
	attr = ig.attribute(x.unsqueeze(0), additional_forward_args=(edge_index.unsqueeze(0)), target=target)
	return attr

def return_attention_scores(G, model):
	x,edge_index=G.x,G.edge_index
	outputs=model.encoder(x,edge_index)
	attention_scores=[]
	for conv in model.encoder.convs:
		conv.attention_coefs['edge_index']=edge_index
		attention_scores.append(conv.attention_coefs)
	return attention_scores
