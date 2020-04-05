from captum.attr import IntegratedGradients
import torch

def captum_interpret_graph(G, model, use_mincut, target=0, method='integrated_gradients'):
    assert use_mincut , "Interpretations only work for min-cut pooling for now"
    x,edge_index=G.x,G.edge_index
    print(x.shape,edge_index.shape)
    def custom_forward(x,edge_index):
        print(x.shape,edge_index.shape)
        return torch.cat([model.encode(x[i], edge_index[i])[1] for i in range(x.shape[0])],0)
    # custom_forward=(lambda x,edge_index:)
    interpretation_method=dict(integrated_gradients=IntegratedGradients)[method]
    ig = interpretation_method(custom_forward)
    attr = ig.attribute(x.unsqueeze(0), additional_forward_args=(edge_index.unsqueeze(0)), target=target)
    return attr

def return_attention_scores():
    pass
