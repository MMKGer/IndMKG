import torch
from torch import nn
import numpy as np
from . import tasks, layers
from sklearn.cluster import KMeans
import torch.nn.functional as F

class IndMKG(nn.Module):
    def __init__(self, rel_model_cfg, entity_model_cfg):
        super(IndMKG, self).__init__()
        self.relation_model = globals()[rel_model_cfg.pop('class')](**rel_model_cfg)
        self.entity_model = globals()[entity_model_cfg.pop('class')](**entity_model_cfg)
        self.GetRepresentations = GetRepresentations(entity_model_cfg.input_dim, num_mlp_layers=2)

    def forward(self, data, batch):
        query_rels = batch[:, 0, 2]
        text_feature, clusters_text, cluster_centers_text, img_feature, clusters_img, cluster_centers_img = self.GetRepresentations(data)
        relation_representations = self.relation_model(data.relation_graph, query=query_rels)
        score = self.entity_model(data, relation_representations, batch, text_feature=text_feature,
                                                  img_feature=img_feature, clusters_text=clusters_text, cluster_centers_text=cluster_centers_text,
                                                  clusters_img=clusters_img, cluster_centers_img=cluster_centers_img ,modality="all")
        return score

class GetRepresentations(torch.nn.Module):
###
###
###
        return text_feature, clusters_text, cluster_centers_text, img_feature, clusters_img, cluster_centers_img


class Relationmodel(Basemodel):

    def __init__(self, input_dim, hidden_dims, num_relation=4, **kwargs):
        super().__init__(input_dim, hidden_dims, num_relation, **kwargs)

        self.layers = nn.ModuleList()
        for i in range(len(self.dims) - 1):
            self.layers.append(
                layers.GeneralizedRelationalConv(
                    self.dims[i], self.dims[i + 1], num_relation,
                    self.dims[0], self.message_func, self.aggregate_func, self.layer_norm,
                    self.activation, dependent=False)
                )

        if self.concat_hidden:
            feature_dim = sum(hidden_dims) + input_dim
            self.mlp = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.ReLU(),
                nn.Linear(feature_dim, input_dim)
            )

    def bellmanford(self, data, h_index, modality="structural", separate_grad=False):
        batch_size = len(h_index)
        query = torch.ones(h_index.shape[0], self.dims[0], device=h_index.device, dtype=torch.float)
        index = h_index.unsqueeze(-1).expand_as(query)
        boundary = torch.zeros(batch_size, data.num_nodes, self.dims[0], device=h_index.device)
        boundary.scatter_add_(1, index.unsqueeze(1), query.unsqueeze(1))
        size = (data.num_nodes, data.num_nodes)
        edge_weight = torch.ones(data.num_edges, device=h_index.device)
        hiddens = []
        edge_weights = []
        layer_input = boundary

        for layer in self.layers:
            hidden = layer(layer_input, query, boundary, data.edge_index, data.edge_type, size, edge_weight)
            if self.short_cut and hidden.shape == layer_input.shape:
                hidden = hidden + layer_input
            hiddens.append(hidden)
            edge_weights.append(edge_weight)
            layer_input = hidden
        node_query = query.unsqueeze(1).expand(-1, data.num_nodes, -1)
        if self.concat_hidden:
            output = torch.cat(hiddens + [node_query], dim=-1)
            output = self.mlp(output)
        else:
            output = hiddens[-1]
        return {
            "node_feature": output,
            "edge_weights": edge_weights,
        }

    def forward(self, rel_graph, query, modality="structural"):
        output = self.bellmanford(rel_graph, h_index=query, modality=modality)["node_feature"]
        return output


class Entitymodel(Basemodel):
#######
#######
#######
#######
#######
        return score

class ClusterCentersAdjuster(torch.nn.Module):
#######
#######
#######
#######
#######
        return adjusted_features, clusters, cluster_centers

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x