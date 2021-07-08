import os, sys, inspect, torch, csv, copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.nn import GATConv

import torch.nn as nn
import torch.nn.functional as F

from rdkit.Chem.Draw import SimilarityMaps

from models.mp_layers import *
from models.tetra import *
from visualization.visualization import visualize_atom_attention

SUPPORTED_ATTN_TYPE =  ['gat', 'tang']

class GNN(nn.Module):
    def __init__(self, args, num_node_features, num_edge_features):
        super(GNN, self).__init__()
        self.args = args
        if args.attn_type == 'tang' and args.heads != 1:
          raise RuntimeError('tang attention must have heads = 1.')
        self.attn_type = args.attn_type

        if not self.attn_type in SUPPORTED_ATTN_TYPE:
          raise RuntimeError(f'Attention type {self.attn_type} not supported')
        self.depth = args.depth
        self.hidden_size = args.hidden_size
        self.dropout = args.dropout
        self.gnn_type = args.gnn_type
        self.graph_pool = args.graph_pool
        self.tetra = args.tetra
        self.task = args.task
        self.n_layers = args.n_layers
        self.skip_coef = args.skip_coef
        self.gat_act = args.gat_act

        # attention layer
        if self.attn_type == 'gat':
          self.gat_depth = args.gat_depth
          self.heads = args.heads
          self.attn_dropout = args.attn_dropout
          self.gat = torch.nn.ModuleList()
          # Set activation function
          if self.gat_act == 'relu':
            self.gat_act_func = nn.ReLU()
          elif self.gat_act == 'leakyrelu':
            if args.alpha is None:
              raise RuntimeError('LeakyRelu activation is used but alpha is not specified')
            self.gat_act_func = nn.LeakyReLU(negative_slope=args.alpha)
          elif self.gat_act == 'sigmoid':
            self.gat_act_func = nn.sigmoid()
          else:
            raise RuntimeError('activation function not supported')

          if 'concat' in dir(args):
            self.concat = args.concat
            if self.concat and self.gat_depth == 1:
                raise RuntimeError('Cannot set concatenation when there is only 1 attention layer')
            if not self.concat:
                for d in range(self.gat_depth):
                    self.gat.append(GATConv(self.hidden_size, self.hidden_size, heads=self.heads, dropout=self.attn_dropout, concat=False))
            else:
                self.gat.append(GATConv(self.hidden_size, self.hidden_size, heads=self.heads, dropout=self.attn_dropout, concat=True))
                for d in range(self.gat_depth-1):
                    if d == self.gat_depth-2: # last layer can't use concat
                        self.gat.append(GATConv(self.heads*self.hidden_size, self.hidden_size, heads=self.heads, dropout=self.attn_dropout, concat=False))
                    else:
                        self.gat.append(GATConv(self.heads*self.hidden_size, self.hidden_size, heads=self.heads, dropout=self.attn_dropout, concat=True))
          elif 'concat' not in dir(args):
            # self.concat = args.concat
            for _ in range(self.gat_depth):
                # self.gat.append(GATLayer(in_dim=self.hidden_size, out_dim=self.hidden_size))
                self.gat.append(GATConv(self.hidden_size, self.hidden_size, heads=self.heads, dropout=self.attn_dropout, concat=False))
            # self.gat.append(GATConv(self.hidden_size, self.hidden_size, heads=self.heads, dropout=self.gat_dropout, concat=False))

        elif self.attn_type == 'tang':
          self.heads = args.heads
          self.attn_dropout = args.attn_dropout

        if self.gnn_type == 'dmpnn':
            self.edge_init = nn.Linear(num_node_features + num_edge_features, self.hidden_size)
            self.edge_to_node = DMPNNConv(args)
        elif self.gnn_type == 'orig_dmpnn':
            self.edge_init = nn.Linear(num_node_features + num_edge_features, self.hidden_size)
            self.edge_to_node = OrigDMPNNConv(args, node_agg=True)
        else:
            self.node_init = nn.Linear(num_node_features, self.hidden_size)
            self.edge_init = nn.Linear(num_edge_features, self.hidden_size)

        # layers
        self.convs = torch.nn.ModuleList()

        for d in range(self.depth):
            if args.ft_boost and d>0:
                custom_hidden_size=args.hidden_size+3
            else:
                custom_hidden_size = None
            if self.gnn_type == 'gin':
                self.convs.append(GINEConv(args, custom_hidden_size=custom_hidden_size))
            elif self.gnn_type == 'gcn':
                self.convs.append(GCNConv(args, custom_hidden_size=custom_hidden_size))
            elif self.gnn_type == 'dmpnn':
                self.convs.append(DMPNNConv(args, custom_hidden_size=custom_hidden_size))
            elif self.gnn_type == 'orig_dmpnn':
                self.convs.append(OrigDMPNNConv(args, custom_hidden_size=custom_hidden_size))
            else:
                ValueError('Undefined GNN type called {}'.format(self.gnn_type))

        # graph pooling
        if self.tetra:
            self.tetra_update = get_tetra_update(args)

        if self.graph_pool == "sum":
            self.pool = global_add_pool
        elif self.graph_pool == "mean":
            self.pool  = global_mean_pool
        elif self.graph_pool == "max":
            self.pool = global_max_pool
        elif self.graph_pool == "attn":
            self.pool = GlobalAttention(
                gate_nn=torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 2 * self.hidden_size),
                                           torch.nn.BatchNorm1d(2 * self.hidden_size),
                                            torch.nn.ReLU(),
                                            torch.nn.Linear(2 * self.hidden_size, 1)))
        elif self.graph_pool == "set2set":
            self.pool = Set2Set(self.hidden_size, processing_steps=2)
        else:
            raise ValueError("Invalid graph pooling type.")

        # ffn
        self.ffn = torch.nn.ModuleList()
        self.mult = 2 if self.graph_pool == "set2set" else 1
        for n in range(self.n_layers):
          if n != self.n_layers - 1:
            self.ffn.append(nn.Linear(self.mult * self.hidden_size, self.mult * self.hidden_size))
          else:
            self.ffn.append(nn.Linear(self.mult * self.hidden_size, 1))


        #### added ####
        self.cached_zero_vector = nn.Parameter(torch.zeros(self.hidden_size), requires_grad=False)
        self.W_a = nn.Linear(self.hidden_size, self.hidden_size)
        self.W_b = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_func = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=self.attn_dropout)

        self.use_input_features = False
        if self.use_input_features:
            features_batch = torch.from_numpy(np.stack(features_batch)).float()
        #############



    def forward(self, data, viz_dir=None,  num_graphs_processed=0, stdzer=None, viz_ids=None):
        x, edge_index, edge_attr, batch, parity_atoms = data.x, data.edge_index, data.edge_attr, data.batch, data.parity_atoms
        row, col = edge_index
        # print('='*20)
        # print('Inputs dimension')
        # print('x:', x.size())
        # print('edge_index', edge_index.size())
        # print('edge_attre:', edge_attr.size())
        # print('parity_atoms:', parity_atoms.size())
        # print('='*20)

        if self.gnn_type == 'dmpnn'or self.gnn_type == 'orig_dmpnn':
            row, col = edge_index
            edge_attr = torch.cat([x[row], edge_attr], dim=1)
            edge_attr = F.relu(self.edge_init(edge_attr))
            # print('='*20)
            # print('After Linear:')
            # print('edge_attr:', edge_attr.size())
            # print('='*20)
        else:
            x = F.relu(self.node_init(x))
            edge_attr = F.relu(self.edge_init(edge_attr))

        x_list = [x]
        edge_attr_list = [edge_attr]

        # convolutions
        for l in range(self.depth):
            if not self.args.ft_boost:
                x_h, edge_attr_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms)
            else:
                if l>0:
                    x_list[-1] = torch.cat((x_list[-1], x_list[0][:,-3:]), 1)
                    x_h, edge_attr_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms)
                else:
                    x_h, edge_attr_h = self.convs[l](x_list[-1], edge_index, edge_attr_list[-1], parity_atoms)
                
            h = edge_attr_h if (self.gnn_type == 'dmpnn' or self.gnn_type == 'orig_dmpnn') else x_h
            # print('='*20)
            # print('After DMPNN:')

            # print('='*20)
            if l == self.depth - 1:
                h = F.dropout(h, self.dropout, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.dropout, training=self.training)

            if self.gnn_type == 'dmpnn' or self.gnn_type == 'orig_dmpnn':
                h += self.skip_coef*edge_attr_h
                edge_attr_list.append(h)
            else:
                h += self.skip_coef*x_h
                x_list.append(h)

        # dmpnn edge -> node aggregation
        if self.gnn_type == 'dmpnn' or self.gnn_type == 'orig_dmpnn':
            h, _ = self.edge_to_node(x_list[-1], edge_index, h, parity_atoms)

        cur_hiddens = h

        # attention

        if self.attn_type == 'tang':
          att_w, att_hiddens = [], []
          for graph_ind in range(data.num_graphs):
            h_this_graph = h[batch==graph_ind]
            att_w_this_graph = torch.matmul(self.W_a(h_this_graph), h_this_graph.t())
            att_w_this_graph = F.softmax(att_w_this_graph, dim=0)
            att_w_this_graph_ = att_w_this_graph
            att_w_this_graph_ = att_w_this_graph_.sum(dim=1)
            att_w_this_graph_.unsqueeze_(-1) # aadd dimension for num heads. CUrrently, num heads is 1
            att_hiddens_this_graph = torch.matmul(att_w_this_graph, h_this_graph)
            att_hiddens_this_graph = self.act_func(self.W_b(att_hiddens_this_graph))
            att_hiddens_this_graph = self.dropout_layer(att_hiddens_this_graph)
            att_w.append(att_w_this_graph_)
            att_hiddens.append(att_hiddens_this_graph)

          att_weights = torch.cat(att_w, 0)
          att_hiddens = torch.cat(att_hiddens, 0)

        elif self.attn_type == 'gat':
          att_hiddens, att_weights = self.gat[0](cur_hiddens, edge_index, return_attention_weights=True)
          if self.gat_depth > 1:
            att_hiddens = self.gat_act_func(att_hiddens)
            att_hiddens = F.dropout(att_hiddens, p=self.attn_dropout, training=self.training)
          for l in range(1, self.gat_depth):
            if l!= self.gat_depth - 1:
              att_hiddens = self.gat[l](att_hiddens, edge_index)
              att_hiddens = self.gat_act_func(att_hiddens)
              att_hiddens = F.dropout(att_hiddens, p=self.attn_dropout, training=self.training)
            else:
              att_hiddens, att_weights = self.gat[l](att_hiddens, edge_index, return_attention_weights= True)

        mol_vec = (self.skip_coef*cur_hiddens + att_hiddens)
        # print('mol_vec size:', mol_vec.size())
        # mol_vec = mol_vec.sum(dim=0) / a_size
        mol_vec = self.pool(mol_vec, batch)

        # #######################
        for n in range(self.n_layers):
          mol_vec = self.ffn[n](mol_vec)
          if n == self.n_layers - 1:
            mol_vec = F.dropout(mol_vec, self.dropout, training=self.training)
          else:
            mol_vec = F.dropout(F.relu(mol_vec), self.dropout, training=self.training)
        if self.task == 'regression':
            output = mol_vec.squeeze(-1)
        elif self.task == 'classification':
            output = torch.sigmoid(mol_vec).squeeze(-1)

        if not viz_dir is None:
          viz_this_batch = False
          for graph_ind in range(data.num_graphs):
            if graph_ind + num_graphs_processed in viz_ids:
                viz_this_batch = True 
          if viz_this_batch:
              # num_nodes = int(att_weights[0].max())+1
              preds = stdzer(output, rev=True)
              if self.attn_type != "tang":
                weights= torch.sparse.FloatTensor(att_weights[0], att_weights[1], torch.Size([data.num_nodes,data.num_nodes, self.args.heads])).to_dense().sum(dim=1)
                weights = weights.cpu().data.numpy()
              else:
                weights = att_weights.cpu().data.numpy()


              for graph_ind in range(data.num_graphs):
                if not num_graphs_processed+graph_ind in viz_ids:
                    continue
    #          for graph_ind in [0,1,2,3]:
                smiles = data.smiles[graph_ind]
                # get attention weights for this graph
                att_w_this_graph = weights[torch.where(batch==graph_ind)[0].cpu().data.numpy()]

                visualize_atom_attention(viz_dir=viz_dir + f'{num_graphs_processed+graph_ind}',
                                         smiles=smiles,
                                         attention_weights=att_w_this_graph,
                                         heads=self.heads)

                y_this_graph = float(data.y[graph_ind])
                pred_this_graph = float(preds[graph_ind])
                num_atoms = att_w_this_graph.shape[0]
                generate_info_file(viz_dir + f'{num_graphs_processed+graph_ind}', viz_dir,num_graphs_processed+graph_ind , num_atoms, smiles, y_this_graph, pred_this_graph)

        return output

    
def generate_info_file(viz_dir, csv_dir, row_order, num_atoms, smiles, y, pred):
   os.makedirs(viz_dir, exist_ok=True)
   f= open(viz_dir+"/info.txt","w+")
   f.write(f'Number of atoms:{num_atoms}\n')
   f.write(smiles)
   f.write(f'Groundtruth: {y:.3f}\n')
   f.write(f'Predicted: {pred:.3f}')
   f.close()
   #wtite to csv
   with open(csv_dir+'0info.csv', mode='a') as csvfile:
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
    # writing the fields  
    if row_order == 1:
      csvwriter.writerow(['order', 'groundtruth', 'pred'])  
    # writing the data rows  
    csvwriter.writerow([row_order, y, pred])
