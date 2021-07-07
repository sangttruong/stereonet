import os, math, torch, kora.install.rdkit, pandas as pd
#from model.parsing import parse_train_args
#from data_model.data import construct_loader
#from util import Standardizer, create_logger, get_loss_func
#from model.main import GNN
import csv
import numpy as np
#from model.training import *

from rdkit import Chem
from rdkit.Chem.rdchem import ChiralType
from rdkit.Chem.Draw import SimilarityMaps
import matplotlib
import matplotlib.pyplot as plt

"""# **Visualization function**
Functions to (1) output png files and (2) write csv of groundtruth and preds for easy tracing
"""

def visualize_atom_attention(viz_dir: str, smiles: str, attention_weights, heads):
    """
    Saves figures of attention maps between atoms. Note: works on a single molecule, not in batch

    :param viz_dir: Directory in which to save attention map figures.
    :param smiles: Smiles string for molecule.
    :param num_atoms: The number of atoms in this molecule.
    :param attention_weights: A num_atoms x num_atoms PyTorch FloatTensor containing attention weights.
    """
    
    mol = Chem.MolFromSmiles(smiles)
    
    os.makedirs(viz_dir, exist_ok=True)
    # atomSum_weights=np.zeros(num_atoms)
    # for a in range(num_atoms):
    #     a_weights = attention_weights[a].cpu().data.numpy()
    #     atomSum_weights+=a_weights
    # Amean_weight=atomSum_weights/num_atoms

    # nanMean=np.nanmean(Amean_weight)
    for head in range(heads):
      attention_weights_this_head = attention_weights[:, head]
      mean_attention = np.mean(attention_weights_this_head)
      fig = SimilarityMaps.GetSimilarityMapFromWeights(mol,
                                                      attention_weights_this_head-mean_attention,
                                                      colorMap=matplotlib.cm.bwr)
      # save_path = os.path.join(smiles_viz_dir, f'atom_{a}.png')
      save_path = os.path.join(viz_dir, f'head_{head}.png')
      fig.savefig(save_path, bbox_inches='tight')
      plt.close(fig)


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