import os, math, torch, kora.install.rdkit, csv
import pandas as pd
from model.parsing import parse_train_args
from data_model.data import construct_loader
from util import Standardizer, create_logger, get_loss_func

import numpy as np

import os, sys, inspect, torch, csv, copy
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from stereonet.models.training import *
from stereonet.models.main import GNN


args = parse_train_args()
torch.manual_seed(args.seed)


train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std, args.task)
loss = get_loss_func(args)


# load model
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
print('Model architecture: ', model)
state_dict = torch.load(args.model_path, map_location=args.device) if 'best_model' in args.model_path else torch.load(args.model_path, map_location=args.device)['model_state_dict']
model.load_state_dict(state_dict)

def train_and_save_predictions(loader, preds_path, viz_dir=None, viz_ids=None):
    # predict on train data
    ys, preds, loss_val, acc, auc = test(model, loader, loss, stdzer, args.device, args.task, viz_dir=viz_dir, viz_ids=viz_ids)
    # save predictions
    smiles = loader.dataset.smiles
#    preds_path = os.path.join(args.log_dir, 'preds_on_train.csv')
    pd.DataFrame(list(zip(smiles, ys, preds)), columns=['smiles', 'label', 'prediction']).to_csv(preds_path, index=False)
    return ys, preds, loss_val, acc, auc

# predict on val data
print('Evaluation on validation data')
train_and_save_predictions(val_loader, preds_path=os.path.join(args.eval_output_dir, 'preds_on_val.csv'), viz_dir=os.path.join(args.eval_output_dir, 'val_viz'), viz_ids=[1,2,9,10,29,30,3,4,13,14,21,22])


# predict on train data
print('Evaluation on training data')
train_and_save_predictions(train_loader, preds_path=os.path.join(args.eval_output_dir, 'preds_on_train.csv'), viz_dir=os.path.join(args.eval_output_dir, 'train_viz'), viz_ids=[1,2,9,10,29,30,3,4,13,14,21,22])
