import os, math, torch, kora.install.rdkit, pandas as pd
from model.parsing import parse_train_args
from data_model.data import construct_loader
from util import Standardizer, create_logger, get_loss_func
from model.main import GNN
import csv
import numpy as np
from model.training import *

args = parse_train_args()
torch.manual_seed(args.seed)

def get_lr(optimizer):
  for param_group in optimizer.param_groups:
      return param_group['lr']


# create path to save logs
if not os.path.exists(args.log_dir):
  os.makedirs(args.log_dir)
  args.checkpoint_dir = args.log_dir

# # create path to save checkpoint
# if not os.path.exists(args.checkpoint_dir):
#   os.makedirs(args.checkpoint_dir)

train_loader, val_loader = construct_loader(args)
mean = train_loader.dataset.mean
std = train_loader.dataset.std
stdzer = Standardizer(mean, std, args.task)
loss = get_loss_func(args)

########################################## RUN IF TRAINING ##########################################
# create model, optimizer, scheduler, and loss fn
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)

model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
print('Number of parameters:',params)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# scheduler = build_lr_scheduler(optimizer, args, len(train_loader.dataset))
scheduler = None

best_val_loss = math.inf
best_epoch = 0

if args.epoch != 0:
  checkpoint = torch.load(os.path.join(args.checkpoint_dir, f'{args.epoch}_model'))
  model.load_state_dict(checkpoint['model_state_dict'])
  optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
  # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
  best_val_loss = checkpoint['best_val_loss']
  best_epoch = checkpoint['best_epoch']

for epoch in range(args.epoch, args.n_epochs):
    print('Learning rate:', get_lr(optimizer))
    train_loss, train_acc = train(model, train_loader, optimizer, loss, stdzer, args.device, scheduler, args.task)
    # writer.add_scalar("Loss/train", train_loss, epoch)
    # if not train_acc is None: writer.add_scalar("Acc/train", train_acc, epoch)

    val_loss, val_acc = eval(model, val_loader, loss, stdzer, args.device, args.task)
    # writer.add_scalar("Loss/val", val_loss, epoch)
    # if not vwriteral_acc is None: writer.add_scalar("Acc/val", val_acc, epoch)

    # write logs
    f= open(args.log_dir+"/log.txt","a")
    f.write(f"Epoch {epoch}: Training Loss {train_loss}")
    if args.task == 'classification':
      f.write(f"Epoch {epoch}: Training Classification Accuracy {train_acc}")
    f.write(f"Epoch {epoch}: Validation Loss {val_loss}")
    if args.task == 'classification':
      f.write(f"Epoch {epoch}: Validation Classification Accuracy {val_acc}")
    f.close()

    # write to csv. This is used to write tensorboard when resuming training
    with open(args.log_dir+'train_data.csv', mode='a') as csvfile:
      # creating a csv writer object
      csvwriter = csv.writer(csvfile)
      if args.task == 'classification':
        csvwriter.writerow([epoch, train_loss, train_acc])
      else:
        csvwriter.writerow([epoch, train_loss])

    with open(args.log_dir+'val_data.csv', mode='a') as csvfile:
      # creating a csv writer object
      csvwriter = csv.writer(csvfile)
      if args.task == 'classification':
        csvwriter.writerow([epoch, val_loss, val_acc])
      else:
        csvwriter.writerow([epoch, val_loss])

    if val_loss <= best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'best_model'))

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # 'scheduler_state_dict': scheduler.state_dict(),
                'best_epoch': best_epoch,
                'best_val_loss': best_val_loss},
                os.path.join(args.checkpoint_dir, f'{epoch+1}_model'))

# logger.info(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}")
f= open(args.log_dir+"/log.txt","w+")
f.write(f"Best Validation Loss {best_val_loss} on Epoch {best_epoch}")
f.close()

###### TEST and VIZ

# load best model
model = GNN(args, train_loader.dataset.num_node_features, train_loader.dataset.num_edge_features).to(args.device)
print('Model architecture: ', model)
state_dict = torch.load(os.path.join(args.log_dir, 'best_model'), map_location=args.device)
model.load_state_dict(state_dict)

# predict test data
test_loader = construct_loader(args, modes='test')
ys, preds, test_loss, test_acc, test_auc = test(model, test_loader, loss, stdzer, args.device, args.task, viz_dir=args.viz_dir)
logger.info(f"Test Loss {test_loss}")
if args.task == 'classification':
    logger.info(f"Test Classification Accuracy {test_acc}")
    logger.info(f"Test ROC AUC Score {test_auc}")

# save predictions
smiles = test_loader.dataset.smiles
preds_path = os.path.join(args.log_dir, 'preds.csv')
pd.DataFrame(list(zip(smiles, ys, preds)), columns=['smiles', 'label', 'prediction']).to_csv(preds_path, index=False)
