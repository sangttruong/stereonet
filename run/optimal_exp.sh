python stereonet/train.py --data_path 'stereonet/data/d4_docking/d4_docking.csv' 
                           --split_path 'stereonet/data/d4_docking/full/split0.npy'
                           --log_dir 'gdrive/My Drive/Colab Notebooks/gcnn/log-s9'
                           --checkpoint_dir 'gdrive/My Drive/Colab Notebooks/gcnn/checkpoint-s9'
                           --n_epochs 100 --batch_size 64 --warmup_epochs 0 
                           --gnn_type gcn --hidden_size 32
                           --depth 2 --dropout 0 --message tetra_permute_concat 
                           --n_layers 2 --attn_type gat --gat_act leakyrelu 
                           --gat_depth 2 --heads 8 --concat