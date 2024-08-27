
dataset_root = './datasets/lung-1vs5.csv'

results_path = './results'

exp_name = 'test'

epochs = 500

random_seed = 777

hp = {
    'batch_size' : 128,
    'lr' : 1e-03,
    'sequence_len' : 64,
    'heads' : 16,
    'dim' : 256,
    'num_layers' : 3,
    'layer_conf' : 'same' #same, smaller, hybrid
}