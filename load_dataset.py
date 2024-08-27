import csv
import numpy as np

def load_tabular_dataset(dataset_root):
    print(' Dataset path:', dataset_root)
    print()

    train_X = []
    train_Y = []

    with open(dataset_root, 'r', newline='') as file:
        reader = csv.reader(file)
        for idx, row in enumerate(reader):
            if idx == 0:
                # print('header:', row)
                print('header length:', len(row))
                continue
            float_row = [float(value) for value in row]

            train_X.append(float_row[:-1])
            train_Y.append([float_row[-1]])
    
    print('------------------------------------------------')
    print('Train X shape:', np.array(train_X).shape)
    print('Train Y shape:', np.array(train_Y).shape)
    print('------------------------------------------------')

    return train_X, train_Y