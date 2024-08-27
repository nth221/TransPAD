import torch.utils.data as data
import torch
import numpy as np
import copy

import parameters as params

class train_Dataset(data.Dataset):
    def __init__(self, tokens, sequence_len, random_seed):
        self.tokens = np.array(tokens)
        self.tokens_index = np.arange(len(self.tokens))

        np.random.seed(random_seed)
        np.random.shuffle(self.tokens_index)

        self.sample_num = len(self.tokens) // sequence_len

        self.sequences = np.array([])
        self.sequences_index = np.array([])
        for idx in range(self.sample_num):
            if idx == 0:
                self.sequences = np.expand_dims(self.tokens[self.tokens_index[:sequence_len]], axis=0)
                self.sequences_index = np.expand_dims(self.tokens_index[:sequence_len], axis=0)
            else:
                self.sequences = np.append(self.sequences, np.expand_dims(self.tokens[self.tokens_index[sequence_len * idx : sequence_len * idx + sequence_len]],axis=0), axis=0)
                self.sequences_index = np.append(self.sequences_index, np.expand_dims(self.tokens_index[sequence_len * idx : sequence_len * idx + sequence_len], axis=0), axis=0)

    def __getitem__(self, index):
        tensor_segment = torch.FloatTensor(self.sequences[index])
        return tensor_segment, self.sequences_index[index]
    
    def __len__(self):
        return len(self.sequences)

class test_Dataset(data.Dataset):
    def __init__(self, tokens, labels, criteria_tokens_index, sequence_len):
        self.tokens = np.array(tokens)
        self.tokens_index = np.arange(len(self.tokens))
        self.sequence_len = sequence_len
        self.criteria_tokens_index = criteria_tokens_index

        self.labels = np.array(labels)

    def __getitem__(self, index):
        current_criteria_tokens_index = copy.deepcopy(self.criteria_tokens_index)
        if index in current_criteria_tokens_index:
            current_criteria_tokens_index.remove(index)
        
        criteria_sequence_not_x = self.tokens[current_criteria_tokens_index[:self.sequence_len - 1]]
        sequence = np.concatenate((np.expand_dims(self.tokens[index], axis=0), criteria_sequence_not_x), axis=0)
        
        tensor_segment = torch.FloatTensor(sequence)

        tensor_label = torch.FloatTensor(self.labels[index])
        
        return tensor_segment, tensor_label
    
    def __len__(self):
        return len(self.tokens)