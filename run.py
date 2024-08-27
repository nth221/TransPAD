import os
import torch
import sys
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve

from dataloader import *
from TransPAD import TransPAD
import parameters as params

def criteria_sequence_selecting(all_tokens_idx, all_attn_list, sequence_len):
    all_tokens_idx = np.array(all_tokens_idx)
    all_attn_list = np.array(all_attn_list)

    all_tokens_idx = np.expand_dims(all_tokens_idx, axis=1)
    all_attn_list = np.expand_dims(all_attn_list, axis=1)

    appended_tokens = np.append(all_tokens_idx, all_attn_list, axis=1)

    sorted_appended_tokens = sorted(appended_tokens, key=lambda x : x[1])
    sorted_appended_tokens = np.array(sorted_appended_tokens)

    criteria_tokens_index = sorted_appended_tokens[:sequence_len, 0]
    criteria_tokens_index = criteria_tokens_index.astype(int)
    return criteria_tokens_index.tolist()

def start_unsupervised_pad(train_tokens, train_labels):
    hp = params.hp

    encoder_dim = []
    decoder_dim = []

    if hp['layer_conf'] == 'same':
        for m_num in range(hp['num_layers'] + 1):
            encoder_dim.append(hp['dim'])
            decoder_dim.append(hp['dim'])
    elif hp['layer_conf'] == 'smaller':
        for m_num in range(hp['num_layers'] + 1):
            if m_num == 0:
                encoder_dim.append(hp['dim'])
                decoder_dim.append(hp['dim'] // (2 ** (hp['num_layers'] - m_num)))
            elif m_num == hp['num_layers']:
                encoder_dim.append(hp['dim'] // (2 ** m_num))
                decoder_dim.append(hp['dim'])
            else:
                encoder_dim.append(hp['dim'] // (2 ** m_num))
                decoder_dim.append(hp['dim'] // (2 ** (hp['num_layers'] - m_num)))
    else:
        for m_num in range(hp['num_layers'] + 1):
            for hybrid_idx in range(2):
                if m_num == 0:
                    encoder_dim.append(hp['dim'])
                    decoder_dim.append(hp['dim'] // (2 ** (hp['num_layers'] - m_num)))
                elif m_num == hp['num_layers']:
                    encoder_dim.append(hp['dim'] // (2 ** m_num))
                    decoder_dim.append(hp['dim'])
                else:
                    encoder_dim.append(hp['dim'] // (2 ** m_num))
                    decoder_dim.append(hp['dim'] // (2 ** (hp['num_layers'] - m_num)))            
        hp['num_layers'] = hp['num_layers'] * 2
    
    print('Encoder dim:', encoder_dim)
    print('Decoder dim:', decoder_dim)
    print('------------------------------------------------')

    exp_prefix = params.results_path + '/' + params.exp_name

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    input_dim = len(train_tokens[0])
    model = TransPAD(hp['num_layers'], encoder_dim, decoder_dim, input_dim, hp['dim'], hp['heads'], hp['dim']//hp['heads'], 0.1)
    model.to(device)
    model_parameter = list(model.parameters())

    optimizer = optim.Adam(model_parameter, lr=hp['lr'])

    print('Start TransPAD Training and Testing ...')
    print ()

    if os.path.isdir(exp_prefix) == False:
        os.mkdir(exp_prefix)

    best_auroc = 0
    best_auprc = 0
    best_epoch = 0

    test_auroc_list = []
    test_auprc_list = []
    for epoch in range(params.epochs):
        model.train()
        print('=====================')
        print('Exp Prefix:', exp_prefix)
        print('Epoch:', epoch)
        print()

        trainset = train_Dataset(train_tokens, hp['sequence_len'], params.random_seed + epoch)
        dataloader = DataLoader(trainset, batch_size=hp['batch_size'], shuffle=True)

        running_loss = 0
        all_tokens_idx = []
        all_attn_list = []
        for idx, (tokens, tokens_idx) in enumerate(dataloader):
            sys.stdout.write('\x1b[F\x1b[K')
            print('Batch Percentage : ', idx / len(dataloader) * 100)
            
            tokens = tokens.to(device)

            output, _, all_att = model(tokens)
            loss = F.mse_loss(output, tokens)

            for b_idx in range(len(tokens_idx)):
                all_tokens_idx.extend(tokens_idx[b_idx].tolist())
                all_attn_list.extend(all_att[b_idx].tolist())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print('Train Loss : ', running_loss / len(dataloader))

        criteria_tokens_index = criteria_sequence_selecting(all_tokens_idx, all_attn_list, hp['sequence_len'])

        testset = test_Dataset(train_tokens, train_labels, criteria_tokens_index, hp['sequence_len'])
        testloader = DataLoader(testset, batch_size=hp['batch_size'], shuffle=False)

        print()
        print()
        with torch.no_grad():
            model.eval()

            predictions = []
            ground_truth = []

            for idx, (sequence, labels) in enumerate(testloader):
                sys.stdout.write('\x1b[F\x1b[K')
                print('Batch Percentage : ', idx / len(testloader) * 100)

                sequence = sequence.to(device)

                _, att, _ = model(sequence)
                predictions.extend(att.tolist())

                labels = labels.squeeze(dim=1)
                ground_truth.extend(labels.tolist())

        fpr, tpr, _ = roc_curve(ground_truth, predictions)
        test_auroc = auc(fpr, tpr)
        
        precision, recall, _ = precision_recall_curve(ground_truth, predictions)
        test_auprc = auc(recall, precision)
            
        print()
        print('Test AUROC:', test_auroc)
        print('Test AUPRC:', test_auprc)
        print('True Ratio:', sum(ground_truth) / len(ground_truth))

        test_auroc_list.append(test_auroc)
        test_auprc_list.append(test_auprc)

        if best_auroc < test_auroc:
            best_auroc = test_auroc
            best_auprc = test_auprc
            best_epoch = epoch

            torch.save(model.state_dict(), exp_prefix + '/best_auroc_model.pt')
            print('model saved ...')
        
        print()
        print('Best AUROC:', best_auroc)
        print('Best AUPRC:', best_auprc)
        print('Best epoch:', epoch)