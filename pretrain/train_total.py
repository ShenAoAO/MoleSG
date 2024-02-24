import argparse
import os
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import pickle as pkl
import numpy
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn import metrics
from sklearn.model_selection import train_test_split
# from dataset_node_zinc import construct_dataset_mask, mol_collate_func_mask
from transformer_graph import make_model
from utils import ScheduledOptim, get_options
# from graph_mae_model import GNNDecoder
from GNN_model import GNNDecoder
from Data_process.zinc_dataset_pretrain import Zinc_data, mol_collate_func_mask
from smiles_model import Smiles_encoder_model, encoder_model, smiles_decoder_model
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import torch.nn as nn
from transformers import RobertaConfig


def loss_function(y_true, y_pred):
    y_true, y_pred = y_true.flatten(), y_pred.flatten()
    y_mask = torch.where(y_true != 0., torch.full_like(y_true, 1), torch.full_like(y_true, 0))
    loss = torch.sum(torch.abs(y_true - y_pred * y_mask)) / torch.sum(y_mask)
    return loss
def sce_loss(x, y, alpha=1):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss

def model_train(model, atom_pred_decoder, smiles_encoder_model,encoder_model ,smiles_decoder_model, train_dataset, model_params, train_params, epochs, experiment_name):

    train_loader = DataLoader(dataset=train_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func_mask,
                              shuffle=True, drop_last=True, num_workers=4, pin_memory=True)


    # build optimizer
    optimizer = ScheduledOptim(torch.optim.Adam(model.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])
    optimizer_dec_pred_atoms = ScheduledOptim(torch.optim.Adam(atom_pred_decoder.parameters(), lr=0),
                               train_params['warmup_factor'], model_params['d_model'],
                               train_params['total_warmup_steps'])
    optimizer_smiles_encoder_model = ScheduledOptim(torch.optim.Adam(smiles_encoder_model.parameters(), lr=0),
                                              train_params['warmup_factor'], model_params['d_model'],
                                              train_params['total_warmup_steps'])
    optimizer_encoder_model = ScheduledOptim(torch.optim.Adam(encoder_model.parameters(), lr=0),
                                              train_params['warmup_factor'], model_params['d_model'],
                                              train_params['total_warmup_steps'])
    optimizer_smiles_decoder_model = ScheduledOptim(torch.optim.Adam(smiles_decoder_model.parameters(), lr=0),
                                              train_params['warmup_factor'], model_params['d_model'],
                                              train_params['total_warmup_steps'])


    best_valid_loss = float('inf')
    loss_accum=0

    if not Path("./Model/" + experiment_name + "/compt/").exists():
        os.makedirs("./Model/" + experiment_name + "/compt/")
    if not Path("./Model/" + experiment_name + "/smiles_encoder/").exists():
        os.makedirs("./Model/" + experiment_name + "/smiles_encoder/")
    if not Path("./Model/" + experiment_name + "/total_encoder/").exists():
        os.makedirs("./Model/" + experiment_name + "/total_encoder/")


    for epoch in range(epochs):
        # train

        model.train()
        atom_pred_decoder.train()
        smiles_encoder_model.train()
        encoder_model.train()
        smiles_decoder_model.train()

        for node_features, bond_features, adjacency_matrix, mask_node_labels, masked_atom_indices, token_ids, labels, edge_attr, edge_index, x, xmasked_atom_indices, xmask_node_labels, num_nodes in tqdm(train_loader):

            adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
            node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
            bond_features = bond_features.to(train_params['device'])
            # mask_node_labels = mask_node_labels.to(train_params['device'])
            # masked_atom_indices = masked_atom_indices.to(train_params['device'])
            token_ids = token_ids.to(train_params['device'])
            labels = labels.to(train_params['device'])
            edge_attr = edge_attr.to(train_params['device'])
            edge_index = edge_index.to(train_params['device'])
            x = x.to(train_params['device'])
            xmasked_atom_indices = xmasked_atom_indices.to(train_params['device'])
            xmask_node_labels =xmask_node_labels.to(train_params['device'])
            num_nodes =num_nodes.to(train_params['device'])

            batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0   # (batch_size, max_length)

            # (batch_size, max_length, 1)
            node_new_feature = model(node_features, batch_mask, adjacency_matrix, bond_features)
            graph_features = node_new_feature + nn.Parameter(torch.zeros(1, 1, 256)).cuda()
            token_ids = token_ids.squeeze()

            smiles_embedding = smiles_encoder_model(token_ids)
            smiles_features = smiles_embedding + nn.Parameter(torch.zeros(1, 1, 256)).cuda()
            new_embedding = torch.cat([graph_features,smiles_features],dim=1)
            total_embedding = encoder_model(new_embedding)
            total_list=[]
            for a in range(total_embedding.size()[0]):
                total_list.append(total_embedding[a,:num_nodes[a],:])
            total_embedding_new = torch.cat(total_list, 0)
            pred_node = atom_pred_decoder(total_embedding_new, edge_index, edge_attr, xmasked_atom_indices)
            loss_graph = sce_loss(xmask_node_labels, pred_node[xmasked_atom_indices])
            loss_fct = CrossEntropyLoss()
            prediction_scores = smiles_decoder_model(total_embedding[:,26:,:])
            masked_lm_loss = loss_fct(prediction_scores.view(-1, 700), labels.long().view(-1))
            loss = loss_graph + masked_lm_loss
            # b = node_new_feature[0].cpu().detach().numpy()
            # numpy.save("node_new_feature.npy", b)
            # loss = loss_function(y_true, y_pred)
            optimizer.zero_grad()
            optimizer_dec_pred_atoms.zero_grad()
            optimizer_smiles_encoder_model.zero_grad()
            optimizer_encoder_model.zero_grad()
            optimizer_smiles_decoder_model.zero_grad()
            loss.backward()
            optimizer.step_and_update_lr()
            optimizer_dec_pred_atoms.step_and_update_lr()
            optimizer_smiles_encoder_model.step_and_update_lr()
            optimizer_encoder_model.step_and_update_lr()
            optimizer_smiles_decoder_model.step_and_update_lr()


        # valid
        # model.eval()
        # with torch.no_grad():
        #     valid_result = dict()
        #     valid_result['label'], valid_result['prediction'], valid_result['loss'] = list(), list(), list()
        #     for batch in tqdm(valid_loader):
        #         adjacency_matrix, node_features, edge_features, y_true = batch
        #         adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
        #         node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
        #         edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)
        #
        #         batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch_size, max_length)
        #         # (batch_size, max_length, 1)
        #         y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)
        #
        #         y_true = y_true.numpy().flatten()
        #         y_pred = y_pred.cpu().detach().numpy().flatten()
        #         y_mask = np.where(y_true != 0., 1, 0)
        #
        #         times = 0
        #         for true, pred in zip(y_true, y_pred):
        #             if true != 0.:
        #                 times += 1
        #                 valid_result['label'].append(true)
        #                 valid_result['prediction'].append(pred)
        #                 valid_result['loss'].append(np.abs(true - pred))
        #         assert times == np.sum(y_mask)
        #
        #     valid_result['r2'] = metrics.r2_score(valid_result['label'], valid_result['prediction'])

        print('Epoch {}, learning rate {:.6f}, train loss: {:.4f}'.format(epoch + 1, optimizer.view_lr(), loss))



        # save the model and valid result
        if loss < best_valid_loss:
            torch.save({'state_dict': model.state_dict(),
                        'best_epoch': epoch, 'best_valid_loss': best_valid_loss},
                       "./Model/"+experiment_name+ "/compt/compt_epoch{}.pth".format(epoch),)
            torch.save({'state_dict': smiles_encoder_model.state_dict(),
                        'best_epoch': epoch, 'best_valid_loss': best_valid_loss},
                       "./Model/"+ experiment_name+ "/smiles_encoder/smiles_encoder_epoch{}.pth".format(epoch),)
            torch.save({'state_dict': encoder_model.state_dict(),
                        'best_epoch': epoch, 'best_valid_loss': best_valid_loss},
                       "./Model/" + experiment_name + "/total_encoder/total_encoder_epoch{}.pth".format(epoch),)
            best_valid_loss = loss

        # temp test
        # if (epoch + 1) % 10 == 0:
        #     checkpoint = torch.load(f'./Model/{dataset_name}/best_model_{dataset_name}_{element}.pt')
        #     print('=' * 20 + ' middle test ' + '=' * 20)
        #     test_result = model_test(checkpoint, test_dataset, model_params, train_params)
        #     print("best epoch: {}, best valid loss: {:.4f}, test loss: {:.4f}, test r2: {:.4f}".format(
        #         checkpoint['best_epoch'], checkpoint['best_valid_loss'], np.mean(test_result['loss']), test_result['r2']
        #     ))
        #     print('=' * 40)
        #
        # # early stop
        # if abs(best_epoch - epoch) >= 20:
        #     print("=" * 20 + ' early stop ' + "=" * 20)
        #     break
        loss_accum += float(loss.cpu().item())

    return loss_accum


# def model_test(checkpoint, test_dataset, model_params, train_params):
#     # build loader
#     test_loader = DataLoader(dataset=test_dataset, batch_size=train_params['batch_size'], collate_fn=mol_collate_func_mask,
#                              shuffle=False, drop_last=True, num_workers=4, pin_memory=True)
#
#     # build model
#     model = make_model(**model_params)
#     model.to(train_params['device'])
#     model.load_state_dict(checkpoint['state_dict'])
#
#     # test
#     model.eval()
#     with torch.no_grad():
#         test_result = dict()
#         test_result['label'], test_result['prediction'], test_result['loss'] = list(), list(), list()
#         for batch in tqdm(test_loader):
#             adjacency_matrix, node_features, edge_features = batch
#             adjacency_matrix = adjacency_matrix.to(train_params['device'])  # (batch_size, max_length, max_length)
#             node_features = node_features.to(train_params['device'])  # (batch_size, max_length, d_node)
#             edge_features = edge_features.to(train_params['device'])  # (batch_size, max_length, max_length, d_edge)
#
#             batch_mask = torch.sum(torch.abs(node_features), dim=-1) != 0  # (batch_size, max_length)
#             # (batch_size, max_length, 1)
#             y_pred = model(node_features, batch_mask, adjacency_matrix, edge_features)
#
#
#             y_true = y_true.numpy().flatten()
#             y_pred = y_pred.cpu().detach().numpy().flatten()
#             y_mask = np.where(y_true != 0., 1, 0)
#
#             times = 0
#             for true, pred in zip(y_true, y_pred):
#                 if true != 0.:
#                     times += 1
#                     test_result['label'].append(true)
#                     test_result['prediction'].append(pred)
#                     test_result['loss'].append(np.abs(true - pred))
#             assert times == np.sum(y_mask)
#     test_result['r2'] = metrics.r2_score(test_result['label'], test_result['prediction'])
#     test_result['best_valid_loss'] = checkpoint['best_valid_loss']
#     return test_result


if __name__ == '__main__':
    # init args
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="random seeds", default=np.random.randint(10000))
    parser.add_argument("--gpu", type=str, help='gpu', default=0)
    parser.add_argument("--dataset", type=str, help='nmrshiftdb/DFT8K_DFT/DFT8K_FF/Exp5K_DFT/Exp5K_FF', default='nmrshiftdb')
    parser.add_argument("--element", type=str, help="1H/13C", default='1H')
    parser.add_argument("--epochs", type=int, help="epochs", default=300)
    parser.add_argument("--experiment_name", type=str, help="experiment_name", default='')
    args = parser.parse_args()

    # load options
    model_params, train_params = get_options(args.dataset)

    # init device and seed
    print(f"Seed: {args.seed}")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        train_params['device'] = torch.device(f'cuda:{args.gpu}')
        torch.cuda.manual_seed(args.seed)
    else:
        train_params['device'] = torch.device('cpu')

    # with open('/home/sa/桌面/Project/compt/CoMPT/Data/zinc15_250k/zinc15_250K.pickle', 'rb') as f:
    #     mol = pkl.load(f)

    # print('=' * 20 + ' begin train ' + '=' * 20)
    # model_params['max_length'] = max([data.GetNumAtoms() for data in mol])
    # print(f"Max padding length is: {model_params['max_length']}")

    # train_mol, test_mol = train_test_split(mol, test_size=0.05,random_state=np.random.randint(10000))

    atom_hidden = 115
    bond_hidden = 13
    train_dataset = Zinc_data()

    # valid_dataset = construct_dataset_mask(test_mol, model_params['d_atom'], model_params['d_edge'], model_params['max_length'], mask_rate = 0.25)
    # test_dataset = construct_dataset_mask(test_mol, model_params['d_atom'], model_params['d_edge'], model_params['max_length'], mask_rate = 0.25)

    # calculate total warmup factor and step
    train_params['warmup_factor'] = 0.2 if args.element == '1H' else 1.0
    train_params['total_warmup_steps'] = \
        int(len(train_dataset) / train_params['batch_size']) * train_params['total_warmup_epochs']
    print('train warmup step is: {}'.format(train_params['total_warmup_steps']))

    # define a model
    model = make_model(**model_params)
    atom_pred_decoder = GNNDecoder(hidden_dim=256, out_dim=116)
    config = RobertaConfig(
        vocab_size=700,
        max_position_embeddings=515,
        num_attention_heads=2,
        num_hidden_layers=2,
        type_vocab_size=1,
        is_gpu=torch.cuda.is_available(),
        hidden_size= 256
    )

    smiles_encoder_model = Smiles_encoder_model(config)
    encoder_model = encoder_model(config)
    smiles_decoder_model = smiles_decoder_model(config)

    model = model.to(train_params['device'])
    atom_pred_decoder =atom_pred_decoder.to(train_params['device'])
    smiles_encoder_model = smiles_encoder_model.to(train_params['device'])
    encoder_model = encoder_model.to(train_params['device'])
    smiles_decoder_model = smiles_decoder_model.to(train_params['device'])

    # train and valid
    print(f"train size: {len(train_dataset)}")
    train_loss=model_train(model, atom_pred_decoder, smiles_encoder_model,encoder_model ,smiles_decoder_model, train_dataset, model_params, train_params, args.epochs, args.experiment_name)
    print(train_loss)

