import numpy as np
import torch
import os
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Zinc_data(Dataset):
    def __init__(self, data_root='../Data/zinc15/zinc15_0.25_geo/preprocess'):
        self.root = os.listdir(data_root)
        self.base_root = data_root
    def __len__(self):
        return len(self.root)

    def __getitem__(self, index):
        file_name = os.path.join(self.base_root, self.root[index])
        data = np.load(file_name)
        node_features = data['node_features']
        bond_features = data['bond_features']
        adjacency_matrix = data['adjacency_matrix']
        mask_node_labels = data['mask_node_labels']
        masked_atom_indices = data['masked_atom_indices']
        token_ids = data['token_ids']
        labels = data['labels']
        edge_attr = data['edge_attr']
        edge_index = data['edge_index']
        num_atoms = data['num_atoms']
        x = data['x']

        node_features = torch.from_numpy(node_features)
        bond_features = torch.from_numpy(bond_features)
        adjacency_matrix = torch.from_numpy(adjacency_matrix)
        mask_node_labels = torch.from_numpy(mask_node_labels)
        masked_atom_indices = torch.from_numpy(masked_atom_indices)
        token_ids = torch.from_numpy(token_ids)
        labels = torch.from_numpy(labels)
        edge_attr = torch.from_numpy(edge_attr)
        edge_index = torch.from_numpy(edge_index)
        num_atoms = torch.from_numpy(num_atoms)
        x = torch.from_numpy(x)




        return node_features, bond_features, adjacency_matrix, mask_node_labels, masked_atom_indices, token_ids, labels, edge_attr, edge_index, num_atoms, x


def mol_collate_func_mask(batch):
    node_features_list, bond_features_list, adjacency_list, mask_node_labels_list, masked_atom_indices_list, token_ids_list, labels_list = [], [], [], [], [], [], []
    # labels = []
    edge_attr_list, edge_index_list, x_list, xmasked_atom_indices_list,num_nodes_list= [], [], [], [],[]
    cumsum_node = 0
    for molecule in batch:
        a = molecule[8] + cumsum_node
        b = molecule[4] + cumsum_node
        cumsum_node += molecule[9]
        c = molecule[9].reshape(1,)
        node_features_list.append(molecule[0])
        bond_features_list.append(molecule[1])
        adjacency_list.append(molecule[2])
        mask_node_labels_list.append(molecule[3])
        masked_atom_indices_list.append(molecule[4])
        token_ids_list.append(molecule[5])
        labels_list.append(molecule[6])
        edge_attr_list.append(molecule[7])
        edge_index_list.append(a)
        x_list.append(molecule[10])
        xmasked_atom_indices_list.append(b)
        num_nodes_list.append(c)




        # labels.append(molecule.label)


    return torch.stack(node_features_list), torch.stack(bond_features_list), torch.stack(adjacency_list), mask_node_labels_list, masked_atom_indices_list, torch.stack(token_ids_list), torch.stack(labels_list), torch.cat(edge_attr_list,0), torch.cat(edge_index_list,-1), torch.cat(x_list,0), torch.cat(xmasked_atom_indices_list,0), torch.cat(mask_node_labels_list,0), torch.cat(num_nodes_list,0)




if __name__ == "__main__":
    dataset = Zinc_data()
    train_dataloader = DataLoader(
        dataset,
        batch_size=4,
        collate_fn=mol_collate_func_mask,
        shuffle=True, drop_last=True, num_workers=4, pin_memory=True
    )



    # for xyz, normal, label, curvature, dists, atom_type_sel in tqdm(dataset):
    #     print(type(xyz))
    #     break

    for node_features, bond_features, adjacency_matrix, mask_node_labels, masked_atom_indices, token_ids, labels, edge_attr, edge_index, x, xmasked_atom_indices, xmask_node_labels in tqdm(train_dataloader):
        print(node_features.shape)
        print(bond_features.shape)
        print(adjacency_matrix.shape)
        print(mask_node_labels.shape)
        print(masked_atom_indices.shape)
        print(token_ids.shape)
        print(labels.shape)
        print(edge_attr.shape)
        print(edge_index.shape)
        print(x.shape)
        print(xmasked_atom_indices.shape)
        print(xmask_node_labels.shape)
        break
