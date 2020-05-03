import numpy as np
from scipy.linalg import norm
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class HARDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = list(samples.astype('float32'))
        self.labels = list(labels)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]
    
def to_phase_trajectory(series, l):
    phase = np.zeros([series.shape[0] - l, l])

    for i in range(0, series.shape[0] - l):
        phase[i] = np.squeeze(series[i:i + l])
    return phase

def get_HARD(l = 32):
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    labels_name = dict()

    with open('./data/activity_labels.txt') as f:
        for line in f:
            line = line.split()
            labels_name[int(line[0]) - 1] = line[1]

    with open('./data/train/y_train.txt') as f:
        for line in f:
            train_labels.append(int(line))

    with open('./data/test/y_test.txt') as f:
        for line in f:
            test_labels.append(int(line))

    data_ax = []
    with open('./data/train/Inertial Signals/body_acc_x_train.txt') as f:
        for line in f:
            vals = [float(val) for val in line.split()]
            data_ax.append(np.array(vals))
    train_data.append(data_ax)

    data_ax = []
    with open('./data/train/Inertial Signals/body_acc_y_train.txt') as f:
        for line in f:
            vals = [float(val) for val in line.split()]
            data_ax.append(np.array(vals))
    train_data.append(data_ax)

    data_ax = []
    with open('./data/train/Inertial Signals/body_acc_z_train.txt') as f:
        for line in f:
            vals = [float(val) for val in line.split()]
            data_ax.append(np.array(vals))
    train_data.append(data_ax)

    data_ax = []
    with open('./data/test/Inertial Signals/body_acc_x_test.txt') as f:
        for line in f:
            vals = [float(val) for val in line.split()]
            data_ax.append(np.array(vals))
    test_data.append(data_ax)

    data_ax = []
    with open('./data/test/Inertial Signals/body_acc_y_test.txt') as f:
        for line in f:
            vals = [float(val) for val in line.split()]
            data_ax.append(np.array(vals))
    test_data.append(data_ax)

    data_ax = []
    with open('./data/test/Inertial Signals/body_acc_z_test.txt') as f:
        for line in f:
            vals = [float(val) for val in line.split()]
            data_ax.append(np.array(vals))
    test_data.append(data_ax)

    train_data = np.array(train_data)
    train_labels = np.array(train_labels) - 1

    test_data = np.array(test_data)
    test_labels = np.array(test_labels) - 1

    print('Data extracted...')
    print('Train data shape:', train_data.shape)
    print('Train labels shape:', train_labels.shape)
    print('Test data shape:', test_data.shape)
    print('Test labels shape:', test_labels.shape)

    train_data_all_axes = norm(train_data, axis=0)
    test_data_all_axes = norm(test_data, axis=0)

    train_data_phase = np.zeros((train_data_all_axes.shape[0], train_data_all_axes.shape[1] - l, l))
    test_data_phase = np.zeros((test_data_all_axes.shape[0], test_data_all_axes.shape[1] - l, l))
    
    for i in range(train_data_all_axes.shape[0]):
        train_data_phase[i] = to_phase_trajectory(train_data_all_axes[i], l)

    for i in range(test_data_all_axes.shape[0]):
        test_data_phase[i] = to_phase_trajectory(test_data_all_axes[i], l)

    print(train_data_phase.shape)
    print(test_data_phase.shape)
    
    train_dataset = HARDataset(train_data_phase, train_labels)
    test_dataset = HARDataset(test_data_phase, test_labels)

    train_dataset_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=8,
                                                   shuffle=True)

    test_dataset_loader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_size=8,
                                                  shuffle=True)
    
    return train_dataset_loader, test_dataset_loader, labels_name