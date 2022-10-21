import numpy as np
import gzip
import _pickle as cPickle
from sklearn.preprocessing import LabelEncoder, normalize
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# uncomment before running code for first time
# wandb.login(key="your key")
wandb.init()

def load_dataset(data_path, class_vectors_path, classes_file_path):
    with open(classes_file_path, 'r') as infile:
        train_classes = [str.strip(line) for line in infile]

    with gzip.GzipFile(data_path, 'rb') as infile:
        data = cPickle.load(infile)

    training_data = [instance for instance in data if instance[0] in train_classes]
    test_data = [instance for instance in data if instance[0] not in train_classes]
    np.random.shuffle(training_data)

    train_size = 300  # per class
    train_data, valid_data = [], []
    for class_label in train_classes:
        ctr = 0
        for instance in training_data:
            if instance[0] == class_label:
                if ctr < train_size:
                    train_data.append(instance)
                    ctr += 1
                else:
                    valid_data.append(instance)

    np.random.shuffle(train_data)
    np.random.shuffle(valid_data)
    vectors = {i: j for i, j in np.load(class_vectors_path, allow_pickle=True)}

    train_data = [(feat, vectors[clss]) for clss, feat in train_data]
    valid_data = [(feat, vectors[clss]) for clss, feat in valid_data]

    train_clss = [clss for clss, feat in train_data]
    valid_clss = [clss for clss, feat in valid_data]
    zsl_class = [clss for clss, feat in test_data]

    x_train, y_train = zip(*train_data)
    x_train, y_train = np.squeeze(np.asarray(x_train)), np.squeeze(np.asarray(y_train))
    x_train = normalize(x_train, norm='l2')

    x_valid, y_valid = zip(*valid_data)
    x_valid, y_valid = np.squeeze(np.asarray(x_valid)), np.squeeze(np.asarray(y_valid))
    x_valid = normalize(x_valid, norm='l2')

    y_zsl, zsl_x = zip(*test_data)
    zsl_x, y_zsl = np.squeeze(np.asarray(zsl_x)), np.squeeze(np.asarray(y_zsl))
    zsl_x = normalize(zsl_x, norm='l2')

    trn_ds = TensorDataset(*[torch.Tensor(t).to(device) for t in [x_train, y_train]])
    val_ds = TensorDataset(*[torch.Tensor(t).to(device) for t in [x_valid, y_valid]])

    trn_dl = DataLoader(trn_ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False)
    return trn_dl, val_dl, zsl_x, zsl_class
