import torch
import numpy as np
from preprocess import load_dataset
from torch_snippets import *
import wandb

wandb.init()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

CLASSVECTORSPATH = "./data/class_vectors.npy"
DATAPATH = "./data/facial_data.pkl"
CLASSESFILEPATH = "./src/train_classes.txt"


def build_model():
    return torch.nn.Sequential(
        torch.nn.Linear(4096, 1024), torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(1024), torch.nn.Dropout(0.8),
        torch.nn.Linear(1024, 512), torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(512), torch.nn.Dropout(0.8),
        torch.nn.Linear(512, 256), torch.nn.ReLU(inplace=True),
        torch.nn.BatchNorm1d(256), torch.nn.Dropout(0.8),
        torch.nn.Linear(256, 300)
    )


def train_batch(model, data, optimizer, criterion):
    ims, labels = data
    _preds = model(ims)
    optimizer.zero_grad()
    loss = criterion(_preds, labels)
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def validate_batch(model, data, criterion):
    ims, labels = data
    _preds = model(ims)
    loss = criterion(_preds, labels)
    return loss.item()


if __name__ == '__main__':
    trn_dl, val_dl, zsl_x, zsl_class = load_dataset(DATAPATH, CLASSVECTORSPATH, CLASSESFILEPATH)
    model = build_model().to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    n_epochs = 60
    log = Report(n_epochs)

    for ex in range(n_epochs):
        N = len(trn_dl)
        for bx, data in enumerate(trn_dl):
            loss = train_batch(model, data, optimizer, criterion)
            log.record(ex + (bx + 1) / N, trn_loss=loss, end='\r')

        N = len(val_dl)
        for bx, data in enumerate(val_dl):
            loss = validate_batch(model, data, criterion)
            log.record(ex + (bx + 1) / N, val_loss=loss, end='\r')

        # change learning rate
        if ex == 10: optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        if ex == 40: optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        if not (ex + 1) % 10: log.report_avgs(ex + 1)

    log.plot(log=True)

    pred_zsl = model(torch.Tensor(zsl_x).to(device)).cpu().detach().numpy()

    class_vectors = sorted(np.load(CLASSVECTORSPATH, allow_pickle=True), key=lambda x: x[0])
    classnames, vectors = zip(*class_vectors)
    classnames = list(classnames)

    vectors = np.array(vectors)

    dists = (pred_zsl[None] - vectors[:, None])
    dists = (dists ** 2).sum(-1).T

    best_classes = []
    for item in dists:
        best_classes.append([classnames[j] for j in np.argsort(item)[:5]])

    print(np.mean([i in J for i, J in zip(zsl_class, best_classes)]))
