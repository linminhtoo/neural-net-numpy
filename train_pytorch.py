''' run this to confirm that the numpy version gets similar val_acc '''

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def get_batches(l, n):
    ''' Yield successive n-sized batches from l '''
    for i in range(0, len(l), n):
        yield (l[i:i + n])

# set up model, criterion & training parameters
class MLP_torch(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(*[
            nn.Linear(30, 15),
            nn.ReLU(),
            nn.Linear(15, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        return self.net(x)

def train():
    model = MLP_torch()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=5e-3)
    criterion = nn.BCELoss(reduction='sum')
    num_epochs = 50
    bsz = 4

    # load data
    cancer_data = datasets.load_breast_cancer()
    X, y = cancer_data['data'], cancer_data['target']

    # train-test split
    X_train, X_val, \
        y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=1337)
    y_train = np.expand_dims(y_train, axis=-1).astype(int)
    y_val = np.expand_dims(y_val, axis=-1).astype(int)

    # normalize input data
    scaler = preprocessing.StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    X_train, X_val = torch.from_numpy(X_train).float(), torch.from_numpy(X_val).float()
    y_train, y_val = torch.from_numpy(y_train).float(), torch.from_numpy(y_val).float()

    # run the training for num_epochs
    epoch_cnt = 0
    while epoch_cnt < num_epochs:
        print('#'*30)
        print(f'epoch {epoch_cnt}')

        model.train()
        train_loss = 0
        epoch_loader = get_batches(range(len(X_train)), bsz)
        for i, batch_idxs in enumerate(epoch_loader):
            x, y_true = X_train[batch_idxs], y_train[batch_idxs]

            y_pred = model.forward(x)

            loss = criterion(y_pred, y_true)
            train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i == 0:
                X_train, y_train = shuffle(X_train, y_train)
                print(f'avg train loss: {train_loss / len(X_train):.4f}')

                # validation at start of every epoch
                model.eval()
                val_loss, val_correct = 0, 0
                for val_batch_idxs in get_batches(range(len(X_val)), bsz):
                    x, y_true = X_val[val_batch_idxs], y_val[val_batch_idxs]

                    y_pred = model.forward(x)

                    val_loss += criterion(y_pred, y_true)

                    y_pred = (y_pred > 0.5).float()
                    val_correct += torch.eq(y_true, y_pred).sum().item()

                print(f'avg val loss: {val_loss / len(X_val):.4f}')
                print(f'avg val acc: {val_correct / len(X_val):.4f}')

        epoch_cnt += 1

if __name__ == "__main__":
    train()