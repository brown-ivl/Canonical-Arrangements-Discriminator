import torch 
import argparse
from numpy import genfromtxt

class Discriminator(torch.nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc = torch.nn.Linear(12,1)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        output = self.fc(x)
        output = self.relu(x)
        return output
class Dataset(torch.utils.data.Dataset):
    #TODO
    def __init__(self, dir_path):
        self.data = genfromtxt(dir_path,delimiter=';')

    def __len__(self):
        return 

    def __getitem__(self, idx):
        dataset = torch.as_tensor(self.data, dtype=torch.float32)
        x_train, y_train, x_test, y_test = dataset[idx]
        return x_train, y_train, x_test, y_test
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="train",
                        help="One of 'train' or 'test'.")

    flags, unparsed = parser.parse_known_args()
    model = Discriminator()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    x_train, y_train, x_test, y_test = Dataset()

    if flags.mode == "train":
        model.train()
        epoch = 20
        for epoch in range(epoch):
            optimizer.zero_grad()
            # Forward pass
            y_pred = model(x_train)
            # Compute Loss
            loss = criterion(y_pred.squeeze(), y_train)
        
            print('Epoch {}: train loss: {}'.format(epoch, loss.item()))
            # Backward pass
            loss.backward()
            optimizer.step()
    elif flags.mode == "test":
        model.eval()
        y_pred = model(x_test)
        before_train = criterion(y_pred.squeeze(), y_test)
        print('Test loss before training' , before_train.item())