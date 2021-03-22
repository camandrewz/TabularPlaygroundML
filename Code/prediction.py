import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# Loading the Test Dataset
data = pd.read_csv('Tabular Playground/Data/test.csv')

submission = data

# Giving Cat Column Values a Numerical Value
data["cat0"] = data["cat0"].astype('category').cat.codes
data["cat1"] = data["cat1"].astype('category').cat.codes
data["cat2"] = data["cat2"].astype('category').cat.codes
data["cat3"] = data["cat3"].astype('category').cat.codes
data["cat4"] = data["cat4"].astype('category').cat.codes
data["cat5"] = data["cat5"].astype('category').cat.codes
data["cat6"] = data["cat6"].astype('category').cat.codes
data["cat7"] = data["cat7"].astype('category').cat.codes
data["cat8"] = data["cat8"].astype('category').cat.codes
data["cat9"] = data["cat9"].astype('category').cat.codes
data["cat10"] = data["cat10"].astype('category').cat.codes
data["cat11"] = data["cat11"].astype('category').cat.codes
data["cat12"] = data["cat12"].astype('category').cat.codes
data["cat13"] = data["cat13"].astype('category').cat.codes
data["cat14"] = data["cat14"].astype('category').cat.codes
data["cat15"] = data["cat15"].astype('category').cat.codes
data["cat16"] = data["cat16"].astype('category').cat.codes
data["cat17"] = data["cat17"].astype('category').cat.codes
data["cat18"] = data["cat18"].astype('category').cat.codes

# Exclude the PassengerId Column
data = data.iloc[:, 1:]
submission = submission.iloc[:, 0:-30]

print(data.head())

# Standardize Input
scaler = StandardScaler()
input = scaler.fit_transform(data)

x = data.iloc[:, 1:-1]

# Ensure PyTorch is Using our GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ' + str(device))


class testData(Dataset):
    def __init__(self, X_data):
        self.X_data = X_data

    def __getitem__(self, index):
        return self.X_data[index]

    def __len__(self):
        return len(self.X_data)


test_data = testData(torch.FloatTensor(input))

# Initialize the Dataloaders
test_loader = DataLoader(dataset=test_data, batch_size=1)

LAYER_NODES = 32

# Define our Neural Network Architecture



class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 30.
        self.layer_1 = nn.Linear(30, LAYER_NODES)
        self.layer_2 = nn.Linear(LAYER_NODES, LAYER_NODES)
        self.layer_3 = nn.Linear(LAYER_NODES, LAYER_NODES)
        self.layer_4 = nn.Linear(LAYER_NODES, LAYER_NODES)
        self.layer_out = nn.Linear(LAYER_NODES, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.batchnorm1 = nn.BatchNorm1d(LAYER_NODES)
        self.batchnorm2 = nn.BatchNorm1d(LAYER_NODES)
        self.batchnorm3 = nn.BatchNorm1d(LAYER_NODES)
        self.batchnorm4 = nn.BatchNorm1d(LAYER_NODES)

    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.relu(self.layer_4(x))
        x = self.batchnorm4(x)
        x = self.dropout(x)
        x = self.layer_out(x)

        return x


# Initialize our Optimizer and Choose a Loss Function
model = torch.load('Tabular Playground/trained_model')
model.to(device)

# Testing our Model
y_pred_list = []
model.eval()

with torch.no_grad():
    for X_batch in test_loader:
        X_batch = X_batch.to(device)
        y_test_pred = model(X_batch)
        y_test_pred = torch.sigmoid(y_test_pred)
        y_pred_tag = torch.round(y_test_pred)
        y_pred_list.append(y_pred_tag.cpu().numpy())

y_pred_list = [int(a.squeeze().tolist()) for a in y_pred_list]

# Save Our Results into The Submission DataFrame and to CSV
submission.insert(len(submission.columns), 'Target', y_pred_list)
submission.to_csv('Tabular Playground/Data/Submission.csv', index=False)
print(submission.head())
print(submission.shape)
