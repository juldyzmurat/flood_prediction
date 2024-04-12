#%%
#import packs 
import os
import urllib
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
import pandas as pd
#%%
adj = np.load("/home/zxu4/stGNN/21-pv-stgnn/data/node_values.npy")
adj 


# %%
power = pd.read_csv("/home/zxu4/stGNN/21-pv-stgnn/data/zxu/rwb-s4p_stgae_imputation.csv")
power


#cretae a weights matrix 

# %%
class METRLADatasetLoader(object):


    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(METRLADatasetLoader, self).__init__()
        ###adjecency matrix based on 1 for the same cluster, 0.5 otherwise"""
        """weights = np.zeros((45, 45))
        counts = [3,3,3,3,2,3,3,3,2,3,3,3,3,3,3,2]
        cntr = 0
        ran = [x for x in range(0,counts[cntr])]
        runsran = 0 
        for i in range(len(weights)):
            for j in range(i,len(weights)):
                if i==j:
                    weights[i][j] =0.0 
                elif j in ran and i in ran:
                    weights[i][j] = 1.0
                else: 
                    weights[i][j] = 0.5
            if runsran<len(ran)-1:
                runsran+=1
            else: 
                if (cntr<len(counts)-1):

                    cntr+=1
                    ran = [x for x in range(ran[-1]+1,ran[-1]+1+counts[cntr])]
                    runsran = 0
        for i in range(1,len(weights)):
            for j in range(0,j+1):
                weights[i][j] = weights[j][i]
        A = weights"""

        ###adjacency matrix based on distance 
        """
        A = distance_based("ss_locations.csv",(45,45))"""

        ###adjacency matrix based on power correlation 
        from scipy.stats import pearsonr

        A = np.zeros(shape=(45,45))
        x = pd.read_csv("/home/zxu4/stGNN/21-pv-stgnn/data/zxu/rwb-s4p_stgae_imputation.csv")
        x = x.astype(np.float32)
        
        for i in range(45):
            new_weights = []
            for n in range(45):
                k1 = x.iloc[:,i]
                k2 = x.iloc[:,n]
                # Compute MAE between each pair of nodes
                correlation_coefficient, p_value = pearsonr(k1, k2)
                new_weights.append(p_value)
            for j in range(len(new_weights)):
                A[i][j] = new_weights[j]


        X = pd.read_csv("/home/zxu4/stGNN/21-pv-stgnn/data/zxu/rwb-s4p_stgae_imputation.csv").values.reshape(99360,45,1).transpose((1, 2, 0))
        X = X.astype(np.float32)


        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X)
        X = X - means
        stds = np.std(X)
        X = X / stds
        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values

    def _generate_task(self, num_timesteps_in: int = 96, num_timesteps_out: int = 96):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:, :, i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 0, i + num_timesteps_in : j]).numpy())

        self.features = features
        self.targets = target
    
    def get_dataset(
        self, num_timesteps_in: int = 96, num_timesteps_out: int = 96) -> StaticGraphTemporalSignal:
        self._get_edges_and_weights()
        self._generate_task(num_timesteps_in, num_timesteps_out)
        dataset = StaticGraphTemporalSignal(
            self.edges, self.edge_weights, self.features, self.targets
        )

        return dataset
# %%
loader = METRLADatasetLoader()
dataset = loader.get_dataset(num_timesteps_in=4, num_timesteps_out=1)
# %%
import seaborn as sns
# Visualize traffic over time
sensor_number = 44
timestamps = 96
sensor_labels = [hour.y[sensor_number][0].item() for hour in list(dataset)[:timestamps]]
sns.lineplot(data=sensor_labels)
# %%
from torch_geometric_temporal.signal import temporal_signal_split
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = self.linear(h)
        return h

# %%
device = torch.device('cpu') # cuda

# Create model and optimizers
model = TemporalGNN(node_features=1, periods=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
model.train()

print("Running training...")
for epoch in range(10): 
    loss = 0
    step = 0
    for snapshot in train_dataset:
        snapshot = snapshot.to(device)
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index)
        # Mean squared error
        loss = loss + torch.mean((y_hat-snapshot.y)**2) 
        step += 1

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
#%%
model.eval()
loss = 0
step = 0
horizon = 1

# Store for analysis
predictions = []
labels = []

for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    # Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index)
    # Mean squared error
    loss = loss + torch.mean((y_hat-snapshot.y)**2)
    # Store for analysis below
    labels.append(snapshot.y)
    predictions.append(y_hat)
    step += 1
    if step > horizon:
          break

loss = loss / (step+1)
loss = loss.item()
print("Test MSE: {:.4f}".format(loss))
# %%
import numpy as np

sensor = 2
timestep = 0
preds = np.asarray([pred[sensor][timestep].detach().cpu().numpy() for pred in predictions])
labs  = np.asarray([label[sensor][timestep].cpu().numpy() for label in labels])
print("Data points:,", preds.shape)
# %%
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,5))
sns.lineplot(data=preds, label="pred")
sns.lineplot(data=labs, label="true")
# %%
# New Adjacency Matrix from Similarity Analysis
# 100 nodes, single channel, in example below
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np
A = np.zeros(shape=(45,45))
x = pd.read_csv("/home/zxu4/stGNN/21-pv-stgnn/data/zxu/rwb-s4p_stgae_imputation.csv")
x = x.astype(np.float32)


def softmax(x):
    return(np.exp(x)/np.exp(x).sum())

for i in range(45):
    new_weights = []
    for n in range(45):
        k1 = x.iloc[:,i]
        k2 = x.iloc[:,n]
        # Compute MAE between each pair of nodes
        d = np.mean(np.abs(k1 - k2))
        new_weights.append(np.exp(-1 * d))
    for j in range(len(new_weights)):
        A[i][j] = new_weights[j]
A

# %%
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import numpy as np
def distance_based(locations,shape):
    def haversine(lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r

    # please change the path according to your setting
    location = pd.read_csv(locations)
    distance = np.zeros(shape=shape)
    dist = []
    for i in range(shape[0]):
        for j in range(shape[0]):
            d = haversine(location.iloc[i][1], location.iloc[i][0], location.iloc[j][1], location.iloc[j][0])
            distance[i][j] = d
            dist.append(d)

    dist_std = np.std(dist)
    distance = pd.DataFrame(distance)
    distance

    # epsilon = 0, 0.25, 0.5, 0.75, 1
    epsilon = 1
    sigma = dist_std
    W = np.zeros(shape=shape)

    for i in range(shape[0]):
        for j in range(shape[0]):
            if i == j: 
                W[i][j] = 0
            else:
                # Compute distance between stations
                d_ij = distance.loc[i][j]
                
                # Compute weight w_ij
                w_ij = np.exp(-d_ij**2 / sigma**2)
                
                if w_ij >= epsilon:
                    W[i, j] = w_ij
    return W


#%%
#create a df with locations of SS schools 
import pandas as pd 

listcol = ['s2001_inv1', 's2001_inv2', 's2001_inv3', 's2004_inv1', 's2004_inv2', 's2004_inv3', 's2005_inv1', 's2005_inv2',
           's2005_inv3', 's2006_inv1', 's2006_inv2', 's2006_inv3', 's2007_inv1', 's2007_inv2', 's2008_inv1', 's2008_inv2',
           's2008_inv3', 's2009_inv1', 's2009_inv2', 's2009_inv3', 's2010_inv1', 's2010_inv2', 's2010_inv3', 's2014_inv2',
           's2014_inv3', 's2017_inv1', 's2017_inv2', 's2017_inv3', 's2020_inv1', 's2020_inv2', 's2020_inv3', 's2021_inv1',
           's2021_inv2', 's2021_inv3', 's2022_inv1', 's2022_inv2', 's2022_inv3', 's2024_inv1', 's2024_inv2', 's2024_inv3',
           's2025_inv1', 's2025_inv2', 's2025_inv3', 's2027_inv1', 's2027_inv3']
listcolgroup = sorted(set(["SS"+x[1:5] for x in listcol]))

ssschools = pd.read_csv("sunsmartschoolsmetadata_-_sheet1__1_ (1).csv")
ssschools = ssschools[ssschools['Subsystem'].isin(listcolgroup)]
ssschools = ssschools[['Subsystem','latd','longd']]
ssschools = ssschools.reset_index(drop=True)

counts = [3,3,3,3,2,3,3,3,2,3,3,3,3,3,3,2]
repeated_rows = ssschools.loc[ssschools.index.repeat(counts)].reset_index(drop=True)
repeated_rows = repeated_rows[['latd','longd']]
repeated_rows.to_csv("ss_locations.csv",index=False)
# %%
W = distance_based("ss_locations.csv",(45,45))
W
# %%
