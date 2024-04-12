#%%
#create the tensor 
import pandas as pd 
import torch
import numpy as np 
directory = '/mnt/rstor/CSE_MSE_RXF131/staging/sdle/pv-stgnn/simulated_sunsmart/power/'
rows = []

for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    df = pd.read_parquet(file)
    site_names = [n for i, n in enumerate(df.columns) if 'site' in n]
    for i in site_names:
        copydf = df[['dni', 'ghi', 'albedo', 'temp_air', 'Dew Point', 'poa', 'sun_angle', i]].copy()
        copydf.fillna(0, inplace=True)
        rows.append(copydf.values) 

tensor = torch.tensor(rows)
tensor_t = tensor.permute(1, 0, 2)
numpy_tensor = tensor.numpy()
np.save('sites_tensor.npy', numpy_tensor)

# %%
#create the adjacency matrix 
import pandas as pd 
import torch
import numpy as np 
from tqdm import tqdm

directory = '/mnt/rstor/CSE_MSE_RXF131/staging/sdle/pv-stgnn/simulated_sunsmart/power/'

rows = []
for filename in os.listdir(directory):
    file = os.path.join(directory, filename)
    df = pd.read_parquet(file)
    site_names = [n for i, n in enumerate(df.columns) if 'site' in n]
    for i in site_names:
        copydf = df['poa'].copy()
        copydf.fillna(0, inplace=True)
        rows.append(copydf.values) 
tensor = torch.tensor(rows)
tensor_t = tensor.permute(1, 0)

#%%
sim = pd.DataFrame(tensor_t).corr()
sim = sim.values
tensor = torch.tensor(sim)
numpy_tensor = tensor.numpy()
np.save('sites_adjacency.npy', tensor)


# %%
#run starting from this chunk
import os
import urllib
import zipfile
import numpy as np
import torch
from torch_geometric.utils import dense_to_sparse
from torch_geometric_temporal.signal import StaticGraphTemporalSignal



class METRLADatasetLoader(object):


    def __init__(self, raw_data_dir=os.path.join(os.getcwd(), "data")):
        super(METRLADatasetLoader, self).__init__()

        A = np.load("/pv_stGNN/21-pv-stgnn/data/sites_adjacency.npy")
        X = np.load("/pv_stGNN/21-pv-stgnn/data/sites_tensor.npy").transpose((0, 2, 1))
        X = X.astype(np.float32)

        # Normalise as in DCRNN paper (via Z-Score Method)
        means = np.mean(X, axis=(1, 2))
        #X = X - means.reshape(1, -1, 1)
        stds = np.std(X, axis=(1, 2))
        #X = X / stds.reshape(1, -1, 1)
        self.A = torch.from_numpy(A)
        self.X = torch.from_numpy(X)
        print(self.A.shape)
        print(self.X.shape)

    def _get_edges_and_weights(self):
        edge_indices, values = dense_to_sparse(self.A)
        edge_indices = edge_indices.numpy()
        values = values.numpy()
        self.edges = edge_indices
        self.edge_weights = values
        

    def _generate_task(self, num_timesteps_in: int = 4, num_timesteps_out: int = 1):
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(self.X.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        # Generate observations
        features, target = [], []
        for i, j in indices:
            features.append((self.X[:,0:7,i : i + num_timesteps_in]).numpy())
            target.append((self.X[:, 7, i + num_timesteps_in : j]).numpy())
        self.features = features
        self.targets = target

    def get_dataset(
        self, num_timesteps_in: int = 4, num_timesteps_out: int = 1) -> StaticGraphTemporalSignal:
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
# Visualize traffic over time
import seaborn as sns
sensor_number = 55
hours = 96
sensor_labels = [bucket.y[sensor_number][0].item() for bucket in list(dataset)[:hours]]
sns.lineplot(data=sensor_labels)
# %%
from torch_geometric_temporal.signal import temporal_signal_split
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.8)
# %%
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
        self.bn = torch.nn.BatchNorm1d(32) 
        self.relu = torch.nn.ReLU()  
        self.dropout = torch.nn.Dropout(p=0.5)
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        h = self.tgnn(x, edge_index)
        for i in range(len(h)):
            if torch.isnan(h[i]).any():
                print(i)
        h = F.leaky_relu(h)
        h = self.bn(h)
        h = self.dropout(h)
        h = self.linear(h)

        return h

TemporalGNN(node_features=7, periods=1)
# %%
# GPU support
device = torch.device('cpu') # cuda
model = TemporalGNN(node_features=7, periods=1).to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.0001)
model.train()
# %%
#subset = 2000
print("Running training...")
for epoch in range(10): 
    loss = 0
    step = 0
    for snapshot in train_dataset:
        snapshot = snapshot.to(device)
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index)
        #loss = loss_function(y_hat,snapshot.y)
        nonzero_idx = torch.nonzero(snapshot.y)
        nonzero_idx = nonzero_idx[:,0]


        if nonzero_idx.numel()!=0:
            #print(nonzero_idx)          
            y_hat_non_zero = torch.index_select(y_hat, 0, nonzero_idx)
            snapshot_y_non_zero = torch.index_select(snapshot.y, 0, nonzero_idx)

            loss = torch.mean((y_hat_non_zero-snapshot_y_non_zero)**2) 
            # Mean squared error
            #loss = torch.mean((y_hat-snapshot.y)**2) 
            #loss = loss_function(y_hat,snapshot.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print("Epoch {} train MSE: {:.4f}".format(epoch, loss.item()))
# %%
#the last one to run 
model.eval()
loss = 0
step = 0
horizon = 24

# Store for analysis
predictions = []
labels = []

for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    # Get predictions
    y_hat = model(snapshot.x, snapshot.edge_index)
    nonzero_idx = torch.nonzero(snapshot.y)
    nonzero_idx = nonzero_idx[:,0]

    if nonzero_idx.numel()!=0:
        #print(nonzero_idx)          
        y_hat_non_zero = torch.index_select(y_hat, 0, nonzero_idx)
        snapshot_y_non_zero = torch.index_select(snapshot.y, 0, nonzero_idx)
        loss = torch.mean((y_hat_non_zero-snapshot_y_non_zero)**2) 

    # Store for analysis below
    labels.append(snapshot.y)
    predictions.append(y_hat)
    step += 1
    if step > horizon:
          break

loss = loss / (step+1)
#loss = loss.item()
print("Test MSE: {:.4f}".format(loss))
# %%
#don't run 
import numpy as np

sensor = 1
timestep = 0
preds = np.asarray([pred[sensor][timestep].detach().cpu().numpy() for pred in predictions])
labs  = np.asarray([label[sensor][timestep].cpu().numpy() for label in labels])
print("Data points:,", preds.shape)
# %%
import seaborn as sns
import matplotlib.pyplot as plt 
plt.figure(figsize=(20,5))
sns.lineplot(data=preds, label="pred")
sns.lineplot(data=labs, label="true")

#%%
#play around
import numpy as np 
arr = np.load('/home/zxu4/stGNN/21-pv-stgnn/data/node_values.npy') 
arr

#%%
#a function to calcualte the distance to the shore 
#is this okay? 

import folium
from geopy.geocoders import Nominatim
from geopy.distance import geodesic


# Replace 'your_location' with the location you want to find the nearest shore for
your_location = "41.0814,-75.5190"

# Initialize a geocoder
geolocator = Nominatim(user_agent="hwMTKQysu5D15SvMvK7IvZj1kNtgeqqU",)

# Use geopy to get the nearest shoreline
location = geolocator.reverse(your_location, exactly_one=True)

# Extract the coordinates of the nearest shore
shore_lat = location.latitude
shore_lon = location.longitude

# Calculate the distance to the nearest shore
distance_to_shore = geodesic(your_location, (shore_lat, shore_lon)).meters

# Create a map centered around your location
m = folium.Map(location=[float(your_location.split(',')[0]), float(your_location.split(',')[1])], zoom_start=10)

# Add a marker for your location
folium.Marker(
    location=[float(your_location.split(',')[0]), float(your_location.split(',')[1])],
    popup=f"Your Location: {your_location}",
    icon=folium.Icon(color='blue')
).add_to(m)

# Add a marker for the nearest shore
folium.Marker(
    location=[shore_lat, shore_lon],
    popup=f"Nearest Shore: {shore_lat}, {shore_lon}\nDistance: {distance_to_shore:.5f} m",
    icon=folium.Icon(color='red')
).add_to(m)

# Display the map
m.save('nearest_shore_map.html')
m
