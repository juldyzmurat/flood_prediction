#%%
import json
import logging
import numpy as np
np.random.seed(0)
import os
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import datetime

import random
random.seed(0)
# %%
import torch
torch.manual_seed(0)
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR 
from torch_geometric.nn import GATConv, GCNConv, GATv2Conv
from torch.utils.tensorboard import SummaryWriter

print(torch.__version__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device}")
# %%
config = {
    'TRAIN_TEST_PROPORTION'             : (0.7, 0.1, 0.2), #(Train %, Validation %, Test %)
    'BATCH_SIZE'                        : 128,
    'EPOCHS'                            : 100,
    'WEIGHT_DECAY'                      : 0,
    'INITIAL_LR'                        : 0.05,
    'DROPOUT'                           : 0.0,
    'ATTENTION_HEADS'                   : 8,
    'RESULTS_DIR'                       : './runs/'+time.strftime("%m-%dT%H-%M-%S")+'/',
    'data_with_already_filled_gaps'     : True,
    'counter_files_path'                : './toy_data/toy_counters/',
    'counters_nontemporal_aggregated'   : './toy_data/toy_counters_non_temporal_aggregated_data.csv',
    'USE_YEAR_PERIODIC_DATA'            : False,
    'HALF_INTERVAL_SIZE'                : 3 * 24,
    'USE_HOLIDAY_FEATURES'              : False,
    'USE_WEEKDAY_FEATURES'              : True,
    'USE_MONTH_FEATURES'                : True,
    'N_GRAPHS'                          : 100*24,
    'F_IN'                              : 7*24,
    'F_OUT'                             : 7*24,
    'N_NODE'                            : 165,
    'target_col'                        : 'Fast',
    'use_tensorboard'                   : False,
    'USE_GAT'                           : True, # if True use GAT, else use GCN
    'USE_LSTM'                          : True, # if True use LSTM, else use GRU
    'LSTM_LAYER_SIZES'                  : [100, 100],  
    'GRU_LAYER_SIZES'                   : [800, 800],  
    'LINEAR_HIDDEN_SIZE'                : 100,     
    'USE_EARLY_STOPPING'                : True,
    "MIN_ITERATIONS_EARLY_STOPPING"     : 40,
    "EARLY_STOPPING_TOLERANCE"          : 10,
    "LOG_BASELINE"                      : True, # if true outputs average rmse on computed on each batch,
    "DATA_DATE_SPLIT"                   : '05/07/22 00:00:00',
    "SCALE_DATA"                        : False,
    "USE_ONEHOT_FEATURES"               : False
}

# Set logging level
logging.getLogger().setLevel(logging.INFO)

# Make a tensorboard writer
if config["use_tensorboard"]:
    writer = SummaryWriter()