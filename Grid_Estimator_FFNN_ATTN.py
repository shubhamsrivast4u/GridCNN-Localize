#!/usr/bin/env python
# coding: utf-8

# Import libraries
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
from sklearn.metrics import confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt
import os
import time
from scipy.io import savemat
import dill as pickle
import thop
from scipy import stats

# GPU configuration
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Select the GPU index

# Define all important parameters at the beginning
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 3000
lr = 5e-4
min_lr = 1e-6  # Define your desired minimum learning rate
num_epochs = 2000
patience = 500
num_runs = 50  # Number of runs for statistical confidence
weight_decay = 1e-3
hidden_size = 4096
input_grid_size = 10  # Grid size in meters
train_size_ratio = 0.8
val_size_ratio = 0.2
folder_path = "/homes/tanmoy/TanProj/"
#file_name = f"all_measurements_{input_grid_size}m_25k.json"
file_name = f"Train_{input_grid_size}m_Dataset.json"
file_name_grid = f"all_measurements_grid_{input_grid_size}m.json"
num_residual_blocks = 5  # Increased from 2 to 5
num_attention_heads = 4  # Increased from 2 to 4

# Create path directory for saving model weights and results
path_directory = f"finalmodels/FFNN_Attention/Grid_{input_grid_size}m/batch_size_{batch_size}/initial_lr_{lr}/epochs{num_epochs}_p{patience}_hs{hidden_size}_blocks{num_residual_blocks}_heads{num_attention_heads}/"

# Create directory if it doesn't exist
if not os.path.exists(path_directory):
    os.makedirs(path_directory)

def load_json_file(folder_path, file_name):
    file_path = Path(folder_path) / file_name
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File '{file_name}' not found in '{folder_path}'")
        return None
    except json.JSONDecodeError:
        print(f"Error: Unable to parse JSON from file '{file_name}'")
        return None

def normalize_serving_cell_measurements(df):
    params = ['RSRP', 'RSRQ', 'RSSI', 'SINR']
    norm_params = {}
    
    for metric in params:
        col_name = f"Serving_{metric}"
        if col_name in df.columns:
            mean_val = df[col_name].mean()
            std_val = df[col_name].std()
            
            norm_params[f"{col_name}_mean"] = mean_val
            norm_params[f"{col_name}_std"] = std_val
            
            if std_val != 0:
                df[col_name] = (df[col_name] - mean_val) / std_val
            else:
                print(f"Warning: Standard deviation is 0 for {col_name}")
                df[col_name] = df[col_name] - mean_val
    
    return df, norm_params

def normalize_neighbor_cell_measurements(df, max_neighbors):
    params = ['RSRP', 'RSRQ', 'RSSI']
    norm_params = {}
    
    for i in range(1, max_neighbors + 1):
        for metric in params:
            col_name = f"Neighbor_{i}_{metric}"
            if col_name in df.columns:
                mean_val = df[col_name].mean()
                std_val = df[col_name].std()
                
                norm_params[f"{col_name}_mean"] = mean_val
                norm_params[f"{col_name}_std"] = std_val
                
                if std_val != 0:
                    df[col_name] = (df[col_name] - mean_val) / std_val
                else:
                    print(f"Warning: Standard deviation is 0 for {col_name}")
                    df[col_name] = df[col_name] - mean_val
    
    return df, norm_params

def normalize_position_minmax(df):
    x_min = df['X_Position'].min()
    x_max = df['X_Position'].max()
    y_min = df['Y_Position'].min()
    y_max = df['Y_Position'].max()
    
    position_params = {
        'x_min': x_min,
        'x_max': x_max,
        'y_min': y_min,
        'y_max': y_max
    }
    
    df['X_Position'] = (df['X_Position'] - x_min) / (x_max - x_min)
    df['Y_Position'] = (df['Y_Position'] - y_min) / (y_max - y_min)
    
    return df, position_params

# def preprocess_data(json_data):
#     ue_data = []
#     max_neighbors = 0
#     for ue_index, ue in enumerate(json_data['UEs']):
#         ue_info = {
#             'UE_ID': ue['id'],
#             'X_Position': ue['position']['x'],
#             'Y_Position': ue['position']['y'],
#             'Serving_PCI': ue['servingCell']['pci'],
#             'Serving_RSRP': ue['servingCell']['rsrp'],
#             'Serving_RSRQ': ue['servingCell']['rsrq'],
#             'Serving_RSSI': ue['servingCell']['rssi'],
#             'Serving_SINR': ue['servingCell']['sinr'],
#             'Closest_Grid_dist': ue['gridDetail']['gridCenterDistance'],
#             'Closest_Grid_X': ue['gridDetail']['gridCenterX'],
#             'Closest_Grid_Y': ue['gridDetail']['gridCenterY'],
#             'Closest_Grid_ID': ue['gridDetail']['gridId']
#         }
        
#         if 'neighborCells' in ue and isinstance(ue['neighborCells'], list):
#             for i, neighbor in enumerate(ue['neighborCells']):
#                 ue_info[f'Neighbor_{i+1}_RSRP'] = neighbor.get('rsrp', 0)
#                 ue_info[f'Neighbor_{i+1}_RSRQ'] = neighbor.get('rsrq', 0)
#                 ue_info[f'Neighbor_{i+1}_RSSI'] = neighbor.get('rssi', 0)
#             max_neighbors = max(max_neighbors, len(ue['neighborCells']))
        
#         ue_data.append(ue_info)
    
#     df = pd.DataFrame(ue_data)
    
#     for i in range(1, max_neighbors + 1):
#         for metric in ['RSRP', 'RSRQ', 'RSSI']:
#             col_name = f'Neighbor_{i}_{metric}'
#             if col_name not in df.columns:
#                 df[col_name] = 0
#             else:
#                 df[col_name] = df[col_name].fillna(0)
    
#     df, position_params = normalize_position_minmax(df)
#     df.attrs.update(position_params)
    
#     df, serving_params = normalize_serving_cell_measurements(df)
#     df.attrs.update(serving_params)
    
#     df, neighbor_params = normalize_neighbor_cell_measurements(df, max_neighbors)
#     df.attrs.update(neighbor_params)
    
#     # Extracting the feature set
#     features = np.array(df.drop(['UE_ID', 'X_Position', 'Y_Position', 'Serving_PCI', 'Closest_Grid_dist', 
#                                 'Closest_Grid_X', 'Closest_Grid_Y', 'Closest_Grid_ID'], axis=1))
    
#     # Extracting labels
#     labels = np.array(df[['X_Position', 'Y_Position']])
    
#     # Ground truth grid details for the UEs
#     grid_id = np.array(df['Closest_Grid_ID'])
    
#     # Shuffle the data before splitting
#     indices = np.arange(len(features))
#     np.random.shuffle(indices)
#     features = features[indices]
#     labels = labels[indices]
#     grid_id = grid_id[indices]
    
#     features = torch.tensor(features, dtype=torch.float32)
#     labels = torch.tensor(labels, dtype=torch.float32)
#     grid_id = torch.tensor(grid_id, dtype=torch.float32)
    
#     return features, labels, grid_id, df

def preprocess_data(json_data):
    ue_data = []
    max_neighbors = 0
    # Store original index to map back to JSON geometry for CRLB
    original_indices = []
    
    for ue in json_data['UEs']:
        # Extract static UE information (constant across samples)
        ue_id = ue['id']
        x_pos = ue['position']['x']
        y_pos = ue['position']['y']
        
        # Grid details
        grid_detail = ue['gridDetail']
        closest_grid_x = grid_detail['gridCenterX']
        closest_grid_y = grid_detail['gridCenterY']
        closest_grid_id = grid_detail['gridId']
        #grid_ids.add(grid_detail['gridId'])
        
        # # Calculate grid distance if missing from JSON
        # if 'gridCenterDistance' in grid_detail:
        #     closest_grid_dist = grid_detail['gridCenterDistance']
        # else:
        #     closest_grid_dist = np.sqrt((x_pos - closest_grid_x)**2 + (y_pos - closest_grid_y)**2)

        # Iterate through each sample for the current UE
        if 'samples' in ue and isinstance(ue['samples'], list):
            for sample in ue['samples']:
                # Base info for this sample
                ue_info = {
                    'UE_ID': ue_id,
                    'Sample_ID': sample.get('sampleId', -1), # Tracking sample ID
                    'X_Position': x_pos,
                    'Y_Position': y_pos,
                    #'Closest_Grid_dist': closest_grid_dist,
                    'Closest_Grid_X': closest_grid_x,
                    'Closest_Grid_Y': closest_grid_y,
                    'GridID': closest_grid_id
                }

                # Serving Cell Info
                serving = sample.get('servingCell', {})
                ue_info.update({
                    'Serving_PCI': serving.get('pci', -1),
                    'Serving_RSRP': serving.get('rsrp', -140), # Default noise floor if missing
                    'Serving_RSRQ': serving.get('rsrq', -20),
                    'Serving_RSSI': serving.get('rssi', -110),
                    'Serving_SINR': serving.get('sinr', -20)
                    # 'Serving_X': serving.get('x', 0),
                    # 'Serving_Y': serving.get('y', 0)
                })

                # Neighbor Cells Info
                neighbors = sample.get('neighborCells', [])
                max_neighbors = max(max_neighbors, len(neighbors))

                for i, neighbor in enumerate(neighbors):
                    # Using 1-based indexing for columns to match your previous logic
                    idx = i + 1
                    # ue_info[f'Neighbor_{idx}_PCI'] = neighbor.get('pci', 0)
                    ue_info[f'Neighbor_{idx}_RSRP'] = neighbor.get('rsrp', -140)
                    ue_info[f'Neighbor_{idx}_RSRQ'] = neighbor.get('rsrq', -20)
                    ue_info[f'Neighbor_{idx}_RSSI'] = neighbor.get('rssi', -110)
                    # ue_info[f'Neighbor_{idx}_X'] = neighbor.get('x', 0)
                    # ue_info[f'Neighbor_{idx}_Y'] = neighbor.get('y', 0)

                ue_data.append(ue_info)

    # Create DataFrame
    df = pd.DataFrame(ue_data)
    df_dataset = df.copy()
    
    neighbor_defaults = {
        'RSRP': -140.0,  # Thermal noise floor
        'RSRQ': -20.0,   # Extremely poor quality
        'RSSI': -110.0  # Baseline noise power
    }

    # # Ensure all neighbor columns exist for all rows (fill with 0 if missing)
    # for i in range(1, max_neighbors + 1):
    #     #for metric in ['RSRP', 'RSRQ', 'RSSI', 'X', 'Y']:
    #     for metric in ['RSRP', 'RSRQ', 'RSSI']:
    #         col_name = f'Neighbor_{i}_{metric}'
    #         if col_name not in df.columns:
    #             df[col_name] = 0
    #         else:
    #             df[col_name] = df[col_name].fillna(0)
    
    # Iterate through all potential neighbor slots
    for i in range(1, max_neighbors + 1):
        for metric in ['RSRP', 'RSRQ', 'RSSI']:
            col_name = f'Neighbor_{i}_{metric}'
            
            # Determine the value to fill based on the metric type
            fill_value = neighbor_defaults.get(metric, 0)
            
            if col_name not in df.columns:
                # Column doesn't exist at all? Create it and fill with noise floor
                df[col_name] = fill_value
            else:
                # Column exists but has NaNs? Fill NaNs with noise floor
                df[col_name] = df[col_name].fillna(fill_value)
    
    df, position_params = normalize_position_minmax(df)
    df.attrs.update(position_params)
    
    df, serving_params = normalize_serving_cell_measurements(df)
    df.attrs.update(serving_params)
    
    df, neighbor_params = normalize_neighbor_cell_measurements(df, max_neighbors)
    df.attrs.update(neighbor_params)
    
    # Extracting the feature set
    features = np.array(df.drop(['UE_ID', 'Sample_ID', 'X_Position', 'Y_Position', 
                                 'Closest_Grid_X', 'Closest_Grid_Y', 'Serving_PCI', 'GridID'], axis=1))
    
    # Extracting labels
    labels = np.array(df[['X_Position', 'Y_Position']])
    
    # Ground truth grid details
    grid_id = np.array(df['GridID'])
    
    original_indices = np.arange(len(features))
    
    # Shuffle the data
    indices = np.arange(len(features))
    np.random.shuffle(indices)
    
    features = features[indices]
    labels = labels[indices]
    grid_id = grid_id[indices]
    
    # IMPORTANT: Return shuffled original indices to map back to geometry
    shuffled_original_indices = original_indices[indices]
    
    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)
    grid_id = torch.tensor(grid_id, dtype=torch.float32)
    
    return features, labels, grid_id, df

class GridDataPreprocessor:
    def __init__(self, json_path, json_file: str):
        """
        Initialize the preprocessor with path to JSON file
        
        Args:
            json_path (str): Path to the JSON file containing grid data
        """
        self.json_path = Path(json_path)/json_file
        self.raw_data = self._load_json()
        
    def _load_json(self) -> Dict:
        """Load JSON file and return the data"""
        with open(self.json_path, 'r') as f:
            return json.load(f)
    
    def _extract_grid_coordinates(self, grid_dict: Dict) -> torch.Tensor:
        """
        Extract coordinates from a single grid dictionary
        
        Args:
            grid_dict (Dict): Dictionary containing grid information
            
        Returns:
            torch.Tensor: Tensor of shape (num_points, 3) containing [x, y, grid_id]
        """
        points = []
        for point in grid_dict['gridDictionary']:
            points.append([
                point['gridCenterX'],
                point['gridCenterY'],
                point['gridId']
            ])
        return torch.tensor(points, dtype=torch.float32)
    
    def process_data(self) -> Dict[str, torch.Tensor]:
        """
        Process all grids in the data
        
        Returns:
            Dict[str, torch.Tensor]: Dictionary containing:
                - 'coordinates': Tensor of shape (num_grids, num_points, 3)
                - 'grid_centers': Tensor of shape (num_grids, num_points, 2)
                - 'grid_ids': Tensor of shape (num_grids, num_points)
        """
        all_grids = []
        
        for grid in self.raw_data['GRIDs']:
            grid_tensor = self._extract_grid_coordinates(grid)
            all_grids.append(grid_tensor)
            
        # Stack all grids into a single tensor
        all_grids_tensor = torch.stack(all_grids)
        
        # Split into coordinates and grid IDs
        grid_centers = all_grids_tensor[..., :2]  # Shape: (num_grids, num_points, 2)
        grid_ids = all_grids_tensor[..., 2]      # Shape: (num_grids, num_points)
        
        return {
            'coordinates': all_grids_tensor,      # Full data
            'grid_centers': grid_centers,         # Just x,y coordinates
            'grid_ids': grid_ids                  # Just grid IDs
        }

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels, in_channels),
            nn.BatchNorm1d(in_channels)
        )
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class SelfAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads):
        super(SelfAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        
    def forward(self, x):
        # Reshape for attention: [batch_size, seq_len=1, hidden_size]
        x_reshaped = x.unsqueeze(0)
        
        # Apply self-attention
        x_attended, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Remove sequence dimension: [batch_size, hidden_size]
        return x_attended.squeeze(0)

class ImprovedPositioningModel(nn.Module):
    def __init__(self, input_size, hidden_size=1024, output_size=2, num_residual_blocks=5, num_attention_heads=4):
        super(ImprovedPositioningModel, self).__init__()
        
        # Wider initial layers with progressive narrowing
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Increased number of residual blocks with improved regularization
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate=0.1) for _ in range(num_residual_blocks)
        ])
        
        # Enhanced attention mechanism with more heads
        self.attention = SelfAttentionLayer(hidden_size, num_attention_heads)
        
        # Second set of residual blocks after attention
        self.post_attention_blocks = nn.ModuleList([
            ResidualBlock(hidden_size, dropout_rate=0.1) for _ in range(2)
        ])
        
        # Deeper decoder with smooth dimension reduction
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_size // 4, hidden_size // 8),
            nn.BatchNorm1d(hidden_size // 8),
            nn.ReLU(),
            
            nn.Linear(hidden_size // 8, output_size),
            nn.Sigmoid()
        )
        
        # Place model on specified device
        self.to(device)
        
    def forward(self, x):
        # Initial feature extraction
        x = self.input_layer(x)
        
        # Apply first set of residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Apply self-attention
        x = self.attention(x)
        
        # Apply second set of residual blocks
        for res_block in self.post_attention_blocks:
            x = res_block(x)
        
        # Apply decoder to get final output
        x = self.decoder(x)
        return x

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(model, train_loader, val_loader, num_epochs=1000, patience=500, learning_rate=1e-3, path_directory=None):
    criterion = nn.MSELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Calculate required final_div_factor based on the desired min_lr
    final_div_factor = learning_rate / min_lr

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=learning_rate,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,  # Controls initial learning rate (max_lr/div_factor)
        final_div_factor=final_div_factor  # Controls final learning rate (max_lr/final_div_factor)
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_features, batch_labels in train_loader:
            # Move batch to device
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                # Move batch to device
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        
        if epoch % 10 == 0:
            print(f'epoch:{epoch+1}/{num_epochs} average TL:{avg_train_loss:.8f} average VL:{avg_val_loss:.8f} epoch time:{int(epoch_time)} seconds, lr:{current_lr:.2e}')
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # Save model to the specified path directory
            if path_directory:
                model_path = path_directory + 'best_model.pth'
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break
    
    return train_losses, val_losses

def gaussian_grid_probability(predicted_coords, grid_centers, sigma=50):
    # Calculate distances
    distances = np.sqrt(((grid_centers - predicted_coords[:, np.newaxis]) ** 2).sum(axis=2))
    # Apply Gaussian kernel
    probabilities = np.exp(-distances**2 / (2 * sigma**2))
    return probabilities / probabilities.sum(axis=1, keepdims=True)

def softmax_grid_probability(predicted_coords, grid_centers, temperature=1.0):
    # Calculate distances
    distances = np.sqrt(((grid_centers - predicted_coords[:, np.newaxis]) ** 2).sum(axis=2))
    # Apply softmax with temperature scaling
    logits = -distances / temperature
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)

def calculate_confidence_interval(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    h = se * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, h

if __name__ == "__main__":
    print(f"Using device: {device}")
    print(f"Model will be saved to: {path_directory}")
    print(f"Model architecture: {hidden_size} hidden units, {num_residual_blocks} residual blocks, {num_attention_heads} attention heads")
    
    # Load data
    json_data = load_json_file(folder_path, file_name)
    if json_data is None:
        print("Error loading data")
        exit()
    
    # Process grid information
    preprocessor_grid = GridDataPreprocessor(folder_path, file_name_grid)
    processed_data_grid = preprocessor_grid.process_data()
    
    coordinates = processed_data_grid['coordinates'].to(device)
    grid_centers = processed_data_grid['grid_centers'].to(device)
    grid_ids = processed_data_grid['grid_ids'].to(device)
    center = grid_centers.cpu().numpy()[0]
    ids = grid_ids.cpu().numpy()[0]
    
    # Initial preprocessing for model training
    features, labels, grid_number, df = preprocess_data(json_data)
    
    # Get position normalization parameters for future use
    x_pos_min = df.attrs['x_min']
    x_pos_max = df.attrs['x_max']
    y_pos_min = df.attrs['y_min']
    y_pos_max = df.attrs['y_max']
    
    # Move tensors to device
    features = features.to(device)
    labels = labels.to(device)
    grid_number = grid_number.to(device)
    
    # Calculate split sizes
    total_samples = len(features)
    train_size = int(train_size_ratio * total_samples)
    
    # Split the data for training and validation/testing
    train_features = features[:train_size]
    train_labels = labels[:train_size]
    val_features = features[train_size:]
    val_labels = labels[train_size:]
    test_features = val_features  # Using validation set as test set
    test_labels = val_labels
    test_grid_ids = grid_number[train_size:]
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_features, train_labels)
    val_dataset = CustomDataset(val_features, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model with enhanced architecture
    input_size = features.shape[1]
    model = ImprovedPositioningModel(
        input_size, 
        hidden_size, 
        output_size=2, 
        num_residual_blocks=num_residual_blocks, 
        num_attention_heads=num_attention_heads
    )
    
    # Measure training time
    start_time = time.time()
    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=num_epochs,
        patience=patience,
        learning_rate=lr,
        path_directory=path_directory
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # --- Complexity Analysis (IEEE Requirement) ---
    dummy_input = torch.randn(1, input_size).to(device)
    # thop.profile returns MACs, convert to FLOPs (x2)
    macs, params = thop.profile(model, inputs=(dummy_input,), verbose=False)
    flops = macs * 2
    print(f'Model MACs: {macs}')
    print(f'Model FLOPs: {flops}')
    print(f'Model Params Num: {params}\n')
    
    losses_dict = {
        'train_losses': np.array(train_losses), 
        'val_losses': np.array(val_losses), 
        'flops': flops, 
        'parameters': params
    }
    
    savemat(path_directory + 'losses.mat', losses_dict)
    with open(path_directory + 'losses.pkl', 'wb') as file:
        pickle.dump(losses_dict, file)
    # Save in both .mat and .pkl formats
    savemat(path_directory + 'losses.mat', losses_dict)
    with open(path_directory + 'losses.pkl', 'wb') as file:
        pickle.dump(losses_dict, file)
        
    # --- Inference and Latency Measurement ---
    print("\n--- Running Inference & Latency Test ---")
    latency_list = []
    
    # Warm up GPU
    with torch.no_grad():
        _ = model(test_features[:10])
        
    # Measure Latency over batches
    test_loader_lat = DataLoader(CustomDataset(test_features, test_labels), batch_size=1)
    
    with torch.no_grad():
        start_inf = time.perf_counter()
        outputs_test = model(test_features)
        end_inf = time.perf_counter()
        
    avg_latency_batch = (end_inf - start_inf) / len(test_features) * 1000 # ms per sample (batched)
    print(f"Inference Latency (Batched): {avg_latency_batch:.4f} ms/sample")

    
    # Save training curves as PDF
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(path_directory + 'training_curves.pdf')
    
    # Load the best model for evaluation
    model = ImprovedPositioningModel(
        input_size, 
        hidden_size, 
        output_size=2, 
        num_residual_blocks=num_residual_blocks, 
        num_attention_heads=num_attention_heads
    )
    model.load_state_dict(torch.load(path_directory + 'best_model.pth'))
    model.to(device)
    model.eval()
    
    # --- Initial single evaluation ---
    # Evaluate on test data
    with torch.no_grad():
        outputs_test = model(test_features)
    
    # Convert normalized predictions back to original scale
    predicted = outputs_test.detach().cpu().numpy()
    real = test_labels.cpu().numpy()
    
    # Denormalize predictions
    predicted[:,0] = x_pos_min + predicted[:,0]*(x_pos_max - x_pos_min)
    predicted[:,1] = y_pos_min + predicted[:,1]*(y_pos_max - y_pos_min)
    
    # Denormalize ground truth
    real[:,0] = x_pos_min + real[:,0]*(x_pos_max - x_pos_min)
    real[:,1] = y_pos_min + real[:,1]*(y_pos_max - y_pos_min)
    
    # Calculate distances for grid prediction
    distances = np.sqrt((center[:, np.newaxis, 0] - predicted[:, 0])**2 + 
                        (center[:, np.newaxis, 1] - predicted[:, 1])**2)
    
    # Calculate prediction accuracy using different methods
    # 1. Using reciprocal of distances
    reciprocals = 1 / distances
    column_sums = reciprocals.sum(axis=0)
    probabilities_rec = reciprocals / column_sums
    grid_predicted = np.argmax(probabilities_rec, axis=0)
    grid_real = test_grid_ids.cpu().numpy()
    accuracy = metrics.accuracy_score(grid_real, grid_predicted)
    
    # 2. Using Gaussian kernel
    probabilities_gauss = gaussian_grid_probability(predicted, center, sigma=50)
    grid_predicted_gauss = np.argmax(probabilities_gauss, axis=1)
    accuracy_gauss = metrics.accuracy_score(grid_real, grid_predicted_gauss)
    
    # 3. Using softmax
    probabilities_softmax = softmax_grid_probability(predicted, center, temperature=1.0)
    grid_predicted_softmax = np.argmax(probabilities_softmax, axis=1)
    accuracy_softmax = metrics.accuracy_score(grid_real, grid_predicted_softmax)
    
    # Calculate positioning error
    distance_error = np.linalg.norm(real - predicted, axis=1)
    mean_error = np.mean(distance_error)
    diff = predicted - real
    rmse_ATTN = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
    print(f"RMSE_ATTN = {rmse_ATTN:.3f} m")
    percentile_90 = np.percentile(distance_error, 90)
    
    # Save the confusion matrix for grid prediction
    conf_matrix = confusion_matrix(grid_real, grid_predicted, labels=ids)
    np.save(path_directory + 'confusion_matrix.npy', conf_matrix)
    
    # Print and save single run results
    print(f"\n--- Single Run Results ---")
    print(f"Grid Size: {input_grid_size}m")
    print(f"Residual Blocks: {num_residual_blocks}")
    print(f"Attention Heads: {num_attention_heads}")
    print(f"Mean 2D Error: {mean_error:.2f} meters")
    print(f"90th Percentile Error: {percentile_90:.2f} meters")
    print(f"Grid Accuracy (Reciprocal): {accuracy:.4f}")
    print(f"Grid Accuracy (Gaussian): {accuracy_gauss:.4f}")
    print(f"Grid Accuracy (Softmax): {accuracy_softmax:.4f}")
    print(f"Training Time: {training_time:.2f} seconds")
    
    # Save results to file
    with open(path_directory + 'results_single.txt', 'w') as f:
        f.write(f"Grid Size: {input_grid_size}m\n")
        f.write(f"Residual Blocks: {num_residual_blocks}\n")
        f.write(f"Attention Heads: {num_attention_heads}\n")
        f.write(f"Mean 2D Error: {mean_error:.2f} meters\n")
        f.write(f"90th Percentile Error: {percentile_90:.2f} meters\n")
        f.write(f"Grid Accuracy (Reciprocal): {accuracy:.4f}\n")
        f.write(f"Grid Accuracy (Gaussian): {accuracy_gauss:.4f}\n")
        f.write(f"Grid Accuracy (Softmax): {accuracy_softmax:.4f}\n")
        f.write(f"Training Time: {training_time:.2f} seconds\n")
        f.write(f"Inference Latency: {avg_latency_batch:.4f} ms\n")
        f.write(f"FLOPs: {flops}\n")
    
    # Plot CDF of positioning errors and save as PDF
    plt.figure(figsize=(8, 5))
    sorted_errors = np.sort(distance_error)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    plt.plot(sorted_errors, cdf, label="CDF", color="blue")
    plt.axvline(percentile_90, color="red", linestyle="--", 
                label=f"90th Percentile = {percentile_90:.2f}")
    plt.xlabel("Distance Error (meters)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"CDF of Distance Errors for {input_grid_size}m grid size FFNN-Attention (Single Run)")
    plt.legend()
    plt.grid()
    plt.savefig(path_directory + 'error_cdf_single.pdf')
    
    # Save performance metrics for single run
    metrics_dict_single = {
        'mean_error': mean_error,
        'percentile_90': percentile_90,
        'accuracy': accuracy,
        'accuracy_gauss': accuracy_gauss,
        'accuracy_softmax': accuracy_softmax,
        'training_time': training_time,
        'sorted_errors': sorted_errors,
        'cdf': cdf,
        'num_residual_blocks': num_residual_blocks,
        'num_attention_heads': num_attention_heads
    }
    
    savemat(path_directory + 'metrics_single.mat', metrics_dict_single)
    with open(path_directory + 'metrics_single.pkl', 'wb') as file:
        pickle.dump(metrics_dict_single, file)
    
    # # --- Multiple runs for statistical confidence ---
    # print(f"\n--- Starting Multiple Run Analysis ({num_runs} runs) ---")
    
    # # Lists to store metrics from all runs
    # dist_prob_list = []  # Reciprocal distance accuracy
    # gauss_prob_list = []  # Gaussian kernel accuracy
    # softm_prob_list = []  # Softmax accuracy
    # percentile_90_list = []  # 90th percentile errors
    # mean_err_list = []  # Mean errors
    # all_distance_errors = []  # All distance errors across all runs
    
    # # Start multiple runs
    # for run in range(num_runs):
    #     if run % 10 == 0:
    #         print(f"Run {run+1}/{num_runs}...")
            
    #     # Preprocess data with different random shuffling
    #     features_run, labels_run, grid_number_run, _ = preprocess_data(json_data)
        
    #     # Move tensors to device
    #     features_run = features_run.to(device)
    #     labels_run = labels_run.to(device)
    #     grid_number_run = grid_number_run.to(device)
        
    #     # Split the data for this run
    #     train_size = int(train_size_ratio * len(features_run))
    #     test_features_run = features_run[train_size:]
    #     test_labels_run = labels_run[train_size:]
    #     test_grid_ids_run = grid_number_run[train_size:]
        
    #     # Load the trained model
    #     model.load_state_dict(torch.load(path_directory + 'best_model.pth'))
    #     model.eval()
        
    #     # Evaluate on test data for this run
    #     with torch.no_grad():
    #         outputs_test_run = model(test_features_run)
        
    #     # Denormalize predictions
    #     predicted_run = outputs_test_run.detach().cpu().numpy()
    #     predicted_run[:,0] = x_pos_min + predicted_run[:,0]*(x_pos_max - x_pos_min)
    #     predicted_run[:,1] = y_pos_min + predicted_run[:,1]*(y_pos_max - y_pos_min)
        
    #     # Denormalize ground truth
    #     real_run = test_labels_run.cpu().numpy()
    #     real_run[:,0] = x_pos_min + real_run[:,0]*(x_pos_max - x_pos_min)
    #     real_run[:,1] = y_pos_min + real_run[:,1]*(y_pos_max - y_pos_min)
        
    #     # Calculate distances for grid prediction
    #     distances_run = np.sqrt((center[:, np.newaxis, 0] - predicted_run[:, 0])**2 + 
    #                         (center[:, np.newaxis, 1] - predicted_run[:, 1])**2)
        
    #     # Calculate reciprocal distance accuracy
    #     reciprocals_run = 1 / distances_run
    #     column_sums_run = reciprocals_run.sum(axis=0)
    #     probabilities_rec_run = reciprocals_run / column_sums_run
    #     grid_predicted_run = np.argmax(probabilities_rec_run, axis=0)
    #     grid_real_run = test_grid_ids_run.cpu().numpy()
    #     accuracy_run = metrics.accuracy_score(grid_real_run, grid_predicted_run)
        
    #     # Calculate Gaussian kernel accuracy
    #     probabilities_gauss_run = gaussian_grid_probability(predicted_run, center, sigma=50)
    #     grid_predicted_gauss_run = np.argmax(probabilities_gauss_run, axis=1)
    #     accuracy_gauss_run = metrics.accuracy_score(grid_real_run, grid_predicted_gauss_run)
        
    #     # Calculate softmax accuracy
    #     probabilities_softmax_run = softmax_grid_probability(predicted_run, center, temperature=1.0)
    #     grid_predicted_softmax_run = np.argmax(probabilities_softmax_run, axis=1)
    #     accuracy_softmax_run = metrics.accuracy_score(grid_real_run, grid_predicted_softmax_run)
        
    #     # Calculate positioning error
    #     distance_error_run = np.linalg.norm(real_run - predicted_run, axis=1)
    #     mean_error_run = np.mean(distance_error_run)
    #     percentile_90_run = np.percentile(distance_error_run, 90)
        
    #     # Store metrics for this run
    #     dist_prob_list.append(accuracy_run)
    #     gauss_prob_list.append(accuracy_gauss_run)
    #     softm_prob_list.append(accuracy_softmax_run)
    #     percentile_90_list.append(percentile_90_run)
    #     mean_err_list.append(mean_error_run)
        
    #     # Store all distance errors for comprehensive CDF
    #     all_distance_errors.extend(distance_error_run.tolist())
    
    # # Calculate mean and confidence intervals for all metrics
    # mean_accuracy, margin_of_error_accuracy = calculate_confidence_interval(dist_prob_list)
    # mean_accuracy_gauss, margin_of_error_accuracy_gauss = calculate_confidence_interval(gauss_prob_list)
    # mean_accuracy_softmax, margin_of_error_accuracy_softmax = calculate_confidence_interval(softm_prob_list)
    # mean_percentile_90, margin_of_error_percentile_90 = calculate_confidence_interval(percentile_90_list)
    # mean_2D_err, margin_of_error_2D_err = calculate_confidence_interval(mean_err_list)

    # # Print results of multiple runs with confidence intervals
    # print("\n--- Multiple Run Results with 95% Confidence Intervals ---")
    # print(f"Mean 2D Error: {mean_2D_err:.2f} ± {margin_of_error_2D_err:.2f} meters")
    # print(f"90th Percentile Error: {mean_percentile_90:.2f} ± {margin_of_error_percentile_90:.2f} meters")
    # print(f"Grid Accuracy (Reciprocal): {mean_accuracy:.4f} ± {margin_of_error_accuracy:.4f}")
    # print(f"Grid Accuracy (Gaussian): {mean_accuracy_gauss:.4f} ± {margin_of_error_accuracy_gauss:.4f}")
    # print(f"Grid Accuracy (Softmax): {mean_accuracy_softmax:.4f} ± {margin_of_error_accuracy_softmax:.4f}")
    
    # # Save results to file
    # with open(path_directory + 'results_multiple.txt', 'w') as f:
    #     f.write(f"Grid Size: {input_grid_size}m\n")
    #     f.write(f"Residual Blocks: {num_residual_blocks}\n")
    #     f.write(f"Attention Heads: {num_attention_heads}\n")
    #     f.write(f"Number of runs: {num_runs}\n\n")
    #     f.write(f"Mean 2D Error: {mean_2D_err:.2f} ± {margin_of_error_2D_err:.2f} meters\n")
    #     f.write(f"90th Percentile Error: {mean_percentile_90:.2f} ± {margin_of_error_percentile_90:.2f} meters\n")
    #     f.write(f"Grid Accuracy (Reciprocal): {mean_accuracy:.4f} ± {margin_of_error_accuracy:.4f}\n")
    #     f.write(f"Grid Accuracy (Gaussian): {mean_accuracy_gauss:.4f} ± {margin_of_error_accuracy_gauss:.4f}\n")
    #     f.write(f"Grid Accuracy (Softmax): {mean_accuracy_softmax:.4f} ± {margin_of_error_accuracy_softmax:.4f}\n")
    
    # # Plot comprehensive CDF from all runs
    # plt.figure(figsize=(8, 5))
    # all_sorted_errors = np.sort(all_distance_errors)
    # all_cdf = np.arange(1, len(all_sorted_errors) + 1) / len(all_sorted_errors)
    
    # # Calculate the 90th percentile from the combined errors
    # all_percentile_90 = np.percentile(all_distance_errors, 90)
    
    # plt.plot(all_sorted_errors, all_cdf, label="CDF", color="blue")
    # plt.axvline(all_percentile_90, color="red", linestyle="--", 
    #             label=f"90th Percentile = {all_percentile_90:.2f}")
    # plt.xlabel("Distance Error (meters)")
    # plt.ylabel("Cumulative Probability")
    # plt.title(f"CDF of Distance Errors for {input_grid_size}m grid size FFNN-Attention ({num_runs} runs)")
    # plt.legend()
    # plt.grid()
    # plt.savefig(path_directory + 'error_cdf_multiple.pdf')
    
    # # Save comprehensive metrics with confidence intervals
    # metrics_dict_multiple = {
    #     'mean_error': mean_2D_err,
    #     'mean_error_margin': margin_of_error_2D_err,
    #     'percentile_90': mean_percentile_90,
    #     'percentile_90_margin': margin_of_error_percentile_90,
    #     'accuracy': mean_accuracy,
    #     'accuracy_margin': margin_of_error_accuracy,
    #     'accuracy_gauss': mean_accuracy_gauss,
    #     'accuracy_gauss_margin': margin_of_error_accuracy_gauss,
    #     'accuracy_softmax': mean_accuracy_softmax,
    #     'accuracy_softmax_margin': margin_of_error_accuracy_softmax,
    #     'num_runs': num_runs,
    #     'num_residual_blocks': num_residual_blocks,
    #     'num_attention_heads': num_attention_heads,
    #     'all_sorted_errors': all_sorted_errors,
    #     'all_cdf': all_cdf,
    #     'all_percentile_90': all_percentile_90,
    #     'mean_errors_per_run': np.array(mean_err_list),
    #     'percentile_90_per_run': np.array(percentile_90_list),
    #     'accuracy_per_run': np.array(dist_prob_list),
    #     'accuracy_gauss_per_run': np.array(gauss_prob_list),
    #     'accuracy_softmax_per_run': np.array(softm_prob_list)
    # }
    
    # savemat(path_directory + 'metrics_multiple.mat', metrics_dict_multiple)
    # with open(path_directory + 'metrics_multiple.pkl', 'wb') as file:
    #     pickle.dump(metrics_dict_multiple, file)
    
    # print(f"\nAll results saved to {path_directory}")
    # print(f"Training time: {training_time:.2f} seconds")
    # print(f"Total evaluation time for {num_runs} runs: {time.time() - start_time - training_time:.2f} seconds")
