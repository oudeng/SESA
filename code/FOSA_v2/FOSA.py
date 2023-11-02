'''
FOSA (Full information maximum likelihood (FIML) Optimized Self-Attention)

# What is FOSA?
  A python programme for missing data imputation.
  More details, please see https://github.com/oudeng/FOSA/
  
# How FOSA works?
  Two steps of FOSA：
  1. Step-I：  FIML estimation。
  2. Step-II： Self-Attention training by FIML estimation input.
  3. Final imputation by trained SA model.
'''

import os
from typing import Any
import torch
import torch.nn as nn
import torch.optim as optim
from semopy import Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore")  # silence warnings

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# Seed for reproducibility
np.random.seed(2023)

def approximate_decode(value, encoding_dict, column_name):
    # 获取指定列的编码字典
    label_dict = encoding_dict[column_name]
    
    # 初始化最小距离和对应的标签
    min_distance = float('inf')
    closest_label = None
    
    for encoded_value,label in label_dict.items():
        # 计算当前值与编码值的距离
        distance = abs(value - encoded_value)
        
        # 如果找到更接近的标签，更新最小距离和标签
        if distance < min_distance:
            min_distance = distance
            closest_label = label
    
    return closest_label

def reverse_encoding(data, encoding_dict):
    reversed_data = data.copy()

    for column in data.columns:
        if column in encoding_dict:
            column_data = data[column].values
            reversed_column_data = [approximate_decode(encoded_value, encoding_dict, column) for encoded_value in column_data]
            reversed_data[column] = reversed_column_data

    return reversed_data

def encode_columns_with_nan_preservation(data):
    struct_data = data.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
    encoding_dict = {}

    for col in non_numeric_columns:
        # Drop NaN values before encoding
        valid_data = struct_data[col].dropna()
        
        # Fit the encoder on the entire column including NaN, 
        # but transform only the non-NaN values
        le = LabelEncoder()
        le.fit(data[col].astype(str))
        encoded_values = le.transform(valid_data.astype(str))
        
        # Create a dictionary mapping for the current column
        encoding_dict[col] = dict(zip(encoded_values, valid_data))
        
        # 记录NaN的下标
        nan_indices = struct_data[col].index[struct_data[col].isna()]

        # Replace the original values with the encoded values in the struct_data DataFrame
        struct_data[col] = le.fit_transform(struct_data[col])
        # 根据记录的NaN下标将NaN填充回去
        struct_data[col].iloc[nan_indices] = np.nan

    return struct_data, encoding_dict

# Self-Attention Model Definition.
class SelfAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, dropout_rate):
        super(SelfAttentionModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = hidden_dim
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(embed_dim=self.embed_dim, num_heads=num_heads)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        batch_size, seq_length, feature_dim = x.size()
        x = x.view(batch_size * seq_length, feature_dim)
        x = self.linear(x)
        x = x.view(batch_size, seq_length, self.hidden_dim)
        x = x.transpose(0, 1)
        attn_output, _ = self.attention(x, x, x)
        attn_output = attn_output.transpose(0, 1)
        attn_output = torch.mean(attn_output, dim=1)
        x = self.fc1(attn_output)
        x = self.dropout(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x.squeeze()

# Evaluation functions: KL, MAPE
def compute_kl_divergence(p, q):
    p = np.asarray(p, dtype=np.float64) + 1e-10
    q = np.asarray(q, dtype=np.float64) + 1e-10
    return np.sum(np.where(p != 0, p * np.log(p / q), 0))

def compute_mape(true_values, predicted_values):
    return np.mean(np.abs((true_values - predicted_values) / true_values)) * 100

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

# Define covariance loss function.
def covariance_loss(predicted, target):
    cov_pred = torch.matmul(predicted.T, predicted) / predicted.size(0)
    cov_target = torch.matmul(target.T, target) / target.size(0)
    loss = torch.norm(cov_pred - cov_target, p='fro')
    return loss

# L1 Regularization function.
def l1_regularizer(model):
    l1 = torch.tensor(0.).to(device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1 = l1 + torch.norm(param, 1)
    return l1

# Introduce missingness in the standardized dataset
def introduce_missing(data, missing_rate=0.4):
    modified_data = data.copy()
    for col in data.columns:
        mask = (np.random.random(len(data)) < missing_rate)
        modified_data[col][mask] = np.nan
    return modified_data

###########################################

class FOSA():
    def run(self,dataset_1,model_desc,test_size=0.2,random_state=2023,num_heads = 4,hidden_dim = 64,dropout_rate = 0.5,l1_lambda = 0.001,num_epochs = 200) -> None:
        
        #default: test_size=0.2,random_state=2023,num_heads = 4,hidden_dim = 64,dropout_rate = 0.5,l1_lambda = 0.001,num_epochs = 200
        
        # Split the original dataset_1 into training and testing set.
        train_data, test_data = train_test_split(dataset_1, test_size=test_size, random_state=random_state)

        # Get index of train_data and test_data.
        train_data_index = train_data.index.tolist()
        test_data_index = test_data.index.tolist()
        
        # Standardize the data using only the training data.
        scaler = StandardScaler()
        train_data_missing = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
        test_data_missing = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)
        # 合并 train_data_missing 和 test_data_missing
        data_missing = pd.DataFrame(scaler.transform(dataset_1), columns=dataset_1.columns)

        # Use column means to fill NaN values as an initial approximation
        initial_filled_train_data = train_data_missing.fillna(train_data_missing.mean())

        # Use the initialized data to estimate using FIML
        mod = Model(model_desc)
        res = mod.fit(initial_filled_train_data, obj="FIML")
        self.mod = mod
        
        # Check if it's successful
        if res.success:
            print("FIML estimation succeeded.")
            fiml_train_predictions = mod.predict(train_data_missing)
            fiml_test_predictions = mod.predict(test_data_missing)
        else:
            print("FIML estimation failed. Please consider adjusting optimization parameters or use another approach.")

        train_inputs_np = np.where(np.isnan(train_data_missing.values), fiml_train_predictions.values, train_data_missing.values)
        test_inputs_np = np.where(np.isnan(test_data_missing.values), fiml_test_predictions.values, test_data_missing.values)

        # 使用之前的scaler进行逆变换（还原）训练数据
        train_data_FIML_restored = pd.DataFrame(scaler.inverse_transform(train_inputs_np), columns=train_data.columns)

        # 使用相同的scaler进行逆变换（还原）测试数据
        test_data_FIML_restored = pd.DataFrame(scaler.inverse_transform(test_inputs_np), columns=test_data.columns)

        FIML_imputed = np.around(np.vstack((train_data_FIML_restored,test_data_FIML_restored)), decimals=3)
        
        # 将 FIML_imputed 转化为 DataFrame
        FIML_imputed = pd.DataFrame(FIML_imputed, columns=dataset_1.columns)
        FIML_imputed['original_index'] = train_data_index+test_data_index
        
        # 按照 'original_index' 列排序
        FIML_imputed = FIML_imputed.sort_values(by='original_index').reset_index(drop=True)

        # 删除 'original_index' 列
        FIML_imputed = FIML_imputed.drop(columns=['original_index'])

        # Convert to tensors
        train_inputs = torch.tensor(train_inputs_np).float().to(device)
        test_inputs = torch.tensor(test_inputs_np).float().to(device)
        train_targets = torch.tensor(fiml_train_predictions.values).float().to(device)
        test_targets = torch.tensor(fiml_test_predictions.values).float().to(device)

        # Model parameters
        input_dim = train_data.shape[1]
        output_dim = test_data.shape[1]

        # Initialize the model with He initialization in weights_init.
        attention_model = SelfAttentionModel(input_dim, hidden_dim, output_dim, num_heads, dropout_rate).to(device)
        attention_model.apply(weights_init)

        # Optimizer and Loss
        optimizer = optim.Adam(attention_model.parameters(), lr=0.001)
        criterion = nn.MSELoss()

        # Training loop
        train_losses = []
        test_losses = []

        for epoch in range(num_epochs):
            attention_model.train()
            optimizer.zero_grad()
            outputs = attention_model(train_inputs)
            
            # Calculate the combined loss
            mse_loss = criterion(outputs, train_targets)
            cov_loss = covariance_loss(outputs, train_targets)
            total_loss = mse_loss + cov_loss + l1_lambda * l1_regularizer(attention_model)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(attention_model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Evaluation
            attention_model.eval()
            with torch.no_grad():
                test_outputs = attention_model(test_inputs)
                test_loss = criterion(test_outputs, test_targets)
            
            train_losses.append(total_loss.item())
            test_losses.append(test_loss.item())

        # Plotting the losses
        plt.figure(figsize=(15, 8))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(test_losses, label='Test Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Testing Losses over Epochs')
        plt.legend()
        plt.grid(True)
        
        # Save the figure to a file
        plt.savefig('FOSA_Epoch_Loss.png', dpi=300)
        plt.show()
        
        # Prediction
        attention_model.eval()
        with torch.no_grad():
            train_predictions = attention_model(torch.tensor(train_inputs_np).float().to(device))
            test_predictions = attention_model(torch.tensor(test_inputs_np).float().to(device))

        train_predictions_np = train_predictions.cpu().numpy()
        test_predictions_np = test_predictions.cpu().numpy()
        
        # print(train_predictions_np)
        train_inputs_np = np.where(np.isnan(train_data_missing.values), train_predictions_np, train_data_missing.values)
        test_inputs_np = np.where(np.isnan(test_data_missing.values), test_predictions_np, test_data_missing.values)

        # 使用之前的scaler进行逆变换（还原）训练数据
        train_data_restored = pd.DataFrame(scaler.inverse_transform(train_inputs_np), columns=train_data.columns)

        # 使用相同的scaler进行逆变换（还原）测试数据
        test_data_restored = pd.DataFrame(scaler.inverse_transform(test_inputs_np), columns=test_data.columns)
        dataset_1_imputed = np.around(np.vstack((train_data_restored,test_data_restored)), decimals=3)
        
        # 将 dataset_1_imputed 转化为 DataFrame
        dataset_1_imputed = pd.DataFrame(dataset_1_imputed, columns=dataset_1.columns)
        dataset_1_imputed['original_index'] = train_data_index+test_data_index
        # 按照 'original_index' 列排序
        dataset_1_imputed = dataset_1_imputed.sort_values(by='original_index').reset_index(drop=True)

        # 删除 'original_index' 列
        dataset_1_imputed = dataset_1_imputed.drop(columns=['original_index'])

        return dataset_1_imputed,FIML_imputed,data_missing
