'''
Sample code of FOSA.
这是FOSA插补的测试sample程序。


'''

import numpy as np
from FOSA import FOSA,encode_columns_with_nan_preservation,reverse_encoding

import pandas as pd
import copy
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

###################### Input start 输入部分开始  ###########################

## Read dataset(with missing data). Here is a sample.
## 读入数据。此处用含缺失值的测试数据文件： heart_2020_test.csv
data = pd.read_csv('./Test_datasets/heart_2020_test.csv', delimiter=',', low_memory=False)

## Select cols of the dataset manually. 
# 手动选取df需要的数据部分。
data_0 = copy.deepcopy(data[['HeartDisease', 'BMI', 'PhysicalHealth', 'MentalHealth','SleepTime','Stroke', 'Sex']])

## Data proceding. Encoding non-numeric data.
# 处理数据。非数值类数据做编码处理。
encoded_data, encoding_dict = encode_columns_with_nan_preservation(data_0.copy())

## Decide the rows of dataset.
## 手动选取数据集大小
first_rows =1000
dataset_1 = copy.deepcopy(encoded_data.head(first_rows))
print(dataset_1)

## FIML Model specification.
## 手动输入FIML模型的描述。
model_desc = """
BMI ~ PhysicalHealth + MentalHealth + HeartDisease + SleepTime + Stroke + Sex
"""

###################### Input end 输入部分结束  ############################

# FOSA imputation for dataset.
# 对data的做FOSA插补
dataset_1_imputed,FIML_imputed,data_missing = FOSA().run(dataset_1,model_desc)

# FOSA Visualization 

## Visualize dataset_1（with missing data).
plt.figure(figsize=(15, 8))
for col, color in zip(dataset_1.columns, ['blue', 'green', 'red', 'cyan', 'magenta']):
    plt.scatter(dataset_1.index, dataset_1[col], s=30, color='grey', alpha=0.6, label="dataset_1" if col == dataset_1.columns[0] else "")
plt.title("Original dataset") # dataset_1
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()

plt.savefig('Original_dataset.png', dpi=300)
plt.show()

## Visualize dataset_1_imputed（with missing data imputed).
## 可视化dataset_1_imputed（缺失值已被插补）。

# One plot for dataset_1 data points and missing data points from test_data_missing
plt.figure(figsize=(15, 8))

for col, color in zip(dataset_1.columns, ['blue', 'green', 'red', 'cyan', 'magenta']):
    plt.scatter(dataset_1.index, dataset_1[col], s=30, color='grey', alpha=0.6, label="True Data Points" if col == dataset_1.columns[0] else "")
for col in dataset_1_imputed.columns:
    missing_rows = data_missing[col].isnull().values  # Convert to boolean array
    if missing_rows.sum() > 0:
        plt.scatter(dataset_1_imputed.index[missing_rows], dataset_1_imputed.loc[missing_rows, col], s=50, color='red', alpha=0.7, marker='+', label="Imputed Missing Data Points" if col == dataset_1_imputed.columns[0] else "")
        
plt.title("True Data Points and Imputed Missing Data Points")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
plt.savefig('Original_dataset_w_Imputed.png', dpi=300)
plt.show()

for col in dataset_1_imputed.columns:
    plt.figure(figsize=(15, 8))
    missing_rows = data_missing[col].isnull().values  # Convert to boolean array
    if missing_rows.sum() > 0:
        print(col)
        plt.scatter(dataset_1.index, dataset_1[col], s=30, color='grey', alpha=0.6, label="True Data Points")
        plt.scatter(dataset_1_imputed.index[missing_rows], dataset_1_imputed.loc[missing_rows, col], s=50, color='red', alpha=0.7, marker='+', label="Imputed Missing Data Points")
    
    plt.title(f"True Data Points and Imputed Missing Data Points for {col}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to a file
    filename = f"FOSAed_{col}_1.png"
    plt.savefig(filename, dpi=300)

    plt.show()

FIML_imputed = reverse_encoding(FIML_imputed, encoding_dict)
dataset_1_imputed = reverse_encoding(dataset_1_imputed, encoding_dict)

## Save results as CSV files.
dataset_1_imputed.to_csv('Dataset_FOSAed.csv', index=False)
FIML_imputed.to_csv('Dataset_FIMLed.csv', index=False)

## Visualize non-metrical data cols.

non_numeric_columns = list(dataset_1_imputed.select_dtypes(exclude=[np.number]).columns)

dataset_1_imputed, encoding_dict = encode_columns_with_nan_preservation(dataset_1_imputed)

for col in non_numeric_columns:
    plt.figure(figsize=(15, 8))
    missing_rows = data_missing[col].isnull().values  # Convert to boolean array
    if missing_rows.sum() > 0:
        print(col)
        plt.scatter(dataset_1.index, dataset_1[col], s=30, color='grey', alpha=0.6, label="True Data Points")
        plt.scatter(dataset_1_imputed.index[missing_rows], dataset_1_imputed.loc[missing_rows, col], s=50, color='red', alpha=0.7, marker='+', label="Imputed Missing Data Points")
    
    plt.title(f"True Data Points and Imputed Missing Data Points for {col}")
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.legend()
    plt.tight_layout()
    
    # Save the figure to a file
    filename = f"FOSAed_{col}_2.png"
    plt.savefig(filename, dpi=300)

    plt.show()
