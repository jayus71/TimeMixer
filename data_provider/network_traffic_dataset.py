import torch
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder

class NetworkTrafficDataset(Dataset):
    def __init__(self, file_path, seq_len=96, label_len=48, pred_len=24, 
                 data_format=1, scale=True, train_ratio=0.7, valid_ratio=0.1, flag='train'):
        """
        网络流量数据加载器
        
        参数:
            file_path: 数据文件路径
            seq_len: 输入序列长度
            label_len: 标签长度（用于TimeMixer）
            pred_len: 预测长度
            data_format: 1表示第一种数据格式(时间,城市,值)，2表示第二种数据格式(日期,小时,小区名,流量)
            scale: 是否标准化数据
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            flag: 'train', 'val' 或 'test'
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_format = data_format
        self.scale = scale
        
        # 设置数据集类型
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]
        
        # 读取并处理数据
        self.__read_data__(file_path, train_ratio, valid_ratio)
        
    def __read_data__(self, file_path, train_ratio, valid_ratio):
        # 读取数据
        df = pd.read_csv(file_path)
        
        # 根据数据格式进行不同的处理
        if self.data_format == 1:
            # 第一种数据格式: 时间,城市,值
            time_col = df.columns[1]  # 时间列
            city_col = df.columns[2]  # 城市列
            value_col = df.columns[3]  # 值列
            
            # 转换城市为数值编码
            if df[city_col].dtype == 'object':
                encoder = LabelEncoder()
                df[city_col] = encoder.fit_transform(df[city_col])
            
            # 提取时间特征
            df['date'] = pd.to_datetime(df[time_col])
            df['hour'] = df['date'].dt.hour
            df['dayofweek'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            # 选择特征列
            data_cols = [city_col, value_col, 'hour', 'dayofweek', 'month', 'day']
            df_data = df[data_cols]
            timestamp_cols = ['hour', 'dayofweek', 'month', 'day']
            # logging.info(f"first 5 rows of df_data: {df_data.head()}")
        else:
            # 第二种数据格式: 日期,小时,小区名,流量
            date_col = df.columns[0]  # 日期列
            hour_col = df.columns[1]  # 小时列
            cell_col = df.columns[2]  # 小区名列
            traffic_col = df.columns[3]  # 流量列
            
            # 转换小区名为数值编码
            if df[cell_col].dtype == 'object':
                encoder = LabelEncoder()
                df[cell_col] = encoder.fit_transform(df[cell_col])
            
            # 提取时间特征
            df['date'] = pd.to_datetime(df[date_col])
            df['dayofweek'] = df['date'].dt.dayofweek
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            # 选择特征列
            data_cols = [hour_col, cell_col, traffic_col, 'dayofweek', 'month', 'day']
            df_data = df[data_cols]
            timestamp_cols = [hour_col, 'dayofweek', 'month', 'day']
        
        # 分割数据集
        num_samples = len(df_data)
        train_end = int(num_samples * train_ratio)
        val_end = int(num_samples * (train_ratio + valid_ratio))
        
        border1s = [0, train_end - self.seq_len, val_end - self.seq_len]
        border2s = [train_end, val_end, num_samples]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        # 标准化数据
        if self.scale:
            self.scaler = StandardScaler()
            train_data = df_data[0:train_end]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            

        df_stamp = df[timestamp_cols][border1:border2]
        data_stamp = df_stamp.values
        
        # 存储处理后的数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
    
    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1
        
    def inverse_transform(self, data):
        """
        反标准化数据
        
        参数:
            data: 形状为[batch_size, pred_len, features]或[pred_len, features]的数组
            
        返回:
            反标准化后的数据，保持原始形状
        """
        if not self.scale:
            return data
            
        # 保存原始形状信息
        original_shape = data.shape
        
        # 将3D数组重塑为2D：[batch_size * pred_len, features]
        if len(original_shape) == 3:
            batch_size, pred_len, features = original_shape
            reshaped_data = data.reshape(-1, features)
        else:
            # 如果已经是2D，则直接使用
            reshaped_data = data
        
        # 应用反标准化
        inverse_data = self.scaler.inverse_transform(reshaped_data)
        
        # 重塑回原始维度
        if len(original_shape) == 3:
            inverse_data = inverse_data.reshape(original_shape)
        
        return inverse_data