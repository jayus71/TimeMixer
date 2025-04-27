import torch
import numpy as np
import pandas as pd
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.timefeatures import time_features

class NetworkTrafficDataset(Dataset):
    def __init__(self, file_path, seq_len=96, label_len=48, pred_len=24, 
                 data_format=1, scale=True, train_ratio=0.7, valid_ratio=0.1, 
                 flag='train', timeenc=0, freq='h', features='S'):
        """
        网络流量数据加载器
        
        参数:
            file_path: 数据文件路径
            seq_len: 输入序列长度
            label_len: 标签长度（用于TimeMixer）
            pred_len: 预测长度
            data_format: 1表示(时间,城市,值)，2表示(日期,小时,小区名,流量)，3表示(时间戳,流量)
            scale: 是否标准化数据
            train_ratio: 训练集比例
            valid_ratio: 验证集比例
            flag: 'train', 'val' 或 'test'
            timeenc: 时间编码方式, 0:手动提取时间特征, 1:使用timefeatures.py中的时间特征
            freq: 数据频率, 可选: 'h'小时, 'd'天, 'm'月, 'b'工作日, 'w'周, 't'分钟, 's'秒
            features: 特征类型, 'S'单变量预测单变量, 'M'多变量预测多变量, 'MS'多变量预测单变量
        """
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.data_format = data_format
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.features = features
        
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
            
            # 根据features参数确定选择的特征列
            if self.features == 'MS' or self.features == 'M':
                # 使用多变量
                data_cols = [city_col, value_col]
            else:  # 'S'
                # 只使用值变量
                data_cols = [value_col]
            
            df_data = df[data_cols]
            
        elif self.data_format == 2:
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
            
            # 根据features参数确定选择的特征列
            if self.features == 'MS' or self.features == 'M':
                data_cols = [hour_col, cell_col, traffic_col]
            else:  # 'S'
                data_cols = [traffic_col]
                
            df_data = df[data_cols]
            
        elif self.data_format == 3:
            # 第三种数据格式: fiveminstamp,datausage
            time_col = df.columns[0]  # fiveminstamp列
            usage_col = df.columns[1]  # datausage列
            
            # 提取时间特征
            df['date'] = pd.to_datetime(df[time_col])
            
            # 根据features参数确定选择的特征列
            if self.features == 'MS' or self.features == 'M':
                data_cols = [usage_col]
            else:  # 'S'
                data_cols = [usage_col]
                
            df_data = df[data_cols]
        
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
        
        # 提取时间标记 - 使用TimeMixer原始的时间特征方法
        df_stamp = df[['date']][border1:border2]
        
        # 根据timeenc参数选择时间特征提取方式
        if self.timeenc == 0:
            # 手动提取时间特征
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            
            if self.freq == 't':
                # 对于分钟级数据，添加分钟特征
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
                # 对于5分钟级别的数据，将分钟转换为对应的时段 (0, 5, 10, ..., 55)
                df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 5 * 5)
            
            data_stamp = df_stamp.drop(['date'], axis=1).values
        
        elif self.timeenc == 1:
            # 使用timefeatures.py中的时间特征函数
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
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