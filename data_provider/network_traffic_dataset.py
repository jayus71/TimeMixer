import torch
import numpy as np
import pandas as pd
import logging
import sys
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from utils.timefeatures import time_features
from scipy import fft

def add_time_series_features(df, time_col, value_col):
    """添加时间序列相关的衍生特征"""
    # 确保时间列是datetime类型
    df['date'] = pd.to_datetime(df[time_col])
    
    # === 高级时间特征 ===
    # 工作日/周末标记
    df['is_weekend'] = df['date'].dt.dayofweek >= 5
    
    # 一天中的时段 (0-23点划分为6个时段)
    df['time_period'] = df['date'].dt.hour // 4
    
    # 每周的相对时刻 (总计168小时)
    df['hour_of_week'] = df['date'].dt.dayofweek * 24 + df['date'].dt.hour
    
    # === 流量统计特征 ===
    # 添加滞后特征
    for lag in [1, 2, 3, 6, 12, 24]:  # 根据数据频率调整
        df[f'lag_{lag}'] = df[value_col].shift(lag)
    
    # 移动平均
    for window in [3, 6, 12, 24]:
        df[f'rolling_mean_{window}'] = df[value_col].rolling(window=window).mean()
    
    # 移动标准差 (捕捉波动性)
    for window in [6, 24]:
        df[f'rolling_std_{window}'] = df[value_col].rolling(window=window).std()
    
    # 同比特征 (例如：前一天同一时刻)
    if len(df) > 24:  
        df['same_hour_yesterday'] = df[value_col].shift(24)
    
    # 同比特征 (例如：上周同一时刻)
    if len(df) > 168:  
        df['same_hour_lastweek'] = df[value_col].shift(168)
    
    # === 变化率特征 ===
    # 与前一个时间点相比的变化率
    df['diff_1'] = df[value_col].diff(1)
    
    # 同比变化率
    if len(df) > 24:
        df['diff_day'] = df[value_col] - df[value_col].shift(24)
    
    # 填充NaN值
    df = df.fillna(method='bfill').fillna(method='ffill')
    
    return df

def add_frequency_domain_features(df, value_col, seq_len=24):
    """添加频域特征来捕捉周期性信息"""
    from scipy import fft
    
    # 计算整个序列的主要频率成分
    signal = df[value_col].values
    
    # 对每个固定长度窗口计算频域特征
    freq_features = []
    
    # 使用滑动窗口
    for i in range(0, max(1, len(signal) - seq_len + 1), max(1, seq_len//2)):  
        end_idx = min(i + seq_len, len(signal))
        window = signal[i:end_idx]
        
        if len(window) < 2:  # 确保窗口至少有2个点
            freq_features.append(np.zeros(5))
            continue
            
        # 快速傅里叶变换
        fft_values = fft.rfft(window)
        fft_magnitudes = np.abs(fft_values)
        
        # 提取前5个主要频率分量
        top_k = min(5, len(fft_magnitudes))
        dominant_freqs = np.argsort(fft_magnitudes)[-top_k:]
        
        # 创建频域特征
        freq_feat = np.zeros(5)  # 固定大小为5
        for j in range(top_k):
            if j < len(dominant_freqs):
                idx = dominant_freqs[j]
                if idx < len(fft_magnitudes):
                    freq_feat[j] = fft_magnitudes[idx]
        
        freq_features.append(freq_feat)
    
    # 确保有足够的特征行
    if len(freq_features) < len(df):
        # 填充到与原数据相同长度
        last_feat = freq_features[-1] if freq_features else np.zeros(5)
        while len(freq_features) < len(df):
            freq_features.append(last_feat)
    
    # 如果特征过多，则截断
    freq_features = freq_features[:len(df)]
    
    # 转换为数组
    freq_features = np.array(freq_features)
    
    # 添加到原始DataFrame
    for i in range(min(5, freq_features.shape[1])):
        df[f'fft_feature_{i}'] = freq_features[:, i]
    
    return df

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
            
            # 添加特征工程
            df = add_time_series_features(df, time_col, value_col)
            df = add_frequency_domain_features(df, value_col)
            
            # 根据features参数确定选择的特征列
            if self.features == 'MS' or self.features == 'M':
                # 使用多变量 - 选择所有特征列，排除时间列和原始日期列
                # 确保value_col在最后一列
                other_cols = [col for col in df.columns if col not in [time_col, 'date', value_col]]
                data_cols = other_cols + [value_col]  # 把流量列放在最后
            else:  # 'S'
                # 只使用衍生特征 + 值变量（放在最后）
                feature_cols = [col for col in df.columns if col.startswith(('lag_', 'rolling_', 'diff_', 'fft_'))]
                data_cols = feature_cols + [value_col]  # 把流量列放在最后
            
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
            
            # 添加特征工程
            df = add_time_series_features(df, date_col, traffic_col)
            df = add_frequency_domain_features(df, traffic_col)
            
            # 根据features参数确定选择的特征列
            if self.features == 'MS' or self.features == 'M':
                # 使用包括小区信息的多变量，但确保流量列在最后
                feature_cols = [hour_col, cell_col] + [
                    col for col in df.columns if col.startswith(('lag_', 'rolling_', 'diff_', 'fft_', 'is_', 'time_', 'hour_')) 
                    and col != traffic_col
                ]
                data_cols = feature_cols + [traffic_col]  # 把流量列放在最后
            else:  # 'S'
                # 只使用衍生特征和流量值（放在最后）
                feature_cols = [
                    col for col in df.columns if col.startswith(('lag_', 'rolling_', 'diff_', 'fft_'))
                    and col != traffic_col
                ]
                data_cols = feature_cols + [traffic_col]  # 把流量列放在最后
                
            df_data = df[data_cols]
            
        elif self.data_format == 3:
            # 第三种数据格式: fiveminstamp,datausage
            time_col = df.columns[0]  # 时间戳列
            usage_col = df.columns[1]  # 流量使用量列
            
            # 提取时间特征
            df['date'] = pd.to_datetime(df[time_col])
            
            # 添加特征工程
            df = add_time_series_features(df, time_col, usage_col)
            df = add_frequency_domain_features(df, usage_col)
            
            # 根据features参数确定选择的特征列
            if self.features == 'MS' or self.features == 'M':
                # 使用多变量，确保流量列在最后
                feature_cols = [
                    col for col in df.columns if col.startswith(('lag_', 'rolling_', 'diff_', 'fft_', 'is_', 'time_', 'hour_'))
                    and col != usage_col
                ]
                data_cols = feature_cols + [usage_col]  # 把流量列放在最后
            else:  # 'S'
                # 只使用衍生特征和流量值（放在最后）
                feature_cols = [
                    col for col in df.columns if col.startswith(('lag_', 'rolling_', 'diff_', 'fft_'))
                    and col != usage_col
                ]
                data_cols = feature_cols + [usage_col]  # 把流量列放在最后
                
            df_data = df[data_cols]
        
        # 保存流量列的索引 - 应该始终是最后一列
        self.traffic_col_idx = len(df_data.columns) - 1
        logging.info(f"流量列索引: {self.traffic_col_idx}, 总特征数: {len(df_data.columns)}")
        # logging.info(f"前五行数据:\n{df_data.head()}")
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
            
            # 保存流量列的均值和标准差，用于单独反标准化
            self.traffic_mean = self.scaler.mean_[self.traffic_col_idx]
            self.traffic_std = self.scaler.scale_[self.traffic_col_idx]
            logging.info(f"流量列均值: {self.traffic_mean}, 标准差: {self.traffic_std}")
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
            
            # 修复drop方法，使用columns参数
            data_stamp = df_stamp.drop(columns=['date']).values
        
        elif self.timeenc == 1:
            # 使用timefeatures.py中的时间特征函数
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)
        
        # 存储处理后的数据
        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        
        # 输出数据形状信息
        logging.info(f"加载数据集完成. X形状: {self.data_x.shape}, 时间标记形状: {self.data_stamp.shape}")
    
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
        只对流量列（最后一列）进行反标准化
        
        参数:
            data: 形状为[batch_size, pred_len, features]或[pred_len, features]的数组
            
        返回:
            反标准化后的数据，保持原始形状
        """
        if not self.scale:
            return data
            
        # 保存原始形状信息
        original_shape = data.shape
        
        # 只选择流量列（始终是最后一个特征）进行反标准化
        if len(original_shape) == 3:
            # 3D数据: [batch_size, pred_len, features]
            # 只选择最后一个特征列（流量列）
            traffic_data = data[:, :, -1:]
            
            # 手动反标准化
            inverse_traffic = traffic_data * self.traffic_std + self.traffic_mean
            
            # 创建一个新数组，用原始数据替换，但流量列使用反标准化的值
            inverse_data = data.copy()
            inverse_data[:, :, -1:] = inverse_traffic
            
        elif len(original_shape) == 2:
            # 2D数据: [samples, features]
            # 只选择最后一个特征列（流量列）
            traffic_data = data[:, -1:]
            
            # 手动反标准化
            inverse_traffic = traffic_data * self.traffic_std + self.traffic_mean
            
            # 创建一个新数组，用原始数据替换，但流量列使用反标准化的值
            inverse_data = data.copy()
            inverse_data[:, -1:] = inverse_traffic
        else:
            raise ValueError(f"不支持的数据维度: {len(original_shape)}")
        
        return inverse_data