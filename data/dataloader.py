"""
数据加载和批处理模块
"""
import os
import numpy as np
import pandas as pd
import torch
import glob
from datetime import datetime
from torch.utils.data import Dataset, DataLoader as TorchDataLoader

def average_downsample(series: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    将时间序列进行平均下采样
    
    Args:
        series: 输入序列，形状为 [seq_len, features]
        scale_factor: 下采样因子
    
    Returns:
        下采样后的序列
    """
    if scale_factor == 1:
        return series
    
    # 计算下采样后的长度
    result_len = series.shape[0] // scale_factor
    
    # 确保可以被scale_factor整除的部分
    valid_len = result_len * scale_factor
    
    # 重塑数组以便计算平均值
    reshaped = series[:valid_len].reshape(result_len, scale_factor, -1)
    
    # 计算平均值
    result = np.mean(reshaped, axis=1)
    
    return result

class MultiScaleTimeSeriesDataset(Dataset):
    """多尺度时间序列数据集类"""
    
    def __init__(self, data, seq_len, pred_len, scale_factors=[1, 2, 4], scale=True, mean=None, std=None):
        """
        初始化多尺度时间序列数据集
        
        Args:
            data (numpy.ndarray): 输入数据
            seq_len (int): 输入序列长度
            pred_len (int): 预测长度
            scale_factors (list): 下采样因子列表
            scale (bool): 是否标准化数据
            mean (numpy.ndarray, optional): 均值，用于标准化
            std (numpy.ndarray, optional): 标准差，用于标准化
        """
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.scale_factors = scale_factors
        self.scale = scale
        
        # 标准化
        if scale:
            self.mean = mean if mean is not None else np.mean(data, axis=0)
            self.std = std if std is not None else np.std(data, axis=0)
            self.data = (data - self.mean) / (self.std + 1e-8)
        
        # 计算样本数量
        self.samples = len(data) - seq_len - pred_len + 1
    
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        s_begin = idx
        s_end = s_begin + self.seq_len
        r_begin = s_end
        r_end = r_begin + self.pred_len
        
        # 获取原始序列
        seq_x = self.data[s_begin:s_end]
        seq_y = self.data[r_begin:r_end]
        
        # 生成多尺度输入
        multi_scale_x = []
        for sf in self.scale_factors:
            if sf == 1:
                # 原始尺度
                downsampled = seq_x
            else:
                # 平均下采样
                downsampled = average_downsample(seq_x, sf)
            
            multi_scale_x.append(torch.FloatTensor(downsampled))
        
        return multi_scale_x, torch.FloatTensor(seq_y)
    
    def inverse_transform(self, data):
        """逆标准化数据"""
        if self.scale:
            return data * (self.std + 1e-8) + self.mean
        return data

class DataLoader:
    """数据加载器类"""
    
    def __init__(self, data_path, batch_size=64, seq_len=96, pred_len=24, 
                 train_ratio=0.7, valid_ratio=0.1, scale=True, num_workers=4,
                 target_col='Value', time_col='Time', scale_factors=None, dataset_type=None):
        """
        初始化数据加载器
        
        Args:
            data_path (str): 数据文件路径或数据目录
            batch_size (int): 批大小
            seq_len (int): 输入序列长度
            pred_len (int): 预测长度
            train_ratio (float): 训练数据比例
            valid_ratio (float): 验证数据比例
            scale (bool): 是否标准化数据
            num_workers (int): 加载数据的worker数量
            target_col (str): 目标列名称，默认为'Value'
            time_col (str): 时间列名称，默认为'Time'
            dataset_type (str): 数据集类型，可选值为'aiia_hour', 'china_unicom', 'lte_network'，如果为None则自动检测
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.scale = scale
        self.num_workers = num_workers
        self.target_col = target_col
        self.time_col = time_col
        self.dataset_type = dataset_type
        self.scale_factors = scale_factors
        
        self.data = self._load_data()
        self._prepare_datasets()
    
    def _load_data(self):
        """加载数据文件"""
        # 检查是否为目录
        if os.path.isdir(self.data_path):
            # 自动检测数据集类型
            if self.dataset_type is None:
                if os.path.basename(self.data_path) == 'AIIA_hour':
                    self.dataset_type = 'aiia_hour'
                elif os.path.basename(self.data_path) == 'China_Unicom_One_Cell_Data':
                    self.dataset_type = 'china_unicom'
                elif os.path.basename(self.data_path) == 'lte_network':
                    self.dataset_type = 'lte_network'
                else:
                    raise ValueError(f"无法自动检测数据集类型: {self.data_path}")
            
            # 根据数据集类型加载数据
            return self._load_dataset_by_type()
        else:
            # 单个文件加载
            return self._load_single_file(self.data_path)
    
    def _load_dataset_by_type(self):
        """根据数据集类型加载数据"""
        if self.dataset_type == 'aiia_hour':
            # AIIA小时级数据集
            return self._load_aiia_dataset()
        elif self.dataset_type == 'china_unicom':
            # 中国联通单小区数据集
            return self._load_china_unicom_dataset()
        elif self.dataset_type == 'lte_network':
            # LTE网络数据集
            return self._load_lte_network_dataset()
        else:
            raise ValueError(f"不支持的数据集类型: {self.dataset_type}")
    
    def _load_aiia_dataset(self):
        """加载AIIA小时级数据集"""
        print(f"正在加载AIIA小时级数据集: {self.data_path}")
        # 获取所有CSV文件
        data_files = glob.glob(os.path.join(self.data_path, "*.csv"))
        
        all_data = []
        for file_path in data_files:
            df = pd.read_csv(file_path)
            # 确保有必要的列
            if self.time_col in df.columns and self.target_col in df.columns:
                # 转换时间列
                df[self.time_col] = pd.to_datetime(df[self.time_col])
                # 确保数据按时间排序
                df = df.sort_values(by=self.time_col)
                # 提取目标列
                values = df[self.target_col].values.reshape(-1, 1)
                all_data.append(values)
        
        # 合并所有数据，沿特征维度拼接
        if all_data:
            combined_data = np.hstack(all_data)
            print(f"AIIA数据集加载完成，形状: {combined_data.shape}")
            return combined_data
        else:
            raise ValueError(f"在{self.data_path}中未找到有效的数据文件")
    
    def _load_china_unicom_dataset(self):
        """加载中国联通单小区数据集"""
        print(f"正在加载中国联通数据集: {self.data_path}")
        file_path = os.path.join(self.data_path, "traffic_one_cell.csv")
        
        df = pd.read_csv(file_path)
        # 提取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        values = df[numeric_cols].values
        
        print(f"中国联通数据集加载完成，形状: {values.shape}")
        return values
    
    def _load_lte_network_dataset(self):
        """加载LTE网络数据集"""
        print(f"正在加载LTE网络数据集: {self.data_path}")
        # 合并训练和测试数据
        train_file = os.path.join(self.data_path, "train.csv")
        test_file = os.path.join(self.data_path, "test.csv")
        
        train_df = pd.read_csv(train_file)
        test_df = pd.read_csv(test_file)
        
        # 合并数据并按时间排序（如果有时间列）
        if self.time_col in train_df.columns:
            train_df[self.time_col] = pd.to_datetime(train_df[self.time_col])
            test_df[self.time_col] = pd.to_datetime(test_df[self.time_col])
            df = pd.concat([train_df, test_df]).sort_values(by=self.time_col)
        else:
            df = pd.concat([train_df, test_df])
        
        # 提取数值列
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        values = df[numeric_cols].values
        
        print(f"LTE网络数据集加载完成，形状: {values.shape}")
        return values
    
    def _load_single_file(self, file_path):
        """加载单个数据文件"""
        print(f"正在加载单个文件: {file_path}")
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            
            # 如果有时间列，确保按时间排序
            if self.time_col in df.columns:
                df[self.time_col] = pd.to_datetime(df[self.time_col])
                df = df.sort_values(by=self.time_col)
            
            # 如果有目标列，只提取目标列
            if self.target_col in df.columns:
                values = df[self.target_col].values.reshape(-1, 1)
            else:
                # 否则提取所有数值列
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                values = df[numeric_cols].values
            
        elif file_path.endswith('.npy'):
            values = np.load(file_path)
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
        print(f"单个文件加载完成，形状: {values.shape}")
        return values
    
    def _prepare_datasets(self):
        """准备训练、验证和测试数据集"""
        data_len = len(self.data)
        train_end = int(data_len * self.train_ratio)
        valid_end = int(data_len * (self.train_ratio + self.valid_ratio))
        
        train_data = self.data[:train_end]
        valid_data = self.data[train_end:valid_end]
        test_data = self.data[valid_end:]
        
        print(f"数据集划分: 训练集 {train_data.shape}, 验证集 {valid_data.shape}, 测试集 {test_data.shape}")
        
        # 计算训练集的均值和标准差用于标准化
        if self.scale:
            self.mean = np.mean(train_data, axis=0)
            self.std = np.std(train_data, axis=0)
            print(f"数据标准化: 均值 {self.mean.shape}, 标准差 {self.std.shape}")
        else:
            self.mean = None
            self.std = None
        
        # 创建数据集
        scale_factors = [1, 2, 4]  # 默认尺度因子
        
        self.train_dataset = MultiScaleTimeSeriesDataset(
            train_data, self.seq_len, self.pred_len, 
            scale_factors, self.scale, self.mean, self.std)
        
        self.valid_dataset = MultiScaleTimeSeriesDataset(
            valid_data, self.seq_len, self.pred_len, 
            scale_factors, self.scale, self.mean, self.std)
        
        self.test_dataset = MultiScaleTimeSeriesDataset(
            test_data, self.seq_len, self.pred_len, 
            scale_factors, self.scale, self.mean, self.std)
    
    def get_train_loader(self, shuffle=True):
        """获取训练数据加载器"""
        return TorchDataLoader(
            self.train_dataset, batch_size=self.batch_size, 
            shuffle=shuffle, num_workers=self.num_workers)
    
    def get_valid_loader(self):
        """获取验证数据加载器"""
        return TorchDataLoader(
            self.valid_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)
    
    def get_test_loader(self):
        """获取测试数据加载器"""
        return TorchDataLoader(
            self.test_dataset, batch_size=self.batch_size, 
            shuffle=False, num_workers=self.num_workers)
    
    def inverse_transform(self, data):
        """逆标准化数据"""
        if self.scale:
            return data * (self.std + 1e-8) + self.mean
        return data


class MultiDatasetLoader:
    """多数据集加载器，能够加载和处理所有可用的数据集"""
    
    def __init__(self, data_root='./data', batch_size=64, seq_len=96, pred_len=24, 
                 train_ratio=0.7, valid_ratio=0.1, scale=True, num_workers=4):
        """
        初始化多数据集加载器
        
        Args:
            data_root (str): 数据根目录路径
            batch_size (int): 批大小
            seq_len (int): 输入序列长度
            pred_len (int): 预测长度
            train_ratio (float): 训练数据比例
            valid_ratio (float): 验证数据比例
            scale (bool): 是否标准化数据
            num_workers (int): 加载数据的worker数量
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.train_ratio = train_ratio
        self.valid_ratio = valid_ratio
        self.scale = scale
        self.num_workers = num_workers
        
        # 查找所有数据集目录
        self.dataset_dirs = self._find_dataset_dirs()
        
        # 存储各数据集的加载器
        self.dataset_loaders = {}
    
    def _find_dataset_dirs(self):
        """查找数据根目录下的所有数据集目录"""
        dataset_dirs = []
        
        # 已知的数据集目录
        known_datasets = ['AIIA_hour', 'China_Unicom_One_Cell_Data', 'lte_network']
        
        for dataset in known_datasets:
            dataset_path = os.path.join(self.data_root, dataset)
            if os.path.isdir(dataset_path):
                dataset_dirs.append(dataset_path)
        
        return dataset_dirs
    
    def load_all_datasets(self):
        """加载所有可用的数据集"""
        for dataset_dir in self.dataset_dirs:
            dataset_name = os.path.basename(dataset_dir)
            print(f"加载数据集: {dataset_name}")
            
            try:
                loader = DataLoader(
                    data_path=dataset_dir,
                    batch_size=self.batch_size,
                    seq_len=self.seq_len,
                    pred_len=self.pred_len,
                    train_ratio=self.train_ratio,
                    valid_ratio=self.valid_ratio,
                    scale=self.scale,
                    num_workers=self.num_workers,
                    dataset_type=dataset_name.lower().replace('_', '_')
                )
                
                self.dataset_loaders[dataset_name] = loader
                print(f"数据集 {dataset_name} 加载成功")
            
            except Exception as e:
                print(f"加载数据集 {dataset_name} 失败: {e}")
        
        return self.dataset_loaders
    
    def get_dataset_loader(self, dataset_name):
        """获取指定数据集的加载器"""
        if dataset_name in self.dataset_loaders:
            return self.dataset_loaders[dataset_name]
        else:
            # 尝试加载数据集
            dataset_path = os.path.join(self.data_root, dataset_name)
            if os.path.isdir(dataset_path):
                try:
                    loader = DataLoader(
                        data_path=dataset_path,
                        batch_size=self.batch_size,
                        seq_len=self.seq_len,
                        pred_len=self.pred_len,
                        train_ratio=self.train_ratio,
                        valid_ratio=self.valid_ratio,
                        scale=self.scale,
                        num_workers=self.num_workers
                    )
                    
                    self.dataset_loaders[dataset_name] = loader
                    return loader
                
                except Exception as e:
                    raise ValueError(f"加载数据集 {dataset_name} 失败: {e}")
            else:
                raise ValueError(f"数据集 {dataset_name} 不存在")