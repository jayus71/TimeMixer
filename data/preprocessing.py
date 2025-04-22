"""
数据预处理和多尺度生成模块
"""
import numpy as np
import pandas as pd
from scipy import signal
from typing import List, Tuple, Optional, Union


class DataPreprocessor:
    """数据预处理器类，负责数据清洗和多尺度特征生成"""
    
    def __init__(self, fill_missing: bool = True, remove_outliers: bool = True):
        """
        初始化数据预处理器
        
        Args:
            fill_missing (bool): 是否填充缺失值
            remove_outliers (bool): 是否移除异常值
        """
        self.fill_missing = fill_missing
        self.remove_outliers = remove_outliers
    
    def preprocess(self, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        预处理数据
        
        Args:
            data: 输入数据，可以是pandas DataFrame或numpy数组
        
        Returns:
            np.ndarray: 预处理后的数据
        """
        # 转换为DataFrame，方便处理
        if isinstance(data, np.ndarray):
            data = pd.DataFrame(data)
        
        # 填充缺失值
        if self.fill_missing:
            data = self._fill_missing_values(data)
        
        # 移除异常值
        if self.remove_outliers:
            data = self._remove_outliers(data)
        
        return data.values
    
    def _fill_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        填充缺失值
        
        Args:
            data: 输入数据
        
        Returns:
            填充缺失值后的数据
        """
        # 使用前向填充和后向填充组合的方式填充缺失值
        return data.fillna(method='ffill').fillna(method='bfill')
    
    def _remove_outliers(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        移除异常值（使用IQR方法）
        
        Args:
            data: 输入数据
        
        Returns:
            移除异常值后的数据
        """
        for col in data.columns:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # 将异常值替换为边界值
            data.loc[data[col] < lower_bound, col] = lower_bound
            data.loc[data[col] > upper_bound, col] = upper_bound
            
        return data


class MultiScaleGenerator:
    """多尺度特征生成器类，生成不同时间尺度的特征"""
    
    def __init__(self, scales: List[int] = None):
        """
        初始化多尺度特征生成器
        
        Args:
            scales: 时间尺度列表，例如[1, 24, 168]表示原始尺度、日尺度和周尺度
        """
        self.scales = scales if scales is not None else [1, 24, 168]
    
    def generate(self, data: np.ndarray) -> List[np.ndarray]:
        """
        生成多尺度特征
        
        Args:
            data: 输入数据
        
        Returns:
            多尺度特征列表
        """
        multi_scale_data = []
        
        # 添加原始数据
        multi_scale_data.append(data)
        
        # 生成不同尺度的数据
        for scale in self.scales[1:]:
            # 使用均值池化生成低频信息
            scale_data = self._pooling(data, scale)
            
            # 上采样回原始大小以便后续处理
            scale_data_upsampled = self._upsample(scale_data, len(data))
            
            multi_scale_data.append(scale_data_upsampled)
        
        return multi_scale_data
    
    def _pooling(self, data: np.ndarray, scale: int) -> np.ndarray:
        """
        使用均值池化降采样数据
        
        Args:
            data: 输入数据
            scale: 池化尺度
        
        Returns:
            池化后的数据
        """
        # 确保数据长度是scale的倍数
        trim_length = len(data) - (len(data) % scale)
        data_trimmed = data[:trim_length]
        
        # 重塑数据以便进行池化
        reshaped = data_trimmed.reshape(-1, scale, data.shape[1])
        
        # 计算均值
        return np.mean(reshaped, axis=1)
    
    def _upsample(self, data: np.ndarray, target_length: int) -> np.ndarray:
        """
        上采样数据到目标长度
        
        Args:
            data: 输入数据
            target_length: 目标长度
        
        Returns:
            上采样后的数据
        """
        # 使用线性插值进行上采样
        n_features = data.shape[1]
        upsampled_data = np.zeros((target_length, n_features))
        
        for i in range(n_features):
            # 创建原始索引和目标索引
            orig_indices = np.arange(len(data))
            target_indices = np.linspace(0, len(data) - 1, target_length)
            
            # 线性插值
            upsampled_data[:, i] = np.interp(target_indices, orig_indices, data[:, i])
        
        return upsampled_data


class TimeFeatureGenerator:
    """时间特征生成器，生成时间相关的特征（小时、星期几、月份等）"""
    
    def __init__(self, freq: str = 'h'):
        """
        初始化时间特征生成器
        
        Args:
            freq: 时间频率，'h'表示小时，'d'表示天
        """
        self.freq = freq
    
    def generate(self, 
                 dates: Union[pd.DatetimeIndex, List[str]], 
                 include_hour: bool = True,
                 include_weekday: bool = True,
                 include_month: bool = True,
                 include_year: bool = False) -> np.ndarray:
        """
        生成时间特征
        
        Args:
            dates: 日期索引或日期字符串列表
            include_hour: 是否包含小时特征
            include_weekday: 是否包含星期几特征
            include_month: 是否包含月份特征
            include_year: 是否包含年份特征
        
        Returns:
            时间特征数组
        """
        # 确保日期是DatetimeIndex类型
        if not isinstance(dates, pd.DatetimeIndex):
            dates = pd.DatetimeIndex(dates)
        
        # 初始化特征列表
        time_features = []
        
        # 小时特征，正弦和余弦变换
        if include_hour and self.freq == 'h':
            hour_sin = np.sin(2 * np.pi * dates.hour / 24.0)
            hour_cos = np.cos(2 * np.pi * dates.hour / 24.0)
            time_features.extend([hour_sin, hour_cos])
        
        # 星期几特征
        if include_weekday:
            weekday_sin = np.sin(2 * np.pi * dates.weekday / 7.0)
            weekday_cos = np.cos(2 * np.pi * dates.weekday / 7.0)
            time_features.extend([weekday_sin, weekday_cos])
        
        # 月份特征
        if include_month:
            month_sin = np.sin(2 * np.pi * (dates.month - 1) / 12.0)
            month_cos = np.cos(2 * np.pi * (dates.month - 1) / 12.0)
            time_features.extend([month_sin, month_cos])
        
        # 年份特征（归一化）
        if include_year:
            # 获取最小和最大年份进行归一化
            min_year = dates.year.min()
            max_year = dates.year.max()
            if max_year > min_year:  # 避免除以零
                norm_year = (dates.year - min_year) / (max_year - min_year)
                time_features.append(norm_year)
        
        # 将特征列表转换为数组并转置
        return np.vstack(time_features).T