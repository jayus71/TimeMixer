import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPredictor(nn.Module):
    """基于CNN的预测器，用于捕获局部模式"""
    
    def __init__(self, input_len, pred_len, d_model, kernel_sizes=[3, 5, 7]):
        super(CNNPredictor, self).__init__()
        
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=k, padding=k//2),
                nn.ReLU()
            ) for k in kernel_sizes
        ])
        
        self.projection = nn.Linear(input_len, pred_len)
    
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        
        # 转换为 [B, C, T] 用于1D卷积
        x_permuted = x.permute(0, 2, 1)  # [B, C, T]
        
        # 应用多个卷积块，具有不同内核大小
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_out = conv_block(x_permuted)
            conv_outputs.append(conv_out)
        
        # 合并来自不同卷积块的输出
        x_conv = torch.stack(conv_outputs).mean(dim=0)  # [B, C, T]
        x_permuted = x_permuted + x_conv  # 残差连接
        
        # 关键修改：正确处理维度变换
        # 对每个特征通道单独进行投影
        outputs = []
        for i in range(C):
            channel_data = x_permuted[:, i, :]  # [B, T]
            channel_output = self.projection(channel_data)  # [B, pred_len]
            outputs.append(channel_output.unsqueeze(2))  # [B, pred_len, 1]
        
        # 合并所有通道的结果
        output = torch.cat(outputs, dim=2)  # [B, pred_len, C]
        
        return output


class LSTMPredictor(nn.Module):
    """基于LSTM的预测器，用于捕获时间依赖性"""
    
    def __init__(self, input_len, pred_len, d_model, num_layers=2):
        super(LSTMPredictor, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.projection = nn.Linear(input_len, pred_len)
    
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        
        # LSTM处理序列
        outputs, _ = self.lstm(x)  # [B, T, C]
        
        # 对每个特征通道单独进行投影
        pred_outputs = []
        for i in range(C):
            channel_data = outputs[:, :, i]  # [B, T]
            channel_output = self.projection(channel_data)  # [B, pred_len]
            pred_outputs.append(channel_output.unsqueeze(2))  # [B, pred_len, 1]
        
        # 合并所有通道的结果
        pred = torch.cat(pred_outputs, dim=2)  # [B, pred_len, C]
        
        return pred


class SimpleTransformerEncoder(nn.Module):
    """简化的Transformer编码器块"""
    
    def __init__(self, d_model, nhead=8, dim_feedforward=512, dropout=0.1):
        super(SimpleTransformerEncoder, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src):
        # 自注意力模块
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络模块
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        return src


class TransformerPredictor(nn.Module):
    """基于Transformer的预测器，用于捕获全局依赖性"""
    
    def __init__(self, input_len, pred_len, d_model, num_layers=2):
        super(TransformerPredictor, self).__init__()
        
        self.transformer_blocks = nn.ModuleList([
            SimpleTransformerEncoder(d_model=d_model)
            for _ in range(num_layers)
        ])
        
        self.projection = nn.Linear(input_len, pred_len)
    
    def forward(self, x):
        # x shape: [B, T, C]
        B, T, C = x.shape
        
        # Transformer处理序列
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # [B, T, C]
        
        # 对每个特征通道单独进行投影
        pred_outputs = []
        for i in range(C):
            channel_data = x[:, :, i]  # [B, T]
            channel_output = self.projection(channel_data)  # [B, pred_len]
            pred_outputs.append(channel_output.unsqueeze(2))  # [B, pred_len, 1]
        
        # 合并所有通道的结果
        pred = torch.cat(pred_outputs, dim=2)  # [B, pred_len, C]
        
        return pred


class HybridFuturePredictor(nn.Module):
    """混合未来预测器，结合CNN、LSTM和Transformer"""
    
    def __init__(self, input_len, pred_len, d_model, d_output):
        super(HybridFuturePredictor, self).__init__()
        
        # 创建不同类型的预测器
        self.cnn_predictor = CNNPredictor(input_len, pred_len, d_model)
        self.lstm_predictor = LSTMPredictor(input_len, pred_len, d_model)
        self.transformer_predictor = TransformerPredictor(input_len, pred_len, d_model)
        
        # 用于集成的可学习权重
        self.ensemble_weights = nn.Parameter(torch.ones(3) / 3.0)
        
        # 输出投影（如果需要）
        if d_model != d_output:
            self.output_projection = nn.Linear(d_model, d_output)
    
    def forward(self, x):
        # 获取每种模型类型的预测
        cnn_out = self.cnn_predictor(x)
        lstm_out = self.lstm_predictor(x)
        transformer_out = self.transformer_predictor(x)
        
        # 对集成权重应用softmax
        weights = F.softmax(self.ensemble_weights, dim=0)
        
        # 加权组合
        combined = (
            weights[0] * cnn_out + 
            weights[1] * lstm_out + 
            weights[2] * transformer_out
        )
        
        # 如果需要，进行最终投影
        if hasattr(self, 'output_projection'):
            combined = self.output_projection(combined)
            
        return combined