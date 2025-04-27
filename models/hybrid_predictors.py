import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNPredictor(nn.Module):
    """增强版基于CNN的预测器，用于捕获局部模式"""
    
    def __init__(self, input_len, pred_len, d_model, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(CNNPredictor, self).__init__()
        
        # 特征转换层
        self.input_projection = nn.Linear(d_model, d_model*2)
        self.feature_dropout = nn.Dropout(dropout)
        
        # 多层次的卷积块
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model*2, d_model*2, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(d_model*2),
                nn.LeakyReLU(0.1),
                nn.Conv1d(d_model*2, d_model*2, kernel_size=k, padding=k//2),
                nn.BatchNorm1d(d_model*2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ) for k in kernel_sizes
        ])
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Conv1d(d_model*2*len(kernel_sizes), d_model*2, kernel_size=1),
            nn.BatchNorm1d(d_model*2),
            nn.LeakyReLU(0.1)
        )
        
        # 时间维度变换层
        self.temp_projection = nn.Linear(input_len, pred_len)
        
        # 输出维度恢复层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x):
        # x形状: [B, T, C]
        B, T, C = x.shape
        
        # 特征维度增强
        x = self.input_projection(x)  # [B, T, 2C]
        x = self.feature_dropout(x)
        
        # 转换为卷积格式
        x_permuted = x.permute(0, 2, 1)  # [B, 2C, T]
        
        # 应用多尺度卷积块
        conv_outputs = []
        for conv_block in self.conv_blocks:
            conv_out = conv_block(x_permuted)
            conv_outputs.append(conv_out)
        
        # 特征融合 - 连接不同核大小的输出
        x_concat = torch.cat(conv_outputs, dim=1)  # [B, 2C*len(kernels), T]
        x_fused = self.feature_fusion(x_concat)  # [B, 2C, T]
        
        # 对每个特征通道单独进行时间投影
        outputs = []
        d_out = x_fused.size(1)
        for i in range(d_out):
            channel_data = x_fused[:, i, :]  # [B, T]
            channel_output = self.temp_projection(channel_data)  # [B, pred_len]
            outputs.append(channel_output.unsqueeze(2))  # [B, pred_len, 1]
        
        # 合并通道输出
        out = torch.cat(outputs, dim=2)  # [B, pred_len, 2C]
        
        # 恢复原始特征维度
        out = self.output_projection(out)  # [B, pred_len, C]
        
        return out


class LSTMPredictor(nn.Module):
    """增强版基于LSTM的预测器，用于捕获时间依赖性"""
    
    def __init__(self, input_len, pred_len, d_model, num_layers=3, dropout=0.2):
        super(LSTMPredictor, self).__init__()
        
        # 特征转换层
        self.input_projection = nn.Linear(d_model, d_model*2)
        
        # 双向LSTM层
        self.bi_lstm = nn.LSTM(
            input_size=d_model*2,
            hidden_size=d_model*2,
            num_layers=num_layers//2 + 1,
            batch_first=True,
            dropout=dropout if num_layers > 2 else 0,
            bidirectional=True
        )
        
        # 带注意力的LSTM层
        self.attn_lstm = nn.LSTM(
            input_size=d_model*4,  # 来自双向LSTM的输出
            hidden_size=d_model*2,
            num_layers=num_layers//2,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 自注意力层 - 捕获序列内部关系
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model*2, 
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model*2)
        
        # 特征融合层 - 修复维度从 d_model*4 变为 d_model*6
        self.feature_fusion = nn.Sequential(
            nn.Linear(d_model*6, d_model*2),  # 修改这里：从 d_model*4 改为 d_model*6
            nn.LayerNorm(d_model*2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # 时间维度变换层
        self.temp_projection = nn.Linear(input_len, pred_len)
        
        # 输出维度恢复层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x):
        # x形状: [B, T, C]
        B, T, C = x.shape
        
        # 特征维度增强
        x = self.input_projection(x)  # [B, T, 2C]
        
        # 双向LSTM处理
        bi_outputs, _ = self.bi_lstm(x)  # [B, T, 4C]
        
        # 带注意力机制的LSTM
        attn_outputs, _ = self.attn_lstm(bi_outputs)  # [B, T, 2C]
        
        # 自注意力处理
        attn_out, _ = self.self_attention(attn_outputs, attn_outputs, attn_outputs)
        attn_out = self.layer_norm(attn_outputs + attn_out)  # 残差连接 [B, T, 2C]
        
        # 特征融合 - 结合双向LSTM和注意力LSTM
        fusion_input = torch.cat([bi_outputs, attn_out], dim=-1)  # [B, T, 6C]
        fused_features = self.feature_fusion(fusion_input)  # [B, T, 2C]
        
        # 时间维度变换 - 对每个特征通道单独处理
        outputs = []
        d_out = fused_features.size(2)
        for i in range(d_out):
            channel_data = fused_features[:, :, i]  # [B, T]
            channel_output = self.temp_projection(channel_data)  # [B, pred_len]
            outputs.append(channel_output.unsqueeze(2))  # [B, pred_len, 1]
        
        # 合并通道输出
        out = torch.cat(outputs, dim=2)  # [B, pred_len, 2C]
        
        # 恢复原始特征维度
        out = self.output_projection(out)  # [B, pred_len, C]
        
        return out


class EnhancedTransformerEncoder(nn.Module):
    """增强版Transformer编码器块"""
    
    def __init__(self, d_model, nhead=8, dim_feedforward=1024, dropout=0.1):
        super(EnhancedTransformerEncoder, self).__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # 两层前馈网络，带激活和归一化
        self.ff_network = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # 引入额外的MLP层
        self.intermediate = nn.Sequential(
            nn.Linear(d_model, dim_feedforward // 2),
            nn.LayerNorm(dim_feedforward // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward // 2, d_model)
        )
        
        # 层归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout层
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, src):
        # 自注意力处理
        src2, _ = self.self_attn(src, src, src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # 前馈网络处理
        src2 = self.ff_network(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        
        # 额外的MLP处理
        src2 = self.intermediate(src)
        src = src + self.dropout3(src2)
        src = self.norm3(src)
        
        return src


class TransformerPredictor(nn.Module):
    """增强版基于Transformer的预测器，用于捕获全局依赖性"""
    
    def __init__(self, input_len, pred_len, d_model, num_layers=3, dropout=0.1):
        super(TransformerPredictor, self).__init__()
        
        # 特征转换层
        self.input_projection = nn.Linear(d_model, d_model*2)
        
        # 位置编码
        self.pos_encoder = nn.Parameter(torch.zeros(1, input_len, d_model*2))
        nn.init.xavier_uniform_(self.pos_encoder)
        
        # 多层Transformer编码器
        self.transformer_blocks = nn.ModuleList([
            EnhancedTransformerEncoder(
                d_model=d_model*2, 
                nhead=8, 
                dim_feedforward=d_model*4,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # 交叉注意力层 - 关注预测相关部分
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model*2,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # 上下文提取器
        self.context_encoder = nn.Sequential(
            nn.Linear(d_model*2, d_model*4),
            nn.LayerNorm(d_model*4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model*4, d_model*2),
            nn.LayerNorm(d_model*2)
        )
        
        # 时间维度变换层
        self.temp_projection = nn.Linear(input_len, pred_len)
        
        # 输出维度恢复层
        self.output_projection = nn.Sequential(
            nn.Linear(d_model*2, d_model),
            nn.Dropout(dropout/2)
        )
    
    def forward(self, x):
        # x形状: [B, T, C]
        B, T, C = x.shape
        
        # 特征维度增强
        x = self.input_projection(x)  # [B, T, 2C]
        
        # 添加位置编码
        x = x + self.pos_encoder
        
        # Transformer编码器序列
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)  # [B, T, 2C]
        
        # 提取全局上下文
        context = self.context_encoder(x.mean(dim=1, keepdim=True))  # [B, 1, 2C]
        
        # 使用上下文进行交叉注意力
        attn_out, _ = self.cross_attention(context.repeat(1, T, 1), x, x)
        x = x + attn_out  # [B, T, 2C]
        
        # 对每个特征通道单独进行时间投影
        outputs = []
        d_out = x.size(2)
        for i in range(d_out):
            channel_data = x[:, :, i]  # [B, T]
            channel_output = self.temp_projection(channel_data)  # [B, pred_len]
            outputs.append(channel_output.unsqueeze(2))  # [B, pred_len, 1]
        
        # 合并通道输出
        out = torch.cat(outputs, dim=2)  # [B, pred_len, 2C]
        
        # 恢复原始特征维度
        out = self.output_projection(out)  # [B, pred_len, C]
        
        return out

class DynamicAttentionWeighting(nn.Module):
    """
    动态注意力加权模块
    
    为每个时间步和特征维度计算不同的权重，使模型能够根据不同预测器
    在不同情况下的表现动态调整权重。
    """
    
    def __init__(self, pred_len, d_model):
        super(DynamicAttentionWeighting, self).__init__()
        
        # 特征提取器 - 提取每个预测器输出的特征
        self.feature_extractor = nn.Linear(d_model, d_model)
        
        # 权重生成器 - 根据三个预测器的输出生成权重
        self.weight_generator = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 3)
        )
        
    def forward(self, cnn_out, lstm_out, transformer_out):
        # 每个预测器输出的形状: [B, T_pred, C]
        B, T_pred, C = cnn_out.shape
        
        # 提取每个预测器输出的特征
        cnn_features = self.feature_extractor(cnn_out)  # [B, T_pred, C]
        lstm_features = self.feature_extractor(lstm_out)  # [B, T_pred, C]
        transformer_features = self.feature_extractor(transformer_out)  # [B, T_pred, C]
        
        # 将所有特征连接起来用于权重生成
        all_features = torch.cat(
            [cnn_features, lstm_features, transformer_features], 
            dim=-1
        )  # [B, T_pred, 3*C]
        
        # 重塑以便于处理每个位置
        all_features_flat = all_features.reshape(B * T_pred, -1)  # [B*T_pred, 3*C]
        
        # 生成每个位置的权重
        weights_flat = self.weight_generator(all_features_flat)  # [B*T_pred, 3]
        weights = weights_flat.reshape(B, T_pred, 3)  # [B, T_pred, 3]
        weights = F.softmax(weights, dim=-1)  # [B, T_pred, 3]
        
        # 使用动态权重组合预测结果
        cnn_weighted = weights[:, :, 0:1] * cnn_out  # [B, T_pred, C]
        lstm_weighted = weights[:, :, 1:2] * lstm_out  # [B, T_pred, C]
        transformer_weighted = weights[:, :, 2:3] * transformer_out  # [B, T_pred, C]
        
        weighted_sum = cnn_weighted + lstm_weighted + transformer_weighted  # [B, T_pred, C]
        
        # 计算每个模型的整体权重（用于分析）
        model_weights = weights.mean(dim=1)  # [B, 3]
        
        return weighted_sum, model_weights

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