import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hybrid_predictors import HybridFuturePredictor

# 从原TimeMixer代码库导入
from layers.Autoformer_EncDec import series_decomp
from layers.Embed import DataEmbedding_wo_pos
from layers.StandardNorm import Normalize
from models.TimeMixer import PastDecomposableMixing, DFT_series_decomp


class EnhancedTimeMixer(nn.Module):
    """增强版TimeMixer模型，使用混合预测器进行未来预测"""
    
    def __init__(self, configs):
        super(EnhancedTimeMixer, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.down_sampling_window = configs.down_sampling_window
        self.channel_independence = configs.channel_independence
        
        # ======= 保留原TimeMixer的PDM模块 =======
        self.pdm_blocks = nn.ModuleList([PastDecomposableMixing(configs)
                                         for _ in range(configs.e_layers)])

        # 选择分解方法
        if configs.decomp_method == 'moving_avg':
            self.preprocess = series_decomp(configs.moving_avg)
        elif configs.decomp_method == "dft_decomp":
            self.preprocess = DFT_series_decomp(configs.top_k)
        else:
            raise ValueError('分解方法错误')

        self.enc_in = configs.enc_in
        self.use_future_temporal_feature = configs.use_future_temporal_feature

        # 设置嵌入
        if self.channel_independence == 1:
            self.enc_embedding = DataEmbedding_wo_pos(1, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)
        else:
            self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                      configs.dropout)

        self.layer = configs.e_layers

        # 标准化层
        self.normalize_layers = nn.ModuleList(
            [
                Normalize(self.configs.enc_in, affine=True, non_norm=True if configs.use_norm == 0 else False)
                for i in range(configs.down_sampling_layers + 1)
            ]
        )

        # ======= 增强版的未来多预测器混合 =======
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            # 为每个尺度创建混合预测器
            self.hybrid_predictors = nn.ModuleList(
                [
                    HybridFuturePredictor(
                        input_len=configs.seq_len // (configs.down_sampling_window ** i),
                        pred_len=configs.pred_len,
                        d_model=configs.d_model,
                        d_output=configs.d_model
                    )
                    for i in range(configs.down_sampling_layers + 1)
                ]
            )
            
            # 尺度重要性权重（可学习）
            self.scale_weights = nn.Parameter(
                torch.ones(configs.down_sampling_layers + 1) / (configs.down_sampling_layers + 1)
            )
            
            # 输出投影
            if self.channel_independence == 1:
                self.projection_layer = nn.Linear(
                    configs.d_model, 1, bias=True)
            else:
                self.projection_layer = nn.Linear(
                    configs.d_model, configs.c_out, bias=True)
                
                self.out_res_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.seq_len // (configs.down_sampling_window ** i),
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

                self.regression_layers = nn.ModuleList(
                    [
                        nn.Linear(
                            configs.seq_len // (configs.down_sampling_window ** i),
                            configs.pred_len,
                        )
                        for i in range(configs.down_sampling_layers + 1)
                    ]
                )

    def out_projection(self, dec_out, i, out_res):
        """投影输出并应用残差连接"""
        dec_out = self.projection_layer(dec_out)
        out_res = out_res.permute(0, 2, 1)
        out_res = self.out_res_layers[i](out_res)
        out_res = self.regression_layers[i](out_res).permute(0, 2, 1)
        dec_out = dec_out + out_res
        return dec_out

    def pre_enc(self, x_list):
        """预处理编码，与原TimeMixer相同"""
        if self.channel_independence == 1:
            return (x_list, None)
        else:
            out1_list = []
            out2_list = []
            for x in x_list:
                x_1, x_2 = self.preprocess(x)
                out1_list.append(x_1)
                out2_list.append(x_2)
            return (out1_list, out2_list)

    def __multi_scale_process_inputs(self, x_enc, x_mark_enc):
        """多尺度处理输入，与原TimeMixer相同"""
        # 选择下采样方法
        if self.configs.down_sampling_method == 'max':
            down_pool = nn.MaxPool1d(self.configs.down_sampling_window, return_indices=False)
        elif self.configs.down_sampling_method == 'avg':
            down_pool = nn.AvgPool1d(self.configs.down_sampling_window)
        elif self.configs.down_sampling_method == 'conv':
            padding = 1 if torch.__version__ >= '1.5.0' else 2
            down_pool = nn.Conv1d(in_channels=self.configs.enc_in, out_channels=self.configs.enc_in,
                                  kernel_size=3, padding=padding,
                                  stride=self.configs.down_sampling_window,
                                  padding_mode='circular',
                                  bias=False)
        else:
            return x_enc, x_mark_enc
            
        # B,T,C -> B,C,T
        x_enc = x_enc.permute(0, 2, 1)

        x_enc_ori = x_enc
        x_mark_enc_mark_ori = x_mark_enc

        x_enc_sampling_list = []
        x_mark_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        x_mark_sampling_list.append(x_mark_enc)

        for i in range(self.configs.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)

            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling

            if x_mark_enc_mark_ori is not None:
                x_mark_sampling_list.append(x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :])
                x_mark_enc_mark_ori = x_mark_enc_mark_ori[:, ::self.configs.down_sampling_window, :]

        x_enc = x_enc_sampling_list
        if x_mark_enc_mark_ori is not None:
            x_mark_enc = x_mark_sampling_list
        else:
            x_mark_enc = x_mark_enc

        return x_enc, x_mark_enc

    def enhanced_future_mixing(self, B, enc_out_list, x_list):
        """增强版未来多预测器混合"""
        dec_out_list = []
        
        if self.channel_independence == 1:
            x_list = x_list[0]
            for i, enc_out in enumerate(enc_out_list):
                # 使用混合预测器
                dec_out = self.hybrid_predictors[i](enc_out)
                
                # 应用未来时间特征（如果需要）
                if self.use_future_temporal_feature:
                    dec_out = dec_out + self.x_mark_dec
                
                # 最终投影
                dec_out = self.projection_layer(dec_out)
                dec_out = dec_out.reshape(B, self.configs.c_out, self.pred_len).permute(0, 2, 1).contiguous()
                dec_out_list.append(dec_out)

        else:
            for i, (enc_out, out_res) in enumerate(zip(enc_out_list, x_list[1])):
                # 使用混合预测器
                dec_out = self.hybrid_predictors[i](enc_out)
                
                # 应用残差连接和投影
                dec_out = self.out_projection(dec_out, i, out_res)
                dec_out_list.append(dec_out)

        return dec_out_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """主要预测函数"""
        # 设置未来时间特征（如果需要）
        if self.use_future_temporal_feature:
            if self.channel_independence == 1:
                B, T, N = x_enc.size()
                x_mark_dec = x_mark_dec.repeat(N, 1, 1)
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)
            else:
                self.x_mark_dec = self.enc_embedding(None, x_mark_dec)

        # 多尺度处理
        x_enc, x_mark_enc = self.__multi_scale_process_inputs(x_enc, x_mark_enc)

        # 准备输入
        x_list = []
        x_mark_list = []
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_enc)), x_enc, x_mark_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                    x_mark = x_mark.repeat(N, 1, 1)
                x_list.append(x)
                x_mark_list.append(x_mark)
        else:
            for i, x in zip(range(len(x_enc)), x_enc):
                B, T, N = x.size()
                x = self.normalize_layers[i](x, 'norm')
                if self.channel_independence == 1:
                    x = x.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)
                x_list.append(x)

        # 嵌入
        enc_out_list = []
        x_list = self.pre_enc(x_list)
        if x_mark_enc is not None:
            for i, x, x_mark in zip(range(len(x_list[0])), x_list[0], x_mark_list):
                enc_out = self.enc_embedding(x, x_mark)
                enc_out_list.append(enc_out)
        else:
            for i, x in zip(range(len(x_list[0])), x_list[0]):
                enc_out = self.enc_embedding(x, None)
                enc_out_list.append(enc_out)

        # 使用原TimeMixer的PDM模块作为过去信息编码器
        for i in range(self.layer):
            enc_out_list = self.pdm_blocks[i](enc_out_list)

        # 使用增强版未来多预测器混合作为未来预测解码器
        dec_out_list = self.enhanced_future_mixing(B, enc_out_list, x_list)

        # 应用尺度权重进行最终聚合
        scale_weights = F.softmax(self.scale_weights, dim=0)
        weighted_outputs = [w * out for w, out in zip(scale_weights, dec_out_list)]
        
        # 合并加权输出
        dec_out = torch.stack(weighted_outputs, dim=0).sum(dim=0)
        
        # 反标准化
        dec_out = self.normalize_layers[0](dec_out, 'denorm')
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """前向传播"""
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        else:
            # 实现其他任务（与原TimeMixer相同）
            raise ValueError('其他任务尚未实现')