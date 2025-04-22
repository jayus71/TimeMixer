import argparse

def get_args():
    """获取增强版TimeMixer的配置参数"""
    parser = argparse.ArgumentParser(description='增强版TimeMixer网络流量预测')
    
    # 基本配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='任务名称，可选:[long_term_forecast, short_term_forecast]')
    parser.add_argument('--data_path', type=str, default='data/AIIA_hour/dataA_fill.csv',
                        help='数据文件路径')
    parser.add_argument('--data_format', type=int, default=1,
                        help='数据格式：1 表示 (时间,城市,值)，2 表示 (日期,小时,小区名,流量)')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints',
                        help='模型检查点保存路径')
    parser.add_argument('--output_path', type=str, default='./output',
                        help='输出结果保存路径')
    
    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=96,
                        help='输入序列长度')
    parser.add_argument('--label_len', type=int, default=48,
                        help='开始标记长度')
    parser.add_argument('--pred_len', type=int, default=24,
                        help='预测序列长度')
    
    # 数据加载配置
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='训练集比例')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='验证集比例')
    parser.add_argument('--scale', action='store_true', default=True,
                        help='是否标准化数据')
    
    # 模型定义
    parser.add_argument('--enc_in', type=int, default=7,
                        help='编码器输入大小')
    parser.add_argument('--c_out', type=int, default=7,
                        help='输出大小')
    parser.add_argument('--d_model', type=int, default=256,
                        help='模型维度')
    parser.add_argument('--e_layers', type=int, default=2,
                        help='编码器层数')
    parser.add_argument('--d_ff', type=int, default=512,
                        help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='dropout率')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='时间特征编码')
    parser.add_argument('--freq', type=str, default='h',
                        help='时间频率')
    parser.add_argument('--moving_avg', type=int, default=25,
                        help='移动平均窗口大小')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='0: 特征依赖 1: 特征独立')
    parser.add_argument('--use_norm', type=int, default=1,
                        help='是否使用归一化')
    parser.add_argument('--down_sampling_layers', type=int, default=2,
                        help='下采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=2,
                        help='下采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='下采样方法，支持 avg, max, conv')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                        help='分解方法，支持 moving_avg 或 dft_decomp')
    parser.add_argument('--top_k', type=int, default=5,
                        help='DFT分解的top_k参数')
    parser.add_argument('--use_future_temporal_feature', type=int, default=0,
                        help='是否使用未来时间特征')
    
    # 优化
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器工作线程数')
    parser.add_argument('--train_epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停耐心值')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help='学习率')
    parser.add_argument('--use_gpu', type=bool, default=True,
                        help='是否使用GPU')
    
    return parser.parse_args()