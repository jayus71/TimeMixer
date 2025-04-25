import torch
import os
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import logging
from rich.progress import (
    Progress, 
    TextColumn, 
    BarColumn, 
    TimeElapsedColumn, 
    TimeRemainingColumn,
    MofNCompleteColumn
)
from torch.utils.data import DataLoader
from data_provider.network_traffic_dataset import NetworkTrafficDataset
from models.enhanced_timemixer import EnhancedTimeMixer
from utils.config import get_args
from utils.losses import *

logging.basicConfig(filename='training.log', filemode='a', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 设置随机种子
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# 早停类
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            logging.info(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
            print(f'验证损失减少 ({self.val_loss_min:.6f} --> {val_loss:.6f})。保存模型...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

# 训练模型
def train_model(args, train_loader, val_loader, model, device):
    model_save_path = os.path.join(args.checkpoints, 'best_model.pth')
    os.makedirs(args.checkpoints, exist_ok=True)
    
    # 设置优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    # criterion = torch.nn.MSELoss()
    if args.loss == 'MSE':
        criterion = torch.nn.MSELoss()
    elif args.loss == 'MAE':
        criterion = torch.nn.L1Loss()
    elif args.loss == 'MAPE':
        criterion = mape_loss()
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    for epoch in range(args.train_epochs):
        model.train()
        train_loss = 0
        
        # 创建Rich进度条
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            train_task = progress.add_task(f"[green]Epoch {epoch+1}/{args.train_epochs} (Train)", total=len(train_loader))
            
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                optimizer.zero_grad()
                
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                # 前向传播
                if args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                else:
                    dec_inp = None
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 计算损失
                loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                train_loss += loss.item()
                
                # 反向传播和优化
                loss.backward()
                optimizer.step()
                
                # 更新进度条
                progress.update(train_task, advance=1, description=f"[green]Epoch {epoch+1}/{args.train_epochs} (Train) - Loss: {loss.item():.4f}")
        
        # 验证
        model.eval()
        val_loss = 0
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=50),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
        ) as progress:
            val_task = progress.add_task(f"[yellow]Epoch {epoch+1}/{args.train_epochs} (Valid)", total=len(val_loader))
            
            with torch.no_grad():
                for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(val_loader):
                    batch_x = batch_x.float().to(device)
                    batch_y = batch_y.float().to(device)
                    batch_x_mark = batch_x_mark.float().to(device)
                    batch_y_mark = batch_y_mark.float().to(device)
                    
                    if args.down_sampling_layers == 0:
                        dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                    else:
                        dec_inp = None
                    
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    
                    # 计算损失
                    loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                    val_loss += loss.item()
                    
                    # 更新进度条
                    progress.update(val_task, advance=1, description=f"[yellow]Epoch {epoch+1}/{args.train_epochs} (Valid) - Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        print(f'Epoch [{epoch+1}/{args.train_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        logging.info(f'Epoch [{epoch+1}/{args.train_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 早停检查
        early_stopping(avg_val_loss, model, model_save_path)
        if early_stopping.early_stop:
            logging.info("Early stopping")
            print("Early stopping")
            break
    
    return model_save_path

def visualize_predictions(preds, trues, output_path, data_format=1):
    """
    网络流量预测结果可视化
    
    参数:
        preds: 预测值数组
        trues: 真实值数组
        output_path: 输出路径
        data_format: 数据格式类型(1或2)
    """
    os.makedirs(output_path, exist_ok=True)
    
    # 确定流量数据的特征索引
    if data_format == 1:
        traffic_feature_idx = 1  # 第一种格式：流量值在索引1
    else:
        traffic_feature_idx = 2  # 第二种格式：流量值在索引2
    
    print(f"预测值形状: {preds.shape}, 真实值形状: {trues.shape}")
    print(f"使用特征索引 {traffic_feature_idx} 作为流量数据")
    
    # 检查流量特征的数据范围
    true_min = trues[:,:,traffic_feature_idx].min()
    true_max = trues[:,:,traffic_feature_idx].max()
    true_mean = trues[:,:,traffic_feature_idx].mean()
    pred_min = preds[:,:,traffic_feature_idx].min()
    pred_max = preds[:,:,traffic_feature_idx].max()
    pred_mean = preds[:,:,traffic_feature_idx].mean()
    
    print(f"流量数据 - 真实值: 最小={true_min:.6f}, 最大={true_max:.6f}, 平均={true_mean:.6f}")
    print(f"流量数据 - 预测值: 最小={pred_min:.6f}, 最大={pred_max:.6f}, 平均={pred_mean:.6f}")
    
    # 选择一些有代表性的样本进行可视化
    num_samples = min(10, preds.shape[0])
    sample_indices = np.linspace(0, preds.shape[0]-1, num_samples, dtype=int)
    
    for i, idx in enumerate(sample_indices):
        plt.figure(figsize=(12, 6))
        
        # 创建x轴时间点
        time_points = np.arange(preds.shape[1])
        
        # 绘制真实值和预测值
        plt.plot(time_points, trues[idx, :, traffic_feature_idx], 'b-', label='真实值', linewidth=2)
        plt.plot(time_points, preds[idx, :, traffic_feature_idx], 'r-', label='预测值', linewidth=2)
        
        # 添加图表信息
        plt.title(f'网络流量预测 - 样本 {idx+1}', fontsize=15)
        plt.xlabel('时间步', fontsize=12)
        plt.ylabel('流量值', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'traffic_sample_{idx+1}.png'), dpi=150)
        plt.close()
    
    # 创建一个汇总图，显示所有样本的平均性能
    plt.figure(figsize=(14, 8))
    
    # 计算每个时间步上所有样本的平均真实值和预测值
    mean_true = np.mean(trues[:, :, traffic_feature_idx], axis=0)
    mean_pred = np.mean(preds[:, :, traffic_feature_idx], axis=0)
    
    plt.plot(mean_true, 'b-', label='平均真实流量', linewidth=2)
    plt.plot(mean_pred, 'r-', label='平均预测流量', linewidth=2)
    plt.title('所有样本的平均网络流量预测性能')
    plt.xlabel('时间步')
    plt.ylabel('流量值')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'average_traffic_performance.png'), dpi=150)
    plt.close()

    print(f"可视化完成，结果保存到: {output_path}")

# 测试模型
def test_model(args, test_loader, model, device, test_dataset):
    model_save_path = os.path.join(args.checkpoints, 'best_model.pth')
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    criterion = torch.nn.MSELoss()
    
    test_loss = 0
    preds = []
    trues = []
    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=50),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    ) as progress:
        test_task = progress.add_task("[red]Testing", total=len(test_loader))
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(device)
                batch_y = batch_y.float().to(device)
                batch_x_mark = batch_x_mark.float().to(device)
                batch_y_mark = batch_y_mark.float().to(device)
                
                if args.down_sampling_layers == 0:
                    dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
                    dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(device)
                else:
                    dec_inp = None
                
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # 计算损失
                loss = criterion(outputs, batch_y[:, -args.pred_len:, :])
                test_loss += loss.item()
                
                # 更新进度条
                progress.update(test_task, advance=1, description=f"[red]Testing - Loss: {loss.item():.4f}")
                # 保存预测和真实值
                pred = outputs.detach().cpu().numpy()
                true = batch_y[:, -args.pred_len:, :].detach().cpu().numpy()
                
                preds.append(pred)
                trues.append(true)
    
    avg_test_loss = test_loss / len(test_loader)
    print(f'Test Loss: {avg_test_loss:.4f}')
    
    # 合并预测结果
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    
    # 反标准化
    if args.scale:
        preds = test_dataset.inverse_transform(preds)
        trues = test_dataset.inverse_transform(trues)
    
    # 保存结果
    os.makedirs(args.output_path, exist_ok=True)
    np.save(os.path.join(args.output_path, 'preds.npy'), preds)
    np.save(os.path.join(args.output_path, 'trues.npy'), trues)
    
    # 确定流量特征索引
    traffic_feature_idx = 1 if args.data_format == 1 else 2
    
    # 计算额外的评估指标
    # 只对流量特征计算评估指标
    pred_traffic = preds[:, :, traffic_feature_idx]
    true_traffic = trues[:, :, traffic_feature_idx]
    
    # 平均绝对误差 (MAE)
    mae = np.mean(np.abs(pred_traffic - true_traffic))
    
    # 均方根误差 (RMSE)
    rmse = np.sqrt(np.mean((pred_traffic - true_traffic) ** 2))
    
    # 平均绝对百分比误差 (MAPE)
    # 避免除以零，添加一个小常数
    epsilon = 1e-10
    mape = np.mean(np.abs((true_traffic - pred_traffic) / (np.abs(true_traffic) + epsilon))) * 100
    
    # 决定系数 (R²)
    true_mean = np.mean(true_traffic)
    ss_tot = np.sum((true_traffic - true_mean) ** 2)
    ss_res = np.sum((true_traffic - pred_traffic) ** 2)
    r2 = 1 - (ss_res / (ss_tot + epsilon))
    
    # 对称平均绝对百分比误差 (SMAPE)
    smape = np.mean(2.0 * np.abs(pred_traffic - true_traffic) / (np.abs(pred_traffic) + np.abs(true_traffic) + epsilon)) * 100
    
    # 归一化的均方根误差 (NRMSE)
    nrmse = rmse / (np.max(true_traffic) - np.min(true_traffic))
    
    # 打印并记录评估指标
    metrics = {
        'MSE': avg_test_loss,
        'MAE': float(mae),
        'RMSE': float(rmse),
        'MAPE': float(mape),
        'SMAPE': float(smape),
        'R2': float(r2),
        'NRMSE': float(nrmse)
    }
    
    # 保存评估指标到CSV文件
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(args.output_path, f'metrics_{args.data_format}.csv'), index=False)
    
    # 可视化结果 - 对特定样本和特征进行绘图
    visualize_predictions(preds, trues, args.output_path, args.data_format)
    
    return avg_test_loss, preds, trues, metrics

def main():
    # 获取参数
    args = get_args()
    
    # 设置随机种子
    setup_seed(2023)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    
    # 创建数据集
    train_dataset = NetworkTrafficDataset(
        file_path=args.data_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        data_format=args.data_format,
        scale=args.scale,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        flag='train'
    )
    
    val_dataset = NetworkTrafficDataset(
        file_path=args.data_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        data_format=args.data_format,
        scale=args.scale,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        flag='val'
    )
    
    test_dataset = NetworkTrafficDataset(
        file_path=args.data_path,
        seq_len=args.seq_len,
        label_len=args.label_len,
        pred_len=args.pred_len,
        data_format=args.data_format,
        scale=args.scale,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        flag='test'
    )
    logging.info("========== 数据集信息 ==========")
    logging.info(f"训练集大小: {len(train_dataset)}")
    logging.info(f"验证集大小: {len(val_dataset)}")
    logging.info(f"测试集大小: {len(test_dataset)}")
    logging.info(f"数据集特征数量: {train_dataset.data_x.shape[1]}")
    logging.info(f"数据集标签长度: {train_dataset.data_y.shape[1]}")
    logging.info(f"数据集预测长度: {args.pred_len}")
    logging.info(f"数据集序列长度: {args.seq_len}")
    logging.info(f"数据集数据格式: {args.data_format}")
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # 更新参数
    args.enc_in = train_dataset.data_x.shape[1]
    args.c_out = train_dataset.data_x.shape[1]
    
    # 创建模型
    model = EnhancedTimeMixer(args)
    model = model.to(device)
    
    # 训练模型
    model_save_path = train_model(args, train_loader, val_loader, model, device)
    
    # 测试模型
    avg_test_loss, preds, trues, metrics = test_model(args, test_loader, model, device, test_dataset)
    
    # 结果汇总
    print(f'测试完成。')
    print(f'测试损失: {avg_test_loss:.4f}')
    print(f'预测结果已保存到: {args.output_path}')
    logging.info("========== 测试结果 ==========")
    logging.info(f'测试损失: {avg_test_loss:.4f}')
    print("\n========== 评估指标 ==========")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    logging.info("========== 评估指标 ==========")
    for name, value in metrics.items():
        logging.info(f"{name}: {value:.4f}")
    
    return model, avg_test_loss, preds, trues, metrics

if __name__ == "__main__":
    main()