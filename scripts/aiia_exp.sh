#!/bin/bash

# 获取脚本所在目录的上一级目录（项目根目录）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "项目根目录: $PROJECT_ROOT"

# 设置基础路径（相对于项目根目录）
# DATA_DIR="$PROJECT_ROOT/data/AIIA_hour/split_data"
DATA_DIR="$PROJECT_ROOT/data/AIIA_hour/split_half_year_data"
# OUTPUT_BASE="$PROJECT_ROOT/output/AIIA_hour"
OUTPUT_BASE="$PROJECT_ROOT/output/AIIA_hour/split_half_year_data"
# CHECKPOINTS_BASE="$PROJECT_ROOT/checkpoints/AIIA_hour"
CHECKPOINTS_BASE="$PROJECT_ROOT/checkpoints/AIIA_hour/split_half_year_data"

# 确保输出和检查点目录存在
mkdir -p $OUTPUT_BASE
mkdir -p $CHECKPOINTS_BASE

# 循环处理数据目录中的所有CSV文件
for data_file in $DATA_DIR/*.csv; do
    # 提取文件名（不带路径和扩展名）
    filename=$(basename "$data_file" .csv)
    echo "处理数据文件: $filename"
    
    # 创建特定于此数据文件的输出和检查点目录
    out_dir="$OUTPUT_BASE/$filename"
    checkpoint_dir="$CHECKPOINTS_BASE/$filename"
    mkdir -p $out_dir
    mkdir -p $checkpoint_dir
    
    # 运行实验（从项目根目录运行main.py）
    echo "运行 $filename 的实验..."
    cd $PROJECT_ROOT
    python main.py \
      --data_path "$data_file" \
      --data_format 1 \
      --output_path "$out_dir" \
      --checkpoints "$checkpoint_dir" \
      --seq_len 96 \
      --pred_len 24 \
      --label_len 48 \
      --train_ratio 0.7 \
      --valid_ratio 0.2 \
      --log_file "$out_dir/training.log" \
      --log_level "INFO"
    
done

echo "所有数据处理完成！"