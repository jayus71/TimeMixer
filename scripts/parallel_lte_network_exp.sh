#!/bin/bash



# 获取脚本所在目录的上一级目录（项目根目录）
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
echo "项目根目录: $PROJECT_ROOT"

# 设置基础路径（相对于项目根目录）
DATA_DIR="$PROJECT_ROOT/data/lte_network/split_cellname_data"
# OUTPUT_BASE="$PROJECT_ROOT/output/AIIA_hour"

OUTPUT_BASE="$PROJECT_ROOT/output/lte_network/split_cellname_data"
# CHECKPOINTS_BASE="$PROJECT_ROOT/checkpoints/AIIA_hour"
CHECKPOINTS_BASE="$PROJECT_ROOT/checkpoints/lte_network/split_cellname_data"


# 确保输出和检查点目录存在
mkdir -p $OUTPUT_BASE
mkdir -p $CHECKPOINTS_BASE

# 设置最大并行任务数
MAX_PARALLEL_JOBS=3

# 创建一个函数来处理单个数据文件
process_file() {
    local data_file=$1
    # 提取文件名（不带路径和扩展名）
    filename=$(basename "$data_file" .csv)
    echo "开始处理数据文件: $filename"
    
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
      --data_format 2 \
      --output_path "$out_dir" \
      --checkpoints "$checkpoint_dir" \
      --seq_len 96 \
      --pred_len 24 \
      --label_len 48 \
      --train_epochs 100 \
      --train_ratio 0.7 \
      --valid_ratio 0.2 \
      --log_file "$out_dir/training.log" \
      --log_level "INFO" \
      > "$out_dir/stdout.log" 2> "$out_dir/stderr.log"
    
    echo "完成处理 $filename"
}

# 导出函数以便在子shell中使用
export -f process_file
export PROJECT_ROOT
export OUTPUT_BASE
export CHECKPOINTS_BASE

# 获取所有CSV文件列表
csv_files=($(ls $DATA_DIR/*.csv))
total_files=${#csv_files[@]}
echo "找到 $total_files 个CSV文件待处理"

# 使用GNU并行工具处理文件
if command -v parallel &> /dev/null; then
    echo "使用GNU parallel并行处理文件，最大并行任务数: $MAX_PARALLEL_JOBS"
    parallel --jobs $MAX_PARALLEL_JOBS --bar process_file ::: "${csv_files[@]}"
else
    # 如果没有安装parallel，使用简单的子shell并行处理
    echo "未找到GNU parallel，使用简单的并行处理，最大并行任务数: $MAX_PARALLEL_JOBS"
    
    # 跟踪运行中的作业数
    running=0
    
    # 处理所有文件
    for data_file in "${csv_files[@]}"; do
        # 如果达到最大并行数，等待任意子进程完成
        if [[ $running -ge $MAX_PARALLEL_JOBS ]]; then
            wait -n
            running=$((running - 1))
        fi
        
        # 在后台启动处理
        process_file "$data_file" &
        running=$((running + 1))
        echo "当前运行中作业数: $running"
    done
    
    # 等待所有后台作业完成
    echo "等待所有作业完成..."
    wait
fi

echo "所有数据处理完成！"