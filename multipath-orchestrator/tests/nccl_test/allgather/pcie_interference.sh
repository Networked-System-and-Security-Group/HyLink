#!/bin/bash

LOG_DIR=./logs


log_file="${LOG_DIR}/nccl_allgather_with_pcie_$(date +%Y%m%d_%H%M%S).log"
mkdir -p "${LOG_DIR}"
# stdbuf -oL ./a.out 2>&1 | tee log.txt
bash ./run.sh ./nccl_allgather_with_pcie --devices=2 --send-count=268435456 --iters=1001 --interference-ranges=100-200,300-600,700-900 --interference-size=268435456 2>&1 | tee "${log_file}" 

# python process_log_and_plot.py ^
#   --dynamic-log logs\nccl_allgather_with_pcie_20260305_130347.log ^
#   --static-log logs\static.log ^
#   --alpha-static 0.6 ^
#   --output-fig figure\my_figure.png

# python batch_process_logs.py \
#   --logs-dir logs \
#   --static-log my_static.log    
#   --alpha-static 0.6
