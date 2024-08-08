#!/bin/bash

# 使用一张卡依次执行所有任务，并将输出重定向到 /dev/null
nohup sh Mamba2_RMS.sh > /dev/null 2>&1 &
wait

nohup sh Mamba2_LN.sh > /dev/null 2>&1 &
wait

nohup sh Mamba2_IN.sh > /dev/null 2>&1 &
wait

nohup sh Mamba2_GN.sh > /dev/null 2>&1 &
wait

nohup sh Mamba2_BN.sh > /dev/null 2>&1 &
wait

nohup sh Mamba.sh > /dev/null 2>&1 &
wait

echo "所有任务已完成。"
