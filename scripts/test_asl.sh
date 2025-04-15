#!/bin/bash

# 默认值
NAME="base"
MODEL="pp_base"
GPU="-1"
EPOCH="latest"

OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
  DATAROOT="--dataroot=/root/autodl-fs"
elif [[ "$OS" == "Darwin" ]]; then
  DATAROOT="--dataroot=./datasets/1217_ASL_T1"
fi

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
  case "$1" in
  -n | --name)
    NAME="$2"
    shift 2
    ;;
  -m | --model)
    MODEL="$2"
    shift 2
    ;;
  -g | --gpu)
    GPU="$2"
    shift 2
    ;;
  -e | --epoch)
    EPOCH="$2"
    shift 2
    ;;
  *)
    echo "未知参数: $1"
    exit 1
    ;;
  esac
done

python test.py $DATAROOT --gpu_ids="$GPU" --epoch=$EPOCH --name=$NAME --model=$MODEL --load_size=64  --results_dir=./results --num_threads=0 --netG=unet_64 --netD=pixel --direction=AtoB
