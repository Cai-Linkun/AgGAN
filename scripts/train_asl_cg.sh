#!/bin/bash

set -ex

# 默认值
NAME="cg_base"
MODEL="cg_base"
GPU="0"
BATCH="2"
EPOCH="latest"
CONTINUE_TRAIN=""
CONTINUE_EPOCH=""
SAVE_FREQ="10"

OS="$(uname)"
if [[ "$OS" == "Linux" ]]; then
  DATAROOT="--dataroot=/root/autodl-fs"
  SAVE_FREQ="50"
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
  -b | --batch)
    BATCH="$2"
    shift 2
    ;;
  -e | --epoch)
    EPOCH="$2"
    shift 2
    ;;
  -c | --continue)
    # EPOCH="$2"
    CONTINUE_TRAIN="--continue_train"
    shift 1
    ;;
  *)
    echo "未知参数: $1"
    exit 1
    ;;
  esac
done

if [[ -n "$CONTINUE_TRAIN" ]]; then
    CONTINUE_EPOCH=" --epoch_count="+$EPOCH
fi

python train.py "$DATAROOT" --epoch="$EPOCH" $CONTINUE_TRAIN $CONTINUE_EPOCH  --batch_size="$BATCH" --gpu_ids="$GPU" --name="$NAME" --model="$MODEL"  --n_epochs=100 --num_threads=0 --netG=resnet_6blocks --netD=basic --load_size=64 --gan_mode=lsgan --crop_size=64 --no_flip --save_epoch_freq="$SAVE_FREQ"
