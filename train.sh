#!/usr/bin/bash

# Set the path to the pretrained model
export PRETRAINED_DIR=/root/autodl-tmp

# set data path
export DATA_DIR=/root/autodl-tmp


# python -m vit_jax.main --workdir=$DATA_DIR/vit-$(date +%s) \
#     --config=$(pwd)/vit_jax/configs/mixer_base16_cifar100.py \
#     --config.pretrained_dir=$PRETRAINED_DIR

# python -m vit_jax.main --workdir=$DATA_DIR/vit-$(date +%s) \
#     --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar100 \
#     --config.dataset=$DATA_DIR/cifar-100 \
#     --config.pretrained_dir=$PRETRAINED_DIR

# python -m vit_jax.main --workdir=$DATA_DIR/vit-s16-$(date +%s) \
#     --config=$(pwd)/vit_jax/configs/vit.py:s16,cifar100 \
#     --config.dataset=$DATA_DIR/cifar-100 \
#     --config.pretrained_dir=$PRETRAINED_DIR \

####### restore checkpoint failed
# python -m vit_jax.main --workdir=$DATA_DIR/vit-1719502028 \
#     --config=$(pwd)/vit_jax/configs/vit.py:b16,cifar100 \
#     --config.dataset=$DATA_DIR/cifar-100 \
#     --config.pretrained_dir=$PRETRAINED_DIR

python -m vit_jax.main --workdir=$DATA_DIR/vit-b16-fine-$(date +%s) \
    --config=$(pwd)/vit_jax/configs/augreg.py:B_16-i21k-300ep-lr_0.001-aug_light0-wd_0.03-do_0.0-sd_0.0--cifar100-steps_10k-lr_0.001-res_224 \
    --config.dataset=$DATA_DIR/cifar-100 \
    --config.pp.train='train[:90%]' \
    --config.base_lr=0.01 \
    --config.pretrained_dir=$PRETRAINED_DIR \
    --config.total_steps=2000