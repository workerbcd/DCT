#!/bin/bash
devicenum=0
dataset='OfficeHome'
alg='DCT'
model='resnet50'
margin=0
checkpoints=200
name='newdct'
useswad=False
lr=1e-4
normfeat=False #DCT:true
usebn=False  #dct:true
if [ $useswad = True ] ; then
  name="${name}_swad"
fi
if [ $normfeat = True ]; then
  name="${name}_norm"
fi
if [ $usebn = True ]; then
  name="${name}_bnn"
fi

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial0-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--trial_seed 0 \
--resnet_dropout 0.2 \
--lr $lr \
--checkpoint_freq $checkpoints \
--steps 5001 --use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn #--use_reidoptim False

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial1-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--trial_seed 1 \
--resnet_dropout 0.2 \
--lr $lr \
--checkpoint_freq $checkpoints \
--steps 5001 --use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn #--use_reidoptim False

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial2-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--trial_seed 2 \
--resnet_dropout 0.2 \
--lr $lr \
--checkpoint_freq $checkpoints \
--steps 5001 --use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn #--use_reidoptim False