#!/bin/bash
devicenum=1
dataset='VLCS'
alg='DCT'
margin=0
model='resnet50'
name='dct'
ckeckpoints=200
usebn=False
useswad=False
normfeat=True
if [ $useswad = True ] ; then
  name="${name}_swad"
fi
if [ $normfeat = True ]; then
  name="${name}_norm"
fi
if [ $usebn = True ]; then
  name="${name}_bnn"
fi

CUDA_VISIBLE_DEVICES=$devicenum python train_all.py "$alg-maigin$margin-$model-trial0-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--holdout_fraction 0.2 \
--trial_seed 0 \
--resnet_dropout 0.1 \
--lr 1e-4 \
--steps 5001 \
--checkpoint_freq $ckeckpoints \
--use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim True

CUDA_VISIBLE_DEVICES=$devicenum python train_all.py "$alg-maigin$margin-$model-trial1-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--holdout_fraction 0.2 \
--trial_seed 1 \
--resnet_dropout 0.1 \
--lr 1e-4 \
--steps 5001 \
--checkpoint_freq $ckeckpoints \
--use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim True

CUDA_VISIBLE_DEVICES=$devicenum python train_all.py "$alg-maigin$margin-$model-trial2-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--holdout_fraction 0.2 \
--trial_seed 2 \
--resnet_dropout 0.1 \
--lr 1e-4 \
--steps 5001 \
--checkpoint_freq $ckeckpoints \
--use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim True