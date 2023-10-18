#!/bin/bash
devicenum=0
dataset='OfficeHome'
alg='DCT'
margin=0
lr=5e-5
useswad=False
normfeat=False
usebn=False
model='resnet50'
name="regnet_newdct"
pretrained=True
if [ $useswad = True ] ; then
  name="${name}_swad"
fi
if [ $normfeat = True ]; then
  name="${name}_norm"
fi
if [ $usebn = True ]; then
  name="${name}_bnn"
fi
if [ $pretrained = False ]; then
  name="${name}_nopretrain"
fi
usereidop=False

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial0-$name" \
--data_dir '' \
--checkpoint_freq 200 \
--dataset $dataset \
--algorithm $alg \
--trial_seed 0 \
--backbone "swag_regnety_16gf" \
--lr $lr \
--steps 5001 --use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim $usereidop --pretrained $pretrained

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial1-$name" \
--data_dir '' \
--checkpoint_freq 200 \
--dataset $dataset \
--algorithm $alg \
--trial_seed 1 \
--backbone "swag_regnety_16gf" \
--lr $lr \
--steps 5001 --use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim $usereidop --pretrained $pretrained

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial2-$name" \
--data_dir '' \
--checkpoint_freq 200 \
--dataset $dataset \
--algorithm $alg \
--trial_seed 2 \
--backbone "swag_regnety_16gf" \
--lr $lr \
--steps 5001 --use_swad $useswad --margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim $usereidop --pretrained $pretrained