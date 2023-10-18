#!/bin/bash
devicenum=1
dataset='PACS'
alg='ERM'
margin=12
lr=5e-5
useswad=False
normfeat=False
usebn=True
model='resnet50' #swag_regnety_16gf，resnet50
name="dct"
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
--dataset $dataset \
--algorithm $alg \
--trial_seed 0 \
--lr $lr \
--steps 6001 \
--use_swad $useswad #--margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim $usereidop --pretrained $pretrained

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial1-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--trial_seed 1 \
--lr $lr \
--steps 6001 \
--use_swad $useswad #--margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim $usereidop --pretrained $pretrained

CUDA_VISIBLE_DEVICES=$devicenum \
python train_all.py "$alg-maigin$margin-$model-trial2-$name" \
--data_dir '' \
--dataset $dataset \
--algorithm $alg \
--trial_seed 2 \
--lr $lr \
--steps 6001 \
--use_swad $useswad #--margin $margin --normfeat $normfeat --use_bnn $usebn --use_reidoptim $usereidop --pretrained $pretrained