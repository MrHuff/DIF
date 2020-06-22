#!/usr/bin/env sh
#
#$ -cwd
#$ -j y
#$ -N output_train_lstm
#$ -S /bin/sh
#
CUDA_VISIBLE_DEVICES=4 python main.py  --hdim=512 --output_height=256 --channels='32, 64, 128, 256, 512, 512' --m_plus=120 --weight_rec=0.05 --weight_kl=1.0  --weight_neg=0.5 --num_vae=0  --dataroot='/homes/rhu/data/data256x256' --trainsize=29000 --test_iter=1000 --save_iter=1 --start_epoch=0  --batchSize=16 --nrow=8 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=500   2>&1   | tee train0.log

#Another appropriate setting defined as below:

#CUDA_VISIBLE_DEVICES=0 python main.py  --hdim=512 --output_height=256 --channels='32, 64, 128, 256, 512, 512' --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0  --weight_neg=0.5 --num_vae=10  --dataroot='/home/huaibo.huang/data/celeba-hq/celeba-hq-images-256' --trainsize=29000 --test_iter=1000 --save_iter=1 --start_epoch=0  --batchSize=16 --nrow=8 --lr_e=0.0002 --lr_g=0.0002   --cuda  --nEpochs=500   2>&1   | tee train0.log 

#CELEBHQ
###benchmark run
#python main_DIF.py --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/data256x256 --trainsize=29000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=500 --class_indicator_file=/homes/rhu/data/celebA_hq_gender.csv --fp_16 --J=0.25 --lambda_me=0
###encoder run
#python main_DIF.py --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/data256x256 --trainsize=29000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=500 --class_indicator_file=/homes/rhu/data/celebA_hq_gender.csv --fp_16 --J=0.25 --lambda_me=0.2 --C=10 --tanh_flag --kernel "linear"
###flow run
#python main_DIF_flow.py --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/data256x256 --trainsize=29000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=500 --class_indicator_file=/homes/rhu/data/celebA_hq_gender.csv --fp_16 --J=0.25 --lambda_me=0.2 --C=10 --tanh_flag --kernel "linear" --flow_depth 4

#FASHION
###benchmark run
#python main_DIF.py --prefix "fashion" --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/fashion_256x256 --trainsize=22000 --test_iter=985 --save_iter=10 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=500 --class_indicator_file=/homes/rhu/data/fashion_price_class.csv --fp_16 --J=0.25 --lambda_me=0
#encoder run
#python main_DIF.py --prefix "fashion" --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/fashion_256x256 --trainsize=22000 --test_iter=985 --save_iter=10 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=500 --class_indicator_file=/homes/rhu/data/fashion_price_class.csv --fp_16 --J=0.25 --lambda_me=0.2 --C=10 --tanh_flag --kernel "linear"
###flow run
#python main_DIF_flow.py --prefix "fashion" --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/fashion_256x256 --trainsize=22000 --test_iter=985 --save_iter=10 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=500 --class_indicator_file=/homes/rhu/data/fashion_price_class.csv --fp_16 --J=0.25 --lambda_me=0.2 --C=10 --tanh_flag --kernel "linear" --flow_depth 4

