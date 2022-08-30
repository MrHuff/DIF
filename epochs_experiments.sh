#!/usr/bin/env sh

#CELEBHQ Experiments
###encoder run
python main_DIF.py --hdim=512 --prefix "facesHQv3_epochs" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/data/ziz/rhu/data/data256x256 --trainsize=29000 --test_iter=10 --save_iter=10 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=100 --class_indicator_file=/data/ziz/rhu/local_deploys/DIF/celebA_hq_gender.csv --fp_16 --J=0.25 --lambda_me=0.25 --C=10 --tanh_flag --kernel "linear" --tensorboard
wait
#MNIST Experiments
python main_DIF.py --prefix "mnist38_epochs" --hdim=16 --output_height=64 "--channels=64, 128, 256, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/data/ziz/rhu/data/mnist_3_8_64x64 --trainsize=13000 --test_iter=10 --save_iter=2 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=25 --class_indicator_file=/data/ziz/rhu/local_deploys/DIF/mnist_3_8.csv --fp_16 --J=0.25 --lambda_me=0.01 --C=10 --tanh_flag --kernel "linear" --cdim 1 --tensorboard
wait
#COVID Experiments
###encoder run
python main_DIF.py --hdim=128 --prefix "covid256_epochs" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=150 --weight_rec=0.25 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/data/ziz/rhu/data/covid_dataset_256x256 --trainsize=1900 --test_iter=10 --save_iter=5 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=100 --class_indicator_file=/data/ziz/rhu/local_deploys/DIF/covid_19_sick.csv --fp_16 --J=0.25 --lambda_me=0.15 --C=10 --tanh_flag --kernel "linear" --cdim 1 --tensorboard
