#!/usr/bin/env sh

#CELEBHQ Experiments
###benchmark run
#python main_DIF.py --hdim=512 --prefix "facesHQv3" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/data256x256 --trainsize=29000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=16 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=300 --class_indicator_file=/homes/rhu/data/celebA_hq_gender.csv --fp_16 --J=0.0 --lambda_me=0 --C=10 --tanh_flag
###encoder run
#python main_DIF.py --hdim=512 --prefix "facesHQv3" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/data256x256 --trainsize=29000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=300 --class_indicator_file=/homes/rhu/data/celebA_hq_gender.csv --fp_16 --J=0.25 --lambda_me=0.25 --C=10 --tanh_flag --kernel "linear"
#linear_benchmark-encoder
##python main_DIF.py --hdim=512 --prefix "facesHQv3" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/data256x256 --trainsize=29000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=300 --class_indicator_file=/homes/rhu/data/celebA_hq_gender.csv --fp_16 --J=0.0 --lambda_me=1.0 --C=10 --tanh_flag --kernel "rbf" --linear_benchmark


#FASHION Experiments
#python main_DIF.py --prefix "fashion" --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=0.1 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/fashion_256x256 --trainsize=22000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=300 --class_indicator_file=/homes/rhu/data/fashion_price_class.csv --fp_16 --J=0 --lambda_me=0 --C=10 --tanh_flag
#encoder run
#python main_DIF.py --prefix "fashion" --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=0.1 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/fashion_256x256 --trainsize=22000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=300 --class_indicator_file=/homes/rhu/data/fashion_price_class.csv --fp_16 --J=0.25 --lambda_me=0.4 --C=10 --tanh_flag --kernel "linear"
#linear benchmark encoder
#python main_DIF.py --prefix "fashion" --hdim=512 --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=0.1 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/fashion_256x256 --trainsize=22000 --test_iter=1000 --save_iter=10 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=300 --class_indicator_file=/homes/rhu/data/fashion_price_class.csv --fp_16 --J=0 --lambda_me=1.0 --C=10 --tanh_flag --kernel "linear" --linear_benchmark



#MNIST Experiments
#python main_DIF.py --prefix "mnist38" --hdim=16 --output_height=64 "--channels=64, 128, 256, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/mnist_3_8_64x64 --trainsize=13000 --test_iter=500 --save_iter=2 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=25 --class_indicator_file=/homes/rhu/data/mnist_3_8.csv --fp_16 --J=0 --lambda_me=0 --cdim 1 --C=10 --tanh_flag
# encoder
#python main_DIF.py --prefix "mnist38" --hdim=16 --output_height=64 "--channels=64, 128, 256, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/mnist_3_8_64x64 --trainsize=13000 --test_iter=500 --save_iter=2 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=25 --class_indicator_file=/homes/rhu/data/mnist_3_8.csv --fp_16 --J=0.25 --lambda_me=0.01 --C=10 --tanh_flag --kernel "linear" --cdim 1
#linear benchmark encoder
#python main_DIF.py --prefix "mnist38" --hdim=16 --output_height=64 "--channels=64, 128, 256, 512" --m_plus=1000 --weight_rec=1.0 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/mnist_3_8_64x64 --trainsize=13000 --test_iter=500 --save_iter=2 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=25 --class_indicator_file=/homes/rhu/data/mnist_3_8.csv --fp_16 --J=0 --lambda_me=1.0 --C=10 --tanh_flag --kernel "linear" --linear_benchmark --cdim 1


#COVID Experiments
###benchmark run
#python main_DIF.py --hdim=512 --prefix "covid256" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=150 --weight_rec=0.25 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/covid_dataset_256x256 --trainsize=1900 --test_iter=136 --save_iter=5 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=150 --class_indicator_file=/homes/rhu/data/covid_19_sick.csv --fp_16 --J=0 --lambda_me=0 --C=10 --tanh_flag --cdim 1
###encoder run
#python main_DIF.py --hdim=128 --prefix "covid256" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=150 --weight_rec=0.25 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/covid_dataset_256x256 --trainsize=1900 --test_iter=136 --save_iter=5 --start_epoch=0 --batchSize=32 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=150 --class_indicator_file=/homes/rhu/data/covid_19_sick.csv --fp_16 --J=0.25 --lambda_me=0.15 --C=10 --tanh_flag --kernel "linear" --cdim 1
#linear_benchmark-encoder
#python main_DIF.py --hdim=512 --prefix "covid256" --output_height=256 "--channels=32, 64, 128, 256, 512, 512" --m_plus=150 --weight_rec=0.25 --weight_kl=1.0 --weight_neg=0.5 --num_vae=10 --dataroot=/homes/rhu/data/covid_dataset_256x256 --trainsize=1900 --test_iter=136 --save_iter=5 --start_epoch=0 --batchSize=24 --nrow=8 --lr_e=0.0002 --lr_g=0.0002 --cuda --nEpochs=150 --class_indicator_file=/homes/rhu/data/covid_19_sick.csv --fp_16 --J=0 --lambda_me=1.0 --C=10 --tanh_flag --kernel "rbf" --linear_benchmark --cdim 1
