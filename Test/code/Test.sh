#!/bin/bash/
# For release
# s-LWSR_BIX2
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
# s-LWSR_BIX3
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
# s-LWSR_BIX4
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5				
# RCAN_BIX8
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
##
# LWSRplus_BIX2
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
# LWSRplus_BIX3
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
# LWSRplus_BIX4
CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
# LWSRplus_BIX8
#CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
