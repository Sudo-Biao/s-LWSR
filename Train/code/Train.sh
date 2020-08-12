## train
# BI, scale 2, 3, 4, 8
##################################################################################################################################
# BI, scale 2, 3, 4, 8
# s-LWSR_BIX2_P48, input=48x48, output=96x96
#LOG=./../experiment/s-LWSR_BIX2_P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model LWSR --save s-LWSR_BIX2_P48 --scale 2 --n_feats 32  --reset --chop --save_results --print_model --patch_size 96 2>&1 | tee $LOG

# s-LWSR_BIX3_P48, input=48x48, output=144x144
#LOG=./../experiment/s-LWSR_BIX3_P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model LWSR --save s-LWSR_BIX3_P48 --scale 3 --n_feats 32  --reset --chop --save_results --print_model --patch_size 144 2>&1 | tee $LOG

# s-LWSR_BIX4_P48, input=48x48, output=192x192
LOG=./../experiment/s-LWSR_BIX4_P48-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model LWSR --save s-LWSR_BIX4_P48 --scale 4  --n_feats 32  --reset --chop --save_results --print_model --patch_size 192 2>&1 | tee $LOG

# s-LWSR_BIX8_P48, input=48x48, output=384x384
#LOG=./../experiment/s-LWSR_BIX8_P48-`date +%Y-%m-%d-%H-%M-%S`.txt
#CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save s-LWSR_BIX8_P48 --scale 8  --n_feats 32  --reset --chop --save_results --print_model --patch_size 384 2>&1 | tee $LOG

