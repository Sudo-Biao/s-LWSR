# s-LWSR: Super Lightweight Super-Resolution Network
This is the code of the paper in following:

[Biao Li](https://github.com/Sudo-Biao), [Jiabin Liu](https://github.com/liujiabin008), [Bo Wang](http://it.uibe.edu.cn/szdw/dsjkxyjzx/50452.htm), [Zhiquan Qi](https://github.com/qizhiquan) and [Yong Shi](http://www.feds.ac.cn/index.php/zh-cn/zxjs/zxld/1447-sy)"s-LWSR: Super Lightweight Super-Resolution Network", [[arXiv]](https://arxiv.org/abs/1909.10774) 


The code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN(Pytorch)](https://github.com/yulunzhang/RCAN), and tested on Ubuntu 18.04 environment (Python3.7, PyTorch_1.0) with Titan Xp GPU. 

## Contents
1. [Introduction](#introduction)
2. [Train](#train)
3. [Test](#test)
4. [Results](#results)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)

## Introduction
Deep learning (DL) architectures for superresolution (SR) normally contain tremendous parameters, which has been regarded as the crucial advantage for obtaining satisfying performance. However, with the widespread use of mobile phones for taking and retouching photos, this character greatly hampers the deployment of DL-SR models on the mobile devices. To address this problem, in this paper, we propose a super lightweight SR network: s-LWSR. There are mainly three contributions in our work. Firstly, in order to efficiently abstract features from the low resolution image, we build an information pool to mix multi-level information from the first half part of the pipeline. Accordingly, the information pool feeds the second half part with the combination of hierarchical features from the previous layers. Secondly, we employ a compression module to further decrease the size of parameters. Intensive analysis confirms its capacity of trade-off between model complexity and accuracy. Thirdly, by revealing the specific role of activation in deep models, we remove several activation layers in our SR model to retain more information for performance improvement. Extensive experiments show that our s-LWSR, with limited parameters and operations, can achieve similar performance to other cumbersome DL-SR methods.


## Train
### Prepare training data 

Our  experiments  are  similar  as  RCAN:
1. Download DIV2K training data (800 training + 100 validtion images) from [DIV2K dataset](https://data.vision.ee.ethz.ch/cvl/DIV2K/) and put in the file DIV2K.

2. Carefully check the dir of HR  and  LR images following the option file. Moreover, '--ext' of  option.py is set as 'sep_reset', which firstly convert .png to .npy. If all the training images (.png) are converted to .npy files, then set '--ext sep' to skip converting files.

For more informaiton, please refer to [EDSR(PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN(Pytorch)](https://github.com/yulunzhang/RCAN).

### Begin to train

1. Cd to 'Train/code', run the following scripts to train models.

    **You can use scripts in file 'Train' to train models as paper. If you want to  more  about  our  model  setting,  you  can  check  in  the  model  folder..**

    ```bash

    BI, scale 2, 3, 4, 8
    #s-LWSR_BIX2_P48, input=48x48, output=96x96
    CUDA_VISIBLE_DEVICES=0 python main.py --model LWSR --save s-LWSR_BIX2_P48 --scale 2 --n_feats 32  --reset --chop --save_results --print_model --patch_size 96 2>&1 | tee $LOG

    #s-LWSR_BIX3_P48, input=48x48, output=144x144
    CUDA_VISIBLE_DEVICES=0 python main.py --model LWSR --save s-LWSR_BIX3_P48 --scale 3 --n_feats 32  --reset --chop --save_results --print_model --patch_size 144 2>&1 | tee $LOG

    #s-LWSR_BIX4_P48, input=48x48, output=192x192
    CUDA_VISIBLE_DEVICES=0 python main.py --model LWSR --save s-LWSR_BIX4_P48 --scale 4  --n_feats 32  --reset --chop --save_results --print_model --patch_size 192 2>&1 | tee $LOG

    #s-LWSR_BIX8_P48, input=48x48, output=384x384
    CUDA_VISIBLE_DEVICES=0 python main.py --model RCAN --save s-LWSR_BIX8_P48 --scale 8  --n_feats 32  --reset --chop --save_results --print_model --patch_size 384 2>&1 | tee $LOG

    ```

## Test
### Quick start
1. Download our  pre-trained  models [s-LWSR(PyTorch)](https://drive.google.com/drive/folders/11eqKn1PsLXRtbrbh_LhJ7WxHtU8Ih2ym) and place them in '/Test/model'. Please be make sure that the code and its corresponding pre-trained model are consistant, because there are several different settings contained in our files. 

    We just  train  our  model on X4 task and  more  information  will  be released soon.

2. Cd to '/Test/code', run the following scripts.

    **You can use scripts in file 'Test' to produce results for our paper.**

    ```bash
    # No self-ensemble: RCAN
    # BI degradation model, X2, X3, X4, X8
    #s-LWSR_BIX2
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    #s-LWSR_BIX3
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    #s-LWSR_BIX4
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5				
    #RCAN_BIX8
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --save 'LWSR' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    
    #s-LWSRplus_BIX2
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 2 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    #s-LWSRplus_BIX3
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 3 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    #s-LWSRplus_BIX4
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 4 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    #s-LWSRplus_BIX8
    CUDA_VISIBLE_DEVICES=0 python main.py --data_test MyImage --scale 8 --model LWSR --n_feats 32 --pre_train ../model/model_latest.pt --test_only --save_results --chop --self_ensemble --save 'LWSRplus' --testpath /home/li/桌面/s-LWSR/Test/LR/LRBI --testset Set5
    ```
### The whole test pipeline
1. Prepare test data.

    Download the standard test sets (our  model using Set5, Set14, BSD100, and  Urban100. All test sets are available from [GoogleDrive](https://drive.google.com/drive/folders/1xyiuTr6ga6ni-yfTP7kyPHRmfBakWovo?usp=sharing) or [Baidu](https://pan.baidu.com/s/1yBI_-rknXT2lm1UAAB_bag)) in 'OriginalTestData'.

2. Conduct image SR. 

    See **Quick start**
3. Evaluate the results  PSNR and  SSIM.

    Run 'Evaluate_PSNR_SSIM.m' to obtain PSNR/SSIM values for paper.



## Results
### Quantitative Results
![Visual_PSNR_SSIM](/Figs/Cont1.PNG)
![Visual_PSNR_SSIM](/Figs/Cont2.PNG)
![Visual_PSNR_SSIM](/Figs/Cont3.PNG)
Quantitative conparing results.All images are chosen from four mentioned test datasets.
###Model structure
![Model structure](/Figs/stru.PNG)
The figure of our proposed s-LWSR.
## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@InProceedings{Lim_2017_CVPR_Workshops,
  author = {Lim, Bee and Son, Sanghyun and Kim, Heewon and Nah, Seungjun and Lee, Kyoung Mu},
  title = {Enhanced Deep Residual Networks for Single Image Super-Resolution},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
  month = {July},
  year = {2017}
}

@inproceedings{zhang2018rcan,
    title={Image Super-Resolution Using Very Deep Residual Channel Attention Networks},
    author={Zhang, Yulun and Li, Kunpeng and Li, Kai and Wang, Lichen and Zhong, Bineng and Fu, Yun},
    booktitle={ECCV},
    year={2018}
}

@article{li2019s,
  title={s-LWSR: Super Lightweight Super-Resolution Network},
  author={Li, Biao and Liu, Jiabin and Wang, Bo and Qi, Zhiquan and Shi, Yong},
  journal={arXiv preprint arXiv:1909.10774},
  year={2019}
}
```
## Acknowledgements
This code is built on [EDSR (PyTorch)](https://github.com/thstkdgus35/EDSR-PyTorch) and [RCAN(Pytorch)](https://github.com/yulunzhang/RCAN). We greatly thank the authors for sharing their codes of EDSR [Torch version](https://github.com/LimBee/NTIRE2017) and [RCAN(Pytorch)](https://github.com/yulunzhang/RCAN).

