# LANet
Pytorch codes for ['LANet: Local Attention Embedding to Improve the Semantic Segmentation of Remote Sensing Images'](https://ieeexplore.ieee.org/document/9102424)

![alt text](https://github.com/ggsDing/LANet/blob/master/Overview.png)

**How to Use**
1. Split the data into training, validation and test set and organize them as follows:

>YOUR_DATA_DIR
>  - Train
>    - image
>    - label
>  - Val
>    - image
>    - label
>  - Test
>    - image
>    - label

2. Change the training parameters in *train_PD.py*, especially the data directory.

3. To evaluate, change also the parameters in *eval_PD.py*, especially the data directory and the checkpoint path.


If you find this work useful, please consider to cite:

>'Ding L, Tang H, Bruzzone L. LANet: Local Attention Embedding to Improve the Semantic Segmentation of Remote Sensing Images[J]. IEEE Transactions on Geoscience and Remote Sensing, 2020.'
