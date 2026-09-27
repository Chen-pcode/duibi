<img width="1650" height="650" alt="image" src="https://github.com/user-attachments/assets/b1fb7f5b-b327-4823-9b05-0ddcee7d6dd6" /><div align="center">
<h1>LMDAU-Net</h1>
<h3>An Effective Lightweight Multi-scale Deformation Aggregation U-Net for Skin Lesion Segmentation</h3>
</div>

<p align="center">
  <img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/Overview.jpg" alt="LMDAU-Net" width="85%" />
</p>


### Abstract
Automatic skin lesion segmentation is a pivotal problem in the medical domain and an indispensable component in the computer-aided diagnosis program. Most convolutional neural network-based segmentation algorithms have demonstrated promising performance due to their ability to encode detail and semantic features efficiently. However, they fail to capture the long-range contextual information at the global level. Therefore, researchers employ Transformer architecture to address this issue. Unfortunately, these methods fail to learn sufficient pixel information at the local level. Motivated by this, some researchers attempt to design a hybrid architecture based on CNN and Transformer. However, the large number of parameters and high computational cost make them challenging to train and use. To alleviate these problems, we propose an effective Lightweight Multi-scale Deformation Aggregation U-Net (LMDAU-Net), which consists of a Lightweight Local-global Learning Module (LLM) and an Adaptive Interactive Fusion Module (AIF). Specifically, we utilize the two branches of the proposed LLM to efficiently learn local fine-grained and global coarse-grained features that assist the model in capturing the complementary feature representations. Moreover, we employ the AIF to selectively learn semantic and detail features at different scales, which can dynamically explore variable feature cues. Extensive experiments on four skin benchmarks, including ISIC2016, ISIC 2017, ISIC2018, and PH2, demonstrate that LMDAU-Net achieves state-of-the-art performance in both qualitative and quantitative aspects.

## Main Results

- ISIC_2016 DataSet 
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/ISIC_2016_table.png" width="60%" />
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/ISIC_2016.png" width="60%" />

- ISIC_2017 DataSet
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/ISIC_2017_table.png" width="60%" />
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/ISIC2017.jpg" width="60%" />

- ISIC_2018 DataSet
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/ISIC_2018_table.png" width="60%" />
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/ISIC2018.jpg" width="60%" />

- PH2 DataSet
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/PH2_table.png" width="60%" />
<img src="https://github.com/Lm0611/LMDAU-Net/blob/main/asset/PH2.jpg" width="60%" />

**0. Main Environments.** </br>
The environment installation procedure can be followed by [VM-UNet](https://github.com/JCruan519/VM-UNet), or by following the steps below (python=3.8):</br>

```
conda create -n vmunet python=3.8
conda activate vmunet
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0  # causal_conv1d-1.0.0+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install mamba_ssm==1.0.1  # mmamba_ssm-1.0.1+cu118torch1.13cxx11abiFALSE-cp38-cp38-linux_x86_64.whl
pip install scikit-learn matplotlib thop h5py SimpleITK scikit-image medpy yacs
```

**1. Datasets.** </br>
Data preprocessing environment installation (python=3.7):
```
conda create -n tool python=3.7
conda activate tool
pip install h5py
conda install scipy==1.2.1  # scipy1.2.1 only supports python 3.7 and below.
pip install pillow
```

*A. ISIC2016* </br>

1. Download the ISIC 2016 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic16/`. </br>
2. Run `Prepare_ISIC2016.py` for data preparation and dividing data to train, validation and test sets. </br>

*B. ISIC2017* </br>

1. Download the ISIC 2017 train dataset from [this](https://drive.google.com/file/d/1XM10fmAXndVLtXWOt5G0puYSQyI2veWy/view) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic17/`. </br>
2. Run `Prepare_ISIC2017.py` for data preparation and dividing data to train, validation and test sets. </br>

*C. ISIC2018* </br>

1. Download the ISIC 2018 train dataset from [this](https://challenge.isic-archive.com/data) link and extract both training dataset and ground truth folders inside the `/data/dataset_isic18/`. </br>
2. Run `Prepare_ISIC2018.py` for data preparation and dividing data to train, validation and test sets. </br>

*D. PH<sup>2</sup>* </br>

1. Download the PH<sup>2</sup> dataset from [this](https://www.dropbox.com/s/k88qukc20ljnbuo/PH2Dataset.rar) link and extract both training dataset and ground truth folders inside the `/data/PH2/`. </br>
2. Run `Prepare_PH2.py` to preprocess the data and form test sets for external validation. </br>

*E. Prepare your own dataset* </br>

1. The file format reference is as follows. (The image is a 24-bit png image. The mask is an 8-bit png image. (0 pixel dots for background, 255 pixel dots for target))
- './your_dataset/'
  - images
    - 0000.png
    - 0001.png
  - masks
    - 0000.png
    - 0001.png
  - Prepare_your_dataset.py
2. In the 'Prepare_your_dataset.py' file, change the number of training sets, validation sets and test sets you want.</br>
3. Run 'Prepare_your_dataset.py'. </br>

**2. Train the LMDAU-Net.**

```
python train.py
```
- After trianing, you could obtain the outputs in './results/' </br>

**3. Test the LMDAU-Net.**  
First, in the test.py file, you should change the address of the checkpoint in 'resume_model'.

```
python test.py
```
- After testing, you could obtain the outputs in './results/' </br>

