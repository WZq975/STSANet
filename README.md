# STSANet
[Spatio-Temporal Self-Attention Network for Video Saliency Prediction](https://ieeexplore.ieee.org/document/9667292)

## Weights & Results
Quantitative Results can be found on [DHF1K Video Saliency Benchmark](https://mmcheng.net/videosal/).

Qualitative Results and Weights of STSANet can be downloaded from:

Baidu Netdisk: https://pan.baidu.com/s/1eTrkmPV65eosgyiJtbKKrw (code: 5vds)

Google Drive: uploading...

## Training
STSANet model uses [S3D](https://github.com/kylemin/S3D) backbone pretrained on Kinetics dataset.
Pretrained S3D backbone: 

Baidu Netdisk: https://pan.baidu.com/s/1z99-PANs_9ZyXAjl672OAA (code: n2il)

Google Drive: uploading

```
$ python train.py --help
usage: train.py [-h] [--lr LR] [--ds DS] [--pd PD]

optional arguments:
  -h, --help  show this help message and exit
  --lr LR     initial learning rate
  --ds DS     dataset (DHF1K, Hollywood-2, UCF, DIEM)
  --pd PD     path of dataset

```

## Citation
```
@ARTICLE{wang2021STSANet,
  author={Wang, Ziqiang and Liu, Zhi and Li, Gongyang and Wang, Yang and Zhang, Tianhong and Xu, Lihua and Wang, Jijun},
  journal={IEEE Transactions on Multimedia (Early Access)}, 
  title={Spatio-Temporal Self-Attention Network for Video Saliency Prediction}, 
  year={2021},
  doi={10.1109/TMM.2021.3139743}
}

```
## Acknowledgements
Some codes (Metrics.py) are borrowed from:
https://github.com/samyak0210/saliency
