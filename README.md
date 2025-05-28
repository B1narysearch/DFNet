# Dual-Domain Feature Fusion: Enhancing Infrared Small Target Detection through Spatial-Frequency Synergy

## Training
The training command is very simple like this:
```
python main --dataset-dir --batch-size --epochs --lr --mode 'train'
```

For example:
```
python main.py --dataset-dir '/dataset/IRSTD-1k' --batch-size 4 --epochs 400 --lr 0.05 --mode 'train'
```

## Testing
You can test the model with the following command:
```
python main.py --dataset-dir '/dataset/IRSTD-1k' --batch-size 4 --mode 'test' --weight-path '/weight/MSHNet_weight.tar'
```


## Quantative Results
| Dataset         | mIoU (x10(-2)) | Pd (x10(-2))|  Fa (x10(-6)) | Weights|
| ------------- |:-------------:|:-----:|:-----:|:-----:|
| IRSTD-1k | 68.27 | 92.34 | 8.42 | [IRSTD-1k_weights](https://drive.google.com/file/d/1Hg97nCqHJfqDIo0EbBYsGzbBGH_xfZoz/view?usp=drive_link) |
| NUDT-SIRST | 84.59 | 96.97 | 4.06 | [NUDT-SIRST_weights](https://drive.google.com/file/d/1xoW9j7RV4N75FOnPMeQe7cEKVw5IeU0z/view?usp=drive_link) |


## Citation
**Please kindly cite the papers if this code is useful and helpful for your research.**

    @inproceedings{liu2024infrared,
      title={Infrared Small Target Detection with Scale and Location Sensitivity},
      author={Liu, Qiankun and Liu, Rui and Zheng, Bolun and Wang, Hongkui and Fu, Ying},
      booktitle={Proceedings of the IEEE/CVF Computer Vision and Pattern Recognition},
      year={2024}
    }
