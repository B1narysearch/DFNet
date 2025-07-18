# Dual-Domain Feature Fusion: Enhancing Infrared Small Target Detection through Spatial-Frequency Synergy

## Notice
The code repository is directly associated with the manuscriptDual-Domain Feature Fusion: Enhancing Infrared Small Target Detection through Spatial-Frequency Synergy that has been submitted to The Visual Computer journal. This implementation constitutes an integral part of the research presented in the submitted work. Readers are kindly requested to cite this manuscript when referencing or utilizing the code in related research.

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

## Dataset Info
```
DFNet
    dataset
        IRSTD-1k
            ...
        NUDT
            ...
```

