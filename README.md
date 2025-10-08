# SFDA-CCRC
The official implementation of SFDA-CCRC.

## Requirements
- python 3.8.0
- pytorch 1.10.0


## Running main experiment
- Download datasets.
- You can train the source domain model weights by yourself or use the provided source domain model weights
```shell
python train_sourceWhs.py
```
- Run ./train_target.py to start the SFDA training process.
```shell
python train_targetWhs.py
```
> Note: Modify the relevant paths and parameters

## Related Dataset

Heart Dataset and details: We used the preprocessed dataset from Dou et al. : https://github.com/carrenD/Medical-Cross-Modality-Domain-Adaptation. 

The data is in tfs records, It is recommended to convert to nii format or npy format.

## Data scheme
Take the CT dataset as an example
```
data
    ct/
        train/
            IMG/
                ctslice0_1.npy
                ctslice1_1.npy
                ctslice2_1.npy
                ...
            GT/
                ctslice0_1.npy
                ctslice1_1.npy
                ctslice2_1.npy
                ...
        test/
            IMG/
                ctslice1003_0.nii
                ctslice1003_1.nii
                ctslice1003_2.nii
                ...
            GT/
                ctslice1003_0.nii
                ctslice1003_1.nii
                ctslice1003_2.nii
                ...
        ...
```

## Acknowledgement 确认
This repo benefits from [SFDA-CBMT](https://github.com/lloongx/SFDA-CBMT). Thanks for their wonderful works.
