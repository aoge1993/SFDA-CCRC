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

For the cardiac dataset, we utilized the preprocessed version provided by SIFA, which is available at their project repository: https://github.com/cchen-cc/SIFA.

The data is in tfs records, It is recommended to convert to npy format and nii format.

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

## Acknowledgement
This repo benefits from [SFDA-CBMT](https://github.com/lloongx/SFDA-CBMT). Thanks for their wonderful works.

## Citation
If you find this work is helpful in your research, please cite:
```
@inproceedings{ma2026source,
  title={Source-Free Domain Adaptation for Cross-Modality Cardiac Image Segmentation with Contrastive Class Relationship Consistency},
  author={Ma, Ao and Zhu, Qingpeng and Li, Jingjing and Nielsen, Mads and Chen, Xu},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2025},
  pages={574--583},
  year={2026},
  publisher={Springer Nature Switzerland},
  address={Cham}
}
```