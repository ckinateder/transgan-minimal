# TransGAN Reimplementation

This is a reimplementation of TransGAN.

## Installation

Before running ```train.py```, check whether you have libraries in ```requirements.txt```! Also, create ```./fid_stat``` folder and download the [fid_stats_cifar10_train.npz](http://bioinf.jku.at/research/ttur/ttur_stats/fid_stats_cifar10_train.npz) file in this folder. To save your model during training, create ```./checkpoint``` folder using ```mkdir checkpoint```.

## Training 

```bash
python train.py
```
## Pretrained Model

You can find pretrained model [here](https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view). You can download using:

```bash
wget https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view
```
or 

```bash
curl gdrive.sh | bash -s https://drive.google.com/file/d/134GJRMxXFEaZA0dF-aPpDS84YjjeXPdE/view
```

## License

MIT
