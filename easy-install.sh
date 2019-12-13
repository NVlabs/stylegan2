#!/bin/bash

pip install --user -r requirements.txt

mkdir pkl
cd pkl
gdown --id 1JLqXE5bGZnlu2BkbLPD5_ZxoO3Nz-AvF --output inception_v3_features.pkl #inception: https://drive.google.com/open?id=1JLqXE5bGZnlu2BkbLPD5_ZxoO3Nz-AvF
cd ../

mkdir networks
cd networks
gdown --id 1JLqXE5bGZnlu2BkbLPD5_ZxoO3Nz-AvF --output stylegan2-ffhq-config-f.pkl
cd ../

mkdir datasets