#!/bin/bash

pip install --user -r requirements.txt

mkdir pkl
cd pkl
gdown --id 1JLqXE5bGZnlu2BkbLPD5_ZxoO3Nz-AvF #inception: https://drive.google.com/open?id=1JLqXE5bGZnlu2BkbLPD5_ZxoO3Nz-AvF
cd ../

mkdir networks
cd networks
gdown --id 1UlDmJVLLnBD9SnLSMXeiZRO6g-OMQCA_
cd ../

mkdir datasets

#run a test
python run_generator.py generate-images --network=./networks/stylegan2-ffhq-config-f.pkl --seeds=6600-6625 --truncation-psi=0.5