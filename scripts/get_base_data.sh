#!/usr/bin/env bash

mkdir -p data
cd data
gdown "https://drive.google.com/uc?id=1WmSYmDWdMSXyNU6ZcFQVF8ULtR_S9L3b&export=download&confirm=t"
unzip base_data.zip
rm base_data.zip
cd ..
mv data/base_data/demo.mp4 .
mkdir -p $HOME/.torch/models/
mv data/base_data/yolov3.weights $HOME/.torch/models/
