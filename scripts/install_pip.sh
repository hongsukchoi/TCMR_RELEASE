#!/usr/bin/env bash

sudo apt install python3.7-venv
echo "Creating virtual environment"
python3.7 -m venv tcmr-env
echo "Activating virtual environment"

source $PWD/tcmr-env/bin/activate

$PWD/tcmr-env/bin/pip install numpy==1.17.5 torch==1.4.0 torchvision==0.5.0
$PWD/tcmr-env/bin/pip install git+https://github.com/giacaglia/pytube.git --upgrade
$PWD/tcmr-env/bin/pip install -r requirements.txt
