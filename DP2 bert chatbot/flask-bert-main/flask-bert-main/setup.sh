#!/bin/bash
sudo apt update && sudo apt upgrade -Y
sudo apt-get install -y python3 
sudo apt-get install -y python3-pip
sudo apt-get install -y libmysqlclient-dev
sudo apt install python3-flask

cd app
pip install gdown
gdown https://drive.google.com/uc?id=1-MoqQMRdVWLb3GqNDxIvAwQTjlM2kGas

pip3 install torch==1.5.1+cpu torchvision==0.6.1+cpu -f https://download.pytorch.org/whl/torch_stable.html

pip3 install pytorch-lightning
pip3 install transformers
pip3 install tensorflow-text
pip3 install tensorflow-hub
pip3 install py-eureka-client
pip3 install flask
pip3 install flask-restx