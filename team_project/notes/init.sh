#!/bin/bash
sudo yum install git -y
git clone https://github.com/torch/distro.git ~/torch --recursive
cd ~/torch;
bash install-deps;
./install.sh -b;