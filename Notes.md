IN WSL 2 ubuntu

cd ~
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion/docker

CAN EDIT DOCKERFILE: before build - repalce some lines with this information
Miniconda3-py38_23.11.0-2-Linux-x86_64.sh

replace line 15/16
RUN conda install -y --override-channels -c conda-forge python=3.8
RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

docker build . -t bevfusion

DOWNLOADED DATA FROM NUSCENES.ORG
mkdir -p ~/data/nuscenes
cd ~/data/nuscenes
tar -xzf /mnt/c/Users/Christian/Downloads/v1.0-mini.tgz
unzip /mnt/c/Users/Christian/Downloads/can_bus.zip
unzip /mnt/c/Users/Christian/Downloads/nuScenes-map-expansion-v1.3.zip -d maps


STARTS UP DOCKER CONTAINER

docker run --gpus all -it --shm-size 16g \
  -v ~/bevfusion:/workspace/bevfusion \
  -v ~/data/nuscenes:/dataset/nuscenes \
  bevfusion


IN BEVFUSION DOCKER CONTAINER RUN THESE COMMANDS

cd /workspace/bevfusion
mkdir -p data && ln -s /dataset/nuscenes data/nuscenes
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
python setup.py develop
./tools/download_pretrained.sh


HAD TO COMMENT OUT RADAR LINES: mmdet3d/models/backbones/radar_encoder.py	

python tools/create_data.py nuscenes --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini

torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox