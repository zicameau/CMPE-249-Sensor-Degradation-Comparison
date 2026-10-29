IN WSL 2 ubuntu

cd ~
git clone https://github.com/mit-han-lab/bevfusion.git
cd bevfusion/docker
docker build . -t bevfusion

mkdir -p ~/data/nuscenes
cd ~/data/nuscenes
tar -xzf /mnt/c/Users/Christian/Downloads/v1.0-mini.tgz
unzip /mnt/c/Users/Christian/Downloads/can_bus.zip
unzip /mnt/c/Users/Christian/Downloads/nuScenes-map-expansion-v1.3.zip -d maps

docker run --gpus all -it --shm-size 16g \
  -v ~/bevfusion:/workspace/bevfusion \
  -v ~/data/nuscenes:/dataset/nuscenes \
  bevfusion

cd /workspace/bevfusion
mkdir -p data && ln -s /dataset/nuscenes data/nuscenes
export TORCH_CUDA_ARCH_LIST="8.6+PTX"
python setup.py develop
./tools/download_pretrained.sh

python tools/create_data.py nuscenes --root-path ./data/nuscenes \
  --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini

torchpack dist-run -np 1 python tools/test.py \
  configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser.yaml \
  pretrained/bevfusion-det.pth --eval bbox