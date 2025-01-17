### Build docker file

`docker build -t fsdp_playground .`

### Run container

increase `shm-size` to enable nvcc distributed training.

`docker run --rm -it --gpus all --shm-size=1g fsdp_playground`

### Run train

`poetry run torchrun --nnodes 1 --nproc_per_node 8 fsdp_playground.py`