### Build docker file

`docker build -t fsdp_playground .`

### Run container
`docker run --rm -it --gpus all --shm-size=1g -e fsdp_playground`