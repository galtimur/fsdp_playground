export CUDA_VISIBLE_DEVICES=1,2,3
tune run --nproc_per_node 3 \
    full_finetune_distributed \
    --config llama3b.yaml \
    epochs=1