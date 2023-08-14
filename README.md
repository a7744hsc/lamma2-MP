# Llama 2 MP(Model Parallel)

This is a folk of [LLaMA](https://github.com/facebookresearch/llama). The purpose of this folk is to run the smallest 7B model on two 8GB GPUs(e.g. 2*2080 8GB).

## How to run
1. Get the model file following the original Repo's instruction.
2. Install the dependencies.
3. Run the codes in `simple-example.py` line by line.

## What did I do
1. Use simple torch layers replace the fairscale's complex layers.
2. Initialize the model on two GPUs. `BLOCKS_IN_GPU0 ` is used to control how the model is split.
3. Minor changes in `generation.py` to move model's output to GPU0(their are some operations in `generation.py` that need be done in GPU0).