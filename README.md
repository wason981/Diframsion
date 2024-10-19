# Diframsion
Video Frame Generation based on Image and Text

## Requirements
* Linux is recommended for performance and compatibility reasons.
* 64-bit Python 3.8 installation. We recommend Anaconda3 with numpy 1.21 or newer.
* We recommend Pytorch 2.0.1, which we used for all experiments in the paper.

## 1. Prepare the datasets
You need to download the pretrained [SD v1.4](https://huggingface.co/CompVis/stable-diffusion-v1-4) in the `./pretrained/`.
```bash
python sample_scripts/sample.py --config sample_scripts/configs/panda.yaml
```

## 2. Inference
You need to use put the image in the `./sample` and segment use [Gounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) to segment the subject.

## Training
Our training process divided into coarse-to-fine manner.
### stage 1:
```bash
srun --mpi=pmi2 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29125 train_stage1.py \
--model TAVU \
--num-frames 16 \
--dataset WebVideoImageStage1  \
--frame-interval 4 \
--ckpt-every 1000 \
--clip-max-norm 0.1 \
--global-batch-size 16 \
--reg-text-weight 0 \
--results-dir ./results \
--pretrained-t2v-model path-to-t2v-model \
--global-mapper-path path-to-elite-global-model
```

###stage 2:
```bash
srun --mpi=pmi2 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29125 train_stage2.py \
--model TAVU \
--num-frames 16 \
--dataset WebVideoImageStage2  \
--frame-interval 4 \
--ckpt-every 1000 \
--clip-max-norm 0.1 \
--global-batch-size 16 \
--reg-text-weight 0 \
--results-dir ./results \
--pretrained-t2v-model path-to-t2v-model \
--global-mapper-path path-to-stage1-model
```
