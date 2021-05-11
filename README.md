# Deformable ViT
Applying deformable multiheadattention to ViT architecture


## To-do list:

- [x] Finishing the logging code and wandb logging
- [x] Implementing timm versions (performer and transformer for 224x224 16-patch-size images) 
- [ ] Code and test deformable attention
    - [ ] Transformer
        - [ ] Convert from C to Python? 
    - [ ] Performer
        - [ ] Evaluate compatibility
        - [ ] Adapt it
        - [ ] Testing 
- [ ] Code and test locality-aware initialization
    - [x] Transformer
        - [x] Implemented
        - [x] Testing
    - [ ] Performer
        - [ ] Implement the masking (Krzysztof)
        - [ ] Testing 
    - [ ] Trainable deformable LAI?
- [ ] Code and test disentangled attention
    - [x] Transformer
        - [x] Making it compatible
        - [x] Adjusting the positions
        - [x] Testing   
    - [ ] Performer
        - [ ] Evaluate compatibility
        - [ ] Adapt it
        - [ ] Testing  
- [ ] Usage
- [x] Define #GPUS and GPU
    - [x] Define models we are going to train
    - [x] Estimate time and cost for training ImageNet and transfer CIFAR
- [ ] Research optimal traninig schedules for ImageNet and transfer CIFAR(e.g. cosine annealing)

## Questions

The locality-aware initialization produces masks like the following:

![LAI_MASK](https://github.com/dinkofranceschi/ViT/blob/main/figures/lai_init_mask.png)

How to implement it in Performers FAVOR+?

## Experiments

Models description:
| Models      | Layers      | Hidden size | MLP size | Heads | Params (M) |
| ------------- |:-------------:| -----:|-----:|-----:|-----:|
| ViT-S/16[1]     | 12 | 768 | 3072 | 12| 48.6|
| ViT-B/16[1]      | 24    | 1024    | 4096| 16| 86.4|
| ViT-L/16[1] | 32  | 1280 | 5120| 16| 304.3|

Training on ImageNet from scratch:
| Models      | Top1-Acc(%)        | Params (M) |
| ------------- |:-------------:| -----:|
| ViT-S/16[1]     | 78.1 | 48.6 |
| ViT-B/16[1]      | 79.8    |   86.4 |
| ViT-L/16[1] | 81.1  | 304.3 |
| ViT-LAI-S |X|X|
| ViT-LAI-B |X|X|
| ViT-LAI-L |X|X|
| ViT-Dis-S |X|X|
| ViT-Dis-B |X|X|
| ViT-Dis-L |X|X|
| ViT-DisLAI-S |X|X|
| ViT-DisLAI-B |X|X|
| ViT-DisLAI-L |X|X|


Training schedule

## Results

Comparison between Locality-aware init vit with regular vit:

#### CIFAR10 from scratch (Locality aware init vs Disentangled vs Regular ViT)
![vit_cifar10](https://github.com/dinkofranceschi/ViT/blob/main/figures/cifar10.png)
#### CIFAR100 from scratch (Locality aware init vs Disentangled vs Regular ViT)
![vit_cifar100](https://github.com/dinkofranceschi/ViT/blob/main/figures/cifar100.png)
## Usage


## Links


- [Deformable MultiScale Attention](https://github.com/fundamentalvision/Deformable-DETR)
- [Disentangeld Attention](https://github.com/microsoft/DeBERTa)
- [Locality-aware initialization](https://github.com/VITA-Group/TransGAN)
- [Performer tensorflow](https://github.com/google-research/google-research/tree/master/performer)
- [Timm library](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py )
- [Wandb logging](https://wandb.ai/ltononro/Deformable%20ViT)


## References

[1] [An image is worth 16x16 words:
Transformers for image recognition at scale](https://arxiv.org/abs/2010.11929)
