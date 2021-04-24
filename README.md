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

## Questions

The locality-aware initialization produces masks like the following:

![LAI_MASK](https://github.com/dinkofranceschi/ViT/blob/main/figures/lai_init_mask.png)

How to implement it in Performers FAVOR+?

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
