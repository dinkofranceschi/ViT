# Deformable ViT
Applying deformable multiheadattention to ViT architecture


To-do list:

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
        - [ ] Implement the masking
        - [ ] Testing 
- [ ] Code and test disentangled attention
    - [ ] Transformer
        - [x] Making it compatible
        - [ ] Adjusting the positions
        - [ ] Testing   
    - [ ] Performer
        - [ ] Evaluate compatibility
        - [ ] Adapt it
        - [ ] Testing  
- [ ] Usage

# Questions

The locality-aware initialization produces masks like the following:

![LAI_MASK](https://github.com/dinkofranceschi/ViT/blob/main/figures/lai_init_mask.png)

How to implement it in Performers FAVOR+?

# Results

Comparison between Locality-aware init vit with regular vit:

![vit_lai](https://github.com/dinkofranceschi/ViT/blob/main/figures/performance_lai.png)

# Usage


# Links

- https://github.com/fundamentalvision/Deformable-DETR (deformable multscaleattention)
- https://github.com/microsoft/DeBERTa (disentangled attention)
- https://github.com/VITA-Group/TransGAN (locality aware initialization)
- https://github.com/google-research/google-research/tree/master/performer (performer tensorflow implementation)
- https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py (timm library)
- https://wandb.ai/ltononro/Deformable%20ViT (wandb logging)
