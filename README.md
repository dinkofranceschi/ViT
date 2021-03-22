# Deformable ViT
Applying deformable multiheadattention to ViT architecture


To-do list:

- [x] Finishing the logging code and wandb logging
- [x] Implementing timm versions (performer and transformer for 224x224 16-patch-size images) 
- [ ] Code and test deformable attention
    - [ ] Transformer
    - [ ] Performer
- [ ] Code and test locality-aware initialization
    - [x] Transformer
    - [ ] Performer
- [ ] Code and test DeBERTa attention
    - [ ] Transformer
    - [ ] Performer
- [ ] Usage

# Questions

The locality-aware initialization produces masks like the following:

![LAI_MASK](https://github.com/dinkofranceschi/ViT/blob/main/figures/lai_init_mask.png)

How to implement it in Performers FAVOR+?

-Idea: Break the matrix Q and K in four chunks, apply the diagonal mask of performer in the left lower and right upper chunks

$\begin{matrix}
1&2&3
\end{matrix}$
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
