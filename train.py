import torch
import torch.nn as nn
from model import VisionTransformer, Transformer, Performer
import torch.optim as optim
import torchvision
from tqdm import tqdm
import argparse
from pathlib import Path
import time
import datetime
import json
import os
from timm.loss import LabelSmoothingCrossEntropy
import timm
import wandb
import time
import random
import numpy as np
from utils.ops import AverageMeter,accuracy,init_distributed_mode,get_rank
from timm.utils import ApexScaler

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False
    
def get_args_parser():
    parser = argparse.ArgumentParser('Deformable ViT',add_help=False)
    '''Training parameters'''
    parser.add_argument('--lr',default=1e-4,type=float,
                        help='initial learning rate')
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--epochs', default=100,type=int)
    parser.add_argument('--gamma_lr',default=0.9,type=float,
                        help='gamma in the step learning rate')
    parser.add_argument('--step_lr',default=70,type=float,
                        help='peridicioty of lr decay by gamma')    
    parser.add_argument('--clip_norm', default=1, type = float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoothing',default=0.1,type=float,
                        help='smoothing classification labels')
    parser.add_argument('--dataset',default='MNIST',type=str,
                        help='MNIST,CIFAR100,CIFAR10,ImageNet...')
    parser.add_argument('--locality_aware_init', nargs='+', default=None, type=int,
                        help='List of epochs for locality aware initialization, it only works for timm models or with performers')
    parser.add_argument('--model',default=None,type=str,
                        help='Types of model: ViT-Small, ViT-Base, ViT-Large')
    parser.add_argument('--dataparallel', default=None, type=str,
                        help='GPU Indexes')
    parser.add_argument('--num_workers', default=2,type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--distributed', action='store_false')
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    '''Model parameters'''
    parser.add_argument('--attention',default='performer',type=str,
                        help='Type of attention among Performer,Deformable Transformer...')
    parser.add_argument('--de_attn_type', default='c2p|p2c', type=str,
                        help='Type of disentangled attention')
    parser.add_argument('--patch_size',default=4,type=int)
    parser.add_argument('--num_layers',default=4,type=int,
                        help='number of encoders')
    parser.add_argument('--embed_dim',default=384,type=int,
                        help='embedding dimension of transformers')
    parser.add_argument('--dim_feedforward', default = 1536, type=int,
                        help='feedforward mlp dimension in transformer')
    parser.add_argument('--num_heads',default=6,type=int,
                        help='number of heads for transformers')
    parser.add_argument('--num_orf',default = 16, type=int,
                        help='number of orthogonal random features for performers')
    parser.add_argument('--kernel',default='softmax',type=str,
                        help='type of kernel approximation for performer')
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--pre_trained', action='store_false',
                        help='load pre-trained model trained on ImageNet (not implemented yet)')    
    '''Logging/wandb'''
    parser.add_argument('--wandb_project', default = 'Deformable ViT' ,type=str)
    parser.add_argument('--wandb_entity', default = 'ltononro' ,type=str)
    parser.add_argument('--wandb_group',default = 'ViT', type=str)
    
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--output_dir', default= './outputs/', type=str)
    parser.add_argument('--save_freq', default= 20, type=int)
    parser.add_argument('--saving_name', default= 'ViT', type=str)

    return parser



def build_model(args):
    if args.attention == 'transformer':
        timm_model=False
        transformer=Transformer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
        )
    elif args.attention == 'performer':
        timm_model=False
        transformer = Performer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
        )

    elif args.attention == 'transformer_timm_base':
        timm_model=True
        print('Building ViT_Base timm...')
        model = timm.create_model('vit_base_patch16_224',pretrained=args.pre_trained)
        #Only works for 224x224 datasets (e.g. CIFAR100_224)

    elif args.attention == 'performer_timm_base':
        timm_model=True
        from timm_classes.timm_vision_performer import PerformerAttention
        print('Building ViT_Base+Performer timm...')
        model = timm.create_model('vit_base_patch16_224',pretrained=args.pre_trained)
        for elem in model.blocks:
            attention = elem.attn
            #This module takes the attention and copy its weights with the favor+ module
            elem.attn = PerformerAttention(attention,768,num_heads=attention.num_heads,n_orf=args.num_orf,kernel=args.kernel)
        #Only works for 224x224 datasets (e.g. CIFAR100_224)
    elif args.attention == 'transformer_timm':
        timm_model = True
        print('Building ViT timm...')
        model = timm.models.vision_transformer.VisionTransformer(img_size=args.img_size,
                                                         patch_size=args.patch_size,
                                                         in_chans=args.in_chans,
                                                         num_classes = args.num_classes,
                                                         embed_dim=args.embed_dim,
                                                         depth=args.num_layers,
                                                         num_heads=args.num_heads,
                                                         mlp_ratio = args.dim_feedforward // args.embed_dim,
                                                         attn_drop_rate=args.dropout,
                                                         )
    elif args.attention == 'performer_timm':
        timm_model = True
        from timm_classes.timm_vision_performer import PerformerAttention
        print('Building ViT+performer timm...')
        model = timm.models.vision_transformer.VisionTransformer(img_size=args.img_size,
                                                         patch_size=args.patch_size,
                                                         in_chans=args.in_chans,
                                                         num_classes = args.num_classes,
                                                         embed_dim=args.embed_dim,
                                                         depth=args.num_layers,
                                                         num_heads=args.num_heads,
                                                         mlp_ratio = args.dim_feedforward // args.embed_dim,
                                                         attn_drop_rate=args.dropout,
                                                         )
        
        for elem in model.blocks:
            attention = elem.attn
            #This module takes the attention and copy its weights + adds the favor+ algorithm
            elem.attn = PerformerAttention(attention,args.embed_dim,num_heads=attention.num_heads,n_orf=args.num_orf,kernel=args.kernel,attn_drop=args.dropout)
            
    elif args.attention == 'transformer_timm_lai':
        timm_model = True
        print('Building ViT timm with locality-aware-intialization')
        from timm_classes.timm_locality_aware_transformer import BlockLAI,VisionTransformerLAI
        old_model = timm.models.vision_transformer.VisionTransformer(img_size=args.img_size,
                                                         patch_size=args.patch_size,
                                                         in_chans=args.in_chans,
                                                         num_classes = args.num_classes,
                                                         embed_dim=args.embed_dim,
                                                         depth=args.num_layers,
                                                         num_heads=args.num_heads,
                                                         mlp_ratio = args.dim_feedforward // args.embed_dim,
                                                         attn_drop_rate=args.dropout,
                                                         )
        

        model=VisionTransformerLAI(img_size=args.img_size,
                                   patch_size=args.patch_size,
                                   in_chans=args.in_chans,
                                   num_classes = args.num_classes,
                                   embed_dim=args.embed_dim,
                                   depth=args.num_layers,
                                   num_heads=args.num_heads,
                                   mlp_ratio = args.dim_feedforward // args.embed_dim,
                                   attn_drop_rate=args.dropout,
                                   mask_epochs=list(args.locality_aware_init)
                                                         )
        model.load_state_dict(old_model.state_dict(),strict=False)
            

    elif args.attention == 'disentangled_transformer':
        timm_model=False
        from disentangled_transformer.disentangled_transformer import DisentangledTransformer
        print('Building Disentangled Attention')
        transformer = DisentangledTransformer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
            pos_att_type=args.de_attn_type,
            position_buckets=-1,
            max_relative_positions=-1,
            max_position_embeddings=args.embed_dim,
            relative_attention=True,
        )
        
    elif args.attention == 'disentangled_performer':
        timm_model=False
        from disentangled_transformer.disentangled_transformer import DisentangledTransformer
        print('Building Disentangled Performer Attention')
        transformer = DisentangledTransformer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
            pos_att_type=args.de_attn_type,
            position_buckets=-1,
            max_relative_positions=-1,
            max_position_embeddings=args.embed_dim,
            relative_attention=True,
            performers=True,
            norf=args.num_orf,
            kernel=args.kernel
        )
    elif args.attention == 'disentangled_lai_attention':
        timm_model=False
        from disentangled_transformer.disentangled_transformer import DisentangledTransformer
        print('Building Disentangled LAI Attention')
        transformer = DisentangledTransformer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
            pos_att_type=args.de_attn_type,
            position_buckets=-1,
            max_relative_positions=-1,
            max_position_embeddings=args.embed_dim,
            relative_attention=True,
            lai=True,
            mask_epochs=list(args.locality_aware_init)
        )
    
    elif args.attention == 'deformable_transformer':
        print('Not implemented')
        pass
    elif args.attention == 'deformable_performer':
        print('Not implemented')
        pass
    else:
        print(f'Unknown attention {args.attention}')
        
    if not timm_model:
        print('Building ViT...')
        model = VisionTransformer(img_size=args.img_size,
                              patch_size=args.patch_size,
                              in_chans=args.in_chans,
                              num_classes=args.num_classes,
                              transformer=transformer,
                              shuffle=False)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(args.device)
    
    optimizer= optim.AdamW(model.parameters(),lr=args.lr)

    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=args.step_lr,gamma=args.gamma_lr) 
    
    return model,criterion, optimizer,scheduler


def build_dataset(args):
    ''' Data '''
    if args.dataset == 'MNIST':
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                    #torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1),interpolation=3),
                                    torchvision.transforms.Resize(32,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=args.batch_size, shuffle=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                        torchvision.transforms.Resize(32,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,))
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True)
        
        args.img_size= 32
        args.num_classes=10
        args.in_chans= 1 #number of in channels
    elif args.dataset == 'CIFAR100':
        
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1),interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True,)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True,)
    
        args.img_size = 32
        args.num_classes= 100
        args.in_chans= 3
    elif args.dataset == 'CIFAR10':
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR10('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True,)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True,)
    
        args.img_size = 32
        args.num_classes= 10
        args.in_chans= 3
    elif args.dataset =='ImageNet':                       
          train_data=torchvision.datasets.ImageFolder('./imagenet/train',
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(224,scale=(0.7,1)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ]))
          
          train_sampler= torch.utils.data.RandomSampler(train_data)
          
          train_loader = torch.utils.data.DataLoader(train_data,          
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True,
                                                     sampler=train_sampler,
                                                     prefetch_factor=6)
        
        
          valid_data =  torchvision.datasets.ImageFolder('./imagenet/val',
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(224),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ]))
          
          valid_sampler= torch.utils.data.SequentialSampler(valid_data)
          
          valid_loader = torch.utils.data.DataLoader(valid_data,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,pin_memory=True,
                                                     sampler=valid_sampler,
                                                     prefetch_factor=4)
          
          args.img_size = 224
          args.num_classes = 1000       
          args.in_chans= 3
         
            
         
    elif args.dataset == "ImageNet_32":
          train_data=torchvision.datasets.ImageFolder('./imagenet/train',
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ]))
          
          train_sampler= torch.utils.data.DistributedSampler(train_data)
          
          train_loader = torch.utils.data.DataLoader(train_data,          
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,
                                                     pin_memory=True,
                                                     sampler=train_sampler)
        
        
          valid_data =  torchvision.datasets.ImageFolder('./imagenet/val',
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ]))
          
          valid_sampler= torch.utils.data.DistributedSampler(valid_data)
          
          valid_loader = torch.utils.data.DataLoader(valid_data,
                                                     batch_size=args.batch_size,
                                                     num_workers=args.num_workers,pin_memory=True,
                                                     sampler=valid_sampler)
          
          args.img_size = 32
          args.num_classes = 1000       
          args.in_chans= 3
        
    elif args.dataset == "CIFAR100_224":
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(224,scale=(0.7,1),interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(224,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True)
        
        args.img_size = 224
        args.num_classes = 100
        args.in_chans= 3
        
    if args.dataset == 'MNIST_224':
        train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.ToPILImage(),
                                    torchvision.transforms.Grayscale(3),
                                    torchvision.transforms.Resize(224,interpolation=3),
                                    torchvision.transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.MNIST('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.ToPILImage(),
                                    torchvision.transforms.Grayscale(3),
                                    torchvision.transforms.Resize(224,interpolation=3),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                         (0.1307,), (0.3081,)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,num_workers=args.num_workers,pin_memory=True)
        
        args.img_size= 224
        args.num_classes=10
        args.in_chans= 3 #number of in channels
        
    return train_loader,valid_loader


def training(model,criterion,optimizer,scheduler,train_loader,valid_loader,epochs,clip_norm):
    
    
    seed = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if args.distributed:
        print('Flag distributed data parallel')
        args.device="cuda:{}".format(args.local_rank)
        init_distributed_mode(args)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank])
        #model =  ApexDDP(model, delay_allreduce=True)
        print(f"Using distributed data parallel: {args.distributed}. Using GPU devices {args.device_ids}.")
        
        model.to(args.device)
        criterion.to(args.device)

    elif args.dataparallel:
        print('Setting data parallel')
        args.device='cuda'
        args.device_ids=[int(elem) for elem in args.dataparallel.split(',')]
        print(f"Using GPU devices {args.device_ids}")
        model=nn.DataParallel(model,device_ids=args.device_ids)
    run = wandb.init(project=args.wandb_project,entity=args.wandb_entity,group=args.wandb_group)
    wandb.config.update(args)
    wandb.watch(model)
    
    log={'train_loss':[],
         'train_accuracy':[],
         'val_loss':[],
         'val_accuracy':[],
         'learning_rate':[]}
    max_norm= clip_norm
    model.train()
    
    #print(model)
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_5_accuracy=0
        
        #batch_time = AverageMeter('Time', ':6.3f')
        #data_time = AverageMeter('Data', ':6.3f')
        
        
        #end = time.time()
        
        for data, label in tqdm(train_loader):
            #data_time.update(time.time() - end)
            
            data = data.to(args.device, non_blocking=True)
            label = label.to(args.device, non_blocking=True)
            
            #print(data.shape)
            if args.locality_aware_init is not None:
                output = model(data,epoch)
            else:
                output = model(data)
            #print(output.shape,label.shape)
            loss = criterion(output, label)
    
            optimizer.zero_grad()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            loss.backward()
            optimizer.step()
            acc1,acc5 = accuracy(output,label,topk=(1,5))
            epoch_accuracy += acc1[0] / len(train_loader)
            epoch_loss += loss / len(train_loader)
            #Compute top 5 accuracy
            epoch_5_accuracy += acc5[0] / len(train_loader)
            #batch_time.update(time.time() - end)
            #end=time.time()
            
            #print(data_time,batch_time)
        scheduler.step()
        new_lr=scheduler.get_last_lr()[0] 
        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            epoch_val_5_accuracy = 0
            for data, label in valid_loader:
                data = data.to(args.device)
                label = label.to(args.device)
                if args.locality_aware_init is not None:
                    val_output=model(data,epoch)
                else:    
                    val_output = model(data)
                val_loss = criterion(val_output, label)
                acc1,acc5 = accuracy(val_output,label,topk=(1,5))
                epoch_val_accuracy += acc1[0] / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
                
                #Compute top 5 accuracy
                epoch_val_5_accuracy += acc5[0] / len(valid_loader)
                
        if args.output_dir:
            checkpoint_paths = [Path(args.output_dir) / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch) % args.save_freq==0:
                checkpoint_paths.append(Path(args.output_dir)/ f'checkpoint_{args.saving_name}_{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                save_on_master({
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n -curr_lr: {scheduler.get_last_lr()[0]}"
        )
        log['train_loss']=log['train_loss']+[epoch_loss.item()]
        log['train_accuracy']=log['train_accuracy']+[epoch_accuracy.item()]
        log['val_loss']=log['val_loss']+[epoch_val_loss.item()]
        log['val_accuracy']=log['val_accuracy']+[epoch_val_accuracy.item()]
        log['learning_rate']=log['learning_rate']+scheduler.get_last_lr()
        run.log({"train_loss":epoch_loss.item(),
                   "train_accuracy":epoch_accuracy.item(),
                   "train_5_accuracy":epoch_5_accuracy.item(),
                   "val_loss":epoch_val_loss.item(),
                   "val_accuracy":epoch_val_accuracy.item(),
                   "val_5_accuracy": epoch_val_5_accuracy.item(),
                   "lr":new_lr})
        final_path=Path(args.output_dir)/ f'final_model_{args.saving_name}_{args.epochs}_epochs.pth'
        save_on_master({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }, final_path)
    
    
        
def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def main(args):

    train_loader,valid_loader=build_dataset(args) 
    model,criterion,optimizer,scheduler = build_model(args)
    model.to(args.device)
    criterion.to(args.device)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_parameters={n_parameters}")
    
    print(f'args:{args}')
    print('Start training')
    start_time=time.time()
    training(model,criterion,optimizer,scheduler,train_loader,valid_loader,args.epochs,args.clip_norm)    
    total_time=time.time()-start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Deformable ViT', parents = [get_args_parser()])
    args = parser.parse_args()
    
    main(args)