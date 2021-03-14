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
import wandb


def get_args_parser():
    parser = argparse.ArgumentParser('Deformable ViT',add_help=False)
    '''Training parameters'''
    parser.add_argument('--lr',default=1e-4,type=float,
                        help='initial learning rate')
    parser.add_argument('--batch_size',default=128,type=int)
    parser.add_argument('--epochs', default=100,type=int)
    parser.add_argument('--gamma_lr',default=0.9,type=float,
                        help='gamma in the step learning rate')
    parser.add_argument('--step_lr',default=30,type=float,
                        help='peridicioty of lr decay by gamma')    
    parser.add_argument('--clip_norm', default=1, type = float)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--smoothing',default=0.1,type=float,
                        help='smoothing classification labels')
    parser.add_argument('--dataset',default='MNIST',type=str,
                        help='MNIST,CIFAR100,CIFAR10,ImageNet...')
    '''Model parameters'''
    parser.add_argument('--attention',default='performer',type=str,
                        help='Type of attention among Performer,Deformable Transformer...')
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

    return parser



def build_model(args):
    if args.attention == 'transformer':
        transformer=Transformer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
        )
    elif args.attention == 'performer':
        transformer = Performer(
            d_model= args.embed_dim,
            dropout=args.dropout,
            nhead=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_encoder_layers=args.num_layers,
            normalize_before=True,
        )

    model = VisionTransformer(img_size=args.img_size,
                              patch_size=args.patch_size,
                              in_chans=args.in_chans,
                              num_classes=args.num_classes,
                              transformer=transformer,
                              shuffle=False)

    criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing).to(args.device)
    
    optimizer= optim.AdamW(model.parameters(),lr=args.lr)

    scheduler=optim.lr_scheduler.StepLR(optimizer,step_size=50,gamma=1e-2) 
    
    return model,criterion, optimizer,scheduler


def build_dataset(args):
    #Data
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
          batch_size=args.batch_size, shuffle=True)
        
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
          batch_size=args.batch_size, shuffle=True,pin_memory=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,pin_memory=True)
    
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
          batch_size=args.batch_size, shuffle=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR10('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True)
    
        args.img_size = 32
        args.num_classes= 10
        args.in_chans= 3
    elif args.dataset =='ImageNet':
          train_loader = torch.utils.data.DataLoader(
          torchvision.datasets.ImageNet('./data/', train=True, download=True,
                                     transform=torchvision.transforms.Compose([
                                      torchvision.transforms.RandomHorizontalFlip(),
                                      torchvision.transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                                       torchvision.transforms.RandomResizedCrop(32,scale=(0.7,1)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])),
          batch_size=args.batch_size, shuffle=True)
        
          valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.ImageNet('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(32),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                     ])),
              batch_size=args.batch_size, shuffle=True)
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
          batch_size=args.batch_size, shuffle=True,pin_memory=True)
        
        valid_loader = torch.utils.data.DataLoader(
          torchvision.datasets.CIFAR100('./data/', train=False, download=True,
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize(224,interpolation=3),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                     ])),
          batch_size=args.batch_size, shuffle=True,pin_memory=True)
        
        args.img_size = 224
        args.num_classes = 100
        args.in_chans= 3
        
    return train_loader,valid_loader


def training(model,criterion,optimizer,scheduler,train_loader,valid_loader,epochs,clip_norm):
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
    
        for data, label in tqdm(train_loader):
            data = data.to(args.device)
            label = label.to(args.device)
            output = model(data)
            #print(output.shape,label.shape)
            loss = criterion(output, label)
    
            optimizer.zero_grad()
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            #REMOVE
            #loss.backward(retain_graph=True)
            loss.backward()
            optimizer.step()
            acc = (output.argmax(dim=1) == label).float().mean()
            epoch_accuracy += acc / len(train_loader)
            epoch_loss += loss / len(train_loader)
        scheduler.step()
        new_lr=scheduler.get_last_lr()[0] 
        with torch.no_grad():
            model.eval()
            epoch_val_accuracy = 0
            epoch_val_loss = 0
            for data, label in valid_loader:
                data = data.to(args.device)
                label = label.to(args.device)
                val_output = model(data)
                val_loss = criterion(val_output, label)
    
                acc = (val_output.argmax(dim=1) == label).float().mean()
                epoch_val_accuracy += acc / len(valid_loader)
                epoch_val_loss += val_loss / len(valid_loader)
        print(
            f"Epoch : {epoch+1} - loss : {epoch_loss:.4f} - acc: {epoch_accuracy:.4f} - val_loss : {epoch_val_loss:.4f} - val_acc: {epoch_val_accuracy:.4f}\n -curr_lr: {scheduler.get_last_lr()[0]}"
        )
        log['train_loss']=log['train_loss']+[epoch_loss.item()]
        log['train_accuracy']=log['train_accuracy']+[epoch_accuracy.item()]
        log['val_loss']=log['val_loss']+[epoch_val_loss.item()]
        log['val_accuracy']=log['val_accuracy']+[epoch_val_accuracy.item()]
        log['learning_rate']=log['learning_rate']+scheduler.get_last_lr()

    
    
        
def save_on_master(*args, **kwargs):
    torch.save(*args, **kwargs)

def main(args):
    train_loader,valid_loader=build_dataset(args) 
    model,criterion,optimizer,scheduler = build_model(args)
    model.to(args.device)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"n_parameters={n_parameters}")
    criterion.to(args.device)
    ''' add args'''
    
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