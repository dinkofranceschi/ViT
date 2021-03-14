import os
import torch
import torch.nn as nn
import timm
from timm.models.vision_transformer import Attention
import math
import torch.nn.functional as F
import copy

class PerformerAttention(Attention):
    def __init__(self,attention,dim,num_heads=8,qkv_bias=False,attn_drop=0.,proj_drop=0.,n_orf=64,kernel='softmax'):
        super().__init__(dim,num_heads=num_heads,qkv_bias=qkv_bias,attn_drop=attn_drop,proj_drop=proj_drop)
        self.m = int(n_orf) 
        self.w=None
        
        self.num_heads=copy.deepcopy(attention.num_heads)
        self.kernel=kernel
        self.head_dim = dim //num_heads
    
        #Given an attention block we would like to copy all of its parameters
        self.qkv =copy.deepcopy(attention.qkv)
        self.attn_drop = copy.deepcopy(attention.attn_drop)
        self.proj = copy.deepcopy(attention.proj)
        self.proj_drop= copy.deepcopy(attention.proj_drop)
    
    def forward(self,x):
        dev=x.device
        #initializing self.w if it does not exist yet, we do this is the forward because of the device
        if self.w is None:
            #print(self.head_dim,self.m)
            self.w=self.gaussian_orthogonal_random_matrix(self.m, self.head_dim, device=dev)
            #print(f'self.w.shape={self.w.shape} m= {self.m}, head_dim = {self.head_dim}')
            
        #print(x.shape) 
        B, N, C = x.shape  #Batch size x Number of Tokens x mini-Channels 
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4) # BxNxC -> BxLx3C -> BxLx3xnum_headsxhead_dim -> 3,B,L,num_heads,head_dim
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple) #BxLxnum_headsxhead_dim

        #attn = (q @ k.transpose(-2, -1)) * self.scale #BxnumHeadxLxL .T
        #attn = attn.softmax(dim=-1)
        
        if self.kernel=='softmax':
            key_prime=self.softmax_kernel_transformation(k,False) # B L H M
            query_prime=self.softmax_kernel_transformation(q,True) # B L H M
        
        elif self.kernel == 'relu':
            key_prime=self.relu_kernel_transformation(k,False) # B L H M
            query_prime=self.relu_kernel_transformation(q,True) # B L H M

        kv=torch.einsum('blhm,blhd->bhmd',key_prime,v)
        qkv=torch.einsum('blhm,bhmd->blhd',query_prime,kv)
        
        #Denominator
        ks_sum=torch.einsum("blhm,l->bhm",key_prime,torch.ones(key_prime.shape[1],device=dev))
        D=torch.einsum("blhm,bhm->blh", query_prime, ks_sum)
        D=D.unsqueeze(-1) #BxLxH->BxLxHx1
        
        out=(qkv/D)  #BxLxHxd
        x=out.flatten(-2) #BxLxHxd->BxLxHd
       
        x = self.attn_drop(x)
       
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



    def softmax_kernel_transformation(self, data, is_query, numerical_stabilizer=0.000001):
         d=data.shape[-1]
         data_normalizer = 1.0 / math.sqrt(math.sqrt(d))
         ratio = 1.0 / math.sqrt(math.sqrt(self.m))
         
         #print(f'shapes-> w={self.w.shape}, data={data.shape} ')
         data_dash = torch.einsum("blhd,md->blhm", data, self.w) 
         diag_data = torch.square(data)
         
         diag_data= data.sum(dim = -1, keepdim = False)
       
         diag_data = (diag_data / 2.0) * data_normalizer * data_normalizer
         diag_data = diag_data.unsqueeze(-1)
       
         if is_query:
           data_dash = torch.exp(data_dash - diag_data - torch.max(data_dash,dim=-1, keepdims=True)[0]) + numerical_stabilizer
           data_dash= ratio * data_dash
           
         else:
           data_dash = torch.exp(data_dash - diag_data - torch.max(data_dash)) +  numerical_stabilizer
           data_dash= ratio * data_dash
       
         return data_dash
     
    def relu_kernel_transformation(self,data,is_query,numerical_stabilizer=0.001):
         #[L,B,H,D]
         del is_query
         ratio = 1.0/ math.sqrt(math.sqrt(self.m))
         #print(f'devices-> w={self.w.device}, data={data.device} ')
         data_dash= ratio * torch.einsum('blhd,md->blhm',data,self.w)
         data_dash=F.relu(data_dash) + numerical_stabilizer
         
         return data_dash
     
    def gaussian_orthogonal_random_matrix(self, nb_rows, nb_columns, scaling = 0, qr_uniform_q = False, device = None):
        nb_full_blocks = int(nb_rows / nb_columns)
        
        block_list = []
    
        for _ in range(nb_full_blocks):
            q = self.orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
            block_list.append(q)
    
        remaining_rows = nb_rows - nb_full_blocks * nb_columns
        #print(f'remaining_rows={remaining_rows}')
        if remaining_rows > 0:
            q = self.orthogonal_matrix_chunk(nb_columns, qr_uniform_q = qr_uniform_q, device = device)
            #print(f'q_shape={q.shape}')
            block_list.append(q[:remaining_rows])
    
        final_matrix = torch.cat(block_list)
    
        if scaling == 0:
            multiplier = torch.randn((nb_rows, nb_columns), device = device).norm(dim = 1)
        elif scaling == 1:
            multiplier = math.sqrt((float(nb_columns))) * torch.ones((nb_rows,), device = device)
        else:
            raise ValueError(f'Invalid scaling {scaling}')
    
        return torch.diag(multiplier) @ final_matrix
    
    def orthogonal_matrix_chunk(self,cols, qr_uniform_q = False, device = None):
        unstructured_block = torch.randn((cols, cols), device = device)
        q, r = torch.qr(unstructured_block, some = True)
    
        if qr_uniform_q:
            d = torch.diag(r, 0)
            q *= d.sign()
        return q.t()
    
    
class VisionPerformerTIMM(nn.Module):
    def __init__(self,n_classes=1000,num_orf=64,kernel='softmax',pretrained=True,image_size=224):
        super(VisionPerformerTIMM,self).__init__()
        self.model = timm.create_model("vit_base_patch16_224",pretrained=pretrained)
        
        if image_size != 224:
            pass
        #For finituning
        #self.model.head = nn.Linear(self.model.head.in_features,n_classes)
        
        for elem in self.model.blocks:
            attention = elem.attn
            elem.attn=PerformerAttention(attention, 768,num_heads=attention.num_heads,n_orf=num_orf,kernel=kernel)
        
    def forward(self,x):
        x = self.model(x)
        return x
    


def test_vip():
    img_size=32
    n_channels=3
    n_classes=1000
    batch_size=3
    device='cpu'
    img=torch.randn(batch_size,n_channels,img_size,img_size)
    norf=16
    kernel = 'relu'
    print(f'device={device},kernel={kernel},num_orf={norf}')
    
    m=timm.create_model("vit_base_patch16_224",pretrained=True)
    #m1=VisionPerformerTIMM(2000,num_orf=norf,kernel='relu')
    m1= copy.deepcopy(m)
    for elem in m1.blocks:
            attention = elem.attn
            elem.attn=PerformerAttention(attention, 768,num_heads=attention.num_heads,n_orf=norf,kernel=kernel)
        
    
    
    img=img.to(device)
    m1.to(device)
    m.to(device)
    import time   
    st2=time.time()
    r2=m1(img)
    stop2=time.time()-st2
    torch.cuda.empty_cache()
    st=time.time()
    r1=m(img)
    stop1=time.time()-st
    torch.cuda.empty_cache()
    print(f'performer in vip= {stop2}, transformer in vip={stop1}')
    
    a=Attention(768,num_heads=12)
    a1=PerformerAttention(a,768,num_heads=12,n_orf=norf)
    a.to(device)
    a1.to(device)
    x=torch.randn(batch_size,2000,768).to(device)
    
    st4=time.time()
    a(x)   
    stop4=time.time()-st4
    torch.cuda.empty_cache()
    st3=time.time()
    a1(x)
    stop3=time.time()-st3
    torch.cuda.empty_cache()
    print(f'performer = {stop3}, transformer={stop4}')
    return r1,r2
#r1,r2=test_vip()
    
    
    