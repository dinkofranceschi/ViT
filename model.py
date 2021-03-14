import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from itertools import repeat
from torch._six import container_abcs
import math
from einops import rearrange


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., hybrid_backbone=None, norm_layer=None, num_queries=100,
                 transformer = None, shuffle = False):

        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = transformer.d_model
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        
        self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim)
        num_patches = self.patch_embed.num_patches
        
        self.cls_token = nn.Parameter(torch.zeros(1,1,self.embed_dim))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.embed_dim))
        self.query_embed = nn.Parameter(torch.randn(num_queries, self.embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        
        self.num_queries = num_queries #cls_tokens for the decoder
        self.transformer = transformer
        
        if shuffle:
            factor=transformer.factor
            num_layers = transformer.num_layers
            self.head=nn.Linear(self.embed_dim * factor **(2*(num_layers-1)),num_classes)
            self.norm = norm_layer(self.embed_dim * factor **(2*(num_layers-1)))
        else:
            self.head = nn.Linear(self.embed_dim, num_classes)
            self.norm = norm_layer(self.embed_dim)
        self.act =nn.GELU()
        self.out = nn.Linear(num_classes,num_classes)
        
        
    def forward(self, img):
        B=img.shape[0]
        #We enter an image img = B x C x H x W
        img = self.patch_embed(img) # B x C x H x W -> B x (H'W') x d
        
        cls_tokens = self.cls_token.expand(B, -1, -1)   # Adding a class token for every element in the batch
        img = torch.cat((cls_tokens, img), dim=1) # Concatenante in the H'W' dim -> B x (H'W'+1) x d
        
        #print(f'Img_shape b4 transformer = {img.shape}')
        out=self.transformer(img,self.query_embed,self.pos_embed)
        #print(f'Img_shape after transformer = {out.shape}')
        
        out=self.norm(out.permute(1,0,2))[:,0]
    
        out=self.head(out.squeeze(0))
                       
        return out
        
        
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        to_2tuple = _ntuple(2)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape #BxCxHxW
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2) 
        # B x embed_dim=Channels x H' x W' -> B x d x (H'*W') -> B x (H'*W') x d
        return x

class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        
        bs, hw, d = src.shape 
        src = src.permute(1, 0, 2)  #-> BS x N x d -> N x BS x d
        
        pos_embed = pos_embed.permute(1, 0, 2) # Same for positional embeddings 1 x num_patchs+1 x d  -> num_patches+1 x 1 x d
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #num_queries x d -> num_queries x 1 x d -> num_queries x batch_size x d
        #print(src.shape)
        memory = self.encoder(src, pos=pos_embed)
        #print(f'memory_shape (out_enc) = {memory.shape}')
        return memory

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                pos: Optional[Tensor] = None):
        output = src
        
        for layer in self.layers:
            output = layer(output,pos=pos)
            #print(f'output_transformer_encoder_shape = {output.shape}')

        if self.norm is not None:
            output = self.norm(output)

        return output
    
class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src , pos)
        return self.forward_post(src, pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
    
    

class DeTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, query_embed, pos_embed):
        
        bs, hw, d = src.shape 
        #src = src.permute(1, 0, 2)  #-> BS x N x d -> N x BS x d
        
        #pos_embed = pos_embed.permute(1, 0, 2) # Same for positional embeddings 1 x num_patchs+1 x d  -> num_patches+1 x 1 x d
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1) #num_queries x d -> num_queries x 1 x d -> num_queries x batch_size x d
        
        memory = self.encoder(src, pos=pos_embed)
        #print(f'memory_shape (out_enc) = {memory.shape}')
        return memory

class DeTransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                pos: Optional[Tensor] = None):
        output = src
        
        for layer in self.layers:
            output = layer(output,pos=pos)
            #print(f'output_transformer_encoder_shape = {output.shape}')

        if self.norm is not None:
            output = self.norm(output)

        return output

class DeTransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.DeMultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src , pos)
        return self.forward_post(src, pos)


class DeMultiheadAttention(nn.Module):
    def __init__(self,d_model,nhead,dropout=0.1):
        super().__init__()
        self.qkv=nn.Linear(d_model,d_model*3,bias=True)
        self.nhead=nhead
        self.d_model=d_model
        self.head_dim=d_model //nhead
        self.dp_attn=nn.Dropout(dropout)
        self.combine_heads = nn.Linear(d_model,d_model)
        self.dp_proj = nn.Dropout(dropout)
        self.pq=nn.Linear(d_model,d_model)
        self.pk=nn.Linear(d_model,d_model)
    def forward(self,x,pos):
        
        qkv = self.qkv(x).chunk(3,dim=-1) # BxLxemb_d -> BxLx (3emb_d) -> tuple (BxLx(emb_d),BxLx(emb_d) ,BxLx(emb_d))
        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b l h d', h = self.nhead), qkv) # every instance to Batch_sizexTokenxnum_headsxhead_dim
        pq=self.pq(pos)
        pk=self.pk(pos)
        pq=rearrange(pq,'b l (h d) -> b l h d',h=self.nhead)
        pk=rearrange(pk,'b l (h d) -> b l h d',h=self.nhead)
        
        CC= torch.einsum('bnhd,blhd->bhnl', q,k) #n,l = num_tokens
        CP= torch.einsum('bnhd,blhd->bhnl', q,pq)
        PC= torch.einsum('bnhd,blhd->bhnl', k,pk)
        
        attn=torch.softmax(CC+CP+PC / (3*self.d_model)**(0.5),dim=-1)
        
        out= torch.einsum('bhnl,blhd->bnhd',attn,v)
        
        return out
    


'''Trying to adapt from
https://github.com/pytorch/pytorch/blob/d54be1a9467db5075256d229ea1b01f1a4bcba8d/torch/nn/functional.py#L4544
'''

class MultiheadAttentionPerformer(nn.MultiheadAttention):
    def __init__(self,d_model,nhead,dropout=0.1,num_orf=32,kernel='softmax'):
        super().__init__(d_model,nhead,dropout=dropout)
        
        self.kernel=kernel
        self.m = int(num_orf) 
        self.w=None
        
    #Non-causal forward
    def forward(self,query,key,value,key_padding_mask=None,
                need_weights=True,attn_mask=None):
        #print(query.shape)
        tgt_len, bsz, embed_dim = query.size()
        dev=query.device
        #initializing self.w if it does not exist yet, we do this is the forward because of the device
        if self.w is None:
            #print(self.head_dim,self.m)
            self.w=self.gaussian_orthogonal_random_matrix(self.m, self.head_dim, device=dev)

        '''Input projections'''
        if self._qkv_same_embed_dim:
            q = nn.functional.linear(query, self.in_proj_weight[0:self.embed_dim,:], self.in_proj_bias[0:self.embed_dim])
            k = nn.functional.linear(key,self.in_proj_weight[self.embed_dim: 2*self.embed_dim,:], self.in_proj_bias[self.embed_dim:2*self.embed_dim])
            v = nn.functional.linear(value,self.in_proj_weight[2*self.embed_dim:,:], self.in_proj_bias[2*self.embed_dim:])
        else:
            q= nn.functional.linear(query,self.q_proj_weight,self.in_proj_bias[0:self.embed_dim])
            k= nn.functional.linear(key,self.k_proj_weight,self.in_proj_bias[self.embed_dim: (2*self.embed_dim)])
            v= nn.functional.linear(value,self.v_proj_weight,self.in_proj_bias[(2*self.embed_dim):])
            
            
        # if self.bias_k is not None and self.bias_v is not None:
        #     k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
        #     v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            
        '''Reshaping'''
        q=q.reshape(q.shape[0],q.shape[1],self.num_heads,self.head_dim)
        k=k.reshape(k.shape[0],k.shape[1],self.num_heads,self.head_dim)
        v=v.reshape(v.shape[0],v.shape[1],self.num_heads,self.head_dim)     
        

        #Multihead attention
        #Kernel transforms
        '''Performer implementation'''
        if self.kernel=='softmax':
            key_prime=self.softmax_kernel_transformation(k,False) # L B H M
            query_prime=self.softmax_kernel_transformation(q,True) # L B H M
 
        
        elif self.kernel == 'relu':
            key_prime=self.relu_kernel_transformation(k,False) # L B H M
            query_prime=self.relu_kernel_transformation(q,True) # L B H M
        
        ''' BMM '''
        kv=torch.einsum('lbhm,lbhd->bhmd',key_prime,v)
        qkv=torch.einsum('lbhm,bhmd->lbhd',query_prime,kv)
        
        '''Dropout'''
        qkv=nn.functional.dropout(qkv,p=self.dropout,training=self.training)
        #Denominator
        '''Concatenation'''
        ks_sum=torch.einsum("lbhm,l->bhm",key_prime,torch.ones(key_prime.shape[0],device=dev))
        D=torch.einsum("lbhm,bhm->lbh", query_prime, ks_sum)
        D=D.unsqueeze(-1) #LxBxH->LxBxHx1
        
        out=(qkv/D)  #LxBxHxd
        out=out.flatten(-2) #LxBxHxd->LxBxHd
        '''Output projections'''
        attn_output = nn.functional.linear(out, self.out_proj.weight, self.out_proj.bias)

        return attn_output


    def softmax_kernel_transformation(self, data, is_query, numerical_stabilizer=0.000001):
         d=data.shape[-1]
         data_normalizer = 1.0 / math.sqrt(math.sqrt(d))
         ratio = 1.0 / math.sqrt(self.m)
         
         #print(f'shapes-> w={self.w.shape}, data={data.shape} ')
         data_dash = torch.einsum("lbhd,md->lbhm", data, self.w) 
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
         #print(f'shapes-> w={self.w.shape}, data={data.shape} ')
         data_dash= ratio * torch.einsum('lbhd,md->lbhm',data,self.w)
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
    
    
class Performer(Transformer):
    def __init__(self,d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 num_orf=64,kernel='relu'):
        super().__init__(d_model=d_model,
                         nhead=nhead,
                         num_encoder_layers=num_encoder_layers,
                         dim_feedforward=dim_feedforward,
                         dropout=dropout,
                         activation=activation,
                         normalize_before=normalize_before,
                         return_intermediate_dec=return_intermediate_dec,
                         )
        
        for layer in self.encoder.layers:
            layer.self_attn = MultiheadAttentionPerformer(d_model,nhead,dropout=dropout,num_orf=num_orf,kernel=kernel)


# #Testing the pipeline
# img_size=224
# patch_size=16
# in_chans=3
# num_classes=100
# embed_dim=384
# enc_layers=6
# dec_layers=6
# num_heads=8
# mlp_ratio=4
# dim_feedforward=mlp_ratio*embed_dim
# batch_size=5

# transformer=Transformer(
#         d_model=embed_dim,
#         dropout=0.1,
#         nhead=num_heads,
#         dim_feedforward=dim_feedforward,
#         num_encoder_layers=enc_layers,
#         num_decoder_layers=dec_layers,
#         normalize_before=True,
#         return_intermediate_dec=False,
#     )



# vit = VisionTransformer(img_size=img_size,patch_size=patch_size,in_chans=in_chans,num_classes=num_classes, num_queries=100,transformer=transformer)

# img = torch.randn(batch_size,in_chans, img_size,img_size)

# print(vit(img).shape)