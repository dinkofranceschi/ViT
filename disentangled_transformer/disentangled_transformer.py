import copy
from typing import Optional, List
import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F

'''Adapting from 
https://github.com/microsoft/DeBERTa/
'''

from utils.ops import build_relative_position


class DisentangledTransformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False,
                 pos_att_type='c2p|p2c',
                 relative_attention=False,
                 position_buckets=-1,
                 max_relative_positions=-1,
                 max_position_embeddings=512,
                 performers=False,
                 norf=32,
                 kernel='softmax',
                 lai=False,
                 mask_epochs=None,
                 ):
        super().__init__()
        
  
        encoder_layer=DisentangledEncoderLayer(d_model=d_model,
                                             nhead=nhead,
                                             dropout=dropout,
                                             dim_feedforward=dim_feedforward,
                                             relative_attention=relative_attention,
                                             pos_att_type=pos_att_type,
                                             position_buckets=position_buckets,
                                             max_relative_positions=max_relative_positions,
                                             max_position_embeddings=max_position_embeddings,
                                             performers=performers,
                                             lai=lai,
                                             mask_epochs=mask_epochs,
                                             norf=norf,
                                             kernel=kernel
                                            )

        
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = DisentangledEncoder(encoder_layer, num_encoder_layers, encoder_norm,performers=performers)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed,epoch=None):
        
        bs, hw, d = src.shape 
        #src = src.permute(1, 0, 2)  #-> BS x N x d -> N x BS x d
        
        pos_embed = pos_embed.permute(1, 0, 2) # Same for positional embeddings 1 x num_patchs+1 x d  -> num_patches+1 x 1 x d

        memory = self.encoder(src, pos=pos_embed,epoch=epoch)

        return memory.permute(1,0,2)
    
class DisentangledEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None,
                 relative_attention=True,max_relative_positions=-1,position_buckets=-1,
                 performers=False
                 ):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        
    
        d_model=encoder_layer.embed_size
        max_position_embeddings=d_model #change later
        self.relative_attention = relative_attention #bool
        if self.relative_attention:
          self.max_relative_positions = max_relative_positions
          if self.max_relative_positions <1:
            self.max_relative_positions = max_position_embeddings
          self.position_buckets = position_buckets
          pos_ebd_size = self.max_relative_positions*2
          if self.position_buckets>0:
            pos_ebd_size = self.position_buckets*2
          self.rel_embeddings = nn.Embedding(pos_ebd_size, d_model)
         
          self.LayerNorm= nn.LayerNorm(d_model)
        
        
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        if rel_embeddings is not None:
          rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings        
    
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
          q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
          relative_pos = build_relative_position(q, hidden_states.size(-2), bucket_size = self.position_buckets, max_position=self.max_relative_positions)
        return relative_pos
    def forward(self, src,
                pos: Optional[Tensor] = None,epoch=None):
        output = src
        
        relative_pos=self.get_rel_pos(src)
        rel_embeddings=self.get_rel_embedding()
        
        #print(output.shape,pos.shape)
        #output=output+pos.permute(1,0,2)
        
        for layer in self.layers:
            output = layer(output,relative_pos=relative_pos,rel_embeddings=rel_embeddings,pos_embed=pos,epoch=epoch)            
        if self.norm is not None:
            output = self.norm(output)

        return output

class DisentangledEncoderLayer(nn.Module):
    def __init__(self,d_model=512,nhead=8,dropout=0.1,
                 relative_attention=True,pos_att_type='c2p|p2c',
                 position_buckets=-1,max_relative_positions=-1,
                 max_position_embeddings=512,
                 dim_feedforward=2048,
                 performers=False,
                 lai=False,
                 mask_epochs=None,
                 norf=32,
                 kernel='softmax'):
        super().__init__()
        
        if performers:
            self.dis_attn=DisentangledSelfAttentionPerformer(d_model=d_model,nhead=nhead,dropout=dropout,
                                                relative_attention=relative_attention,pos_att_type=pos_att_type,
                                                position_buckets=position_buckets,max_relative_positions=max_relative_positions,
                                                max_position_embeddings=max_position_embeddings,
                                                kernel=kernel,
                                                norf=norf,
                                                )
        elif lai:
            self.dis_attn=DisentangledLAISelfAttention(d_model=d_model,nhead=nhead,dropout=dropout,
                                                relative_attention=relative_attention,pos_att_type=pos_att_type,
                                                position_buckets=position_buckets,max_relative_positions=max_relative_positions,
                                                max_position_embeddings=max_position_embeddings,
                                                mask_epochs=mask_epochs)
        else:
            self.dis_attn=DisentangledSelfAttention(d_model=d_model,nhead=nhead,dropout=dropout,
                                                relative_attention=relative_attention,pos_att_type=pos_att_type,
                                                position_buckets=position_buckets,max_relative_positions=max_relative_positions,
                                                max_position_embeddings=max_position_embeddings)
        
        self.embed_size=d_model
        _attention_head_size = int(d_model / nhead)
        self.num_attention_heads = nhead
        self.attention_head_size =  _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        
        self.dense_AttnOut = nn.Linear(d_model,d_model)
        self.layer_norm_AttnOut = nn.LayerNorm(d_model)
        self.dropout_AttnOut = nn.Dropout(dropout)
        
        self.dense_intermediate=nn.Linear(d_model,dim_feedforward)
        self.act_intermediate= nn.GELU()
        
        self.dense_out = nn.Linear(dim_feedforward,d_model)
        self.layer_norm_out = nn.LayerNorm(d_model)
        self.dropout_out = nn.Dropout(dropout)
        
    def forward_intermediate(self,hidden_states):
        hidden_states=self.dense_intermediate(hidden_states)
        return self.act_intermediate(hidden_states)
    
    def forward_self_output(self,hidden_states,input_states):
        hidden_states=self.dense_AttnOut(hidden_states)
        hidden_states=self.dropout_AttnOut(hidden_states)
        hidden_states+= input_states
        hidden_states=self.layer_norm_AttnOut(hidden_states)
        return hidden_states
    def forward_output(self,hidden_states,input_states):
        hidden_states=self.dense_out(hidden_states)
        hidden_states=self.dropout_out(hidden_states)
        hidden_states+= input_states
        hidden_states=self.layer_norm_out(hidden_states)
        return hidden_states
    
    def forward_attention(self,src,
                relative_pos=None,
                rel_embeddings=None,
                query=None,
                pos_embed=None,
                epoch=None):
        
        # q=self.query_proj(src) if query is None else self.query_proj(query) 
        # k=self.key_proj(src)
        # value=self.value_proj(src)
        q,k,value=src,src,src

        if epoch is None:
            output = self.dis_attn(q,k,value,
                               relative_pos=relative_pos,
                               rel_embeddings=rel_embeddings,pos_embed=pos_embed)
        
        else:
            output = self.dis_attn(q,k,value,
                               relative_pos=relative_pos,
                               rel_embeddings=rel_embeddings,pos_embed=pos_embed,epoch=epoch)
            
        self_output, att_matrix, att_logits_=output['hidden_states'], output['attention_probs'], output['attention_logits']
        
        attn_output = self.forward_self_output(self_output,q)
        
        return attn_output
    
    def forward(self,src,
                relative_pos=None,
                rel_embeddings=None,
                query=None,
                pos_embed=None,epoch=None):
        
        attn_output= self.forward_attention(
                src,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                query=query,
                pos_embed=pos_embed,epoch=epoch)
        
        out = self.forward_intermediate(attn_output)
        
        out= self.forward_output(out,attn_output)
        
        #print(f'out_shape = {out.shape}') #BsxNxd
        return out
    
class DisentangledSelfAttention(nn.Module):
    def __init__(self, d_model=512,nhead=8,dropout=0.1,
                 relative_attention=True, pos_att_type='c2p|p2c',
                 position_buckets=-1,max_relative_positions=-1,
                 max_position_embeddings=512):
        '''
        param pos_att_type: (str) relative position attention e.g. 'p2c|c2p','c2p|p2p','p2p' c2c is always implicit
        param relative_attention (bool) use relative position encoding
        param max_position_embeddings (int) maximum sequence length
        param max_relative_positions (int) range of relative positions
        param position_buckets (int)
            
        '''
        super().__init__()
        self.num_attention_heads = nhead
        _attention_head_size = int(d_model / nhead)
        self.attention_head_size =  _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')] # c2p|p2c
        self.relative_attention = relative_attention
        
        self.share_att_key=True

        self.query_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        
        if self.relative_attention:
            self.position_buckets = position_buckets
            self.max_relative_positions = max_relative_positions
            if self.max_relative_positions <1:
                self.max_relative_positions = max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets>0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable

            self.pos_dropout = nn.Dropout(dropout)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_query_proj = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))


    def forward(self,q,k,value,relative_pos=None,rel_embeddings=None,query=None,pos_embed=None):
        #Bs,N,d we want to receive this
        q=self.query_proj(q) if query is None else self.query_proj(query) 
        k=self.key_proj(k)
        value=self.value_proj(value)
       #print(f'query_shape={q.shape}') #BsxLxd
        query_layer = self.transpose_for_scores(q, self.num_attention_heads)
        #print(f'query_shape={query_layer.shape}') #Bs*n_heads x N x head_dim
       
        key_layer = self.transpose_for_scores(k, self.num_attention_heads)
        #print(f'key_shape={key_layer.transpose(-1,-2).shape}')
        value_layer = self.transpose_for_scores(value, self.num_attention_heads)
        #print(f'v_shape={value_layer.shape}')
        rel_att = None  #initialization of relative attention
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1)*scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))/scale
        #print(key_layer.shape)
        #print(query_layer.shape)
        #print(value_layer.shape)
        #print(attention_scores.shape)
    
        if self.relative_attention:
            #Fix input for rel_embeddings
            rel_embeddings = self.pos_dropout(rel_embeddings)
            #Fix dimensions, we enter with Bs*H x N x d
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)
        
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))

        # bxhxlxd
        _attention_probs = torch.softmax(attention_scores, -1)
        attention_probs = self.dropout(_attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        

        return {
            'hidden_states': context_layer,
            'attention_probs': _attention_probs,
            'attention_logits': attention_scores
            }

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        #print(relative_pos)
        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
                    
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        
        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
            score += c2p_att/scale
            
        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att/scale

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:,:,att_span:,:]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att

        return score
    
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class DisentangledSelfAttentionPerformer(nn.Module):
    def __init__(self, d_model=512,nhead=8,dropout=0.1,
                 relative_attention=False, pos_att_type='c2p|p2c',
                 position_buckets=-1,max_relative_positions=-1,
                 max_position_embeddings=512,
                 norf=32,
                 kernel='softmax'):
        '''
        param pos_att_type: (str) relative position attention e.g. 'p2c|c2p','c2p|p2p','p2p' c2c is always implicit
        param relative_attention (bool) use relative position encoding
        param max_position_embeddings (int) maximum sequence length
        param max_relative_positions (int) range of relative positions
        param position_buckets (int)
            
        '''
        super().__init__()
        self.m = int(norf) 
        self.w=None
        
        self.kernel=kernel
        self.num_attention_heads = nhead
        _attention_head_size = int(d_model / nhead)
        self.attention_head_size = self.head_dim = _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')] # c2p|p2c
        self.relative_attention = relative_attention
        
        self.share_att_key=True

        self.query_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        
        if self.relative_attention:
            self.position_buckets = position_buckets
            self.max_relative_positions = max_relative_positions
            if self.max_relative_positions <1:
                self.max_relative_positions = max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets>0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable

            self.pos_dropout = nn.Dropout(dropout)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_query_proj = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    def transpose_for_performers(self,x):
        B,L,d= x.shape
        return x.reshape(B,L,self.num_attention_heads,self.attention_head_size)
    
    def untranspose_for_performers(self,x):
        return x.view(-1,self.num_attention_heads,self.attention_head_size)
        
    def forward(self,q,k,value,relative_pos=None,rel_embeddings=None,query=None,pos_embed=None):
        dev=q.device
        #initializing self.w if it does not exist yet, we do this is the forward because of the device
        if self.w is None:
            #print(self.head_dim,self.m)
            self.w=self.gaussian_orthogonal_random_matrix(self.m, self.head_dim, device=dev)
            #print(f'self.w.shape={self.w.shape} m= {self.m}, head_dim = {self.head_dim}')
        
        #Bs,N,d we want to receive this
        q=self.query_proj(q) if query is None else self.query_proj(query) 
        k=self.key_proj(k)
        value=self.value_proj(value)
        
        q_perf=self.transpose_for_performers(q)
        k_perf=self.transpose_for_performers(k)
        value_perf=self.transpose_for_performers(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        #scale = math.sqrt(q_perf.size(-1)*scale_factor)
        #attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))/scale
        
        #Regular performer matmult CORRECT SCALES LATER MAYBE USE =!= ORF for cp and pc
        
        
        if self.kernel=='softmax':
            key_prime=self.softmax_kernel_transformation(k_perf,False) # B L H M
            query_prime=self.softmax_kernel_transformation(q_perf,True) # B L H M
        
        elif self.kernel == 'relu':
            key_prime=self.relu_kernel_transformation(k_perf,False) # B L H M
            query_prime=self.relu_kernel_transformation(q_perf,True) # B L H M
        
        '''
        kv=torch.einsum('blhm,blhd->bhmd',key_prime,value_perf)
        qkv=torch.einsum('blhm,bhmd->blhd',query_prime,kv)
        
        #Denominator
        ks_sum=torch.einsum("blhm,l->bhm",key_prime,torch.ones(key_prime.shape[1],device=dev))
        D=torch.einsum("blhm,bhm->blh", query_prime, ks_sum)
        D=D.unsqueeze(-1) #BxLxH->BxLxHx1
        
        out=(qkv/D)  #BxLxHxd
        CC=out.flatten(-2) #BxLxHxd->BxLxHd
        
        rel_embeddings = self.pos_dropout(rel_embeddings)
       
        
        Pq,Pk= self.disentangled_attention_perf_bias(q, k, relative_pos, rel_embeddings, scale_factor,pos_embed)
        Pq=Pq.expand(*key_prime.shape)
        Pk=Pk.expand(*key_prime.shape)
        #CP and PC
        
        pqkv=torch.einsum('blhm,bhmd->blhd',Pq,kv)
        
        #Denominator
        ks_sum=torch.einsum("blhm,l->bhm",key_prime,torch.ones(key_prime.shape[1],device=dev))
        D=torch.einsum("blhm,bhm->blh", query_prime, ks_sum)
        D=D.unsqueeze(-1) #BxLxH->BxLxHx1
        
        out=(pqkv/D)  #BxLxHxd
        CP = out.flatten(-2)
        
        
        
        pkv=torch.einsum('blhm,blhd->bhmd',Pk,value_perf)        
        qpkv=torch.einsum('blhm,bhmd->blhd',query_prime,pkv)
        
        #Denominator
        ks_sum=torch.einsum("blhm,l->bhm",Pk,torch.ones(key_prime.shape[1],device=dev))
        D=torch.einsum("blhm,bhm->blh", query_prime, ks_sum)
        D=D.unsqueeze(-1) #BxLxH->BxLxHx1
        
        out=(qpkv/D)  #BxLxHxd
        
        PC = out.flatten(-2)
        
        
        w_cc=4
        w_cp=2
        w_pc=2
        context_layer = w_cc*CC + w_cp*CP + w_pc*PC/(w_cc+w_cp+w_pc)
        '''
        
        Pq,Pk= self.disentangled_attention_perf_bias(q, k, relative_pos, rel_embeddings, scale_factor,pos_embed)
        Pq=Pq.expand(*key_prime.shape)
        Pk=Pk.expand(*key_prime.shape)
        
        new_Q=Pq+query_prime
        new_K=Pk+key_prime
        
        kv=torch.einsum('blhm,blhd->bhmd',new_K,value_perf)
        qkv=torch.einsum('blhm,bhmd->blhd',new_Q,kv)
        
        #Denominator
        ks_sum=torch.einsum("blhm,l->bhm",new_K,torch.ones(new_K.shape[1],device=dev))
        D=torch.einsum("blhm,bhm->blh", new_Q, ks_sum)
        D=D.unsqueeze(-1) #BxLxH->BxLxHx1
        
        out=(qkv/D)  #BxLxHxd
        context_layer=out.flatten(-2) #BxLxHxd->BxLxHd
        
        context_layer = self.dropout(context_layer)

        return {
            'hidden_states': context_layer,
            'attention_probs': None,
            'attention_logits': None
            }

    def disentangled_attention_perf_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor,pos_embed):
        
        rel_embeddings=pos_embed.permute(1,0,2) # 1 x L x d
        
        pos_key_layer= self.transpose_for_performers(self.pos_key_proj(rel_embeddings)) #Both in B  x L x num_head x head_shape
        pos_query_layer= self.transpose_for_performers(self.pos_query_proj(rel_embeddings))
        

        # content->position
        if 'c2p' in self.pos_att_type:
            if self.kernel == 'softmax':
                Pk = self.softmax_kernel_transformation(pos_key_layer, False)
            if self.kernel == 'relu':
                Pk = self.relu_kernel_transformation(pos_key_layer, False)
        

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            if key_layer.size(-3) != query_layer.size(-3):
                r_pos = build_relative_position(key_layer.size(-3), key_layer.size(-3), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos


        if 'p2c' in self.pos_att_type:
            if self.kernel == 'softmax':
                Pq = self.softmax_kernel_transformation(pos_query_layer, True)
            if self.kernel == 'relu':
                Pq = self.relu_kernel_transformation(pos_query_layer, True)
        
        #Expand Pq and Pk
        return Pq,Pk
    
    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)

        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
            score += c2p_att/scale

        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att/scale

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:,:,att_span:,:]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att

        return score
        
    #Performer part
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
    
    
class DisentangledLAISelfAttention(nn.Module):
    def __init__(self, d_model=512,nhead=8,dropout=0.1,
                 relative_attention=True, pos_att_type='c2p|p2c',
                 position_buckets=-1,max_relative_positions=-1,
                 max_position_embeddings=512,
                 mask_epochs=None):
        '''
        param pos_att_type: (str) relative position attention e.g. 'p2c|c2p','c2p|p2p','p2p' c2c is always implicit
        param relative_attention (bool) use relative position encoding
        param max_position_embeddings (int) maximum sequence length
        param max_relative_positions (int) range of relative positions
        param position_buckets (int)
            
        '''
        super().__init__()
        
        
        self.mask_4 = None
        self.mask_6 = None
        self.mask_8 = None
        self.mask_10 = None 
        
        self.mask_epochs = mask_epochs
        
        self.num_attention_heads = nhead
        _attention_head_size = int(d_model / nhead)
        self.attention_head_size =  _attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.pos_att_type = [x.strip() for x in pos_att_type.lower().split('|')] # c2p|p2c
        self.relative_attention = relative_attention
        
        self.share_att_key=True

        self.query_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(d_model, self.all_head_size, bias=True)
        
        if self.relative_attention:
            self.position_buckets = position_buckets
            self.max_relative_positions = max_relative_positions
            if self.max_relative_positions <1:
                self.max_relative_positions = max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets>0:
                self.pos_ebd_size = self.position_buckets
                # For backward compitable

            self.pos_dropout = nn.Dropout(dropout)

            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_key_proj = nn.Linear(d_model, self.all_head_size, bias=True)
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                self.pos_query_proj = nn.Linear(d_model, self.all_head_size)

        self.dropout = nn.Dropout(dropout)

    def transpose_for_scores(self, x, attention_heads):
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))


    def forward(self,q,k,value,relative_pos=None,rel_embeddings=None,query=None,pos_embed=None,epoch=0):
        #Bs,N,d we want to receive this
        B, N, C = q.shape
        q=self.query_proj(q) if query is None else self.query_proj(query) 
        k=self.key_proj(k)
        value=self.value_proj(value)
       #print(f'query_shape={q.shape}') #BsxLxd
        query_layer = self.transpose_for_scores(q, self.num_attention_heads)
        #print(f'query_shape={query_layer.shape}') #Bs*n_heads x N x head_dim
       
        key_layer = self.transpose_for_scores(k, self.num_attention_heads)
        #print(f'key_shape={key_layer.transpose(-1,-2).shape}')
        value_layer = self.transpose_for_scores(value, self.num_attention_heads)
        #print(f'v_shape={value_layer.shape}')
        rel_att = None  #initialization of relative attention
        # Take the dot product between "query" and "key" to get the raw attention scores.
        scale_factor = 1
        if 'c2p' in self.pos_att_type:
            scale_factor += 1
        if 'p2c' in self.pos_att_type:
            scale_factor += 1
        if 'p2p' in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.size(-1)*scale_factor)
        attention_scores = torch.bmm(query_layer, key_layer.transpose(-1, -2))/scale
        #print(key_layer.shape)
        #print(query_layer.shape)
        #print(value_layer.shape)
        #print(attention_scores.shape)
    
        if self.relative_attention:
            #Fix input for rel_embeddings
            rel_embeddings = self.pos_dropout(rel_embeddings)
            #Fix dimensions, we enter with Bs*H x N x d
            rel_att = self.disentangled_attention_bias(query_layer, key_layer, relative_pos, rel_embeddings, scale_factor)

        if rel_att is not None:
            attention_scores = (attention_scores + rel_att)
        
        attention_scores = attention_scores.view(-1, self.num_attention_heads, attention_scores.size(-2), attention_scores.size(-1))
        
        
        if self.mask_epochs is not None:
           if epoch < self.mask_epochs[-1]:
               if epoch < self.mask_epochs[-4]:
                   if self.mask_4 is None:
                       self.mask_4 = get_attn_mask(N,8)
                   mask = self.mask_4
               elif epoch < self.mask_epochs[-3]:
                   if self.mask_6 is None:
                       self.mask_6= get_attn_mask(N,12)
                   mask = self.mask_6
               elif epoch < self.mask_epochs[-2]:
                   if self.mask_8 is None:
                       self.mask_8 = get_attn_mask(N,18)
                   mask = self.mask_8
               else:
                   if self.mask_10 is None:
                       self.mask_10 = get_attn_mask(N,20)
                   mask = self.mask_10
               attention_scores = attention_scores.masked_fill(mask.to(attention_scores.get_device()) == 0, -1e9)

        # bxhxlxd
        _attention_probs = torch.softmax(attention_scores, -1)
        attention_probs = self.dropout(_attention_probs)
        context_layer = torch.bmm(attention_probs.view(-1, attention_probs.size(-2), attention_probs.size(-1)), value_layer)
        context_layer = context_layer.view(-1, self.num_attention_heads, context_layer.size(-2), context_layer.size(-1)).permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (-1,)
        context_layer = context_layer.view(*new_context_layer_shape)
        
        

        return {
            'hidden_states': context_layer,
            'attention_probs': _attention_probs,
            'attention_logits': attention_scores
            }

    def disentangled_attention_bias(self, query_layer, key_layer, relative_pos, rel_embeddings, scale_factor):
        if relative_pos is None:
            q = query_layer.size(-2)
            relative_pos = build_relative_position(q, key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions)
        if relative_pos.dim()==2:
            relative_pos = relative_pos.unsqueeze(0).unsqueeze(0)
        elif relative_pos.dim()==3:
            relative_pos = relative_pos.unsqueeze(1)
        # bxhxqxk
        elif relative_pos.dim()!=4:
            raise ValueError(f'Relative postion ids must be of dim 2 or 3 or 4. {relative_pos.dim()}')
        #print(relative_pos)
        att_span = self.pos_ebd_size
        relative_pos = relative_pos.long().to(query_layer.device)

        rel_embeddings = rel_embeddings[self.pos_ebd_size - att_span:self.pos_ebd_size + att_span, :].unsqueeze(0) #.repeat(query_layer.size(0)//self.num_attention_heads, 1, 1)
        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings), self.num_attention_heads)\
                .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        else:
            if 'c2p' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
                    
            if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings), self.num_attention_heads)\
                    .repeat(query_layer.size(0)//self.num_attention_heads, 1, 1) #.split(self.all_head_size, dim=-1)
        
        score = 0
        # content->position
        if 'c2p' in self.pos_att_type:
            scale = math.sqrt(pos_key_layer.size(-1)*scale_factor)
            c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
            c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span*2-1)
            c2p_att = torch.gather(c2p_att, dim=-1, index=c2p_pos.squeeze(0).expand([query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]))
            score += c2p_att/scale
            
        # position->content
        if 'p2c' in self.pos_att_type or 'p2p' in self.pos_att_type:
            scale = math.sqrt(pos_query_layer.size(-1)*scale_factor)
            if key_layer.size(-2) != query_layer.size(-2):
                r_pos = build_relative_position(key_layer.size(-2), key_layer.size(-2), bucket_size = self.position_buckets, max_position = self.max_relative_positions).to(query_layer.device)
                r_pos = r_pos.unsqueeze(0)
            else:
                r_pos = relative_pos

            p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span*2-1)
            if query_layer.size(-2) != key_layer.size(-2):
                pos_index = relative_pos[:, :, :, 0].unsqueeze(-1)

        if 'p2c' in self.pos_att_type:
            p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=p2c_pos.squeeze(0).expand([query_layer.size(0), key_layer.size(-2), key_layer.size(-2)])).transpose(-1,-2)
            if query_layer.size(-2) != key_layer.size(-2):
                p2c_att = torch.gather(p2c_att, dim=-2, index=pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2))))
            score += p2c_att/scale

        # position->position
        if 'p2p' in self.pos_att_type:
            pos_query = pos_query_layer[:,:,att_span:,:]
            p2p_att = torch.matmul(pos_query, pos_key_layer.transpose(-1, -2))
            p2p_att = p2p_att.expand(query_layer.size()[:2] + p2p_att.size()[2:])
            if query_layer.size(-2) != key_layer.size(-2):
                p2p_att = torch.gather(p2p_att, dim=-2, index=pos_index.expand(query_layer.size()[:2] + (pos_index.size(-2), p2p_att.size(-1))))
            p2p_att = torch.gather(p2p_att, dim=-1, index=c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)]))
            score += p2p_att

        return score

def get_attn_mask(N, w):
    mask = torch.zeros(1, 1, N, N).cuda()
    for i in range(N):
        if i <= w:
            mask[:, :, i, 0:i+w+1] = 1
        elif N - i <= w:
            mask[:, :, i, i-w:N] = 1
        else:
            mask[:, :, i, i:i+w+1] = 1
            mask[:, :, i, i-w:i] = 1
    return mask