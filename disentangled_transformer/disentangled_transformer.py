import copy
from typing import Optional, List
import torch
from torch import nn, Tensor
import math


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
                 ):
        super().__init__()

        encoder_layer = DisentangledEncoderLayer(d_model=d_model,
                                                 nhead=nhead,
                                                 dropout=dropout,
                                                 dim_feedforward=dim_feedforward,
                                                 relative_attention=relative_attention,
                                                 pos_att_type=pos_att_type,
                                                 position_buckets=position_buckets,
                                                 max_relative_positions=max_relative_positions,
                                                 max_position_embeddings=max_position_embeddings,
                                                 )
        
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = DisentangledEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        
        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, pos_embed):
        
        bs, hw, d = src.shape 
        #src = src.permute(1, 0, 2)  #-> BS x N x d -> N x BS x d
        
        pos_embed = pos_embed.permute(1, 0, 2) # Same for positional embeddings 1 x num_patchs+1 x d  -> num_patches+1 x 1 x d

        memory = self.encoder(src, pos=pos_embed)

        return memory.permute(1,0,2)
    
class DisentangledEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None,
                 relative_attention=True,max_relative_positions=-1,position_buckets=-1,
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
                pos: Optional[Tensor] = None):
        output = src
        
        relative_pos=self.get_rel_pos(src)
        rel_embeddings=self.get_rel_embedding()
        
        for layer in self.layers:
            output = layer(output,relative_pos=relative_pos,rel_embeddings=rel_embeddings)
            
        if self.norm is not None:
            output = self.norm(output)

        return output

class DisentangledEncoderLayer(nn.Module):
    def __init__(self,d_model=512,nhead=8,dropout=0.1,
                 relative_attention=False,pos_att_type='c2p|p2c',
                 position_buckets=-1,max_relative_positions=-1,
                 max_position_embeddings=512,
                 dim_feedforward=2048):
        super().__init__()
        
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
                query=None):
        
        # q=self.query_proj(src) if query is None else self.query_proj(query) 
        # k=self.key_proj(src)
        # value=self.value_proj(src)
        q,k,value=src,src,src

        
        output = self.dis_attn(q,k,value,
                               relative_pos=relative_pos,
                               rel_embeddings=rel_embeddings)
        
        self_output, att_matrix, att_logits_=output['hidden_states'], output['attention_probs'], output['attention_logits']
        
        attn_output = self.forward_self_output(self_output,q)
        
        return attn_output
    
    def forward(self,src,
                relative_pos=None,
                rel_embeddings=None,
                query=None):
        
        attn_output= self.forward_attention(
                src,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
                query=query)
        
        out = self.forward_intermediate(attn_output)
        
        out= self.forward_output(out,attn_output)
        
        #print(f'out_shape = {out.shape}') #BsxNxd
        return out
    
class DisentangledSelfAttention(nn.Module):
    def __init__(self, d_model=512,nhead=8,dropout=0.1,
                 relative_attention=False, pos_att_type='c2p|p2c',
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


    def forward(self,q,k,value,relative_pos=None,rel_embeddings=None,query=None):
        #Bs,N,d we want to receive this
        q=self.query_proj(q) if query is None else self.query_proj(query) 
        k=self.key_proj(k)
        value=self.value_proj(value)
        
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