'''
Alternative implementation of deformable attention from
https://arxiv.org/abs/2010.04159 
'''

from torch import nn
import torch
import torch.nn.functional as F



class DeformableTransformer(nn.Module):
    def __init__(self,embed_dim=256,num_heads=8,num_layers=6,dim_feedforward=1024,
                 dropout=0.1, num_feature_levesl=4, encoder_n_points=4,
                 ):
        super().__init__()
        
        self.embed_dim=embed_dim
        self.num_heads=num_heads
        



class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,embed_dim=256,dim_feedforward=1024,dropout=0.1,
                 num_feature_levels=4,num_heads=8,num_points=4):
        super().__init__()
        
        pass
    
class MSDeformAttn(nn.Module):
    def __init__(self,embed_dim=256,n_levels=4,n_heads=8,n_points=4):
        super().__init__()
        
        self.im2col_step = 64
        
        ''' '''
        self.embed_dim=256
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points
        
        ''' Creation of 3MK channels, page 5'''
        self.sampling_offsets = nn.Linear(embed_dim,n_heads * n_levels * n_points * 2) 
        self.attention_weights = nn.Linear(embed_dim, n_heads * n_levels * n_points)
        
        self.value_proj = nn.Linear(embed_dim,embed_dim)
        self.output_proj = nn.Linear(embed_dim,embed_dim)
        
        #take care of initialization
    def forward(self,query,reference_points,input_flatten,input_spatial_shapes,input_level_start_index):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements
        :return output                     (N, Length_{query}, C)
        """
        
        N, len_q,_ = query.shape
        N, len_in,_ = input_flatten.shape
        
        
        value = self.value_proj(input_flatten)
        value = value.view(N,len_in, self.n_heads,self.embed_dim//self.n_heads)
        sampling_offsets = self.sampling_offsets(query).view(N,len_q,self.n_heads,self.n_levels,self.n_points,2)
        attn_weights = self.attention_weights(query).view(N,len_q,self.n_heads,self.n_levels*self.n_points)
        attn_weights = F.softmax(attn_weights,-1).view(N,len_q,self.n_heads,self.n_levels,self.n_points)
        
        
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                                 + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                                 + sampling_offsets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                'Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))
        
        '''MSDA not available?'''
        output = None #Import MultiScaleDeformableAttention
        
        output = self.output_proj(output)
        return output