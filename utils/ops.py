import torch
from packaging import version
import numpy as np

if version.Version(torch.__version__) >= version.Version('1.0.0'):
  from torch import _softmax_backward_data as _softmax_backward_data
else:
  from torch import softmax_backward_data as _softmax_backward_data


class XSoftmax(torch.autograd.Function):
  """ Masked Softmax which is optimized for saving memory
  Args:
      
    input (:obj:`torch.tensor`): The input tensor that will apply softmax.
    mask (:obj:`torch.IntTensor`): The mask matrix where 0 indicate that element will be ignored in the softmax caculation.
    dim (int): The dimenssion that will apply softmax.
    
  Example::
    import torch
    from DeBERTa.deberta import XSoftmax
    # Make a tensor
    x = torch.randn([4,20,100])
    # Create a mask
    mask = (x>0).int()
    y = XSoftmax.apply(x, mask, dim=-1)
      
  """

  @staticmethod
  def forward(self, input, mask, dim):
    """
    """

    self.dim = dim
    if version.Version(torch.__version__) >= version.Version('1.2.0a'):
      rmask = ~(mask.bool())
    else:
      rmask = (1-mask).byte() # This line is not supported by Onnx tracing.

    output = input.masked_fill(rmask, float('-inf'))
    output = torch.softmax(output, self.dim)
    output.masked_fill_(rmask, 0)
    self.save_for_backward(output)
    return output

  @staticmethod
  def backward(self, grad_output):
    """
    """

    output, = self.saved_tensors
    inputGrad = _softmax_backward_data(grad_output, output, self.dim, output)
    return inputGrad, None, None



def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1):
  q_ids = np.arange(0, query_size)
  k_ids = np.arange(0, key_size)
  rel_pos_ids = q_ids[:, None] - np.tile(k_ids, (q_ids.shape[0],1))
  if bucket_size>0 and max_position > 0:
    rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
  rel_pos_ids = torch.tensor(rel_pos_ids, dtype=torch.long)
  rel_pos_ids = rel_pos_ids[:query_size, :]
  rel_pos_ids = rel_pos_ids.unsqueeze(0)
  return rel_pos_ids

def make_log_bucket_position(relative_pos, bucket_size, max_position):
  sign = np.sign(relative_pos)
  mid = bucket_size//2
  abs_pos = np.where((relative_pos<mid) & (relative_pos > -mid), mid-1, np.abs(relative_pos))
  log_pos = np.ceil(np.log(abs_pos/mid)/np.log((max_position-1)/mid) * (mid-1)) + mid
  bucket_pos = np.where(abs_pos<=mid, relative_pos, log_pos*sign).astype(np.int)
  return bucket_pos
