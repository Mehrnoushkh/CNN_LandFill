# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 17:24:37 2020

@author: Arthur

TODOs:
-Try some standard image classification network whose last layer you'll change
- change the color map of plots
- study different values of time indices
------BUGS-----
-when we run less than 100 epochs the figures from previous runs are
logged.
"""
# TODO Log the data run that is used to create the dataset. Log any
# transformation applied to the data. Later we might want to allow from
# stream datasets.

import torch
from torch.nn import Module, Parameter, Sequential, ModuleList
from torch.nn import functional as F
from torch.nn.modules.utils import _pair
from torch.nn.functional import pad
import torch.nn as nn

import numpy as np
from .base import DetectOutputSizeMixin, FinalTransformationMixin

class Identity(Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: torch.Tensor):
        return input


class ScaledModule(Module):
    def __init__(self, factor: float, module: torch.nn.Module):
        super().__init__()
        self.factor = factor
        self.module = module

    def forward(self, input: torch.Tensor):
        return self.factor * self.module.forward(input)

class LocallyConnected2d(nn.Module):
    """Class based on the code provided on the following link:
        https://discuss.pytorch.org/t/locally-connected-layers/26979
    """
    def __init__(self, input_h, input_w, in_channels, out_channels,
                 kernel_size, padding, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        def padding_extract(padding):
            new = []
            for el in padding:
                new.append(int(el))
                new.append(int(el))
            return tuple(new)
        self.padding_long = padding_extract(self.padding)
        output_size = self.calculate_output_size(input_h, input_w, 
                                                 self.kernel_size, 
                                                 self.padding,
                                                 self.stride)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, output_size[0],
                        output_size[1], 
                        self.kernel_size[0] * self.kernel_size[1])
        )
        # Scaling of the weight parameters according to number of inputs
        self.weight.data = self.weight / np.sqrt(in_channels * 
                                                 self.kernel_size[0]
                                                 * self.kernel_size[1])
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, output_size[0], output_size[1])
            )
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = pad(x, self.padding_long)
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x = x.contiguous().view(*x.size()[:-2], -1)
        # Sum in in_channel and kernel_size dims
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        if self.bias is not None:
            out += self.bias
        return out

    @staticmethod
    def calculate_output_size(input_h : int, input_w : int, kernel_size : int,
                              padding : int = None, stride : int = 1):
        # TODO add the stride bit. Right now it assumes 1.
        output_h = int(input_h - (kernel_size[0] - 1) / 2 + padding[0])
        output_w = int(input_w - (kernel_size[1] - 1) / 2 + padding[1])
        return output_h, output_w


class Divergence2d(Module):
    """Class that defines a fixed layer that produces the divergence of the
    input field. Note that the padding is set to 2, hence the spatial dim
    of the output is larger than that of the input."""
    def __init__(self, n_input_channels: int, n_output_channels: int):
        super().__init__()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_input_channels = n_input_channels
        self.n_output_channels = n_output_channels
        factor = n_input_channels // n_output_channels
        lambda_ = 1 / (factor * 2)
        shape = (1, n_input_channels//4, 1, 1)
        self.lambdas1x = Parameter(torch.ones(shape)) * lambda_
        self.lambdas2x = Parameter(torch.ones(shape)) * lambda_
        self.lambdas1y = Parameter(torch.ones(shape)) * lambda_
        self.lambdas2y = Parameter(torch.ones(shape)) * lambda_
        self.lambdas1x = self.lambdas1x.to(device=device)
        self.lambdas2x = self.lambdas2x.to(device=device)
        self.lambdas1y = self.lambdas1y.to(device=device)
        self.lambdas2y = self.lambdas2y.to(device=device)
        x_derivative = torch.tensor([[[[0, 0, 0],
                                       [-1, 0, 1],
                                       [0, 0, 0]]]])
        x_derivative = x_derivative.expand(1, n_input_channels // 4, -1, -1)
        y_derivative = torch.tensor([[0, 1, 0],
                                     [0, 0, 0],
                                     [0, -1, 0]])
        y_derivative = y_derivative.expand(1, n_input_channels // 4, -1, -1)
        x_derivative = x_derivative.to(dtype=torch.float32,
                                                 device=device)
        y_derivative = y_derivative.to(dtype=torch.float32,
                                                 device=device)
        self.x_derivative = x_derivative
        self.y_derivative = y_derivative

    def forward(self, input: torch.Tensor):
        n, c, h, w = input.size()
        y_derivative1 = self.y_derivative * self.lambdas1y
        x_derivative2 = self.x_derivative * self.lambdas2x
        y_derivative2 = self.y_derivative * self.lambdas2y
        output11 = F.conv2d(input[:, :c//4, :, :], self.x_derivative * self.lambdas1x,
                           padding=2)
        output12 = F.conv2d(input[:, c//4:c//2, :, :], self.y_derivative,
                            padding=2)
        output1 = output11 + output12
        output21 = F.conv2d(input[:, c//2:c//2+c//4, :, :], self.x_derivative,
                           padding=2)
        output22 = F.conv2d(input[:, c//2+c//4:, :, :], self.y_derivative,
                            padding=2)
        output2 = output21 + output22
        res =  torch.stack((output1, output2), dim=1)
        res = res[:, :, 0, :, :]
        return res


#new FullyCNN to solve boundary issue
#->
def replicate_nans(_x):
  '''
  x is 4D tensor N x C x Ny x Nx
  returns nans filled with nearest neighbour
  We assume that NaNs occur consistently
  in the full NxC dimensitons
  '''
  mask_array = torch.isnan(_x[0,0,:,:]).clone()
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
  x = _x.clone()
  for j in range(x.shape[-2]):
    for i in range(x.shape[-1]):
      if mask(j,i):
        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          continue # nothing to interoplate

        x[:,:,j,i] = 0.
        n = 0
        if (not mask(j,i+1)):
          x[:,:,j,i] += x[:,:,j,i+1]
          n += 1

        if (not mask(j,i-1)):
          x[:,:,j,i] += x[:,:,j,i-1]
          n += 1

        if (not mask(j+1,i)):
          x[:,:,j,i] += x[:,:,j+1,i]
          n += 1

        if (not mask(j-1,i)):
          x[:,:,j,i] += x[:,:,j-1,i]
          n += 1

        if (not mask(j+1,i+1)):
          x[:,:,j,i] += x[:,:,j+1,i+1]
          n += 1

        if (not mask(j+1,i-1)):
          x[:,:,j,i] += x[:,:,j+1,i-1]
          n += 1

        if (not mask(j-1,i+1)):
          x[:,:,j,i] += x[:,:,j-1,i+1]
          n += 1

        if (not mask(j-1,i-1)):
          x[:,:,j,i] += x[:,:,j-1,i-1]
          n += 1

        if n==0:
          print('Error')
        else:
          x[:,:,j,i] *= 1/n
  return x

def replicate_mat(maskn):
  '''
  create coefficient matrix A for maskn
  Then filling nan in x with nearest neighbour
  can be achieved by A*x
  '''
  mask_array = torch.isnan(maskn[0,0,:,:]).clone()
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
    
  idi = []
  idj = []
  val = []
  
  for j in range(ny):
    for i in range(nx):
      if mask(j,i):

        n = 0
        k = j*nx+i

        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          idi.append(k)
          idj.append(k)

        if (not mask(j,i+1)):
          idi.append(k+1)
          idj.append(k)
          n += 1

        if (not mask(j,i-1)):
          idi.append(k-1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i)):
          idi.append(k+nx)
          idj.append(k)
          n += 1

        if (not mask(j-1,i)):
          idi.append(k-nx)
          idj.append(k)
          n += 1

        if (not mask(j+1,i+1)):
          idi.append(k+nx+1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i-1)):
          idi.append(k+nx-1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i+1)):
          idi.append(k-nx+1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i-1)):
          idi.append(k-nx-1)
          idj.append(k)
          n += 1

        if n==0:
          val.append(torch.tensor(float('nan')))
        else:
          n_val = [1/n] * n
          val.extend(n_val)
      else:
          k = j*nx+i
          idi.append(k)
          idj.append(k)
          val.append(1)

  indices = torch.tensor([idj,idi])
  values = torch.tensor(val, dtype=torch.float32)
  A = torch.sparse_coo_tensor(indices=indices, values=values, size=[nx*ny,nx*ny])
  return A

def replicate_mat_new(maskn):
  '''
  create coefficient matrix A for mask0 (without include any nan)
  mask0 has values either 1 or 0
  can be achieved by A*x to fill land points with nearest neighbours
  '''
  mask_array = torch.isnan(maskn[0,0,:,:]).clone()
#   mask_array = maskn[0,0,:,:] == 0
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
    
  idi = []
  idj = []
  val = []
  
  for j in range(ny):
    for i in range(nx):
      if mask(j,i):

        n = 0
        k = j*nx+i

        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          idi.append(k)
          idj.append(k)

        if (not mask(j,i+1)):
          idi.append(k+1)
          idj.append(k)
          n += 1

        if (not mask(j,i-1)):
          idi.append(k-1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i)):
          idi.append(k+nx)
          idj.append(k)
          n += 1

        if (not mask(j-1,i)):
          idi.append(k-nx)
          idj.append(k)
          n += 1

        if (not mask(j+1,i+1)):
          idi.append(k+nx+1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i-1)):
          idi.append(k+nx-1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i+1)):
          idi.append(k-nx+1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i-1)):
          idi.append(k-nx-1)
          idj.append(k)
          n += 1

        if n==0:
          val.append(torch.tensor(float(0.0)))
        else:
          n_val = [1/n] * n
          val.extend(n_val)
      else:
          k = j*nx+i
          idi.append(k)
          idj.append(k)
          val.append(1)

  indices = torch.tensor([idj,idi])
  values = torch.tensor(val, dtype=torch.float32)
  A = torch.sparse_coo_tensor(indices=indices, values=values, size=[nx*ny,nx*ny])
  return A

def replicate_mat_zero(mask0):
  '''
  create coefficient matrix A for mask0 (without include any nan)
  mask0 has values either 1 or 0
  can be achieved by A*x to fill land points with nearest neighbours
  '''
  mask_array = mask0[0,0,:,:] == 0
  ny, nx = mask_array.shape
  def mask(j,i):
    if j < ny and j > -1 and i < nx and i > -1:
      return mask_array[j,i]
    else:
      return True
    
  idi = []
  idj = []
  val = []
  
  for j in range(ny):
    for i in range(nx):
      if mask(j,i):

        n = 0
        k = j*nx+i

        if mask(j,i+1) and mask(j,i-1) and mask(j+1,i) and mask(j-1,i) and mask(j+1,i+1) and mask(j+1,i-1) and mask(j-1,i+1) and mask(j-1,i-1):
          idi.append(k)
          idj.append(k)

        if (not mask(j,i+1)):
          idi.append(k+1)
          idj.append(k)
          n += 1

        if (not mask(j,i-1)):
          idi.append(k-1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i)):
          idi.append(k+nx)
          idj.append(k)
          n += 1

        if (not mask(j-1,i)):
          idi.append(k-nx)
          idj.append(k)
          n += 1

        if (not mask(j+1,i+1)):
          idi.append(k+nx+1)
          idj.append(k)
          n += 1

        if (not mask(j+1,i-1)):
          idi.append(k+nx-1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i+1)):
          idi.append(k-nx+1)
          idj.append(k)
          n += 1

        if (not mask(j-1,i-1)):
          idi.append(k-nx-1)
          idj.append(k)
          n += 1

        if n==0:
          val.append(torch.tensor(float(0.0)))
        else:
          n_val = [1/n] * n
          val.extend(n_val)
      else:
          k = j*nx+i
          idi.append(k)
          idj.append(k)
          val.append(1)

  indices = torch.tensor([idj,idi])
  values = torch.tensor(val, dtype=torch.float32)
  A = torch.sparse_coo_tensor(indices=indices, values=values, size=[nx*ny,nx*ny])
  return A

def replicate_nans_new(x,A):
  '''
  x is 4D tensor N x C x Ny x Nx
  returns nans filled with nearest neighbour
  We assume that NaNs occur consistently
  in the full NxC dimensitons
  We use A*x to update x
  '''
#   print('x.shape',_x.shape)
  _x=x.clone()
#   _A=A.clone()
#   flat_x = _x.reshape(_x.size(0), _x.size(1), -1)
#   print(flat_x.shape)
#   print('_x.size(2)* _x.size(3)',_x.size(2)* _x.size(3))
  flat_x = _x.reshape(-1, _x.size(2)* _x.size(3))
#   print(flat_x.shape)
  # print(A.shape)
  xx = torch.sparse.mm(A, flat_x.T).T
  # print(xx.shape)
  _x = xx.reshape(_x.size(0), _x.size(1), _x.size(2), _x.size(3))
  return _x


class FullyCNN_BC(DetectOutputSizeMixin, Module):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        super(FullyCNN_BC, self).__init__()
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(n_in_channels, 128, 5, padding=padding_5)
        if self.batch_norm:
          self.batch_norm1 = nn.BatchNorm2d(self.conv1.out_channels) 
        self.conv2 = nn.Conv2d(128, 64, 5, padding=padding_5)
        if self.batch_norm:
          self.batch_norm2 = nn.BatchNorm2d(self.conv2.out_channels) 
        self.conv3 = nn.Conv2d(64, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm3 = nn.BatchNorm2d(self.conv3.out_channels) 
        self.conv4 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm4 = nn.BatchNorm2d(self.conv4.out_channels) 
        self.conv5 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm5 = nn.BatchNorm2d(self.conv5.out_channels) 
        self.conv6 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm6 = nn.BatchNorm2d(self.conv6.out_channels) 
        self.conv7 = nn.Conv2d(32, 32, 3, padding=padding_3)
        if self.batch_norm:
          self.batch_norm7 = nn.BatchNorm2d(self.conv7.out_channels) 
        self.conv8 = nn.Conv2d(32, n_out_channels, 3, padding=padding_3)
        self.relu = nn.ReLU()

    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x, maskn=None, replicate=True, matrix_dict=None,use_cuda=False,gpu_id=0):
        if maskn is not None:
            mask0=maskn.nan_to_num(0.)
        else:
            mask0 = None
        halo = 0
        # _cache = 0 #only conv(mask) in first time
        # halo_old = 0 #old way
        # x_old=x.clone() #old way
        for i in range(1, 9): #8 depth
            # #old way
            # conv = getattr(self, f'conv{i}') #get the conv layer name
            # if i<8 and self.batch_norm:
            #     batchnorm = getattr(self, f'batch_norm{i}') #get the batch norm layer name
            # if maskn is not None:
            #     #to adapt the mask size to the input of each layer
            #     mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x_old.shape[0], x_old.shape[1], -1, -1)
            #     mask_halo = mask[:, :, halo_old:-halo_old, halo_old:-halo_old] if halo_old > 0 else mask
            #     #fill values to land points by nearest values
            #     if replicate is True:
            #       for n in range(1,conv.kernel_size[0]//2+1):# 3x3 kernel needs one replicate and 5x5 needs two replicates
            #         if f'A_{i}_{n}_old' not in matrix_dict:
            #             matrix_dict[f'A_{i}_{n}_old'] = replicate_mat(x_old*mask_halo) if n == 1 else replicate_mat(x_old)
            #             if use_cuda:
            #                 matrix_dict[f'A_{i}_{n}_old']=matrix_dict[f'A_{i}_{n}_old'].cuda(gpu_id)
            #         x_old = replicate_nans_new(x_old,matrix_dict[f'A_{i}_{n}_old'])
            #         # print('x_old'+f'A_{i}_{n}',x_old[0,0,:,:])
            #     else:
            #         # print(i,'x_old',x_old[0,0,:,:])
            #         x_old = x_old * mask_halo
            #         # print(i,'x_old',x_old[0,0,:,:])
            # # print('before conv: x_old',x_old[0,0,:,:])
            # x_old = conv(x_old)
            # # print('after conv: x_old',x_old[0,0,:,:])
            # halo_old = halo_old + conv.kernel_size[0]//2
            # if maskn is not None and replicate is False:
            #     x_old = x_old.nan_to_num(0.)
            # if i<8:
            #   x_old = self.relu(x_old)
            #   x_old = batchnorm(x_old) if self.batch_norm else x_old
            # else:
            #   if maskn is not None:
            #     mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x_old.shape[0], x_old.shape[1], -1, -1)
            #     x_old = x_old.nan_to_num(0.)*mask[:, :, halo_old:-halo_old, halo_old:-halo_old]
            #   x_old = self.final_transformation(x_old)
            
            # #new way
            # conv = getattr(self, f'conv{i}') #get the conv layer name
            # if i<8 and self.batch_norm:
            #     batchnorm = getattr(self, f'batch_norm{i}') #get the batch norm layer name
            # if maskn is not None:
            #     #to adapt the mask size to the input of each layer
            #     mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
            #     mask_halo = mask[:, :, halo:-halo, halo:-halo] if halo > 0 else mask
            #     #fill values to land points by nearest values
            #     if replicate is True:
            #       for n in range(1,conv.kernel_size[0]//2+1):# 3x3 kernel needs one replicate and 5x5 needs two replicates
            #         if f'A_{i}_{n}' not in matrix_dict:
            #             _cache += 1
            #             if i>1 and n==1:
            #                 mask_halo = mask_halo*mask_halo1
            #                 mask_halo[(mask_halo != 0) & (~torch.isnan(mask_halo))] = 1.0
            #             matrix_dict[f'A_{i}_{n}'] = replicate_mat_new(mask_halo) #create matrix and save in a tuple only in first time
            #             # print('mask_halo'+f'A_{i}_{n}',mask_halo[0,0,:,:])
            #             if use_cuda:
            #                 matrix_dict[f'A_{i}_{n}']=matrix_dict[f'A_{i}_{n}'].cuda(gpu_id)
            #             mask_halo = replicate_nans_new(mask_halo,matrix_dict[f'A_{i}_{n}'])
            #             # print('mask_halo'+f'A_{i}_{n}',mask_halo[0,0,:,:])
            #         x = replicate_nans_new(x,matrix_dict[f'A_{i}_{n}'])
            #         x[(x != 0) & (~torch.isnan(x))] = 1.0
            #         print('x'+f'A_{i}_{n}',x[0,0,:,:])
            #         # print(i,n,x[0,0,:,:])
            #     else:
            #         x = x * mask_halo.nan_to_num(0.)
            # # x = x.nan_to_num(0.)
            # # print('before conv: x',x[0,0,:,:])
            # x = conv(x)
            # if maskn is not None and _cache>0:
            #     # print(f'A_{i+1}_{1}')
            #     mask_halo1 = conv(mask_halo)
            #     # print(mask_halo1[0,0,:,:])
            # # print('after conv: x',x[0,0,:,:])
            # halo = halo + conv.kernel_size[0]//2
            # if maskn is not None and replicate is False:
            #     x = x.nan_to_num(0.)
            # if i<8:
            #   x = self.relu(x)
            #   x = batchnorm(x) if self.batch_norm else x
            #   if maskn is not None and _cache>0:
            #       mask_halo1 = self.relu(mask_halo1)
            #       mask_halo1 = batchnorm(mask_halo1) if self.batch_norm else mask_halo1
            # else:
            #   if maskn is not None:
            #       mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
            #       x = x.nan_to_num(0.)*mask[:, :, halo:-halo, halo:-halo].nan_to_num(0.)
            #   x = self.final_transformation(x)

            # #new way (no nan involved)
            conv = getattr(self, f'conv{i}') #get the conv layer name
            if i<8 and self.batch_norm:
                batchnorm = getattr(self, f'batch_norm{i}') #get the batch norm layer name
            if mask0 is not None:
                #to adapt the mask size to the input of each layer
                mask = mask0[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
                mask_halo = mask[:, :, halo:-halo, halo:-halo] if halo > 0 else mask
                #fill values to land points by nearest values
                if replicate is True:
                  for n in range(1,conv.kernel_size[0]//2+1):# 3x3 kernel needs one replicate and 5x5 needs two replicates
                    if f'A_{i}_{n}' not in matrix_dict:
                        matrix_dict[f'A_{i}_{n}'] = replicate_mat_zero(mask_halo) #create matrix and save in a tuple only in first time
                        # print('mask_halo'+f'A_{i}_{n}',mask_halo[0,0,:,:])
                        if use_cuda:
                            matrix_dict[f'A_{i}_{n}']=matrix_dict[f'A_{i}_{n}'].cuda(gpu_id)
                        mask_halo = replicate_nans_new(mask_halo,matrix_dict[f'A_{i}_{n}'])
                        # print('mask_halo'+f'A_{i}_{n}',mask_halo[0,0,:,:])
                    # print(i,n,x[0,0,:,:])
                    x = replicate_nans_new(x,matrix_dict[f'A_{i}_{n}'])
                    # print(i,n,x[0,0,:,:])
                else:
                    # print(i,x[0,0,:,:])
                    x = x * mask_halo
                    # print(i,x[0,0,:,:])
            # x = x.nan_to_num(0.)
            # print('before conv: x',x[0,0,:,:])
            x = conv(x)
            # print('after conv: x',x[0,0,:,:])
            halo = halo + conv.kernel_size[0]//2
            if i<8:
              x = self.relu(x)
              x = batchnorm(x) if self.batch_norm else x
            else:
              if mask0 is not None:
                  mask = mask0[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
                  x = x.nan_to_num(0.)*mask[:, :, halo:-halo, halo:-halo].nan_to_num(0.)
              x = self.final_transformation(x)

        # #Compare old and new
        # # x_old=x_old
        # x_new = x.clone()
        # x_new[(torch.isnan(x_old))] = torch.tensor(float('nan'))
        # x_old=x_old.nan_to_num(0.)
        # # print('x_old',x_old[0,0,:,:])
        # x_new=x_new.nan_to_num(0.)
        # # print('x_new',x_new[0,0,:,:])
        # are_equal = torch.equal(x_old, x_new)
        # if are_equal is False:
        #    print('old and new is same:',are_equal)
        return self.final_transformation(x), matrix_dict
    
    # def forward(self, x, maskn=None, replicate=True, matrix_dict=None,use_cuda=False,gpu_id=0):
    #     x = self.conv1(x)
    #     x = self.relu(x)
    #     x = self.conv2(x)
    #     x = self.relu(x)
    #     x = self.conv3(x)
    #     x = self.relu(x)
    #     x = self.conv4(x)
    #     x = self.relu(x)
    #     x = self.conv5(x)
    #     x = self.relu(x)
    #     x = self.conv6(x)
    #     x = self.relu(x)
    #     x = self.conv7(x)
    #     x = self.relu(x)
    #     x = self.conv8(x)
    #     x = self.final_transformation(x)
            
    #     return self.final_transformation(x), matrix_dict
#<-


class FullyCNN(DetectOutputSizeMixin, Sequential):

    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 128, 5, padding=padding_5)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(128, 64, 5, padding=padding_5)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(64, 32, 3, padding=padding_3)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(32, 32, 3, padding=padding_3)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(32, n_out_channels, 3, padding=padding_3)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)

    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        # for i in range(15):
        #     x = self[i](x)
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
    

class CNN15x15(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 186, 3, padding=padding_3)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(186, 93, 3, padding=padding_3)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(93, 46, 3, padding=padding_3)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(46, 46, 3, padding=padding_3)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(46, 46, 3, padding=padding_3)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(46, 46, 3, padding=padding_3)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(46, 46, 2, padding=padding_2)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(46, n_out_channels, 2, padding=padding_2)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc


class CNN11x11(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 208, 3, padding=padding_3)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(208, 104, 3, padding=padding_3)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(104, 52, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(52, 52, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(52, 52, 2, padding=padding_2)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(52, 52, 2, padding=padding_2)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(52, 52, 2, padding=padding_2)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(52, n_out_channels, 2, padding=padding_2)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc


class CNN9x9(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 271, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(271, 136, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(136, 68, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(68, 68, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(68, 68, 2, padding=padding_2)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(68, 68, 2, padding=padding_2)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(68, 68, 2, padding=padding_2)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(68, n_out_channels, 2, padding=padding_2)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc


class CNN7x7(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 279, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(279, 140, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(140, 70, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(70, 70, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(70, 70, 2, padding=padding_2)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(70, 70, 2, padding=padding_2)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(70, 70, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(70, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
        

#new CNN5x5 to solve boundary issue
#->
class CNN5x5_BC(DetectOutputSizeMixin, Module):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        super(CNN5x5_BC, self).__init__()
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm

        self.conv1 = nn.Conv2d(n_in_channels, 232, 3, padding=padding_3)
        self.batch_norm1 = nn.BatchNorm2d(self.conv1.out_channels) 
        self.conv2 = nn.Conv2d(232, 116, 3, padding=padding_3)
        self.batch_norm2 = nn.BatchNorm2d(self.conv2.out_channels) 
        self.conv3 = nn.Conv2d(116, 58, 1, padding=padding_1)
        self.batch_norm3 = nn.BatchNorm2d(self.conv3.out_channels) 
        self.conv4 = nn.Conv2d(58, 58, 1, padding=padding_1)
        self.batch_norm4 = nn.BatchNorm2d(self.conv4.out_channels) 
        self.conv5 = nn.Conv2d(58, 58, 1, padding=padding_1)
        self.batch_norm5 = nn.BatchNorm2d(self.conv5.out_channels) 
        self.conv6 = nn.Conv2d(58, 58, 1, padding=padding_1)
        self.batch_norm6 = nn.BatchNorm2d(self.conv6.out_channels) 
        self.conv7 = nn.Conv2d(58, 58, 1, padding=padding_1)
        self.batch_norm7 = nn.BatchNorm2d(self.conv7.out_channels) 
        self.conv8 = nn.Conv2d(58, n_out_channels, 1, padding=padding_1)
        self.relu = nn.ReLU()
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation
    
    def forward(self, x, maskn=None, replicate=True, matrix_dict=None,use_cuda=False,gpu_id=0):
        halo = 0
        for i in range(1, 9):
            # print('i:',i)
            # print('x.shape:',x.shape)
            # print('maskn:',maskn.shape)
            conv = getattr(self, f'conv{i}')
            if i<8:
                batchnorm = getattr(self, f'batch_norm{i}')
            if maskn is not None:
                mask = maskn[0,0,:,:].unsqueeze(0).unsqueeze(0).expand(x.shape[0], x.shape[1], -1, -1)
                # print('mask:',mask.shape)
                if halo == 0:
                    if replicate is True:
                        # import time
                        # print(maskn[0,0,:,:])
                        if f'A_{i}' not in matrix_dict:
                            matrix_dict[f'A_{i}'] = replicate_mat(x*mask)
                            if use_cuda:
                                matrix_dict[f'A_{i}']=matrix_dict[f'A_{i}'].cuda(gpu_id)
                        """
                        start_time1 = time.time()
                        reshaped_x = replicate_nans_new(x,matrix_dict[f'A_{i}'])
                        end_time1 = time.time()
                        print("--- %s seconds for new ---" % (end_time1 - start_time1))
                        # print(reshaped_x)
                        start_time2 = time.time()
                        x = replicate_nans(x * mask)
                        end_time2 = time.time()
                        print("--- %s seconds for old ---" % (end_time2 - start_time2))
                        # print(x)
                        x1=x.clone()
                        nan_mask = torch.isnan(x1)
                        x1[nan_mask] = 0.0
                        reshaped_x1=reshaped_x.clone()
                        nan_mask = torch.isnan(reshaped_x1)
                        reshaped_x1[nan_mask] = 0.0
                        print(torch.equal(x1[0,0,:,:],reshaped_x1[0,0,:,:]))
                        # print(torch.equal(x[0,0,:,:].nan_to_num(0.),reshaped_x[0,0,:,:].nan_to_num(0.)))
                        if torch.equal(x1[0,0,:,:],reshaped_x1[0,0,:,:]) is False:
                            raise ValueError("New and old methods for replicate are not equivilent.")
                        """
                        x = replicate_nans_new(x,matrix_dict[f'A_{i}'])
                    else:
                        x = x * mask.nan_to_num(0.)
                else:
                    if replicate is True:
                        if f'A_{i}' not in matrix_dict:
                            matrix_dict[f'A_{i}'] = replicate_mat(x*mask[:,:,halo:-halo,halo:-halo])
                            if use_cuda:
                                matrix_dict[f'A_{i}']=matrix_dict[f'A_{i}'].cuda(gpu_id)
                        """
                        # if i==2:
                        #   B = replicate_mat_update(x,mask[:,:,halo:-halo,halo:-halo],matrix_dict[f'A_{i}'])
                        # if i==2:
                        #   print('x:',x[0,0,:,:])
                        #   print(matrix_dict['A_2'])
                        start_time1 = time.time()
                        reshaped_x = replicate_nans_new(x,matrix_dict[f'A_{i}'])
                        end_time1 = time.time()
                        print("--- %s seconds for new ---" % (end_time1 - start_time1))
                        # if i==2:
                        #   print(reshaped_x[0,0,:,:])
                        start_time2 = time.time()
                        x = replicate_nans(x * mask[:,:,halo:-halo,halo:-halo])
                        end_time2 = time.time()
                        print("--- %s seconds for old ---" % (end_time2 - start_time2))
                        # if i==2:
                        #   print(x[0,0,:,:])
                        x1=x.clone()
                        nan_mask = torch.isnan(x1)
                        x1[nan_mask] = 0.0
                        reshaped_x1=reshaped_x.clone()
                        nan_mask = torch.isnan(reshaped_x1)
                        reshaped_x1[nan_mask] = 0.0
                        print(torch.equal(x1[0,0,:,:],reshaped_x1[0,0,:,:]))
                        # print(torch.equal(x[0,0,:,:].nan_to_num(0.),reshaped_x[0,0,:,:].nan_to_num(0.)))
                        if torch.equal(x1[0,0,:,:],reshaped_x1[0,0,:,:]) is False:
                            raise ValueError("New and old methods for replicate are not equivilent.")
                        """
                        x = replicate_nans_new(x,matrix_dict[f'A_{i}'])
                    else:
                        x = x * mask[:,:,halo:-halo,halo:-halo].nan_to_num(0.)
            # print(x[0,0,:,:])
            x = conv(x)
            if maskn is not None and replicate is False:
                x = x.nan_to_num(0.)
            # print(x[0,0,:,:])
            if i<8:
              x = self.relu(x)
              x = batchnorm(x) if self.batch_norm else x
            halo = halo + conv.kernel_size[0]//2
        return self.final_transformation(x), matrix_dict
#<-


class CNN5x5(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 296, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(296, 148, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(148, 74, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(74, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
    

class CNN5x5_c1(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 148, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(148, 74, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(74, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
    

class CNN5x5_c2(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 74, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(74, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc


class CNN5x5_c3(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 232, 3, padding=padding_3)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(232, 116, 3, padding=padding_3)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(116, 58, 1, padding=padding_1)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(58, 58, 1, padding=padding_1)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(58, 58, 1, padding=padding_1)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(58, 58, 1, padding=padding_1)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(58, 58, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(58, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
    

class CNN5x5_d1(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 296, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(296, 148, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(148, 74, 2, padding=padding_2)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(74, 74, 2, padding=padding_2)
        block4 = self._make_subblock(conv4)
        # conv5 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        # block5 = self._make_subblock(conv5)
        # conv6 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        # block6 = self._make_subblock(conv6)
        # conv7 = torch.nn.Conv2d(74, 74, 1, padding=padding_1)
        # block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(74, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, 
                             conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
        

class CNN3x3(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 327, 2, padding=padding_2)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(327, 164, 2, padding=padding_2)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(164, 82, 1, padding=padding_1)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(82, 82, 1, padding=padding_1)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(82, 82, 1, padding=padding_1)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(82, 82, 1, padding=padding_1)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(82, 82, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(82, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
        

class CNN1x1(DetectOutputSizeMixin, Sequential):
    def __init__(self, n_in_channels: int = 2, n_out_channels: int = 4,
                 padding=None, batch_norm=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
            padding_2 = 0
            padding_1 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
            padding_2 = 0
            padding_1 = 0
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.n_in_channels = n_in_channels
        self.batch_norm = batch_norm
        conv1 = torch.nn.Conv2d(n_in_channels, 528, 1, padding=padding_1)
        block1 = self._make_subblock(conv1)
        conv2 = torch.nn.Conv2d(528, 264, 1, padding=padding_1)
        block2 = self._make_subblock(conv2)
        conv3 = torch.nn.Conv2d(264, 132, 1, padding=padding_1)
        block3 = self._make_subblock(conv3)
        conv4 = torch.nn.Conv2d(132, 132, 1, padding=padding_1)
        block4 = self._make_subblock(conv4)
        conv5 = torch.nn.Conv2d(132, 132, 1, padding=padding_1)
        block5 = self._make_subblock(conv5)
        conv6 = torch.nn.Conv2d(132, 132, 1, padding=padding_1)
        block6 = self._make_subblock(conv6)
        conv7 = torch.nn.Conv2d(132, 132, 1, padding=padding_1)
        block7 = self._make_subblock(conv7)
        conv8 = torch.nn.Conv2d(132, n_out_channels, 1, padding=padding_1)
        Sequential.__init__(self, *block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        
    @property
    def final_transformation(self):
        return self._final_transformation

    @final_transformation.setter
    def final_transformation(self, transformation):
        self._final_transformation = transformation

    def forward(self, x):
        x = super().forward(x)
        
        # mean,prec = torch.split(x,x.shape[1]//2,dim = 1)
        # prec = torch.nn.functional.softplus(prec)
        # return mean,prec
        # Temporary fix for the student loss
        return self.final_transformation(x)

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc


class MixedModel(Module):

    net_cls = FullyCNN

    def __init__(self, *args, **kwargs):
        if 'n_in_channels' in kwargs:
            kwargs['n_in_channels'] -= 2
        args = list(args)
        if len(args) > 0:
            args[0] -= 2
        args = tuple(args)
        self.net = self.net_cls(*args, **kwargs)
        self.n_in_channels = self.net.n_in_channels + 2

    def forward(self, x):
        uv = x[:, :2, ...]
        equations = x[:, 2:, ...]
        out = self.net.forward(uv)
        equations = self.crop_like(equations, out)
        out[:, 0, ...] = (out[:, 0, ...]) * equations[:, 0, ...]
        out[:, 1, ...] = (out[:, 1, ...]) * equations[:, 1, ...]
        return out

    def crop_like(self, x, y):
        shape_x = x.shape
        shape_y = y.shape
        m = (shape_x[-2] - shape_y[-2]) // 2
        n = (shape_x[-1] - shape_y[-1]) // 2
        return x[..., m: shape_x[-2] - m, n: shape_x[-1] - n]

    def __getattr__(self, attr_name):
        return getattr(self.net, attr_name)

    def __setattr__(self, key, value):
        if key == 'net' or key == 'n_in_channels':
            self.__dict__[key] = value
        else:
            setattr(self.net, key, value)

    def __repr__(self):
        return self.net.__repr__()


if __name__ == '__main__':
    import numpy as np

    net = FullyCNN()
    net._final_transformation = lambda x: x
    input_ = torch.randint(0, 10, (17, 2, 35, 30)).to(dtype=torch.float)
    input_[0, 0, 0, 0] = np.nan
    output = net(input_)
