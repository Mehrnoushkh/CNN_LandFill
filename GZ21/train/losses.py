#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  2 23:08:26 2020

@author: arthur
In this module we define custom loss functions. In particular we define
a loss function based on the Gaussian likelihood with two parameters, 
mean and precision.
"""
import torch
from torch.nn.modules.loss import _Loss
from enum import Enum
from abc import ABC
import numpy as np
from torch.autograd import Function


class VarianceMode(Enum):
    variance = 0
    precision = 1



# DEPRECIATED
class HeteroskedasticGaussianLoss(_Loss):
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, 2, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        term1 = - 1 / 2 * torch.log(precision)
        term2 = 1 / 2 * (target - mean)**2 * precision
        return (term1 + term2).mean()


class StudentLoss(_Loss):
    def __init__(self, nu: float = 30, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Temporary fix
        input, nu = input
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.lgamma((nu + 1) / 2)
        term2 = 1 / 2 * torch.log(nu) + torch.lgamma(nu / 2)
        term3 = - torch.log(precision)
        temp = (target - mean) * precision
        term4 = (nu + 1) / 2 * torch.log(1 + 1 / nu * temp**2) 
        return term1 + term2 + term3 + term4

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        input, nu = input
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    def predict_mean(self, input: torch.Tensor):
        input, nu = input
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))


class CauchyLoss(_Loss):
    def __init__(self, n_target_channels: int = 1):
        super().__init__()
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        mean, scale = torch.split(input, self.n_target_channels, dim=1)
        term1 = - torch.log(scale)
        term2 = torch.log((target - mean)**2 + scale**2)
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))


class HeteroskedasticGaussianLossV2(_Loss):
    """Class for Gaussian likelihood"""

    def __init__(self, n_target_channels: int = 1, bias: float = 0.,
                 mode=VarianceMode.precision):
        super().__init__()
        self.n_target_channels = n_target_channels
        self.bias = bias
        self.mode = mode

    @property
    def n_required_channels(self):
        """Return the number of input channel required per target channel.
        In this case, two, one for the mean, another one for the precision"""
        return 2 * self.n_target_channels

    @property
    def channel_names(self):
        return ['S_x', 'S_y', 'S_xscale', 'S_yscale']

    @property
    def precision_indices(self):
        return list(range(self.n_target_channels, self.n_required_channels))

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        # Split the target into mean (first half of channels) and scale
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        if not torch.all(precision > 0):
            raise ValueError('Got a non-positive variance value. \
                             Pre-processed variance tensor was: \
                                 {}'.format(torch.min(precision)))
        if self.mode is VarianceMode.precision:
            term1 = - torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias))**2 * precision**2
        elif self.mode is VarianceMode.variance:
            term1 = torch.log(precision)
            term2 = 1 / 2 * (target - (mean + self.bias))**2 / precision**2
        return term1 + term2

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        lkhs = self.pointwise_likelihood(input, target)
        # Ignore nan values in targets.
        lkhs = lkhs[~torch.isnan(target)]
        return lkhs.mean()

    def predict(self, input: torch.Tensor):
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias

    def predict_mean(self, input: torch.Tensor):
        """Return the mean of the conditional distribution"""
        mean, precision = torch.split(input, self.n_target_channels, dim=1)
        return mean + self.bias


class HeteroskedasticGaussianLossV3(_Loss):
    """Loss to be used with transform2 from models/submodels.py"""

    def __init__(self, *args, **kargs):
        super().__init__()
        self.base_loss = HeteroskedasticGaussianLossV2(*args, **kargs)

    def __getattr__(self, name: str):
        try:
            # This is necessary as the class Module defines its own __getattr__
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_loss, name)

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        return self.base_loss.forward(input, target)

    def pointwise_likelihood(self, input: torch.Tensor, target: torch.Tensor):
        raw_loss = self._base_loss(input, target[:, :self.n_target_channels, ...])
        return raw_loss + torch.log(target[:, self.n_target_channels: self.n_target_channels + 1, ...])


class MultimodalLoss(_Loss):
    """General class for a multimodal loss. Each location on
    each channel can choose its mode independently."""

    def __init__(self, n_modes, n_target_channels, base_loss_cls,
                 base_loss_params=[], share_mode='C'):
        super().__init__()
        self.n_modes = n_modes
        self.n_target_channels = n_target_channels
        self.target_names = ['target' + str(i) for i in range(
            n_target_channels)]
        self.losses = []
        for i in range(n_modes):
            if i < len(base_loss_params):
                params = base_loss_params[i]
                self.losses.append(base_loss_cls(n_target_channels, **params))
            else:
                self.losses.append(base_loss_cls(n_target_channels))
        self.share_mode = share_mode

    @property
    def target_names(self):
        return self._target_names

    @target_names.setter
    def target_names(self, value):
        assert len(value) == self.n_target_channels
        self._target_names = value

    @property
    def n_required_channels(self):
        if self.share_mode == 'C':
            return sum(self.splits)

    @property
    def channel_names(self):
        """Automatically assigns names to output channels depending on the
        target names. For now not really implemented"""
        return [str(i) for i in range(self.n_required_channels)]

    @property
    def precision_indices(self):
        indices = []
        for i, loss in enumerate(self.losses):
            sub_indices = loss.precision_indices
            for j in range(len(sub_indices)):
                sub_indices[j] += self.n_modes * self.n_target_channels + i * loss.n_required_channels
            indices.extend(sub_indices)
        return indices

    @property
    def splits(self):
        """Return how to split the input to recover the different parts:
            - probabilities of the modes
            - quantities definining each mode
        """
        return ([self.n_modes, ] * self.n_target_channels 
                + [loss.n_required_channels for loss in self.losses])

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        splits = torch.split(input, self.splits, dim=1)
        probas, inputs = (splits[:self.n_target_channels],
                          splits[self.n_target_channels:])
        probas = [torch.softmax(proba, dim=1) for proba in probas]
        losses_values = []
        for i, (loss, input) in enumerate(zip(self.losses, inputs)):
            proba_i = torch.stack([proba[:, i, ...] for proba in probas], dim=1)
            loss_i = torch.log(proba_i) - loss.pointwise_likelihood(input, target)
            losses_values.append(loss_i)
        loss = torch.stack(losses_values, dim=2)
        final_loss = -torch.logsumexp(loss, dim=2)
        final_loss = final_loss.mean()
        return final_loss

    def predict(self, input: torch.Tensor):
        splits = torch.split(input, self.splits, dim=1)
        probas, inputs = (splits[:self.n_target_channels],
                          splits[self.n_target_channels:])
        probas = [torch.softmax(proba, dim=1) for proba in probas]
        predictions = [loss.predict(input) for loss, input in
                       zip(self.losses, inputs)]
        weighted_predictions = []
        for i, pred in enumerate(predictions):
            proba_i = torch.stack([proba[:, i, ...] for proba in probas], dim=1)
            weighted_predictions.append(proba_i * pred)
        final_predictions = sum(weighted_predictions)
        return final_predictions

class BimodalGaussianLoss(MultimodalLoss):
    """Class for a bimodal Gaussian loss."""

    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class BimodalStudentLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(2, n_target_channels, base_loss_cls=StudentLoss)


class TrimodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(3, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class PentamodalGaussianLoss(MultimodalLoss):
    def __init__(self, n_target_channels: int):
        super().__init__(5, n_target_channels,
                         base_loss_cls=HeteroskedasticGaussianLossV2)


class QuantileLoss(_Loss):

    def __init__(self, n_target_channels: int = 2, n_quantiles: int = 5):
        super().__init__()
        self.n_quantiles = n_quantiles
        self.n_target_channels = n_target_channels

    @property
    def n_required_channels(self):
        return (4 + self.n_quantiles) * self.n_target_channels

    def forward(self, input, target):
        quantiles_x = torch.cumsum(input[:, :self.n_quantiles, ...], dim=1)
        quantiles_y = torch.cumsum(input[:,
                                   self.n_quantiles: 2 *self.n_quantiles, ...],
                                   dim=1)
        # The below tensor indicates which quantiles the observed data
        # belongs to
        i_quantiles_x = torch.argmax((target[:, :1, ...] <= quantiles_x) * 1.,
                                     dim=1, keepdim=True)
        i_quantiles_y = torch.argmax((target[:, 1:2, ...] <= quantiles_y) * 1.,
                                     dim=1, keepdim=True)
        # lkh
        lkh_x = torch.log(input[:, :self.n_quantiles, ...])
        lkh_x = torch.gather(lkh_x, 1, i_quantiles_x)
        lkh_y = torch.log(input[:, self.n_quantiles: 2 * self.n_quantiles, ...])
        lkh_y = torch.gather(lkh_y, 1, i_quantiles_y)
        lkh = torch.cat((lkh_x, lkh_y), dim=1)
        # Treat the case i_quantile = 0 or i_quantile = self.n_quantiles - 1
        # which are modelled via Generalized Pareto
        lower_quantile = torch.cat((quantiles_x[:, :1, ...],
                                   quantiles_y[:, :1, ...]), dim=1)
        higher_quantile = torch.cat((quantiles_x[:, -1:, ...],
                                    quantiles_y[:, -1:, ...]), dim=1)
        pareto_right = - self.generalized_pareto(target,
                                               input[:, 2 * self.n_quantiles:
                                                        2 * self.n_quantiles
                                                        + 2,
                                               ...],
                                               input[:, 2 * self.n_quantiles + 2:
                                                        2 * self.n_quantiles + 4,
                                               ...],
                                               higher_quantile)
        pareto_left = - self.generalized_pareto(target,
                                              input[:, 2 * self.n_quantiles + 4:
                                                       2 * self.n_quantiles + 6,
                                              ...],
                                              input[:, 2 * self.n_quantiles + 6:
                                                       2 * self.n_quantiles + 8,
                                              ...],
                                              lower_quantile,
                                              direction='left')
        #lkh[target < lower_quantile] = - pareto_left[target < lower_quantile]
        #lkh[target > higher_quantile] = - pareto_right[target >
        # higher_quantile]
        lkh = torch.cat((lkh, pareto_left, pareto_right), dim=1)
        sel = (target < lower_quantile) * 2 + (target > higher_quantile) * 4
        sel[:, 1] += 1
        lkh = torch.gather(lkh, 1, sel)
        return lkh.mean()

    def quantiles(self, input):
        quantiles_x = torch.cumsum(input[:, :self.n_quantiles, ...], dim=1)
        quantiles_y = torch.cumsum(input[:,
                                   self.n_quantiles: 2 * self.n_quantiles, ...],
                                   dim=1)
        return quantiles_x, quantiles_y


    @property
    def precision_indices(self):
        return list(range(1, self.n_quantiles)) + list(
            range(self.n_quantiles + 1, self.n_required_channels))

    @staticmethod
    def generalized_pareto(x, shape, scale, location, direction = 'right'):
        if direction == 'right':
            z = torch.abs((x - location)) / scale
        if direction == 'left':
            z = torch.abs((location - x)) / scale
        return (-1-1/shape) * torch.log(1 + shape * z)

    def predict(self, input):
        quantiles_x = torch.cumsum(input[:, :self.n_quantiles, ...], dim=1)
        quantiles_y = torch.cumsum(input[:, self.n_quantiles:
                                            2 * self.n_quantiles, ...], dim=1)
        return torch.cat((quantiles_x[:,
                          self.n_quantiles // 2: self.n_quantiles // 2 + 1,
                          ...],
                         quantiles_y[:, self.n_quantiles // 2:
                                        self.n_quantiles // 2 + 1, ...]), dim=1)


class Tuckey_g_h_inverse(Function):

    @staticmethod
    def tuckey_g_h(z, g, h):
        out = 1 / g * torch.expm1(g * z) * torch.exp(h * z ** 2 / 2)
        return out

    @staticmethod
    def forward(ctx, z_tilda, g, h):
        """New inverse method based on dichotomial algorithm. Finds true
        inverse up to numerical precision"""
        g[g == 0.] = torch.finfo().eps
        min_ = -20
        max_ = 20
        min_ = torch.ones_like(z_tilda) * min_
        max_ = torch.ones_like(z_tilda) * max_
        middle = min_
        for i in range(100):
            old_middle = middle
            middle = (min_ + max_) / 2.
            if torch.all(abs(middle - old_middle) <= torch.finfo(
                    middle.dtype).eps):
                break
            value = Tuckey_g_h_inverse.tuckey_g_h(middle, g, h)
            max_[z_tilda < value] = middle[z_tilda < value]
            min_[z_tilda > value] = middle[z_tilda > value]
        middle = (min_ + max_) / 2
        middle[torch.isnan(z_tilda)] = np.nan
        ctx.save_for_backward(middle, g, h)
        return middle

    @staticmethod
    def old_forward(ctx, z_tilda, g, h):
        g[g == 0.] = torch.finfo().eps
        nodes = torch.linspace(-4, 4, 1000, device=z_tilda.device)
        nodes = nodes.reshape([1, ] * z_tilda.ndim + [1000, ])
        new_g = g.unsqueeze(-1)
        new_h = h.unsqueeze(-1)
        init_shape = z_tilda.shape
        z_tilda = z_tilda.unsqueeze(-1)
        node_values = Tuckey_g_h_inverse.tuckey_g_h(nodes, new_g, new_h)
        if torch.any(node_values.isnan()):
            indices = torch.nonzero(node_values.isnan())
            first_index = list(indices[0])
            print(first_index)
            print(g[first_index[:-1]])
            print(h[first_index[:-1]])
            print(nodes.flatten()[first_index[-1]])
            raise RuntimeError('Got nan in node values.')

        assert not torch.any(node_values.isinf()), "Got inf in node values"
        i_node = torch.argmax((z_tilda <= node_values) * 1., dim=-1,
                              keepdim=True)
        i_node[z_tilda > node_values[..., -1:]] = 999
        nodes = nodes.flatten()
        z = nodes[i_node]
        z = z.reshape(init_shape)
        ctx.save_for_backward(z, g, h)
        return z

    @staticmethod
    def backward(ctx, grad_output):
        z, g, h = ctx.saved_tensors
        if grad_output is None:
            return None
        d_input = 1 / Tuckey_g_h_inverse.d_tau_d_z(z, g, h)
        d_g = - Tuckey_g_h_inverse.d_tau_d_g(z, g, h) * d_input
        d_h = - Tuckey_g_h_inverse.d_tau_d_h(z, g, h) * d_input
        return d_input * grad_output, d_g * grad_output, d_h * grad_output

    @staticmethod
    def d_tau_d_g(z, g, h):
        out = - 1 / g * Tuckey_g_h_inverse.tuckey_g_h(z, g, h)
        out = out + 1 / g * z * torch.exp(g * z + 1 / 2 * h * z ** 2)
        return out

    @staticmethod
    def d_tau_d_h(z, g, h):
        return 1 / 2 * z ** 2 * Tuckey_g_h_inverse.tuckey_g_h(z, g, h)

    @staticmethod
    def d_tau_d_z(z, g, h):
        out = torch.exp(g * z + h * z ** 2 / 2)
        out = out + h * z * Tuckey_g_h_inverse.tuckey_g_h(z, g, h)
        return out


class TuckeyGandHloss(_Loss):
    def __init__(self, n_target_channels: int = 2):
        super().__init__()
        self.n_target_channels = n_target_channels
        self.inverse_tuckey = Tuckey_g_h_inverse()

    @property
    def n_required_channels(self):
        """Return the number of required channels for the input. For each
        component of the target, 4 input channels are required: 1 for the
        constant, one for the scale, two for the Tuckey g and h parameters"""
        return self.n_target_channels * 4

    def forward(self, input, target):
        epsilon, beta, g, h = torch.split(input, self.n_target_channels, dim=1)
        g, h = self._transform_g_h(g, h)
        z_tilda = (target - epsilon) * beta
        z = self.inverse_tuckey.apply(z_tilda, g, h)
        assert not torch.any(z.isnan()), "Got nan values in the inversion"
        assert not torch.any(z.isinf()), "Got inf values in the inversion"
        for_log = (h * z * 1 / g * torch.expm1(g * z) *
                   torch.exp(h * z ** 2 / 2)
                   + torch.exp(g * z + 1 / 2 * h * z ** 2))
        assert not torch.any(for_log.isnan()), "Got nan values in for_log"
        assert not torch.any(for_log.isinf()), "Got inf values in for inf"
        # print("Max of g", g.abs().max().item())
        # print("Min of g", g.abs().min().item())
        # print("Max of h", h.max().item())
        # print("Min of h", h.min().item())
        # print("Max of for_log", for_log.max().item())
        # print("Min of for_log", for_log.min().item())
        lkh = torch.log(for_log)
        lkh = lkh + 1 / 2 * z ** 2
        lkh = lkh - torch.log(beta)
        lkh = lkh.mean()
        return lkh

    @property
    def precision_indices(self):
        return [2, 3]

    @property
    def channel_names(self):
        return ['epsilon_Sx', 'epsilon_Sy', 'beta_Sx', 'beta_Sy',
                'g_Sx', 'g_Sy', 'h_Sx', 'h_Sy']

    def predict(self, input):
        epsilon, sigma, g, h = torch.split(input, self.n_target_channels, dim=1)
        g, h = self._transform_g_h(g, h)
        return epsilon

    def _transform_g_h(self, g, h):
        g = (torch.sigmoid(g) - 0.5) * 4
        h = torch.sigmoid((h - 2.))
        return g, h



if __name__ == '__main__':
    input = np.random.rand(1, 4, 1, 1)
    input[:, [0, 2]] -= 0.5
    input = torch.tensor(input)
    input.requires_grad = True
    target = (np.random.rand(1, 1, 1, 1) - 0.5)
    target = torch.tensor(target)
    tgh = TuckeyGandHloss(n_target_channels=1)
    z = tgh(input, target)
    print(z)

    from torch.autograd import gradcheck

    # gradcheck takes a tuple of tensors as input, check if your gradient
    # evaluated with these tensors are close enough to numerical
    # approximations and returns True if they all verify this condition.
    target.requires_grad = True

    test2 = gradcheck(tgh.forward, (input, target), eps=0.01, atol=0.01)
    print(test2)

