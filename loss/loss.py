# coding: UTF-8
"""
    @author: samuel ko
"""

import paddle.fluid.dygraph as dygraph
# import torch.autograd as autograd
import paddle.fluid.layers as layers
import paddle

import numpy as np

def gradient_penalty(x, y, f):
    # interpolation
    shape = [x.shape[0]] + [1] * (x.dim() - 1)
    alpha = layers.rand(shape)
    z = x + alpha * (y - x)

    # gradient penalty
    z = dygraph.to_variable(z)
    o = f(z)
    g = dygraph.grad(o, z, grad_outputs=layers.ones(o.size()), create_graph=True)[0].view(z.size(0), -1)
    gp = ((g.norm(p=2, dim=1) - 1)**2).mean()
    return gp


def R1Penalty(real_img, f):
    # gradient penalty
    reals = real_img
    reals.stop_gradient = False
    #reals = real_img
    real_logit = f(reals)
    apply_loss_scaling = lambda x: x * layers.exp(x * np.log(2.0,dtype='float32'))
    undo_loss_scaling = lambda x: x * layers.exp(-x * np.log(2.0,dtype='float32'))

    real_logit = apply_loss_scaling(layers.sum(real_logit))
    grads = dygraph.grad(real_logit, reals, create_graph=True)
    real_grads = layers.reshape(grads[0],(reals.shape[0], -1))
    real_grads = undo_loss_scaling(real_grads)
    r1_penalty = layers.sum(layers.elementwise_mul(real_grads, real_grads))
    return r1_penalty


def R2Penalty(fake_img, f):
    # gradient penalty
    fakes = fake_img
    fakes.stop_gradient = False
    #fakes = fake_img
    fake_logit = f(fake)
    #apply_loss_scaling = lambda x: x * layers.exp(x * torch.Tensor([np.float32(np.log(2.0))]))
    #undo_loss_scaling = lambda x: x * layers.exp(-x * torch.Tensor([np.float32(np.log(2.0))]))
    apply_loss_scaling = lambda x: x * layers.exp(x * np.log(2.0))
    undo_loss_scaling = lambda x: x * layers.exp(-x * np.log(2.0))

    fake_logit = apply_loss_scaling(layers.sum(fake_logit))
    grads = dygraph.grad(fake_logit, fakes,create_graph=True)
    fake_grads = layers.reshape(grads[0],(fakes.shape[0], -1))
    fake_grads = undo_loss_scaling(fake_grads)
    r2_penalty = layers.sum(layers.elementwise_mul(fake_grads, fake_grads))
    return r2_penalty
