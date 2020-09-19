import paddle.fluid as fluid
from paddle.fluid.layer_helper import LayerHelper
from paddle.fluid.dygraph import Conv2D,Linear, Pool2D, BatchNorm, PRelu, SpectralNorm, LayerNorm, InstanceNorm
from paddle.fluid.dygraph import Sequential
import paddle.fluid.dygraph.nn as nn
import numpy as np


def init_w():
    w_param_attrs = fluid.ParamAttr(
                                initializer = fluid.initializer.NormalInitializer(loc=0.0,scale=0.02),
                                learning_rate=1.0, #0.0001
                                regularizer=fluid.regularizer.L2Decay(0.0001),
                                trainable=True)
    return w_param_attrs

def init_bias(use_bias=True):
    if use_bias:
        bias_param_attrs = fluid.ParamAttr(
                                    initializer = fluid.initializer.ConstantInitializer(value=0.0),
                                    learning_rate=1.0, #0.0001
                                    #regularizer=fluid.regularizer.L2Decay(0.0001),
                                    trainable=True)
    else:
        bias_param_attrs = False

    return bias_param_attrs

class Conv2D_(Conv2D):
    def __init__(self,num_channels, num_filters, filter_size, stride=1, padding=0, use_bias=True, act=None):
        super(Conv2D_,self).__init__(num_channels, num_filters, filter_size, stride=stride, padding=padding, param_attr=init_w(),bias_attr=init_bias(use_bias),act=act)


class Linear_(Linear):
    def __init__(self, input_dim, output_dim, use_bias=True, act=None):
        super(Linear_,self).__init__(input_dim=input_dim, output_dim=output_dim, param_attr=init_w(),bias_attr=init_bias(use_bias), act=act)


class MLP(fluid.dygraph.Layer):
    def __init__(self,in_nc=64,out_nc=64,light=True,use_bias=True):
        super(MLP,self).__init__()
                # ops for  Gamma, Beta block
        self.light = light
        
        FC = [
                Linear(in_nc, out_nc,param_attr=init_w(),bias_attr=init_bias(use_bias),act='relu'),
                Linear(out_nc, out_nc,param_attr=init_w(),bias_attr=init_bias(use_bias), act='relu')
                ]
            
        self.gamma = Linear(out_nc, out_nc,param_attr=init_w(),bias_attr=init_bias(use_bias))  # FC256
        self.beta = Linear(out_nc, out_nc,param_attr=init_w(),bias_attr=init_bias(use_bias)) # FC256
        self.FC = Sequential(*FC)
  
    def forward(self,x):
        # alpha, beta
        if self.light:
            # 1/3,shape(N,256,64,64) -->(N,256,1,1)
            x_ = fluid.layers.adaptive_pool2d(x, 1,pool_type='avg') # shape=(N,256,1,1)
            x_ = fluid.layers.reshape(x_, shape=[x.shape[0], -1]) # shape=(N,256)  
            # 2/3~3/3, x2 (N,256)-->(N,256) by FC256
            x_ = self.FC(x_)
        else:
            # 1/3 (N,256,64,64)-->(N,256*64*64)
            x_ = fluid.layers.reshape(x, shape=[x.shape[0], -1]) # shape=(N,256*64*64)
            # 2/3 (N,64*64*256)-->(N,256), 2/3 (N,256)-->(N,256) by FC
            x_ = self.FC(x_)
        # (N,256)-->(N,256), parameters for AdaILN
        gamma, beta = self.gamma(x_), self.beta(x_) # gamma torch.Size([N, 256]) beta torch.Size([N, 256])

        return gamma,beta


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias=True):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, param_attr=init_w(),bias_attr=init_bias(use_bias)),
                       InstanceNorm(dim),
                       ReLU()                         
                       ]

        conv_block += [ReflectionPad2d(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, param_attr=init_w(),bias_attr=init_bias(use_bias)),
                       InstanceNorm(dim)
                       ]

        self.conv_block = Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias=True):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2d(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, param_attr=init_w(),bias_attr=init_bias(use_bias))
        self.norm1 = AdaILN(dim)
        self.relu1 = ReLU()
        self.pad2 = ReflectionPad2d(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, param_attr=init_w(),bias_attr=init_bias(use_bias))
        self.norm2 = AdaILN(dim)

    def forward(self, x_init, gamma, beta):
        x = self.pad1(x_init)
        x = self.conv1(x)
        x = self.norm1(x, gamma, beta)
        x = self.relu1(x)
        x = self.pad2(x)
        x = self.conv2(x)
        x = self.norm2(x, gamma, beta)
        return  x + x_init


class AdaILN(fluid.dygraph.Layer):

    def __init__(self, in_channels, eps=1e-5):
        super(AdaILN, self).__init__()
        self.eps = eps

        self.rho = fluid.dygraph.to_variable(0.9*np.ones((1, in_channels, 1, 1)).astype('float32'))
        #self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=False,
        #default_initializer=fluid.initializer.ConstantInitializer(0.9))
        #self.clip = RhoClipper(0.0,1.0)

    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input, gamma, beta):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        self.rho = (fluid.layers.clamp(self.rho-0.1,0.0, 1.0)) ## start training with smoothing clip
        #self.rho = (fluid.layers.clamp(self.rho,0.0, 1.0))
        out = self.rho * out_in + (1 - self.rho)*out_ln
        out=out*fluid.layers.unsqueeze(gamma,[2,3])+fluid.layers.unsqueeze(beta,[2,3])
        return out

class LIN(fluid.dygraph.Layer):

    def __init__(self, in_channels, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps
        self.rho = fluid.dygraph.to_variable(np.zeros((1, in_channels, 1, 1)).astype('float32'))  # 0.0 -- LN, 1.0 --IN
        self.gamma = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=False,
        default_initializer=fluid.initializer.ConstantInitializer(1.0))
        self.beta = self.create_parameter((1, in_channels, 1, 1), dtype='float32', is_bias=False,
        default_initializer=fluid.initializer.ConstantInitializer(0.0))
        #self.clip = RhoClipper(0.0,1.0)
        
    def var(self, input, dim):
        mean = fluid.layers.reduce_mean(input, dim, keep_dim=True)
        tmp = fluid.layers.reduce_mean((input - mean)**2, dim, keep_dim=True)
        return tmp

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2,3], keep_dim=True), self.var(input, dim =[2,3])
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1,2,3], keep_dim=True), self.var(input, dim=[1,2,3])
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        self.rho = (fluid.layers.clamp(self.rho-0.1,0.0, 1.0)) ## start training with smoothing clip
        #self.rho = (fluid.layers.clamp(self.rho,0.0, 1.0))
        out = self.rho * out_in + (1 - self.rho) * out_ln
        out = out * self.gamma + self.beta
        return out

class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale)
        return out

class RhoClipper(object):
    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.weight
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.weight = w

class BCEWithLogitsLoss():
    def __init__(self, weight=None, reduction='mean'):
        self.weight = weight
        self.reduction = 'mean'

    def __call__(self, x, label):
        out = fluid.layers.sigmoid_cross_entropy_with_logits(x, label)
        if self.reduction == 'sum':
            return fluid.layers.reduce_sum(out)
        elif self.reduction == 'mean':
            return fluid.layers.reduce_mean(out)
        else:
            return out
        
        
class Spectralnorm(fluid.dygraph.Layer):
    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = SpectralNorm(layer.weight.shape, dim=dim, power_iters=power_iters, eps=eps, dtype=dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out


class ReflectionPad2d(fluid.dygraph.Layer):
    def __init__(self, size):
        super(ReflectionPad2d, self).__init__()
        self.size = size

    def forward(self, x):
        return fluid.layers.pad2d(x, [self.size] * 4, mode="reflect")
    
        
class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self.inplace=inplace

    def forward(self, x):
        if self.inplace:
            x.set_value(fluid.layers.relu(x))
            return x
        else:
            y=fluid.layers.relu(x)
            return y
        
    
class Leaky_ReLU(fluid.dygraph.Layer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.leaky_relu = lambda x: fluid.layers.leaky_relu(x, alpha=alpha)

    def forward(self, x):
        return self.leaky_relu(x)
        

class SoftPlus(fluid.dygraph.Layer):
    def __init__(self):
        super(SoftPlus,self).__init__()
        self.softplus = lambda x: fluid.layers.softplus(x)

    def forward(self,input):
        x = self.softplus(input)
        return x