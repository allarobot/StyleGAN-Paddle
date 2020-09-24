# coding: UTF-8
"""
    @author: samuel ko
    @date:   2019.04.11
    @notice:
             1) refactor the module of Gsynthesis with
                - LayerEpilogue.
                - Upsample2d.
                - GBlock.
                and etc.
             2) the initialization of every patch we use are all abided by the original NvLabs released code.
             3) Discriminator is a simplicity version of PyTorch.
             4) fix bug: default settings of batchsize.

"""
import numpy as np
import os
from collections import OrderedDict

import paddle.fluid.layers as layers
import paddle.fluid.dygraph as dygraph
import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import InstanceNorm#, Conv2D
import ops

class ApplyNoise(dygraph.Layer):
    def __init__(self, channels):
        super().__init__()
        self.weight = layers.create_parameter(shape=(channels,),
                                            dtype='float32',
                                            is_bias=True,
                                            default_initializer=fluid.initializer.ConstantInitializer(value=1.0))  # zeros???,需要对照TF

    def forward(self, x, noise):
        if noise is None:
            noise = layers.randn(x.shape[0], 1, x.shape[2], x.shape[3])
        x_ = layers.reshape(self.weight,(1, -1, 1, 1))
        #print("shape: ",x.shape,x_.shape,noise.shape)
        return x + x_ * noise


class ApplyStyle(dygraph.Layer):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels, use_wscale):
        super(ApplyStyle, self).__init__()
        self.linear = FC(latent_size,
                      channels * 2,
                      gain=1.0,
                      use_wscale=use_wscale)

    def forward(self, x, latent):
        style = self.linear(latent)  #[batch_size,latent_size(512)] x ==> style: [batch_size, n_channels*2]
        shape = [-1, 2, x.shape[1], 1, 1]
        style = layers.reshape(style, shape)   # [batch_size, 2, n_channels ,1 ,1] 
        x = x * (style[:, 0] + 1.) + style[:, 1] # x+x*style[0]+style[1]
        return x


class FC(dygraph.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 gain=2**(0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        """
            The complete conversion of Dense/FC/Linear Layer of original Tensorflow version.
        """
        super(FC, self).__init__()
        self.out_channels = out_channels

        he_std = gain * in_channels ** (-0.5)  # He init
        if use_wscale:
            # init_std = 1.0 / lrmul
            # self.w_lrmul = he_std * lrmul
            self.w_lrmul = lrmul
        else:
            # init_std = he_std / lrmul
            # self.w_lrmul = lrmul
            self.w_lrmul = 1.0

        w = np.random.randn(in_channels,out_channels)*he_std*self.w_lrmul
        self.weight_attr = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(w))
        #self.weight = layers.create_parameter((in_channels,out_channels),'float32')
        if bias:
            self.b_lrmul = lrmul
            b = np.random.randn(out_channels)*self.b_lrmul
            self.bias_attr = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(b))
            # self.bias = layers.create_parameter((out_channels,),'float32')
        else:
            self.bias_attr = False

        self.linear = dygraph.Linear(in_channels,out_channels,param_attr=self.weight_attr,bias_attr=self.bias_attr,dtype='float32')

    def forward(self, x):
        # print(type(x))
        # x = layers.fc(x,self.out_channels,num_flatten_dims=len(x.shape)-1,param_attr=self.weight_attr,bias_attr=self.bias_attr)
        x = fluid.layers.cast(x,dtype='float32')
        x = self.linear(x)
        x = layers.leaky_relu(x, alpha=0.2)
        return x


class Blur2d(dygraph.Layer):
    def __init__(self, f=[1,2,1], normalize=True, flip=False, stride=1):
        """
            depthwise_conv2d:
            https://blog.csdn.net/mao_xiao_feng/article/details/78003476
        """
        super(Blur2d, self).__init__()
        assert isinstance(f, list) or f is None, "kernel f must be an instance of python built_in type list!"

        if f is not None:
            f = layers.tensor(f, dtype='float32')
            print("f[:, None]: ",f[:, None])
            print("f[None, :]: ",f[None, :])
            f = f[:, None] * f[None, :]
            print("f=f[:, None] * f[None, :]: ",f)
            f = f[None, None]
            print("then f[None, None]: ",f)
            if normalize:
                f = f / f.sum()
            if flip:
                # f = f[:, :, ::-1, ::-1]
                f = layers.flip(f, [2, 3])
            self.f = f
        else:
            self.f = None
        self.stride = stride

    def forward(self, x):
        if self.f is not None:
            # expand kernel channels
            kernel = self.f.expand(x.size(1), -1, -1, -1)
            x = layers.conv2d(
                x,
                kernel,
                stride=self.stride,
                padding=int((self.f.shape[2]-1)/2),
                groups=x.shape[1]
            )
            return x
        else:
            return x


class Conv2d(dygraph.Layer):
    def __init__(self,
                 input_channels,
                 output_channels,
                 kernel_size,
                 gain=2 ** (0.5),
                 use_wscale=False,
                 lrmul=1.0,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.output_channels = output_channels

        he_std = gain * (input_channels * kernel_size ** 2) ** (-0.5)  # He init
        if use_wscale:
            init_std = 1.0 / lrmul
            self.w_lrmul = he_std * lrmul
        else:
            init_std = he_std / lrmul
            self.w_lrmul = lrmul

        w = np.random.randn(output_channels,input_channels,kernel_size,kernel_size)*he_std
        self.weight_attr = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(w))

        if bias:
            self.b_lrmul = lrmul
            b = np.random.randn(output_channels)*self.b_lrmul
            self.bias_attr = fluid.ParamAttr(initializer=fluid.initializer.NumpyArrayInitializer(b))
        else:
            self.bias_attr = False
        self.conv2d = dygraph.Conv2D(input_channels,output_channels,kernel_size,padding=kernel_size//2,param_attr=self.weight_attr,bias_attr=self.bias_attr)

    def forward(self, x):
        return self.conv2d(x)

class Upscale2d(dygraph.Layer):
    def __init__(self, factor=2, gain=1):
        """
            the first upsample method in G_synthesis.
        :param factor:
        :param gain:
        """
        super().__init__()
        self.gain = gain
        self.factor = factor

    def forward(self, x):
        if self.gain != 1:
            x = x * self.gain
        if self.factor > 1:
            shape = x.shape
            x = layers.expand(layers.reshape(x,(shape[0], shape[1], shape[2], 1, shape[3], 1)),(1, 1, 1, self.factor, 1, self.factor))
            x = layers.reshape(x,(shape[0], shape[1], self.factor * shape[2], self.factor * shape[3]))
        return x

class PixelNorm(dygraph.Layer):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        tmp  = layers.elementwise_mul(x, x) # or x ** 2
        tmp1 = layers.sqrt(layers.reduce_mean(tmp, dim=1, keep_dim=True) + self.epsilon)

        return x * tmp1


# class InstanceNorm(dygraph.Layer):
#     def __init__(self, epsilon=1e-8):
#         """
#             @notice: avoid in-place ops.
#             https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
#         """
#         super(InstanceNorm, self).__init__()
#         self.epsilon = epsilon

#     def forward(self, x):
#         mean =layers.reduce_mean(x, dim=(2, 3), keep_dim=True)
#         tmp = layers.reduce_mean((x-mean)**2,dim=(2,3),keep_dim=True) # or x ** 2
#         tmp = layers.sqrt(tmp + self.epsilon)
#         return x * tmp


class LayerEpilogue(dygraph.Layer):
    def __init__(self,
                 channels,
                 dlatent_size,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 use_styles):
        '''
        noise and style AdaIN operation
        '''
        super(LayerEpilogue, self).__init__()
        #print("channels: ", channels)
        if use_noise:
            self.noise = ApplyNoise(channels)
        self.act = ops.Leaky_ReLU()

        if use_pixel_norm:
            self.pixel_norm = PixelNorm()
        else:
            self.pixel_norm = None

        if use_instance_norm:
            self.instance_norm = InstanceNorm(channels)
        else:
            self.instance_norm = None

        if use_styles:
            self.style_mod = ApplyStyle(dlatent_size, channels, use_wscale=use_wscale)
        else:
            self.style_mod = None

    def forward(self, x, noise, dlatents_in_slice=None):
        x = self.noise(x, noise)
        x = self.act(x)
        if self.pixel_norm is not None:
            x = self.pixel_norm(x)
        if self.instance_norm is not None:
            x = self.instance_norm(x)
        if self.style_mod is not None:
            x = self.style_mod(x, dlatents_in_slice)

        return x


class GBlock(dygraph.Layer):
    def __init__(self,
                 res,
                 use_wscale,
                 use_noise,
                 use_pixel_norm,
                 use_instance_norm,
                 noise_input,        # noise
                 dlatent_size=512,   # Disentangled latent (W) dimensionality.
                 use_style=True,     # Enable style inputs?
                 f=None,        # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 factor=2,           # upsample factor.
                 fmap_base=8192,     # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,     # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,       # Maximum number of feature maps in any layer.
                 ):
        super(GBlock, self).__init__()
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

        # res
        self.res = res

        # # blur2d
        # self.blur = Blur2d(f)

        # noise
        self.noise_input = noise_input

        if res < 7:
            # upsample method 1
            self.up_sample = Upscale2d(factor)
        else:
            # upsample method 2
            self.up_sample = dygraph.Conv2DTranspose(self.nf(res-3), self.nf(res-2), 4, stride=2, padding=1)

        # A Composition of LayerEpilogue and Conv2d.
        self.conv1  = Conv2d(input_channels=self.nf(res-2), output_channels=self.nf(res-2),
                              kernel_size=3, use_wscale=use_wscale)
        self.adaIn1 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv2  = Conv2d(input_channels=self.nf(res-2), output_channels=self.nf(res-2),
                              kernel_size=3, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(res-2), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)

    def forward(self, x, dlatent):
        x = self.up_sample(x)
        x = self.conv1(x)
        x = self.adaIn1(x, self.noise_input[self.res*2-4], dlatent[:, self.res*2-4])
        x = self.conv2(x)
        x = self.adaIn2(x, self.noise_input[self.res*2-3], dlatent[:, self.res*2-3])
        return x


class G_mapping(dygraph.Layer):
    def __init__(self,
                 mapping_fmaps=512,       # Z space dimensionality
                 dlatent_size=512,        # W space dimensionality
                 resolution=1024,         # image resolution
                 normalize_latents=True,  # Normalize latent vectors (Z) before feeding them to the mapping layers?
                 use_wscale=True,         # Enable equalized learning rate?
                 lrmul=0.01,              # Learning rate multiplier for the mapping layers.
                 gain=2**(0.5)            # original gain in tensorflow.
                 ):
        '''
        The mapping of generator, it will product w for style y afterwards.
        parameters: 
        - mapping_fmaps: default 512, Z space dimensionality
        - dlatent_size: default 512, W space dimensionality
        - resolution: default 1024 image resolution
        - normalize_latents: default True,  Normalize latent vectors (Z) before feeding them to the mapping layers?
        - use_wscale: default True,Enable equalized learning rate?
        - lrmul: default 0.01, Learning rate multiplier for the mapping layers.
        - gain: default 2**(0.5), original gain in tensorflow.
        returns:
        a tensor with size dlatent_size 
        '''

        super(G_mapping, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.func = dygraph.Sequential(*[
            FC(self.mapping_fmaps, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale),
            FC(dlatent_size, dlatent_size, gain, lrmul=lrmul, use_wscale=use_wscale)
            ]
        )

        self.normalize_latents = normalize_latents

        # 4^2~1024^2 --> 2~18layers
        # - 2 means we start from feature map with height and width equals 4.
        # as this example, 1024*1024 pixel image we get num_layers = 18.
        self.resolution_log2 = int(np.log2(resolution))
        self.num_layers = self.resolution_log2 * 2 - 2 

        self.pixel_norm = PixelNorm()


    def forward(self, x):
        if self.normalize_latents:
            x = self.pixel_norm(x)
        out = self.func(x)
        return out, self.num_layers


class G_synthesis(dygraph.Layer):
    def __init__(self,
                 dlatent_size,                       # Disentangled latent (W) dimensionality.
                 resolution=1024,                    # Output resolution (1024 x 1024 by default).
                 fmap_base=8192,                     # Overall multiplier for the number of feature maps.
                 num_channels=3,                     # Number of output color channels.
                 structure='fixed',                  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, 'auto' = select automatically.
                 fmap_max=512,                       # Maximum number of feature maps in any layer.
                 fmap_decay=1.0,                     # log2 feature map reduction when doubling the resolution.
                 f=None,                             # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 use_pixel_norm      = False,        # Enable pixelwise feature vector normalization?
                 use_instance_norm   = True,         # Enable instance normalization?
                 use_wscale = True,                  # Enable equalized learning rate?
                 use_noise = True,                   # Enable noise inputs?
                 use_style = True                    # Enable style inputs?
                 ):                             # batch size.
        """
        synthesis of generator, the second part of gnerator
        parameters:
        dlatent_size: 512 Disentangled latent(W) dimensionality.
        resolution: 1024 x 1024.
        fmap_base:
        num_channels:
        structure: only support 'fixed' mode.
        fmap_max:
        """
        super(G_synthesis, self).__init__()

        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        self.structure = structure

        # - 2 means we start from feature map with height and width equals 4.
        # as this example, we get num_layers = 18.
        self.resolution_log2 = int(np.log2(resolution))
        num_layers = self.resolution_log2 * 2 - 2

        self.num_layers = num_layers

        # Noise inputs.
        self.noise_inputs = []
        for layer_idx in range(num_layers): #2~18
            res = layer_idx // 2 + 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noise_inputs.append(layers.randn(shape))

        # Blur2d
        self.blur = Blur2d(f)

        # torgb: fixed mode
        # channel 16 -> channel 8
        self.channel_shrinkage = Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2),3,use_wscale=use_wscale) 
        # channel 8 -> channel 3
        self.torgb = Conv2d(self.nf(self.resolution_log2), num_channels, 1, gain=1, use_wscale=use_wscale) 

        # Initial Input Block
        self.const_input = layers.create_parameter((1, self.nf(1), 4, 4),'float32',default_initializer=fluid.initializer.ConstantInitializer(value=1.0))
        self.bias = layers.create_parameter((self.nf(1),),'float32',default_initializer=fluid.initializer.ConstantInitializer(value=1.0))
        self.adaIn1 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise,
                                    use_pixel_norm, use_instance_norm, use_style)
        self.conv1  = Conv2d(self.nf(1), self.nf(1),3, gain=1, use_wscale=use_wscale)
        self.adaIn2 = LayerEpilogue(self.nf(1), dlatent_size, use_wscale, use_noise, use_pixel_norm,
                                    use_instance_norm, use_style)

        # Common Block
        # 4 x 4 -> 8 x 8
        res = 3
        self.GBlock1 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 8 x 8 -> 16 x 16
        res = 4
        self.GBlock2 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 16 x 16 -> 32 x 32
        res = 5
        self.GBlock3 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 32 x 32 -> 64 x 64
        res = 6
        self.GBlock4 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 64 x 64 -> 128 x 128
        res = 7
        self.GBlock5 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 128 x 128 -> 256 x 256
        res = 8
        self.GBlock6 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 256 x 256 -> 512 x 512
        res = 9
        self.GBlock7 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

        # 512 x 512 -> 1024 x 1024
        res = 10
        self.GBlock8 = GBlock(res, use_wscale, use_noise, use_pixel_norm, use_instance_norm,
                              self.noise_inputs)

    def forward(self, dlatent):
        """
           dlatent: Disentangled latents (W), shape为[minibatch, num_layers, dlatent_size].
        :param dlatent:
        :return:
        """
        images_out = None
        # Fixed structure: simple and efficient, but does not support progressive growing.
        if self.structure == 'fixed':
            # initial block 0:
            x = layers.expand(self.const_input,(dlatent.shape[0], 1, 1, 1))
            x = x + layers.reshape(self.bias,(1, -1, 1, 1))
            x = self.adaIn1(x, self.noise_inputs[0], dlatent[:, 0])
            x = self.conv1(x)
            x = self.adaIn2(x, self.noise_inputs[1], dlatent[:, 1])

            # block 1:
            # 4 x 4 -> 8 x 8
            x = self.GBlock1(x, dlatent)

            # block 2:
            # 8 x 8 -> 16 x 16
            x = self.GBlock2(x, dlatent)

            # block 3:
            # 16 x 16 -> 32 x 32
            x = self.GBlock3(x, dlatent)

            # block 4:
            # 32 x 32 -> 64 x 64
            x = self.GBlock4(x, dlatent)

            # block 5:
            # 64 x 64 -> 128 x 128
            x = self.GBlock5(x, dlatent)

            # block 6:
            # 128 x 128 -> 256 x 256
            x = self.GBlock6(x, dlatent)

            # block 7:
            # 256 x 256 -> 512 x 512
            x = self.GBlock7(x, dlatent)

            # block 8:
            # 512 x 512 -> 1024 x 1024
            x = self.GBlock8(x, dlatent)

            x = self.channel_shrinkage(x)
            images_out = self.torgb(x)
            return images_out


class StyleGenerator(dygraph.Layer):
    def __init__(self,
                 mapping_fmaps=512,
                 style_mixing_prob=0.9,       # Probability of mixing styles during training. None = disable.
                 truncation_psi=0.7,          # Style strength multiplier for the truncation trick. None = disable.
                 truncation_cutoff=8,         # Number of layers for which to apply the truncation trick. None = disable.
                 **kwargs
                 ):
        super(StyleGenerator, self).__init__()
        self.mapping_fmaps = mapping_fmaps
        self.style_mixing_prob = style_mixing_prob
        self.truncation_psi = truncation_psi
        self.truncation_cutoff = truncation_cutoff

        self.mapping = G_mapping(self.mapping_fmaps, **kwargs)
        self.synthesis = G_synthesis(self.mapping_fmaps, **kwargs)

    def forward(self, latents1):
        dlatents1, num_layers = self.mapping(latents1)
        # let [N, O] -> [N, num_layers, O]
        dlatents1 = layers.unsqueeze(dlatents1,1)
        #print("dlatents1 shape:",dlatents1.shape,dlatents1.dtype,(-1, int(num_layers), -1))
        dlatents1 = layers.expand(dlatents1,(1, int(num_layers), 1))

        # Apply truncation trick.
        # coefs = [0.8]*truncation_cutoff+ [1]*(num_layers-truncation_cutoff)
        if self.truncation_psi and self.truncation_cutoff:
            coefs = np.ones([1, num_layers, 1], dtype=np.float32)
            for i in range(num_layers):
                if i < self.truncation_cutoff:
                    coefs[:, i, :] *= self.truncation_psi
            
            # print("shapes: ",dlatents1.shape,coefs.shape)  #shapes:  [1, 18, 512] (1, 18, 1)
            dlatents1 = dlatents1*dygraph.to_variable(coefs)

        # 18 row -> 18 layers
        img = self.synthesis(dlatents1)
        return img


class StyleDiscriminator(dygraph.Layer):
    def __init__(self,
                 resolution=1024,
                 fmap_base=8192,
                 num_channels=3,
                 structure='fixed',  # 'fixed' = no progressive growing, 'linear' = human-readable, 'recursive' = efficient, only support 'fixed' mode now.
                 fmap_max=512,
                 fmap_decay=1.0,
                 # f=[1, 2, 1]       # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 f=None         # (Huge overload, if you dont have enough resouces, please pass it as `f = None`)Low-pass filter to apply when resampling activations. None = no filtering.
                 ):
        """

            if H or W >= 128, we use avgpooling2d to do feature map shrinkage.
            else: we use ordinary conv2d.
        """
        super().__init__()
        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4
        self.nf = lambda stage: min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        # fromrgb: fixed mode
        self.fromrgb = dygraph.Conv2D(num_channels, self.nf(self.resolution_log2-1), 1)
        self.structure = structure

        # blur2d
        self.blur2d = Blur2d(f)

        # down_sample
        self.down1 = dygraph.Pool2D(pool_size=2,pool_stride=2,pool_type='avg')
        self.down21 = dygraph.Conv2D(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-5), 2, stride=2)
        self.down22 = dygraph.Conv2D(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-6), 2, stride=2)
        self.down23 = dygraph.Conv2D(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-7), 2, stride=2)
        self.down24 = dygraph.Conv2D(self.nf(self.resolution_log2-8), self.nf(self.resolution_log2-8), 2, stride=2)

        # conv1: padding=same
        self.conv1 = Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-1), 3)
        self.conv2 = Conv2d(self.nf(self.resolution_log2-1), self.nf(self.resolution_log2-2), 3)
        self.conv3 = Conv2d(self.nf(self.resolution_log2-2), self.nf(self.resolution_log2-3), 3)
        self.conv4 = Conv2d(self.nf(self.resolution_log2-3), self.nf(self.resolution_log2-4), 3)
        self.conv5 = Conv2d(self.nf(self.resolution_log2-4), self.nf(self.resolution_log2-5), 3)
        self.conv6 = Conv2d(self.nf(self.resolution_log2-5), self.nf(self.resolution_log2-6), 3)
        self.conv7 = Conv2d(self.nf(self.resolution_log2-6), self.nf(self.resolution_log2-7), 3)
        self.conv8 = Conv2d(self.nf(self.resolution_log2-7), self.nf(self.resolution_log2-8), 3)

        # calculate point:
        self.conv_last = Conv2d(self.nf(self.resolution_log2-8), self.nf(1), 3)
        self.dense0 = dygraph.Linear(fmap_base, self.nf(0))
        self.dense1 = dygraph.Linear(self.nf(0), 1)
        #self.sigmoid = dygraph.Sigmoid()

    def forward(self, input):
        if self.structure == 'fixed':
            x = layers.leaky_relu(self.fromrgb(input), 0.2)
            # 1. 1024 x 1024 x nf(9)(16) -> 512 x 512
            res = self.resolution_log2
            x = layers.leaky_relu(self.conv1(x), 0.2)
            x = layers.leaky_relu(self.down1(self.blur2d(x)), 0.2)

            # 2. 512 x 512 -> 256 x 256
            res -= 1
            x = layers.leaky_relu(self.conv2(x), 0.2)
            x = layers.leaky_relu(self.down1(self.blur2d(x)), 0.2)

            # 3. 256 x 256 -> 128 x 128
            res -= 1
            x = layers.leaky_relu(self.conv3(x), 0.2)
            x = layers.leaky_relu(self.down1(self.blur2d(x)), 0.2)

            # 4. 128 x 128 -> 64 x 64
            res -= 1
            x = layers.leaky_relu(self.conv4(x), 0.2)
            x = layers.leaky_relu(self.down1(self.blur2d(x)), 0.2)

            # 5. 64 x 64 -> 32 x 32
            res -= 1
            x = layers.leaky_relu(self.conv5(x), 0.2)
            x = layers.leaky_relu(self.down21(self.blur2d(x)), 0.2)

            # 6. 32 x 32 -> 16 x 16
            res -= 1
            x = layers.leaky_relu(self.conv6(x), 0.2)
            x = layers.leaky_relu(self.down22(self.blur2d(x)), 0.2)

            # 7. 16 x 16 -> 8 x 8
            res -= 1
            x = layers.leaky_relu(self.conv7(x), 0.2)
            x = layers.leaky_relu(self.down23(self.blur2d(x)), 0.2)

            # 8. 8 x 8 -> 4 x 4
            res -= 1
            x = layers.leaky_relu(self.conv8(x), 0.2)
            x = layers.leaky_relu(self.down24(self.blur2d(x)), 0.2)

            # 9. 4 x 4 -> point
            x = layers.leaky_relu(self.conv_last(x), 0.2)
            # N x 8192(4 x 4 x nf(1)).
            x = layers.reshape(x,(x.shape[0],-1))
            x = layers.leaky_relu(self.dense0(x), 0.2)
            # N x 1
            x = layers.leaky_relu(self.dense1(x), 0.2)
            return x

