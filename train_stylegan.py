# coding: UTF-8
"""
    @author: samuel ko
"""
from networks_stylegan import StyleGenerator
from networks_stylegan import StyleDiscriminator
#from networks_gan import Generator, Discriminator
from utils.utils import plotLossCurve,save_image,LOG,save_checkpoint,load_checkpoint
from loss.loss import gradient_penalty, R1Penalty, R2Penalty
from opts.opts import TrainOptions, INFO
from dataset import data_loader
# from torchvision.utils import save_image
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import random
import os

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
import paddle.fluid.dygraph as dygraph
import paddle.fluid.optimizer as optim
from paddle.fluid.layers import exponential_decay
from ops import SoftPlus

# Set random seem for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
#torch.manual_seed(manualSeed)

# Hyper-parameters
CRITIC_ITER = 5
log = LOG()

def main(opts):
    # Create the data loader
    # loader = sunnerData.DataLoader(sunnerData.ImageDataset(
    #     root=[[opts.path]],
    #     transforms=transforms.Compose([
    #         sunnertransforms.Resize((1024, 1024)),
    #         sunnertransforms.ToTensor(),
    #         sunnertransforms.ToFloat(),
    #         #sunnertransforms.Transpose(sunnertransforms.BHWC2BCHW),
    #         sunnertransforms.Normalize(),
    #     ])),
    #     batch_size=opts.batch_size,
    #     shuffle=True,
    # )
    loader = data_loader()

    device = fluid.CUDAPlace(0) if opts.device=='GPU' else fluid.CPUPlace(0)
    with fluid.dygraph.guard(device):
                # Create the model
        start_epoch = 0
        G = StyleGenerator()
        D = StyleDiscriminator()

        # Load the pre-trained weight
        if os.path.exists(opts.resume):
            INFO("Load the pre-trained weight!")
            #state = fluid.dygraph.load_dygraph(opts.resume)
            state = load_checkpoint(opts.resume)
            G.load_dict(state['G'])
            D.load_dict(state['D'])
            start_epoch = state['start_epoch']
        else:
            INFO("Pre-trained weight cannot load successfully, train from scratch!")

        # # Multi-GPU support
        # if torch.cuda.device_count() > 1:
        #     INFO("Multiple GPU:" + str(torch.cuda.device_count()) + "\t GPUs")
        #     G = nn.DataParallel(G)
        #     D = nn.DataParallel(D)

        scheduler_D = exponential_decay(learning_rate=0.00001,decay_steps=1000, decay_rate=0.99)
        scheduler_G = exponential_decay(learning_rate=0.00001,decay_steps=1000, decay_rate=0.99)
        optim_D = optim.Adam(parameter_list=D.parameters(), learning_rate=scheduler_D)
        optim_G = optim.Adam(parameter_list=G.parameters(), learning_rate=scheduler_G)

        # Train
        fix_z = np.random.randn(opts.batch_size, 512)#.to(opts.device)
        fix_z = dygraph.to_variable(fix_z)
        softplus = SoftPlus()
        Loss_D_list = [0.0]
        Loss_G_list = [0.0]
        D.train()
        G.train()
        for ep in range(start_epoch, opts.epoch):
            bar = tqdm(loader())
            loss_D_list = []
            loss_G_list = []
            for i, data in enumerate(bar):
                # =======================================================================================================
                #   (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                # =======================================================================================================
                # Compute adversarial loss toward discriminator
                real_img = np.array([item for item in data ],dtype='float32').reshape((-1,3,1024,1024))
            #x,y = data[0] ## data的形状为[(x,y),...]

                D.clear_gradients()
                real_img = dygraph.to_variable(real_img)
                real_logit = D(real_img)
                z =np.float32(np.random.randn(real_img.shape[0],512))
                fake_img = G(dygraph.to_variable(z))
                #fake_logit = D(fake_img.detach())
                fake_logit = D(fake_img)
                d_loss = layers.mean(softplus(fake_logit))
                d_loss = d_loss + layers.mean(softplus(-real_logit))

                if opts.r1_gamma != 0.0:
                    r1_penalty = R1Penalty(real_img.detach(), D)
                    #r1_penalty = R1Penalty(real_img, D)
                    d_loss = d_loss + r1_penalty * (opts.r1_gamma * 0.5)

                if opts.r2_gamma != 0.0:
                    r2_penalty = R2Penalty(fake_img.detach(), D)
                    #r2_penalty = R2Penalty(fake_img, D)
                    d_loss = d_loss + r2_penalty * (opts.r2_gamma * 0.5)

                loss_D_list.append(d_loss.numpy())

                # Update discriminator
                d_loss.backward()
                optim_D.minimize(d_loss)

                # =======================================================================================================
                #   (2) Update G network: maximize log(D(G(z)))
                # =======================================================================================================
                if i % CRITIC_ITER == 0:
                    G.clear_gradients()
                    fake_logit = D(fake_img.detach())
                    g_loss = layers.mean(softplus(-fake_logit))
                    #print("g_loss",g_loss)
                    loss_G_list.append(g_loss.numpy())

                    # Update generator
                    g_loss.backward()
                    optim_G.minimize(g_loss)

                # Output training stats
                bar.set_description("Epoch {} [{}, {}] [G]: {} [D]: {}".format(ep, i+1, 52000, loss_G_list[-1], loss_D_list[-1]))

            # Save the result
            Loss_G_list.append(np.mean(loss_G_list))
            Loss_D_list.append(np.mean(loss_D_list))

            # Check how the generator is doing by saving G's output on fixed_noise
            G.eval()
            #fake_img = G(fix_z).detach().cpu()
            fake_img = G(fix_z).numpy().squeeze()
            log(f"fake_img.shape: {fake_img.shape}")
            save_image(fake_img, os.path.join(opts.det, 'images', str(ep) + '.png'))
            G.train()

            # Save model
            # print("type:",type(G.state_dict()).__name__)
            # print("type:",type(D.state_dict()).__name__)
            states = {
                'G': G.state_dict(),
                'D': D.state_dict(),
                'Loss_G': Loss_G_list,
                'Loss_D': Loss_D_list,
                'start_epoch': ep,
            }
            #dygraph.save_dygraph(state, os.path.join(opts.det, 'models', 'latest'))
            save_checkpoint(states,os.path.join(opts.det, 'models', 'latest.pp'))
            # scheduler_D.step()
            # scheduler_G.step()            

        # Plot the total loss curve
        Loss_D_list = Loss_D_list[1:]
        Loss_G_list = Loss_G_list[1:]
        plotLossCurve(opts, Loss_D_list, Loss_G_list)


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    opts = TrainOptions().parse()
    main(opts)
