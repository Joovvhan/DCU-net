from glob import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
from dataloader import MultiStreamLoader, concat_variable_length_files, get_data_loader

import torch
from torch import nn
from torch import optim

from model import UNet

from tensorboardX import SummaryWriter

if __name__ == "__main__":
    wav_files = sorted(glob('./data/background/YD/*.wav'))
    background_stream_loader = MultiStreamLoader(wav_files)

    speech_files = sorted(glob('./data/speech/KSS/*/*.wav'))
    before_folding = len(speech_files)
    speech_files = concat_variable_length_files(speech_files)
    print(f'{before_folding:5} => {len(speech_files):5}')

    background_stream_loader = MultiStreamLoader(wav_files)
    dataloader = get_data_loader([f[0] for f in speech_files], 
                                 background_stream_loader)
    
    '''
    Our implementation of U-Net is similar to that of [11]. 
    Each encoder layer consists of a strided 2D convolution of stride 2 and kernel size 5x5, 
    batch normalization, and leaky rectified linear units (ReLU) with leakiness 0.2. 
    In the decoder we use strided deconvolution (sometimes referred to as transposed convolution) 
    with stride 2 and kernel size 5x5, batch normalization, plain ReLU, 
    and use 50% dropout to the first three layers, as in [11]. 
    In the final layer we use a sigmoid activation function. 
    The model is trained using the ADAM [12] optimizer.
    '''


    writer = SummaryWriter()
    
    l1_loss = nn.L1Loss(reduction='mean')
    l2_loss = nn.MSELoss(reduction='mean')

    signal_model = UNet()
    noise_model = UNet()

    signal_optimizer = optim.Adam(signal_model.parameters(), lr=0.0001)
    noise_optimizer = optim.Adam(noise_model.parameters(), lr=0.0001)
    
    steps = 0

    signal_loss_1 = list()
    signal_loss_2 = list()
    
    noise_loss_1 = list()
    noise_loss_2 = list()
    
    
    for i in range(100):

        for i, batch in enumerate(dataloader):
            Zxxs, log_spectrograms = batch

            signal, noise, mixed = log_spectrograms

            output = signal_model(mixed)
            loss1 = l1_loss(output, signal)
            loss2 = l2_loss(output, signal)
            loss = loss1 + loss2
            loss.backward()
            signal_optimizer.step()
            signal_optimizer.zero_grad()
            
            signal_loss_1.append(loss1.item())
            signal_loss_2.append(loss2.item())
            
            output = noise_model(mixed)
            loss1 = l1_loss(output, noise)
            loss2 = l2_loss(output, noise)
            loss = loss1 + loss2
            loss.backward()
            noise_optimizer.step()
            noise_optimizer.zero_grad()
            
            noise_loss_1.append(loss1.item())
            noise_loss_2.append(loss2.item())

            steps += 1
            
            if steps % 100 == 0:
                writer.add_scalar('speech/L1', np.mean(signal_loss_1), steps)
                writer.add_scalar('speech/L2', np.mean(signal_loss_2), steps)
                writer.add_scalar('noise/L1', np.mean(noise_loss_1), steps)
                writer.add_scalar('noise/L2', np.mean(noise_loss_2), steps)
                
                signal_loss_1 = list()
                signal_loss_2 = list()

                noise_loss_1 = list()
                noise_loss_2 = list()
            
            print(loss.item())