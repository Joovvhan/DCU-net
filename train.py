from glob import glob
import scipy
import numpy as np
import matplotlib.pyplot as plt
from dataloader import MultiStreamLoader, concat_variable_length_files, get_data_loader, complex_tensor_to_audio

import torch
from torch import nn
from torch import optim

from model import UNet

import matplotlib
matplotlib.use('agg')

from tensorboardX import SummaryWriter

def mel_tensor_to_plt_image(tensor_list, step):
    
    assert len(tensor_list) == 5, f'Length of tensor list is not 5 ({len(tensor_list)})'

    titles = ['Mixed', 'Speech', 'Reconstructed Speech',
              'Background', 'Reconstructed Background']
    
    fig, axes = plt.subplots(5, 1, sharey=True, figsize=(20, 15))
    fig.suptitle(f'Mel-spectrogram from Step #{step:07d}', fontsize=24, y=0.95)
    axes = axes.flatten()
    for i in range(5):
        im = axes[i].imshow(tensor_list[i].T, origin='lower', aspect='auto')
        im.set_clim(0, 1)
        axes[i].axes.xaxis.set_visible(False)
        axes[i].axes.yaxis.set_visible(False)
        axes[i].set_title(titles[i])
    fig.colorbar(im, ax=axes, location='right')
    fig.canvas.draw()

    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    image_array = np.swapaxes(image_array, 0, 2)
    image_array = np.swapaxes(image_array, 1, 2)

    # plt.show()
    plt.close()

    return image_array

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
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    writer = SummaryWriter()
    
    l1_loss = nn.L1Loss(reduction='mean')
    l2_loss = nn.MSELoss(reduction='mean')

    signal_model = UNet().to(device)
    noise_model = UNet().to(device)

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
            signal, noise, mixed = signal.to(device), noise.to(device), mixed.to(device)
            
            # Zxxs[0].shape
            # log_spectrograms[0].shape
            # torch.Size([4, 381, 1021, 2])
            # torch.Size([4, 1, 381, 1021])

            output_signal, mask_signal = signal_model(mixed)
            loss1 = l1_loss(output_signal, signal)
            loss2 = l2_loss(output_signal, signal)
            loss = loss1 + loss2
            loss.backward()
            signal_optimizer.step()
            signal_optimizer.zero_grad()
            
            signal_loss_1.append(loss1.item())
            signal_loss_2.append(loss2.item())
            
            output_noise, mask_noise = noise_model(mixed)
            loss1 = l1_loss(output_noise, noise)
            loss2 = l2_loss(output_noise, noise)
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
                
                tensor_list = [mixed[0, 0], signal[0, 0], output_signal[0, 0].detach(),
                               noise[0, 0], output_noise[0, 0].detach()]
                image = mel_tensor_to_plt_image(tensor_list, steps)
                writer.add_image('spectrograms', image, steps)
                
                speech = complex_tensor_to_audio(Zxxs[0])[0]
                background = complex_tensor_to_audio(Zxxs[1])[0]
                mix = complex_tensor_to_audio(Zxxs[2])[0]
                
                # [4, 1, 381, 1021] => [4, 381, 1021] => [4, 381, 1021, 1]
                mask_signal = mask_signal.squeeze(1).unsqueeze(-1)
                mask_noise = mask_noise.squeeze(1).unsqueeze(-1)
                # print(mask_signal.shape, Zxxs[2].shape)
                
                reconstructed_speech = complex_tensor_to_audio(Zxxs[2] * mask_signal)[0]
                reconstructed_background = complex_tensor_to_audio(Zxxs[2] * mask_noise)[0]
                
                writer.add_audio('speech/input', speech, steps, 22050)
                writer.add_audio('background/input', background, steps, 22050)
                writer.add_audio('speech/output', reconstructed_speech, steps, 22050)
                writer.add_audio('background/output', reconstructed_background, steps, 22050)
                writer.add_audio('mix', mix, steps, 22050)
                
                signal_loss_1 = list()
                signal_loss_2 = list()

                noise_loss_1 = list()
                noise_loss_2 = list()
            
            print(loss.item())