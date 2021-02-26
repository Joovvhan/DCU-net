import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.conv_layers = nn.ModuleList([nn.Conv2d(1, 16, 5, 2),
              nn.Conv2d(16, 32, 5, 2),
              nn.Conv2d(32, 64, 5, 2),
              nn.Conv2d(64, 128, 5, 2),
              nn.Conv2d(128, 256, 5, 2),
              nn.Conv2d(256, 512, 5, 2)])
        
        self.conv_batch_norms = nn.ModuleList([nn.BatchNorm2d(16),
                                               nn.BatchNorm2d(32),
                                               nn.BatchNorm2d(64),
                                               nn.BatchNorm2d(128),
                                               nn.BatchNorm2d(256),
                                               nn.BatchNorm2d(512)])
    
        self.deconv_layers = nn.ModuleList([nn.ConvTranspose2d(512, 256, 5, 2),
                                            nn.ConvTranspose2d(512, 128, 5, 2),
                                            nn.ConvTranspose2d(256, 64, 5, 2),
                                            nn.ConvTranspose2d(128, 32, 5, 2),
                                            nn.ConvTranspose2d(64, 16, 5, 2),
                                            nn.ConvTranspose2d(32, 1, 5, 2)])
        
        self.deconv_batch_norms = nn.ModuleList([nn.BatchNorm2d(256),
                                                 nn.BatchNorm2d(128),
                                                 nn.BatchNorm2d(64),
                                                 nn.BatchNorm2d(32),
                                                 nn.BatchNorm2d(16),
                                                 nn.Sigmoid()])
        
        self.decoder_dropouts = nn.ModuleList([nn.Dropout(p=0.5),
                                               nn.Dropout(p=0.5),
                                               nn.Dropout(p=0.5)])
        
    def forward(self, input_tensor):
        
        tensor = input_tensor
        
        intermediate_tensor = list()
        
        for layer, norm in zip(self.conv_layers, self.conv_batch_norms):
            tensor = layer(tensor)
            tensor = F.leaky_relu(tensor, 0.2)
            tensor = norm(tensor)
            intermediate_tensor.append(tensor)
        
        for i, (layer, norm) in enumerate(zip(self.deconv_layers, self.deconv_batch_norms)):
            if i == 0:
                tensor = intermediate_tensor.pop()
            else:
                past_tensor = intermediate_tensor.pop()
                tensor = torch.cat((tensor, past_tensor), 1)
                                
            tensor = layer(tensor)

            tensor = F.relu(tensor)
            tensor = norm(tensor)
            
            if i < 3:
                tensor = self.decoder_dropouts[i](tensor)
            
        output_tensor = input_tensor * tensor
        
        return output_tensor, tensor.detach()
    
if __name__ == "__main__":
    model = UNet()
    