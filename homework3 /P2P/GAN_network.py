import argparse
import os
import numpy as np
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            # 输入: (3, 256, 256)
            *discriminator_block(3, 64, normalize=False),  # (64, 128, 128)
            *discriminator_block(64, 128),                # (128, 64, 64)
            *discriminator_block(128, 256),               # (256, 32, 32)
            *discriminator_block(256, 512),               # (512, 16, 16)
            
            # 额外的卷积层
            nn.Conv2d(512, 512, 4, stride=2, padding=1),  # (512, 8, 8)
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最后的判别层
            nn.Conv2d(512, 1, 4, stride=1, padding=0),    # (1, 5, 5)
            nn.AdaptiveAvgPool2d(1),                      # (1, 1, 1)
            nn.Sigmoid()
        )

    def forward(self, img):
        # img shape: (batch_size, 3, 256, 256)
        validity = self.model(img)
        return validity.view(img.size(0), -1)                            ## 鉴别器返回的是一个[0, 1]间的概率


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )

        # Decoder
        self.upscore1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # 64 + 64 = 128 (因为concat)
        self.conv_up1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.upscore2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        # 32 + 32 = 64 (因为concat)
        self.conv_up2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.upscore3 = nn.Sequential(
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        # Final convolution
        self.final_conv = nn.Conv2d(16, 3, kernel_size=1)
        
        # Final activation
        self.tanh = nn.Tanh()  # 改用sigmoid确保输出在[0,1]范围

    def forward(self, x):
        # Encoder
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        conv3_out = self.conv3(conv2_out)

        # Decoder with concatenated skip connections
        upscore1_out = self.upscore1(conv3_out)
        # Concat skip connection
        upscore1_cat = torch.cat([upscore1_out, conv2_out], dim=1)
        upscore1_conv = self.conv_up1(upscore1_cat)

        upscore2_out = self.upscore2(upscore1_conv)
        # Concat skip connection
        upscore2_cat = torch.cat([upscore2_out, conv1_out], dim=1)
        upscore2_conv = self.conv_up2(upscore2_cat)

        upscore3_out = self.upscore3(upscore2_conv)
        final_conv = self.final_conv(upscore3_out)
        
        # Final output activation
        output = self.tanh(final_conv)
        return output