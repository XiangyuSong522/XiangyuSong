import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from facades_dataset import FacadesDataset
from GAN_network import Generator, Discriminator, ImprovedGenerator
from torch.optim.lr_scheduler import StepLR

import argparse
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torchvision import datasets
from torch.autograd import Variable

def tensor_to_image(tensor):
    """
    Convert a PyTorch tensor to a NumPy array suitable for OpenCV.

    Args:
        tensor (torch.Tensor): A tensor of shape (C, H, W).

    Returns:
        numpy.ndarray: An image array of shape (H, W, C) with values in [0, 255] and dtype uint8.
    """
    # Move tensor to CPU, detach from graph, and convert to NumPy array
    image = tensor.cpu().detach().numpy()
    # Transpose from (C, H, W) to (H, W, C)
    image = np.transpose(image, (1, 2, 0))
    # Denormalize from [-1, 1] to [0, 1]
    image = (image + 1) / 2
    # Scale to [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    return image

def save_images(inputs, targets, outputs, folder_name, epoch, num_images=5):
    """
    Save a set of input, target, and output images for visualization.

    Args:
        inputs (torch.Tensor): Batch of input images.
        targets (torch.Tensor): Batch of target images.
        outputs (torch.Tensor): Batch of output images from the model.
        folder_name (str): Directory to save the images ('train_results' or 'val_results').
        epoch (int): Current epoch number.
        num_images (int): Number of images to save from the batch.
    """
    os.makedirs(f'{folder_name}/epoch_{epoch}', exist_ok=True)
    for i in range(num_images):
        # Convert tensors to images
        input_img_np = tensor_to_image(inputs[i])
        target_img_np = tensor_to_image(targets[i])
        output_img_np = tensor_to_image(outputs[i])

        # Concatenate the images horizontally
        comparison = np.hstack((input_img_np, target_img_np, output_img_np))

        # Save the comparison image
        cv2.imwrite(f'{folder_name}/epoch_{epoch}/result_{i + 1}.png', comparison)


def validate(model, dataloader, criterion, device, epoch, num_epochs):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for i, (image_rgb, image_semantic) in enumerate(dataloader):
            # Move data to the device
            image_rgb = image_rgb.to(device)
            image_semantic = image_semantic.to(device)

            # Forward pass
            outputs = model(image_rgb)
            # Apply sigmoid to outputs to convert them to probabilities (if needed)
            outputs = torch.sigmoid(outputs)  # Apply sigmoid for BCELoss

            # Compute the loss
            loss = criterion(outputs, image_semantic.float())  # Ensure targets are float type
            # Compute the loss
            # loss = criterion(outputs, image_semantic)
            val_loss += loss.item()

            # Save sample images every 5 epochs
            if epoch % 5 == 0 and i == 0:
                save_images(image_rgb, image_semantic, outputs, 'val_results', epoch)

    # Calculate average validation loss
    avg_val_loss = val_loss / len(dataloader)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}')

def main():
    """
    Main function to set up the training and validation processes.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--b1", type=float, default=0.5)
    parser.add_argument("--b2", type=float, default=0.999)
    parser.add_argument("--n_cpu", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=100)
    parser.add_argument("--img_size", type=int, default=256)  # 修改为256
    parser.add_argument("--channels", type=int, default=3)    # 修改为3
    parser.add_argument("--sample_interval", type=int, default=500)
    opt = parser.parse_args()

    # Set device to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 数据加载器设置
    train_dataset = FacadesDataset(list_file='train_list.txt')
    val_dataset = FacadesDataset(list_file='val_list.txt')
    
    train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=100, shuffle=False, num_workers=4)

    # 模型初始化
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)

    # 损失函数设置
    criterion = nn.BCELoss().to(device)
    l1_loss = nn.L1Loss().to(device)
    lambda_l1 = 8

    # 优化器设置
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    # 训练循环
    for epoch in range(800):
        generator.train()
        discriminator.train()
        
        for i, (img_rgb, img_semantic) in enumerate(train_loader):
            batch_size = img_rgb.size(0)
            
            # 将数据移至设备
            img_rgb = img_rgb.to(device)
            img_semantic = img_semantic.to(device)
            
            # 创建标签
            real_label = torch.ones(batch_size, 1).to(device)
            fake_label = torch.zeros(batch_size, 1).to(device)

            # ---------------------
            #  训练判别器
            # ---------------------
            optimizer_D.zero_grad()
            
            # 真实图像的损失
            real_out = discriminator(img_semantic)
            loss_real_D = criterion(real_out, real_label)
            
            # 生成假图像
            with torch.no_grad():
                fake_img = generator(img_rgb)
            fake_out = discriminator(fake_img.detach())
            loss_fake_D = criterion(fake_out, fake_label)
            
            # 判别器总损失
            loss_D = (loss_real_D + loss_fake_D) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            #  训练生成器
            # ---------------------
            optimizer_G.zero_grad()
            
            # 生成假图像并计算判别器输出
            fake_img = generator(img_rgb)
            fake_out = discriminator(fake_img)
            
            # 生成器损失
            adv_loss = criterion(fake_out, real_label)
            content_loss = l1_loss(fake_img, img_semantic)
            loss_G = adv_loss + lambda_l1 * content_loss

            # 反向传播和优化
            loss_G.backward()
            optimizer_G.step()

            # 保存训练结果和打印日志
            if epoch % 5 == 0 and i == 0:
                save_images(img_rgb, img_semantic, fake_img.detach(), 'train_results', epoch)
                print(f'Epoch [{epoch}/800], D_loss: {loss_D.item():.4f}, G_loss1: {adv_loss.item():.4f}, G_loss2: {lambda_l1 * content_loss.item():.4f}')

        # 每5个epoch保存一次模型
        if (epoch + 1) % 5 == 0:
            print(f'Saving models at epoch {epoch + 1}')
            torch.save(generator.state_dict(), f'checkpoints/pix2pix_Gmodel_epoch_{epoch + 1}.pth')
            torch.save(discriminator.state_dict(), f'checkpoints/pix2pix_Dmodel_epoch_{epoch + 1}.pth')
if __name__ == '__main__':
    main()
