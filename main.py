import os
import torch
import torch.nn as nn # 定义神经网络需要
import torch.nn.functional as F # 自带某些函数
import torch.optim as optim # 自带优化器
import numpy as np
import matplotlib.pyplot as plt
import torchvision # 自带数据集
from torchvision import transforms # 数据的转换操作

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 28*28),
            nn.Tanh()
        )
    def forward(self, x):  # 长度为长100的噪声
        img = self.main(x)
        img = img.view(-1, 28, 28, 1)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.main(x)
        return x

def gen_img_plot(model, epoch, test_input): # 绘图函数
    prediction = np.squeeze(model(test_input).detach().cpu().numpy())
    fig = plt.figure(figsize=(4, 4))
    for i in range(16):
        plt.subplot(4, 4, i+1)
        plt.imshow((prediction[i]+1)/2)
        plt.axis('off')
    img_name = 'gan'+str(epoch)+'.jpg'
    plt.savefig(os.path.join('images',img_name), dpi=300)
    plt.show()

if __name__ == '__main__':
    # 对数据进行归一化（-1，1）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 0-1之间的tensor数据；channel, high, width
        transforms.Normalize(0.5, 0.5)
    ])
    train_ds = torchvision.datasets.MNIST('data',
                                        train=True,
                                        transform=transform,
                                        download=True)
    dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    test_input = torch.randn(16, 100, device=device)

    
    gen = Generator().to(device)
    dis = Discriminator().to(device)

    d_optim = torch.optim.Adam(dis.parameters(), lr=0.0001)
    g_optim = torch.optim.Adam(gen.parameters(), lr=0.0001)

    loss_fn = torch.nn.BCELoss()

    # 训练循环
    D_loss = []
    G_loss = []
    for epoch in range(20):
        d_epoch_loss = 0
        g_epoch_loss = 0
        count = len(dataloader)
        for step, (img,_) in enumerate(dataloader):
            img = img.to(device)
            size = img.size(0)
            random_noise = torch.randn(size, 100, device=device)
            # 判别器损失
            d_optim.zero_grad()
            real_output = dis(img)  # 判别器输入真实的图片，real_output对真实图片的预测结果
            d_real_loss = loss_fn(real_output,
                                torch.ones_like(real_output))   # 判别器在真实图像的损失
            d_real_loss.backward()
            # 生成器损失
            gen_img = gen(random_noise)
            fake_output = dis(gen_img.detach()) # 判别器输入生成器图片，fake_output对生成图片的预测结果
            d_fake_loss = loss_fn(fake_output,
                                torch.zeros_like(fake_output))   # 判别器在生成器图像上的损失
            d_fake_loss.backward()
            d_loss = d_real_loss + d_fake_loss
            d_optim.step()
            
            # 生成器优化
            g_optim.zero_grad()
            fake_output = dis(gen_img)
            g_loss = loss_fn(fake_output,
                            torch.ones_like(fake_output)) # 生成器损失
            g_loss.backward()
            g_optim.step()
            
            with torch.no_grad():
                d_epoch_loss += d_loss
                g_epoch_loss += g_loss
        with torch.no_grad():
            d_epoch_loss /= count
            g_epoch_loss /= count
            D_loss.append(d_epoch_loss)
            G_loss.append(g_epoch_loss)
            print('Epoch:', epoch)
            gen_img_plot(gen, epoch, test_input)

