import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

#读取数据
train_data = MNIST(root='./mnist/',train=True,transform=tfs.ToTensor())#60000张训练集
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy())#生成第第三张图片，显示的为彩色图像
#plt.show()

train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)#分批并打乱顺序

#定义生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=3, padding=1),  # (b, 16, 10, 10)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
            nn.Conv2d(16, 8, kernel_size=3, stride=2, padding=1),  # (b, 8, 3, 3)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, kernel_size=3, stride=2),  # (b, 16, 5, 5)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 8, kernel_size=5, stride=3, padding=1),  # (b, 8, 15, 15)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, kernel_size=2, stride=2, padding=1),  # (b, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

#定义判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 5,padding=2),  # (b, 32, 28, 28)
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2, stride=2),      # (b, 32, 14, 14)
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(32,64,5,padding=2),    #（b，64，14, 14）
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2,stride=2)         #（b，64，7, 7）
        )

        self.fc=nn.Sequential(
            nn.Linear(64*7*7,1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024,1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

#定义参数
gen=generator()
dis=discriminator()
loss_func=nn.BCELoss()              #用于二分类的一个损失函数
#loss_func=nn.CrossEntropyLoss()    #用于多分类的一个损失函数
loss_func_auto=nn.MSELoss()
d_optimizer=torch.optim.Adam(dis.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(gen.parameters(),lr=0.0003)

#开始训练
for epoch in range(5):
    D_loss=0
    G_loss=0
    for step, (img, label) in enumerate(train_loader):   #可同时获得索引和值
        #print(label.shape)
        size=img.shape[0]                #100
        #print(size)
        real_img=Variable(img)

       # #自动编码器训练
        #encoded, decoded = gen(real_img)
        #a_loss = loss_func_auto(decoded, real_img)  # 计算损失函数
        #g_optimizer.zero_grad()  # 梯度清零
        #a_loss.backward()  # 反向传播
        #g_optimizer.step()  # 梯度优化

        # 判别器训练

        real_label=Variable(torch.ones(size,1))          #真的图片label则为1
        false_label=Variable(torch.zeros(size,1))        #假的图片label则为0
        real_out=dis(real_img)
        d_loss_real=loss_func(real_out,real_label)       #输入真实图片的损失函数

        encoder,false_img=gen(img)                       #得到假的图片
        false_img=Variable(false_img)
        false_out=dis(false_img)
        d_loss_false=loss_func(false_out,false_label)     #计算输入假的图片的损失函数

        d_loss=d_loss_real+d_loss_false                   #总的损失函数包括假的图片和真的图片分别产生的损失函数
        d_optimizer.zero_grad()                           #梯度清零
        d_loss.backward()                                 #反向传播
        d_optimizer.step()                                #梯度优化更新判别器网络参数
        D_loss += d_loss.item()
        if step % 100 == 0:  # 每100步显示一次
            print('Epoch: ', epoch, '| d_loss: %.4f' % d_loss.data.numpy())

        #训练生成器
        encoded, decoded = gen(real_img)              #生成假的图片
        output=dis(decoded)                           #生成假的图片丢进判别器当中
        #print(type(output))
       # output=torch.LongTensor(output)
        g_loss = loss_func(output,real_label)         # 计算生成器的损失函数  假的图片与真实label的loss
        g_optimizer.zero_grad()                       # 梯度清零
        g_loss.backward()                             # 反向传播
        g_optimizer.step()                            # 梯度优化更新生成网络参数
        G_loss+=g_loss.item()

        #训练自动编码器

        if step % 100 == 0:  # 每100步显示一次
            print('Epoch: ', epoch,'| d_loss: %.4f' % d_loss.data.numpy(), '| g_loss: %.4f' % g_loss.data.numpy())
    print('epoch: {}, D_Loss: {:.6f}, G_Loss: {:.6f}'
          .format(epoch, D_loss / len(train_loader), G_loss / len(train_loader)))


#创建一个画布
f, a = plt.subplots(2, 10, figsize=(10, 2)) #初始化数字 在图表中创建子图显示的图像是2行5列的
plt.ion()
#在交互模式下：plt.plot(x)或plt.imshow(x)是直接出图像，不需要plt.show()
#如果在脚本中使用ion()命令开启了交互模式，没有使用ioff()关闭的话，则图像会一闪而过，并不会常留。要想防止这种情况，
# 需要在plt.show()之前加上ioff()命令。

# 用于查看原始数据
view_data = train_data.train_data[:10].view(-1,1,28,28).type(torch.Tensor)/255.
#print(view_data.shape)  10,1,28,28
for i in range(10):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)))
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())

encoded_data, decoded_data = gen(view_data)
for i in range(10):
    a[1][i].clear()
    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)))
    a[1][i].set_xticks(())
    a[1][i].set_yticks(())
plt.draw()
plt.pause(0.05)  # 暂停0.05秒
plt.ioff()
plt.show()
