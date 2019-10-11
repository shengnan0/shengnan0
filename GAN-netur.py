import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision.datasets import MNIST
from torchvision import transforms as tfs
import matplotlib.pyplot as plt
from torch.autograd import Variable
import numpy as np

#读取数据
train_data = MNIST(root='./mnist/',train=True,transform=tfs.ToTensor(),)#60000张训练集
print(train_data.train_data.size())     # (60000, 28, 28)
print(train_data.train_labels.size())   # (60000)
plt.imshow(train_data.train_data[0].numpy())#生成第第1张图片，显示为彩色
plt.show()
train_loader = Data.DataLoader(dataset=train_data, batch_size=100, shuffle=True)#分批并打乱顺序

#定义生成器
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(784, 400),
            nn.ReLU(),
            nn.Linear(400, 200),
            nn.ReLU(),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Linear(100,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3, 100),
            nn.ReLU(),
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 400),
            nn.ReLU(),
            nn.Linear(400, 784),
            nn.Tanh()
        )
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


#定义判别器
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(784, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x=self.dis(x)
        return x

gen=generator()
dis=discriminator()
loss_func=nn.BCELoss()              #用于二分类的损失函数
loss_func_auto=nn.MSELoss()
#g_loss_func=nn.CrossEntropyLoss()  #用于多分类的损失函数
d_optimizer=torch.optim.Adam(dis.parameters(),lr=0.0003)
g_optimizer=torch.optim.Adam(gen.parameters(),lr=0.0003)


#创建画布
f, a = plt.subplots(2, 10, figsize=(10, 2)) #初始化数字 在图表中创建子图显示的图像是2行10列的.figize(长，宽)
plt.ion()

#读取原始数据的前十个数据
view_data = train_data.train_data[:10].view(-1, 28*28).type(torch.Tensor)/255
#print(view_data)
for i in range(10):
    a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)))
    a[0][i].set_xticks(())
    a[0][i].set_yticks(())  #设置位置

#开始训练
for epoch in range(1):
    D_loss=0
    G_loss=0
    for step, (img, label) in enumerate(train_loader):   #可同时获得索引和值
        #print(x.shape)           #64,1,28,28
        #print(label.shape)
        img = img.view(-1, 28*28)       # batch x, shape (batch, 28*28)
        size=img.shape[0]               #100
        #print(size)
        real_img=Variable(img)
        #自动编码器训练
        encoded, decoded = gen(real_img)

        a_loss = loss_func_auto(decoded,real_img)  # 计算损失函数
        g_optimizer.zero_grad()  # 梯度清零
        a_loss.backward()  # 反向传播
        g_optimizer.step()  # 梯度优化

        #判别器训练
        real_label=Variable(torch.ones(size,1))          #真的图片label则为1
        false_label=Variable(torch.zeros(size,1))        #假的图片label则为0
        real_out=dis(real_img)
        d_loss_real=loss_func(real_out,real_label)       #输入真实图片的损失函数

        encoder,false_img=gen(img)                       #得到假的图片
        false_img=Variable(false_img)
        false_out=dis(false_img)
        d_loss_false=loss_func(false_out,false_label)     #计算输入假的图片的损失函数 假的图片与真实label的loss

        d_loss=d_loss_real+d_loss_false                   #总的损失函数包括假的图片和真的图片分别产生的损失函数
        d_optimizer.zero_grad()                           #梯度清零
        d_loss.backward()                                 #反向传播
        d_optimizer.step()                                #梯度优化更新判别器网络参数
        D_loss+=d_loss.item()
        if step % 100 == 0:  # 每100步显示一次
            print('Epoch: ', epoch, '| a_loss: %.4f' % a_loss.data.numpy())

        #训练生成器
        encoded, decoded = gen(real_img)              #生成假的图片
        output=dis(decoded)                           #生成假的图片丢进判别器当中
        #print(type(output))
       # output=torch.LongTensor(output)
        g_loss = loss_func(output,real_label)         # 计算生成器的损失函数
        g_optimizer.zero_grad()                       # 梯度清零
        g_loss.backward()                             # 反向传播
        g_optimizer.step()                            # 梯度优化更新生成网络参数
        G_loss+=g_loss.item()

        if step % 100 == 0:  # 每100步显示一次
            print('Epoch: ', epoch, '| g_loss: %.4f' % g_loss.data.numpy())
    print('epoch: {}, D_Loss: {:.6f}, G_Loss: {:.6f}'
          .format(epoch, D_loss / len(train_loader), G_loss / len(train_loader)))

    # 绘制解码图像
encoded_data, decoded_data = gen(view_data)
 # print(encoded_data.shape)
for i in range(10):
    a[1][i].clear()
    a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)))
    a[1][i].set_xticks(());
    a[1][i].set_yticks(())
plt.draw();
plt.pause(0.05)  # 暂停0.05秒

plt.ioff()
plt.show()