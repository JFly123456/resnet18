import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch import nn, optim
# from lenet5 import Lenet5
from resnet import ResNet18
from matplotlib import pyplot as plt


def main():
    batchsz = 32

    cifar_train = datasets.CIFAR10('cifar',True,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)

    cifar_train = DataLoader(cifar_train,batch_size=batchsz,shuffle=True)

    cifar_test = datasets.CIFAR10('cifar',False,transform=transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor()
    ]),download=True)

    cifar_test = DataLoader(cifar_test,batch_size=batchsz,shuffle=True)


    x,label = iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)

    device = torch.device('cuda')
    # model = Lenet5().to(device)
    model = ResNet18().to(device)
    criteon = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(),lr=1e-3)
    print(model)

    train_loss = []

    for epoch in range(1000):
        model.train()
        for batchidx,(x,label) in enumerate(cifar_train):
            # [b,3,32,32]
            # [b]
            x, label = x.to(device), label.to(device)
            logits = model(x)
            loss = criteon(logits,label)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss.append(loss.item())

        print(epoch,loss.item())
        total_correct = 0
        total_num = 0
        model.eval()
        with torch.no_grad():
            for x,label in cifar_test:
                x,label = x.to(device), label.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                total_correct += torch.eq(pred,label).float().sum().item()
                total_num += x.size(0)

            acc = total_correct / total_num
            print(epoch,acc)
    plt_curve(train_loss)

def plt_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)),data,color='blue')
    plt.legend(['value'],loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()





if __name__ == '__main__':
    main()