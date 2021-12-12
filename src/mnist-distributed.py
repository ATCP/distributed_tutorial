import os
from datetime import datetime
import argparse
import torch.multiprocessing as mp
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
#from apex.parallel import DistributedDataParallel as DDP
#from apex import amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F
import torchvision.models as models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '192.168.233.17'
    os.environ['MASTER_PORT'] = '12355'
    #mp.spawn(train, nprocs=args.gpus, args=(args,))
    train(args)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

class VGG16(torch.nn.Module):
    def __init__(self, num_features, num_classes):
      super(VGG16, self).__init__()
      # calculate same padding:
      # (w - k + 2*p)/s + 1 = o
      # => p = (s(o-1) - w + k)/2

      self.block_1 = nn.Sequential(
        nn.Conv2d(in_channels=3,
          out_channels=64,
          kernel_size=(3, 3),
          stride=(1, 1),
          # (1(32-1)- 32 + 3)/2 = 1
          padding=1), 
        nn.ReLU(),
        nn.Conv2d(in_channels=64,
          out_channels=64,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2),
          stride=(2, 2))
      )

      self.block_2 = nn.Sequential(
        nn.Conv2d(in_channels=64,
          out_channels=128,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=128,
          out_channels=128,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2),
          stride=(2, 2))
      )

      self.block_3 = nn.Sequential(        
        nn.Conv2d(in_channels=128,
          out_channels=256,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=256,
          out_channels=256,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),        
        nn.Conv2d(in_channels=256,
          out_channels=256,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=(2, 2),
          stride=(2, 2))
      )

      self.block_4 = nn.Sequential(   
        nn.Conv2d(in_channels=256,
          out_channels=512,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),        
        nn.Conv2d(in_channels=512,
          out_channels=512,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),        
        nn.Conv2d(in_channels=512,
          out_channels=512,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),            
        nn.MaxPool2d(kernel_size=(2, 2),
          stride=(2, 2))
      )

      self.block_5 = nn.Sequential(
        nn.Conv2d(in_channels=512,
          out_channels=512,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),            
        nn.Conv2d(in_channels=512,
          out_channels=512,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),            
        nn.Conv2d(in_channels=512,
          out_channels=512,
          kernel_size=(3, 3),
          stride=(1, 1),
          padding=1),
        nn.ReLU(),    
        nn.MaxPool2d(kernel_size=(2, 2),
          stride=(2, 2))             
      )

      self.classifier = nn.Sequential(
        nn.Linear(512, 4096),
        nn.ReLU(True),
        #nn.Dropout(p=0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(True),
        #nn.Dropout(p=0.5),
        nn.Linear(4096, num_classes),
      )

      for m in self.modules():
        if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
          nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
          if m.bias is not None:
            m.bias.detach().zero_()

          #self.avgpool = nn.AdaptiveAvgPool2d((7, 7))


    def forward(self, x):

      x = self.block_1(x)
      x = self.block_2(x)
      x = self.block_3(x)
      x = self.block_4(x)
      x = self.block_5(x)
      #x = self.avgpool(x)
      x = x.view(x.size(0), -1)
      logits = self.classifier(x)
      probas = F.softmax(logits, dim=1)

      return logits, probas


def train(args):
    gpu = 0
    rank = args.nr * args.gpus + gpu
    print("rank " + str(rank))
    
    dist.init_process_group(backend='gloo', init_method='env://', world_size=args.world_size, rank=rank)
    torch.manual_seed(0)
    #model = ConvNet()
    #model = MyModel()
    model = VGG16(num_features=784, num_classes=10)

    torch.cuda.set_device(gpu)
    model.cuda(gpu)
    batch_size = 100
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(gpu)
    optimizer = torch.optim.SGD(model.parameters(), 1e-4)
    # Wrap the model
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    # Data loading code
    #train_dataset = torchvision.datasets.MNIST(root='./data',
    #                                           train=True,
    #                                           transform=transforms.ToTensor(),
    #                                           download=True)
    train_dataset = datasets.CIFAR10(root='./data', 
                                    train=True, 
                                    transform=transforms.ToTensor(),
                                    download=True)

    #train_dataset = torchvision.datasets.ImageNet(root='./data',
    #                                           train=True,
    #                                           transform=transforms.ToTensor(),
    #                                           download=True)
    
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
    #                                                                num_replicas=args.world_size,
    #                                                                rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)
                                              # sampler=train_sampler)

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, args.epochs, i + 1, total_step,
                                                                         loss.item()))
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


if __name__ == '__main__':
    main()
