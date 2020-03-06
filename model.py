import torch
import torchvision
import torch.nn as nn

root = 'data/chap12'
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.ImageFolder(root, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, batch_size=4)
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

class Encoder(BaseModel):
    def __init__(self,in_channels, hidden_channels,z_num):
        super().__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels,hidden_channels,3,1,1))
        layers.append(nn.ELU(True))
        prev = hidden_channels
        for idx in range(1,5):
            stride = 1
            if idx != 4:
                next = (idx+1)*hidden_channels
                stride = 2
            layers.append(nn.Conv2d(prev,prev,3,1,1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(prev,next,3,stride,1))
            layers.append(nn.ELU(True))
            prev = next
        self.conv1 = nn.Sequential(*layers)
        self.conv_ouput_dim = [next,8,8]
        self.fc1 = nn.Linear(8*8*next, z_num)
    def forward(self,x):
        x= self.conv1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        return x


class Decoder(BaseModel):
    def __init__(self,cin,cout, zn):
        super().__init__()
        self.conv_input_dim = [cin,8,8]
        self.fc1 = nn.Linear(zn,8*8*cin)
        layers = []
        for i in range(4):
            layers.append(nn.Conv2d(cin,cin,3,1,1))
            layers.append(nn.ELU(True))
            layers.append(nn.Conv2d(cin, cin, 3, 1, 1))
            layers.append(nn.ELU(True))
            if i != 3:
                layers.append(nn.UpsamplingNearest2d(scale_factor=2))
        layers.append(nn.Conv2d(cin,cout,3,1,1))
        layers.append(nn.ELU(True))
        self.conv1 = nn.Sequential(*layers)
    def forward(self,x):
        x = self.fc1(x).view([-1] + self.conv_input_dim)
        x = self.conv1(x)
        return x

class Discriminator(nn.Module):
    def __init__(self,cin,cout,nz):
        super().__init__()
        self.encoder = Encoder(cin,cout,nz)
        self.decoder = Decoder(cout,cin,nz)
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Generator(nn.Module):
    def __init__(self,cin,cout,nz):
        super().__init__()
        self.decoder = Decoder(cin,cout,nz)
    def forward(self,x):
        x = self.decoder(x)
        return x

if __name__ =='__main__':
    e = Encoder(3,64,100)
    d = Decoder(64,3,100)
    G = Generator(64,3,100)
    res = G(torch.randn(16,100))
    D = Discriminator(3,64,100)
    res = D(res)
    print(res.size())








