from model import Generator, Discriminator
import argparse
import torch
import torchvision
import logging
import os
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--cin',type=int, default=3)
parser.add_argument('--cout',type=int, default=64)
parser.add_argument('--nz',type=int,default=100)
parser.add_argument('--ngpu', type=int,default=1)
parser.add_argument('--batch_size', type=int,default=16)
parser.add_argument('--lr', type=float,default=0.0001)
parser.add_argument('--lk', type=float, default= 0.)
parser.add_argument('--epochs',type=int, default=100)
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--ulr',type=int, default=100)
parser.add_argument('--input_size',type=int, default=64)
parser.add_argument('--betas',type=int, nargs='+', default=[0.5,0.999])
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--iter_log', type=int, default=100)
parser.add_argument('--iter_save',type=int, default=100)
parser.add_argument('--iter_sample',type=int, default=100)


args = parser.parse_args()

sample_dir = os.path.join(args.checkpoint, 'samples')

os.makedirs(args.checkpoint, exist_ok=True)
os.makedirs(sample_dir,exist_ok=True)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.input_size,args.input_size]),
    torchvision.transforms.ToTensor()
])
dataset = torchvision.datasets.ImageFolder(args.data,transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, shuffle= True, batch_size=args.batch_size)
args.ngpu = list(range(args.ngpu))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator(args.cout, args.cin, args.nz).to(device)
D = Discriminator(args.cin, args.cout, args.nz).to(device)

if len(args.ngpu) > 0:
    G = torch.nn.DataParallel(G, args.ngpu)
    D = torch.nn.DataParallel(D, args.ngpu)

optimG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
optimD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)


k_t = 0
g_iteration = 0
gamma = 0.5
lr_k = 0.001
img_sample_size = (5,5)
n_img_sample = np.prod(img_sample_size)
fixed_noise = torch.empty(args.batch_size,args.nz, device=device).uniform_(-1,1)
loss_D_history = []
loss_G_history = []

def train(config,G,D):

    for i in range(config.epochs):
        for iter, (inputs, _) in enumerate(dataloader):
            #train discriminator
            inputs = inputs.to(device)
            z_d = torch.empty(args.batch_size,config.nz).uniform_(-1,1)
            z_g = torch.empty(args.batch_size,config.nz).uniform_(-1,1)
            fake = G(z_d)

            D.zero_grad()
            recon_real = D(inputs)
            recon_fake = D(fake.detach())
            errD_real = torch.mean(torch.abs(recon_real - inputs))
            errD_fake = torch.mean(torch.abs(recon_fake - fake))
            loss_D = errD_real - k_t * errD_fake
            loss_D.backward()
            optimD.step()

            G.zero_grad()
            fake = G(z_g)
            recon_fake = D(fake)
            loss_G = torch.mean(torch.abs(recon_fake - fake))
            loss_G.backward()
            optimG.step()

            loss_D_history.append(loss_D)
            loss_G_history.append(loss_G)
            g_iteration += 1

            # k update
            k_t = k_t + lr_k *(gamma*recon_real - recon_fake)


            #log
            if config.iter_log and g_iteration % config.iter_log == 0:
                logging.debug('loss_D: {} loss_G:{}'.format(np.mean(loss_D_history[-config.log:]), np.mean(loss_G_history[-config.log:])))
            #sample
            if config.iter_sample and g_iteration % config.iter_sample == 0:
                img = torchvision.utils.make_grid(G(fixed_noise))
                img = img.numpy().transpose((0,2,3,1))
                Image.fromarray(img).save(os.path.join(sample_dir,f'{g_iteration}.jpg'))
            #save
            if config.iter_save and g_iteration % config.iter_save ==0:
                torch.save({
                    'iteration': g_iteration,
                    'state': G.state_dict()
                },os.path.join(args.checkpoint, f'{g_iteration}_{loss_D}_{loss_G}_gen.pth'))
                torch.save({
                    'iteration': g_iteration,
                    'state': D.state_dict()
                }, os.path.join(args.checkpoint, f'{g_iteration}_{loss_D}_{loss_G}_dis.pth'))

if __name__ =='__main__':

    train(args, G,D)