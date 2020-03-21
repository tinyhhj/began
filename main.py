from model import Generator, Discriminator
import argparse
import torch
import torchvision
import logging
import os
import numpy as np
from PIL import Image
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument('--cin',type=int, default=3)
parser.add_argument('--cout',type=int, default=64)
parser.add_argument('--nz',type=int,default=256)
parser.add_argument('--ngpu', type=int,default=1)
parser.add_argument('--batch_size', type=int,default=16)
parser.add_argument('--lr', type=float,default=0.0001)
parser.add_argument('--lk', type=float, default= 0.)
parser.add_argument('--max_iter',type=int, default=200000)
parser.add_argument('--data', type=str, default='data')
parser.add_argument('--input_size',type=int, default=64)
parser.add_argument('--betas',type=int, nargs='+', default=[0.5,0.999])
parser.add_argument('--checkpoint', type=str, default='checkpoint')
parser.add_argument('--iter_log', type=int, default=1000)
parser.add_argument('--iter_save',type=int, default=1000)
parser.add_argument('--iter_sample',type=int, default=1000)
parser.add_argument('--iter_update_lr',type=int, default=1000)

def load_model(sample_dir):
    global G, D, g_iteration
    paths = glob(os.path.join(sample_dir, '*_gen.pth'))
    if len(paths) == 0:
        print(f'[!] no checkpoint found in {sample_dir}')
        return
    paths = sorted(paths)
    indexes = [int(os.path.basename(p).split('_')[0]) for p in paths]
    start = max(indexes)

    print(f'load from {sample_dir}/{start} iteration...')
    g_iteration = torch.load(f'{sample_dir}/{start}_gen.pth')['iteration']
    G.load_state_dict(torch.load(f'{sample_dir}/{start}_gen.pth')['state'])
    D.load_state_dict(torch.load(f'{sample_dir}/{start}_dis.pth')['state'])

args = parser.parse_args()

sample_dir = os.path.join(args.checkpoint, 'samples')

os.makedirs(args.checkpoint, exist_ok=True)
os.makedirs(sample_dir,exist_ok=True)


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize([args.input_size,args.input_size]),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])
# dataset = torchvision.datasets.ImageFolder(args.data,transform=transform)
dataset = torchvision.datasets.LSUN(args.data,classes=['bedroom_train'],transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, shuffle= True, batch_size=args.batch_size)

args.ngpu = list(range(args.ngpu))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x_fixed = []
for i in [10735, 18040,  9928,  3293, 18634,  9515, 11419,  8482,  9744,6432, 16820,  4729,  2476,  5816, 10218,  5094]:
    x_fixed.append(dataset[i][0])
x_fixed = torch.stack(x_fixed).to(device)


# x_fixed, _ = next(iter(dataloader))
if not os.path.exists(f'{sample_dir}/x_fixed.png'):
    torchvision.utils.save_image(x_fixed, f'{sample_dir}/x_fixed.png',normalize=True)

G = Generator(args.cout, args.cin, args.nz).to(device)
D = Discriminator(args.cin, args.cout, args.nz).to(device)
g_iteration = 0
load_model(args.checkpoint)

if len(args.ngpu) > 1:
    G = torch.nn.DataParallel(G, args.ngpu)
    D = torch.nn.DataParallel(D, args.ngpu)

optimG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
optimD = torch.optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)
schedulerG = torch.optim.lr_scheduler.StepLR(optimG, step_size=1,gamma=0.5)
schedulerD = torch.optim.lr_scheduler.StepLR(optimD, step_size=1,gamma=0.5)

def denormalize(img):
    return img * 0.5 + 0.5

def train(config,G,D,optimG,optimD):
    global g_iteration
    G.train()
    D.train()
    prev_measure = 1
    arr_k_t = [0.6]
    arr_gamma = [0.7]
    arr_lr_k = [0.0001]
    orig_checkpoint = args.checkpoint
    fixed_noise = torch.empty(args.batch_size, args.nz, device=device).uniform_(-1, 1)
    for idx,k_t in enumerate(arr_k_t):
        for gamma in arr_gamma:
            for lr_k in arr_lr_k:
                g_iteration = 0
                args.checkpoint = f'{orig_checkpoint}/{arr_k_t[idx]}_{gamma}_{lr_k}'
                sample_dir = os.path.join(args.checkpoint, 'samples')
                os.makedirs(sample_dir, exist_ok=True)
                g_iteration = 0
                epoch = 0
                tolerance = 3
                measure_history = []
                need_train = True
                while need_train:
                    epoch += 1
                    print(f'{epoch} epoch running')
                    for it, (inputs, _) in enumerate(dataloader):
                        if config.max_iter < g_iteration:
                            need_train = False
                            break
                        #train discriminator
                        inputs = inputs.to(device)
                        # z_d = torch.empty(args.batch_size,config.nz).uniform_(-1,1).to(device)
                        z_g = torch.empty(args.batch_size,config.nz).uniform_(-1,1).to(device)
                        # fake = G(z_d).detach()


                        optimD.zero_grad()
                        optimG.zero_grad()
                        fake_g = G(z_g)
                        # real & fake reconstruction error
                        recon_real = D(inputs)
                        recon_fake = D(fake_g.detach())

                        # mean
                        errD_real = torch.mean(torch.abs(recon_real - inputs))
                        errD_fake = torch.mean(torch.abs(recon_fake - fake_g.detach()))

                        # loss
                        loss_D = errD_real - k_t * errD_fake
                        loss_D.backward()
                        optimD.step()
                        recon_fake_g = D(fake_g)
                        loss_G = torch.mean(torch.abs(recon_fake_g - fake_g))
                        loss_G.backward()
                        optimG.step()


                        #
                        # loss_G = torch.mean(torch.abs(recon_fake_g - fake_g))
                        # loss_G.backward()


                        g_iteration += 1

                        # k update
                        balance = (gamma*errD_real - errD_fake).item()
                        k_t = k_t + lr_k *balance
                        k_t = max(min(1, k_t),0)

                        measure = errD_real.item() + abs(balance)
                        measure_history.append(measure)


                        #log
                        if config.iter_log and g_iteration % config.iter_log == 0:
                            measure_mean = np.mean(measure_history[-config.iter_log:])
                            print(f'[[{g_iteration}/{config.max_iter}] d_loss: {loss_D.item():.4f} '
                                          f'd_loss_real:{errD_real.item():.4f} d_loss_fake: {errD_fake.item():.4f} g_loss: {loss_G:.4f} k_t:{k_t:.4f} lr: {[g["lr"] for g in optimD.param_groups]} '
                                          f'measure: {measure_mean:.4f}'.format())
                        #sample
                        if config.iter_sample and g_iteration % config.iter_sample == 0:
                            print(f'[{g_iteration}] save images..')
                            fake_img = denormalize(G(fixed_noise))
                            dis_real_img = denormalize(D(x_fixed))
                            dis_fake_img = denormalize(D(fake_img))
                            torchvision.utils.save_image(fake_img,os.path.join(sample_dir,f'{g_iteration}_G.png'))
                            torchvision.utils.save_image(dis_real_img, f'{sample_dir}/{g_iteration}_D.png')
                            torchvision.utils.save_image(dis_fake_img, f'{sample_dir}/{g_iteration}_fake_D.png')
                            # img = img.numpy().transpose((0,2,3,1))
                            # Image.fromarray(img).save(os.path.join(sample_dir,f'{g_iteration}.jpg'))
                        #save
                        if config.iter_save and g_iteration % config.iter_save ==0:
                            torch.save({
                                'iteration': g_iteration,
                                'state': G.state_dict()
                            },os.path.join(args.checkpoint, f'{g_iteration}_gen.pth'))
                            torch.save({
                                'iteration': g_iteration,
                                'state': D.state_dict()
                            }, os.path.join(args.checkpoint, f'{g_iteration}_dis.pth'))
                        # update lr
                        if config.iter_update_lr and g_iteration % config.iter_update_lr == 0:
                            current_measure = np.mean(measure_history[-config.iter_update_lr:])
                            if current_measure > prev_measure * 0.999:
                                tolerance -= 1
                                if tolerance == 0:
                                    tolerance = 3
                                    schedulerD.step()
                                    schedulerG.step()
                            else:
                                tolerance = 3
                            print(f'prev_measure: {prev_measure} cur_messure: {current_measure} tolerance: {tolerance}')
                            prev_measure = current_measure

if __name__ =='__main__':

    train(args, G,D,optimG,optimD)
    # load_model('checkpoint')