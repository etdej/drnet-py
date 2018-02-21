import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import os
import random
from torch.autograd import Variable
from torch.utils.data import DataLoader
import utils
import itertools
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.002, type=float, help='learning rate')
parser.add_argument('--beta1', default=0.9, type=float, help='momentum term for adam')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--log_dir', default='/misc/vlgscratch4/FergusGroup/denton/drnetpy_logs/', help='base directory to save logs')
parser.add_argument('--name', default='', help='identifierfor directory')
parser.add_argument('--data_root', default='', help='root directory for data')
parser.add_argument('--optimizer', default='adam', help='optimizer to train with')
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--seed', default=1, type=int, help='manual seed')
parser.add_argument('--epoch_size', type=int, default=600, help='epoch size')
parser.add_argument('--content_dim', type=int, default=64, help='size of the content vector')
parser.add_argument('--mvmt_dim', type=int, default=10, help='size of the pose vector')
parser.add_argument('--mvmt_kernel_size', type=int, default=5, help='size of the pose vector')
parser.add_argument('--image_width', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--channels', default=3, type=int)
parser.add_argument('--data', default='moving_mnist', help='dataset to train with')
parser.add_argument('--max_step', type=int, default=12, help='maximum distance between frames')
parser.add_argument('--sd_weight', type=float, default=0.01, help='weight on adversarial loss')
parser.add_argument('--model', default='conv_mvmt', help='model type (dcgan | unet | resnet)')


opt = parser.parse_args()

name = 'model=%s-kernel_mvmt_size=%d-mvmt=%d-max_step=%d-sd_weight=%.3f-lr=%.4f' % (opt.model, opt.mvmt_kernel_size, opt.mvmt_dim, opt.max_step, opt.sd_weight, opt.lr)
opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.data, name)

os.makedirs('%s/rec/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/analogy/' % opt.log_dir, exist_ok=True)

sys.stdout = open('%s/output.txt' % (opt.log_dir), 'w')

print(opt)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
#torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


dtype = torch.cuda.FloatTensor
#dtype = torch.FloatTensor


# ---------------- create the models  ----------------
if opt.model == 'conv_mvmt':
    if opt.image_width == 64:
        import models.net_64 as models
    elif opt.image_width == 128:
        raise ValueError('dcgan_128 not implemented yet!')
        import models.dcgan_128 as models
if opt.model == 'resnet':
    if opt.image_width == 64:
        raise ValueError('resnet_64 not implemented yet!')
    elif opt.image_width == 128:
        raise ValueError('resnet_128 not implemented yet!')
        import models.resnet_128 as models
elif opt.model == 'unet':
    if opt.image_width == 64:
        raise ValueError('unet_64not implemented yet!')
        import models.unet_64 as models
    elif opt.image_width == 128:
        raise ValueError('unet_128 not implemented yet!')

netC = models.scene_discriminator(opt.mvmt_dim)
netEM = models.mvmt_encoder(opt.mvmt_dim, opt.channels, mvmt_kernel_size=opt.mvmt_kernel_size)
netD = models.decoder(opt.mvmt_dim, opt.channels, batch_size=opt.batch_size, mvmt_kernel_size=opt.mvmt_kernel_size)

# ---------------- optimizers ----------------
if opt.optimizer == 'adam':
    opt.optimizer = optim.Adam
elif opt.optimizer == 'rmsprop':
    opt.optimizer = optim.RMSprop
elif opt.optimizer == 'sgd':
    opt.optimizer = optim.SGD
else:
  raise ValueError('Unknown optimizer: %s' % opt.optimizer)

optimizerC = opt.optimizer(netC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerEM = opt.optimizer(netEM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = opt.optimizer(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

# --------- loss functions ------------------------------------
mse_criterion = nn.MSELoss()
bce_criterion = nn.MSELoss()

# --------- transfer to gpu ------------------------------------
netEM.cuda()
netD.cuda()
netC.cuda()
mse_criterion.cuda()
bce_criterion.cuda()

# --------- load a dataset ------------------------------------
train_data, test_data, load_workers = utils.load_dataset(opt)

train_loader = DataLoader(train_data,
                          num_workers=load_workers,
                          batch_size=opt.batch_size,
                          shuffle=True,
                          drop_last=True,
                          pin_memory=False)
test_loader = DataLoader(test_data,
                         num_workers=load_workers,
                         batch_size=opt.batch_size,
                         shuffle=True,
                         drop_last=True,
                         pin_memory=True)

def get_training_batch():
    while True:
        for sequence in train_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
training_batch_generator = get_training_batch()

def get_testing_batch():
    while True:
        for sequence in test_loader:
            batch = utils.normalize_data(opt, dtype, sequence)
            yield batch
testing_batch_generator = get_testing_batch()

# --------- plotting funtions ------------------------------------
def plot_rec(x, epoch):
      x_1 = x[0]
      x_m = x[random.randint(1, opt.max_step-1)]

      h_m = netEM([x_1, x_m])
      rec = netD([x_1, h_m])

      x_1, x_m, rec = x_1.data, x_m.data, rec.data
      fname = '%s/rec/%d.png' % (opt.log_dir, epoch)
      to_plot = []
      row_sz = 5
      nplot = 20
      for i in range(0, nplot-row_sz, row_sz):
          row = [[x1, xm, xr] for x1, xm, xr in zip(x_1[i:i+row_sz], x_m[i:i+row_sz], rec[i:i+row_sz])]
          to_plot.append(list(itertools.chain(*row)))
      utils.save_tensors_image(fname, to_plot)


#def plot_analogy(x, epoch):
#    x_c = x[0]
#
#
#    nrow = 10
#    row_sz = opt.max_step
#    to_plot = []
#    row = [xi[0].data for xi in x]
#    zeros = torch.zeros(opt.channels, opt.image_width, opt.image_width)
#    to_plot.append([zeros] + row)
#    for i in range(nrow):
#        to_plot.append([x[0][i].data])
#
#    for j in range(0, row_sz):
#        h_m = netEM([x_c, x[j]]).data
#        for i in range(nrow):
#            h_m[i] = h_m[0]
#        rec = netD([x_c, Variable(h_m)])
#        for i in range(nrow):
#            to_plot[i+1].append(rec[i].data.clone())
#
#    fname = '%s/analogy/%d.png' % (opt.log_dir, epoch)
#    utils.save_tensors_image(fname, to_plot)

#def plot_ind(x, epoch):
#    x_c = x[0]
#
#    nrow = 10
#    row_sz = opt.max_step
#    to_plot = []
#    row = [xi[0].data for xi in x]
#    zeros = torch.zeros(opt.channels, opt.image_width, opt.image_width)
#    to_plot.append([zeros] + row)
#    for i in range(nrow):
#        to_plot.append([x[0][i].data])
#
#    for j in range(0, row_sz):
#        h_m = netEM([x_c, x[j]]).data
#        for i in range(nrow):
#            h_p[i] = h_p[0]
#        rec = netD([x_c, Variable(h_p)])
#        for i in range(nrow):
#            to_plot[i+1].append(rec[i].data.clone())
#
#    fname = '%s/analogy/%d.png' % (opt.log_dir, epoch)
#    utils.save_tensors_image(fname, to_plot)

# --------- training funtions ------------------------------------
def train(x):
    netEM.zero_grad()
    netD.zero_grad()

    x_1 = x[0]
    x_m1 = x[random.randint(1, opt.max_step-1)]
#    x_m2 = x[random.randint(1, opt.max_step-1)]

    h_m1 = netEM([x_1, x_m1])
#    h_m2 = netEM([x_1, x_m2])

    # reconstruction loss: ||D(h_c1, h_p1), x_p1||
    rec = netD([x_1, h_m1])
    rec_loss = mse_criterion(rec, x_m1)

    # scene discriminator loss: maximize entropy of output
    #target = torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.5)
#    target = torch.FloatTensor(opt.batch_size, 1).fill_(0.5)
#    out = netC([h_m1, h_m2])
#    sd_loss = bce_criterion(out, Variable(target))

    # full loss
    loss = rec_loss
#    loss = rec_loss + opt.sd_weight*sd_loss
    loss.backward()

    optimizerEM.step()
    optimizerD.step()

    return rec_loss.data.cpu().numpy()

#def train_scene_discriminator(x):
#    netC.zero_grad()
#
#    #target = torch.cuda.FloatTensor(opt.batch_size, 1)
#    target = torch.FloatTensor(opt.batch_size, 1)
#
#    x1 = x[0]
#    x_m1 = x[random.randint(1, opt.max_step-1)]
#    x_m2 = x[random.randint(1, opt.max_step-1)]
#
#    h_m1 = netEM([x1, x_m1]).detach()
#    h_m2 = netEM([x1, x_m2]).detach()
#
#    half = int(opt.batch_size/2)
#    #rp = torch.randperm(half).cuda()
#    rp = torch.randperm(half)
#    h_m2[:half] = h_m2[rp]
#    target[:half] = 1
#    target[half:] = 0
#
#    out = netC([h_m1, h_m2])
#    bce = bce_criterion(out, Variable(target))
#
#    bce.backward()
#    optimizerC.step()
#
#    acc =out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()
#    return bce.data.cpu().numpy(), acc.data.cpu().numpy()/opt.batch_size

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    netEM.train()
    netD.train()
    netC.train()
    epoch_rec_loss, epoch_sd_loss, epoch_sd_acc =  0, 0, 0
    for i in range(opt.epoch_size):
        x = next(training_batch_generator)

        # train scene discriminator
#        sd_loss, sd_acc = train_scene_discriminator(x)
#        epoch_sd_loss += sd_loss
#        epoch_sd_acc += sd_acc

        # train main model
        rec_loss = train(x)
#        print(rec_loss)
        epoch_rec_loss += rec_loss


    #netEP.eval()
    #netEC.eval()
    #netD.eval()
    #netC.eval()
    # plot some stuff
    x = next(testing_batch_generator)
    plot_rec(x, epoch)
#    plot_analogy(x, epoch)

    #print('[%02d] rec loss: %.4f | sim loss: %.4f | scene disc acc: %.3f%% (%d)' % (epoch, epoch_rec_loss/opt.epoch_size, epoch_sim_loss/opt.epoch_size, 100*epoch_sd_acc/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))
    print('[%02d] rec loss: %.4f (%d)' % (epoch, epoch_rec_loss/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # save the model
    torch.save({
        'netD': netD,
        'netEM': netEM,
        'opt': opt},
        '%s/model.pth.tar' % opt.log_dir)
