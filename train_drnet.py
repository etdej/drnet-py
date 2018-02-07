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

name = 'model=%s-content_dim=%d-pose_dim=%d-max_step=%d-sd_weight=%.3f-lr=%.3f' % (opt.model, opt.content_dim, opt.pose_dim, opt.max_step, opt.sd_weight, opt.lr)
opt.log_dir = '%s/%s/%s' % (opt.log_dir, opt.data, name)

os.makedirs('%s/rec/' % opt.log_dir, exist_ok=True)
os.makedirs('%s/analogy/' % opt.log_dir, exist_ok=True)

print(opt)

print("Random Seed: ", opt.seed)
random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)
dtype = torch.cuda.FloatTensor


# ---------------- create the models  ----------------
if opt.model == 'dcgan':
    if opt.image_width == 64:
        import models.dcgan_64 as models
    elif opt.image_width == 128:
        raise ValueError('dcgan_128 not implemented yet!')
        import models.dcgan_128 as models
if opt.model == 'resnet':
    if opt.image_width == 64:
        raise ValueError('resnet_64 not implemented yet!')
    elif opt.image_width == 128:
        import models.resnet_128 as models
elif opt.model == 'unet':
    if opt.image_width == 64:
        import models.unet_64 as models
    elif opt.image_width == 128:
        raise ValueError('unet_128 not implemented yet!')

netC = models.scene_discriminator(opt.pose_dim)
netEM = models.mvmt_encoder(opt.content_dim, opt.channels)
netD = models.decoder(opt.content_dim, opt.pose_dim, opt.channels)

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
optimizerEM = opt.optimizer(netEC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
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
                          pin_memory=True)
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
      x_c = x[0]
      x_p = x[random.randint(1, opt.max_step-1)]

      h_c = netEC(x_c)
      h_p = netEP(x_p)
      rec = netD([h_c, h_p])

      x_c, x_p, rec = x_c.data, x_p.data, rec.data
      fname = '%s/rec/%d.png' % (opt.log_dir, epoch)
      to_plot = []
      row_sz = 5
      nplot = 20
      for i in range(0, nplot-row_sz, row_sz):
          row = [[xc, xp, xr] for xc, xp, xr in zip(x_c[i:i+row_sz], x_p[i:i+row_sz], rec[i:i+row_sz])]
          to_plot.append(list(itertools.chain(*row)))
      utils.save_tensors_image(fname, to_plot)


def plot_analogy(x, epoch):
    x_c = x[0]


    h_c = netEC(x_c)
    nrow = 10
    row_sz = opt.max_step
    to_plot = []
    row = [xi[0].data for xi in x]
    zeros = torch.zeros(opt.channels, opt.image_width, opt.image_width)
    to_plot.append([zeros] + row)
    for i in range(nrow):
        to_plot.append([x[0][i].data])

    for j in range(0, row_sz):
        h_p = netEP(x[j]).data
        for i in range(nrow):
            h_p[i] = h_p[0]
        rec = netD([h_c, Variable(h_p)])
        for i in range(nrow):
            to_plot[i+1].append(rec[i].data.clone())

    fname = '%s/analogy/%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

def plot_ind(x, epoch):
    x_c = x[0]

    h_c = netEC(x_c)
    nrow = 10
    row_sz = opt.max_step
    to_plot = []
    row = [xi[0].data for xi in x]
    zeros = torch.zeros(opt.channels, opt.image_width, opt.image_width)
    to_plot.append([zeros] + row)
    for i in range(nrow):
        to_plot.append([x[0][i].data])

    for j in range(0, row_sz):
        h_p = netEP(x[j]).data
        for i in range(nrow):
            h_p[i] = h_p[0]
        rec = netD([h_c, Variable(h_p)])
        for i in range(nrow):
            to_plot[i+1].append(rec[i].data.clone())

    fname = '%s/analogy/%d.png' % (opt.log_dir, epoch)
    utils.save_tensors_image(fname, to_plot)

# --------- training funtions ------------------------------------
def train(x):
    netEP.zero_grad()
    netEC.zero_grad()
    netD.zero_grad()

    x_c1 = x[0]
    x_c2 = x[random.randint(1, opt.max_step-1)]
    x_p1 = x[random.randint(1, opt.max_step-1)]
    x_p2 = x[random.randint(1, opt.max_step-1)]

    h_c1 = netEC(x_c1)
    h_c2 = Variable(netEC(x_c2)[0].data if opt.model == 'unet' else netEC(x_c2).data, requires_grad=False) # used as target for sim loss
    h_p1 = netEP(x_p1) # used for scene discriminator
    h_p2 = netEP(x_p2).detach()


    # similarity loss: ||h_c1 - h_c2||
    sim_loss = mse_criterion(h_c1[0] if opt.model == 'unet' else h_c1, h_c2)


    # reconstruction loss: ||D(h_c1, h_p1), x_p1||
    rec = netD([h_c1, h_p1])
    rec_loss = mse_criterion(rec, x_p1)

    # scene discriminator loss: maximize entropy of output
    target = torch.cuda.FloatTensor(opt.batch_size, 1).fill_(0.5)
    out = netC([h_p1, h_p2])
    sd_loss = bce_criterion(out, Variable(target))

    # full loss
    loss = sim_loss + rec_loss + opt.sd_weight*sd_loss
    loss.backward()

    optimizerEC.step()
    optimizerEP.step()
    optimizerD.step()

    return sim_loss.data.cpu().numpy(),rec_loss.data.cpu().numpy()

def train_scene_discriminator(x):
    netC.zero_grad()

    target = torch.cuda.FloatTensor(opt.batch_size, 1)

    x1 = x[0]
    x2 = x[random.randint(1, opt.max_step-1)]
    h_p1 = netEP(x1).detach()
    h_p2 = netEP(x2).detach()

    half = int(opt.batch_size/2)
    rp = torch.randperm(half).cuda()
    h_p2[:half] = h_p2[rp]
    target[:half] = 1
    target[half:] = 0

    out = netC([h_p1, h_p2])
    bce = bce_criterion(out, Variable(target))

    bce.backward()
    optimizerC.step()

    acc =out[:half].gt(0.5).sum() + out[half:].le(0.5).sum()
    return bce.data.cpu().numpy(), acc.data.cpu().numpy()/opt.batch_size

# --------- training loop ------------------------------------
for epoch in range(opt.niter):
    netEP.train()
    netEC.train()
    netD.train()
    netC.train()
    epoch_sim_loss, epoch_rec_loss, epoch_sd_loss, epoch_sd_acc = 0, 0, 0, 0
    for i in range(opt.epoch_size):
        x = next(training_batch_generator)

        # train scene discriminator
        sd_loss, sd_acc = train_scene_discriminator(x)
        epoch_sd_loss += sd_loss
        epoch_sd_acc += sd_acc

        # train main model
        sim_loss, rec_loss = train(x)
        epoch_sim_loss += sim_loss
        epoch_rec_loss += rec_loss


    #netEP.eval()
    #netEC.eval()
    #netD.eval()
    #netC.eval()
    # plot some stuff
    x = next(testing_batch_generator)
    plot_rec(x, epoch)
    plot_analogy(x, epoch)

    print('[%02d] rec loss: %.4f | sim loss: %.4f | scene disc acc: %.3f%% (%d)' % (epoch, epoch_rec_loss/opt.epoch_size, epoch_sim_loss/opt.epoch_size, 100*epoch_sd_acc/opt.epoch_size, epoch*opt.epoch_size*opt.batch_size))

    # save the model
    torch.save({
        'netD': netD,
        'netEP': netEP,
        'netEC': netEC,
        'opt': opt},
        '%s/model.pth.tar' % opt.log_dir)
