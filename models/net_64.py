import torch
import torch.nn as nn
import math
from torch.autograd import Variable

dtype = torch.cuda.FloatTensor

class dcgan_conv(nn.Module):
  def __init__(self, nin, nout, stride=1):
    super(dcgan_conv, self).__init__()
    self.main = nn.Sequential(
      nn.Conv2d(nin, nout, 5, stride, 2),
      nn.BatchNorm2d(nout),
      nn.LeakyReLU(0.2, inplace=True),
      )

  def forward(self, input):
    return self.main(input)


class dcgan_upconv(nn.Module):
  def __init__(self, nin, nout):
    super(dcgan_upconv, self).__init__()
    self.main = nn.Sequential(
      nn.ConvTranspose2d(nin, nout, 4, 2, 1),
      nn.BatchNorm2d(nout),
      nn.LeakyReLU(0.2, inplace=True),
      )

  def forward(self, input):
    return self.main(input)

class mvmt_encoder(nn.Module):
  def __init__(self, nb_out_kernel = 10, nc=1, nf=32, input_size=64, mvmt_kernel_size = 5):
    super(mvmt_encoder, self).__init__()
    self.nb_out_kernel = nb_out_kernel
    self.out_kernel_size = mvmt_kernel_size
    self.nf = nf
    self.input_size = input_size

    self.extract = nn.Sequential(
      # input is (nc) x 64 x 64
      dcgan_conv(2*nc, nf, 2),
      # state size. (nf) x 32 x 32
      dcgan_conv(nf, nf),
      # state size. (nf) x 32 x 32
      dcgan_conv(nf, nf),
      # state size. (nf) x 32 x 32
      dcgan_conv(nf, nf, 2),
      # state size. (nf) x 16 x 16
      dcgan_conv(nf, nf * 2),
      # state size. (nf*2) x 16 x 16
      dcgan_conv(nf * 2, nf * 4, 2),
      # state size. (nf*4) x 8 x 8
    )
    self.linear = nn.Linear(int(nf *4*input_size*input_size/64), nb_out_kernel*mvmt_kernel_size*mvmt_kernel_size)
  def forward(self, input):
    concat = torch.cat(input, 1)
    out = self.extract(concat)
    out = out.view(out.size()[0], -1)
    out = self.linear(out)
    out = out.view(-1, self.nb_out_kernel, self.out_kernel_size, self.out_kernel_size)

    return out

class decoder(nn.Module):
    def __init__(self, nb_mvmt_kernel=10,nc=1, nf=64, mvmt_kernel_size=5, batch_size=32):
        super(decoder, self).__init__()
        self.nc = nc
        self.mvmt_kernel_size= mvmt_kernel_size
        self.batch_size = batch_size
        self.nb_mvmt_kernel = nb_mvmt_kernel
# input is (nc) x 64 x 64
        self.conv1 = dcgan_conv(nc, nf, 2)
# state size. (nf) x 32 x 32
        self.conv2 = dcgan_conv(nf, nf)
# state size. (nf) x 32 x 32
        self.conv3 = dcgan_conv(nf, nf)
# state size. (nf) x 32 x 32
        self.conv4 = dcgan_conv(nf, nf, 2)
# state size. (nf) x 16 x 16
        self.conv5 = dcgan_conv(nf, nf * 2)
# state size. (nf*2) x 16 x 16
        self.conv6 = dcgan_conv(nf * 2, nf * 4, 2)
# state size. (nf*4) x 8 x 8

        self.linear = nn.Linear(8*8*nf*4 + nb_mvmt_kernel* mvmt_kernel_size*mvmt_kernel_size, 8*8*nf*4)
# state size. (nf*4) x 8 x 8
        self.upconv1 = dcgan_upconv(nf*4, nf*4)
# state size. (nf*4) x 16 x 16
        self.upconv2 = dcgan_upconv(nf*4, nf*2)
# state size. (nf*2) x 32 x 32
        self.upconv3 = dcgan_upconv(nf*2, nb_mvmt_kernel+ 1)
# state size. mvmt_dim + 1 x 64 x 64


    def forward(self, input):
        input_frame, mvmt_kernel = input
#    if type(content) == list:
#      content = torch.cat(content, 1)
#    if type(pose) == list:
#      pose = torch.cat(pose, 1)
        content = self.conv1(input_frame)
        content = self.conv2(content)
        content = self.conv3(content)
        content = self.conv4(content)
        content = self.conv5(content)
        content = self.conv6(content)

        concat = torch.cat([content.view(self.batch_size,-1), mvmt_kernel.view(self.batch_size, -1)], 1)
        concat = self.linear(concat)
        concat = concat.view(self.batch_size, -1, 8, 8)

        masks = self.upconv1(concat)
        masks = self.upconv2(masks)
        masks = nn.functional.sigmoid(self.upconv3(masks))

        mvmt_kernel = mvmt_kernel.view(-1, self.nb_mvmt_kernel, 1, self.mvmt_kernel_size, self.mvmt_kernel_size)

        input_frame_view = input_frame.contiguous()
        input_frame_view = input_frame.view(self.batch_size, self.nc, 1, 64, 64)

        conved_images = [ torch.nn.functional.conv2d(input_frame_view[i], mvmt_kernel[i], padding=int((self.mvmt_kernel_size - 1)/2)) for i in range(self.batch_size) ]
        '''
        conved_images = Variable(torch.cuda.FloatTensor(self.batch_size, self.nc, self.nb_mvmt_kernel, 64, 64), requires_grad=True)
        # How to do better (depthwise_conv2d in torch)??
        for i in range(self.batch_size):
           conved_images[i] = torch.nn.functional.conv2d(input_frame_view[i], mvmt_kernel[i], padding=int((self.mvmt_kernel_size - 1)/2))
       '''
        conved_images = torch.cat(conved_images, 0).view(self.batch_size, self.nc, self.nb_mvmt_kernel, 64, 64)
        #if conved_images.size()[0] == self.batch_size:
        #    assert(False)
        transformed_images = torch.cat([input_frame_view, conved_images], 2)
        # Expected Size : (-1, nb_mvmt_kernel + 1, 3, 64, 64)
        output = self.masking_function(transformed_images, masks)

        return output

    def masking_function(self, transformed_images, masks):
        ## LOWER_BOUDING?
        ## normalizing masks accross channels
        #norm = torch.sum(masks, 0)
        norm = masks.norm(dim=1).view(self.batch_size, 1, 64, 64).expand(self.batch_size, self.nb_mvmt_kernel+1, 64, 64)
        
        masks = torch.div(masks, norm)

        masks = masks.view(self.batch_size,  1, self.nb_mvmt_kernel + 1,  64, 64)
        masks = masks.expand(self.batch_size, self.nc, self.nb_mvmt_kernel + 1, 64, 64)
        images = torch.mul(transformed_images, masks)
        return torch.mean(images, 2)

class scene_discriminator(nn.Module):
    def __init__(self, nb_mvmt_kernel=10, mvmt_kernel_size=5, nf=256):
        super(scene_discriminator, self).__init__()
        self.kernel_dim = nb_mvmt_kernel*mvmt_kernel_size*mvmt_kernel_size
        self.main = nn.Sequential(
          nn.Linear(self.kernel_dim*2, nf),
          nn.ReLU(True),
          nn.Linear(nf, nf),
          nn.ReLU(True),
          nn.Linear(nf, 1),
          nn.Sigmoid(),
        )

    def forward(self, input):
        output = self.main(torch.cat(input, 1).view(-1, self.kernel_dim*2))
        return output

class scene_conv_discriminator(nn.Module):
    def __init__(self, nb_mvmt_kernel=10, mvmt_kernel_size=5, nf=32):
        super(scene_conv_discriminator, self).__init__()
        self.linear_input_dim = 2*nf*math.ceil(mvmt_kernel_size/4)**2
        self.conv1 = dcgan_conv(2*nb_mvmt_kernel, nf, 2)
        self.conv2 = dcgan_conv(nf, 2*nf, 2)
        self.linear = nn.Linear(self.linear_input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.conv1(torch.cat(input, 1))
        output = self.conv2(output)
        output = output.view(-1, self.linear_input_dim)
        output = self.linear(output)
        return self.sigmoid(output)

