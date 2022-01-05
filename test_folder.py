
import torch
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
from PIL import Image
import torch.optim as optim
import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Add Noise")
    parser.add_argument('--model', type=str, help='model to be loaded')
    parser.add_argument('--indir', type=str, help='input image directory')
    parser.add_argument('--outdir', type=str, help='output image dicretory')

    return parser.parse_args()

args = parse_args()

def load_checkpoint(checkpoint_path, model, optimizer):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['state_dict'])
    optimizer.load_state_dict(state['optimizer'])
    print('model loaded from %s' % checkpoint_path)

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )

def add_merge_stage(ch_coarse, ch_fine, in_coarse, in_fine, upsample):
  conv = nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
  torch.cat(conv, in_fine)

  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False)
  )
  upsample(in_coarse)

def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

class Net(nn.Module):
  def __init__(self, useBN=False):
    super(Net, self).__init__()

    self.conv1   = add_conv_stage(1, 32, useBN=useBN)
    self.conv2   = add_conv_stage(32, 64, useBN=useBN)
    self.conv3   = add_conv_stage(64, 128, useBN=useBN)
    self.conv4   = add_conv_stage(128, 256, useBN=useBN)
    self.conv5   = add_conv_stage(256, 512, useBN=useBN)

    self.conv4m = add_conv_stage(512, 256, useBN=useBN)
    self.conv3m = add_conv_stage(256, 128, useBN=useBN)
    self.conv2m = add_conv_stage(128,  64, useBN=useBN)
    self.conv1m = add_conv_stage( 64,  32, useBN=useBN)

    self.conv0  = nn.Sequential(
        nn.Conv2d(32, 1, 3, 1, 1),
        nn.Sigmoid()
    )

    self.max_pool = nn.MaxPool2d(2)

    self.upsample54 = upsample(512, 256)
    self.upsample43 = upsample(256, 128)
    self.upsample32 = upsample(128,  64)
    self.upsample21 = upsample(64 ,  32)

    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()


  def forward(self, x):
    conv1_out = self.conv1(x)
    #return self.upsample21(conv1_out)
    conv2_out = self.conv2(self.max_pool(conv1_out))
    conv3_out = self.conv3(self.max_pool(conv2_out))
    conv4_out = self.conv4(self.max_pool(conv3_out))
    conv5_out = self.conv5(self.max_pool(conv4_out))

    conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
    conv4m_out = self.conv4m(conv5m_out)

    conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv3m_out = self.conv3m(conv4m_out_)

    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv2m_out = self.conv2m(conv3m_out_)

    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
    conv1m_out = self.conv1m(conv2m_out_)

    conv0_out = self.conv0(conv1m_out)

    return conv0_out

def model_RGB(img_in):
    img_R = model(img_in[:,0,:,:].unsqueeze(1))
    img_G = model(img_in[:,1,:,:].unsqueeze(1))
    img_B = model(img_in[:,2,:,:].unsqueeze(1))
    img_out = torch.cat((img_R,img_G,img_B),dim=1)
    return img_out

model = Net()
model.eval()
optimizer = optim.Adam(model.parameters(), lr=1e-5)
criterion = nn.MSELoss()
load_checkpoint(args.model, model, optimizer)

t2i = transforms.Compose([
    #transforms.Resize(256),
    #transforms.CenterCrop(256),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
i2t = transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize((720,720)),
])

for img_fn in os.listdir(args.indir):
    img = Image.open(os.path.join(args.indir,img_fn))
    im_t = t2i(img)
    # stn = train_set[305][0].cuda()
    # im_t = train_set[305][1].cuda()
    im_t = torch.unsqueeze(im_t,0)
    # im_t = im_t.unfold(2, 64, 64).unfold(3, 64, 64)
    # output = torch.zeros(1,3,768,768)
    # for i in range(12):
    #     for j in range(12):
    #         output[:,:,i*64:(i+1)*64,j*64:(j+1)*64] = model(im_t[:,:,i,j,:,:])
    #         #output[:,:,i*64:(i+1)*64,j*64:(j+1)*64] = im_t[:,:,i,j,:,:]

    # for i in range(5):
    #     for j in range(5):
    #         im_t[:,:,i*128:(i+1)*128,j*128:(j+1)*128] = model(im_t[:,:,i*128:(i+1)*128,j*128:(j+1)*128])
    im_t = model_RGB(im_t)

    # loss = criterion(stn_t,im_t)
    im_t = torch.squeeze(im_t,0)
    im_o = i2t(im_t)

    out_fn = img_fn[:-4] + '_denoise' + '.png'
    im_o.save(os.path.join(args.outdir,out_fn))
