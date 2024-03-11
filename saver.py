import os
import torch
import torchvision
# from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
import shutil


# tensor to PIL Image
def tensor2img(img):
  img = img.cpu().float().numpy()
  if img.shape[0] == 1:
    img = np.tile(img, (3, 1, 1))
  img = (img - img.min()) / (img.max() - img.min())
  img = np.transpose(img, (1, 2, 0))  * 255.0
  return img.astype(np.uint8)

  
def tensor2content(content):
  img = content[0].cpu().float().numpy()
  img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
  return img.astype(np.uint8)

# save a set of images
def save_imgs(imgs, names, path):
  if not os.path.exists(path):
    os.mkdir(path)
  for img, name in zip(imgs, names):
    img = tensor2img(img)
    img = Image.fromarray(img)
    img.save(os.path.join(path, name + '.png'))
    
def save_img_single(img, name, width=None, height=None):
  img = tensor2img(img)
  img = Image.fromarray(img)
  if not width==None and not height==None:
    img = img.resize((width.numpy(), height.numpy()))
  img.save(name)

def save_content(content, dir):
  for i in range(content.shape[1]):
    sub_content = tensor2content(content[:, i, :, :])
    img = Image.fromarray(sub_content)
    img.save(os.path.join(dir, '%03d.jpg'%(i + 1)))
    
def resume(model, optimizer=None, model_save_path=None, device=None, is_train=True):
    # weight
    checkpoint = torch.load(model_save_path, map_location=device)
    model.load_state_dict(checkpoint['MTAN'])
    if is_train:
        optimizer.load_state_dict(checkpoint['optimizer'])
        ep = checkpoint['ep']
        total_it = checkpoint['total_it']
        return model, optimizer, ep, total_it
    else:
        return model
class Saver():
  def __init__(self, opts):
    self.display_dir = os.path.join(opts.display_dir, opts.name)
    self.model_dir = os.path.join(opts.result_dir, opts.name)
    self.image_dir = os.path.join(self.model_dir, 'images')    
    # self.model_dir = os.path.join(self.model_dir, 'checkpoints')
    
    # make directory
    if os.path.exists(self.display_dir):
      shutil.rmtree(self.display_dir)
    os.makedirs(self.display_dir, exist_ok=True)
    if not os.path.exists(self.model_dir):
      os.makedirs(self.model_dir)
    if not os.path.exists(self.image_dir):
      os.makedirs(self.image_dir)

  # save result images
  def write_img(self, ep, input, output):
    input = torch.cat((input[0][:1, ::].repeat(1, 3, 1, 1), input[1][:1, ::], input[2][:1, ::].repeat(1, 3, 1, 1), input[3][:1, ::], input[4][:1, ::].repeat(1, 3, 1, 1)), 3)
    pred = torch.cat((output[0][:1, ::].repeat(1, 3, 1, 1), output[1][:1, ::], output[2][:1, ::].repeat(1, 3, 1, 1), output[3][:1, ::], output[4][:1, ::].repeat(1, 3, 1, 1)), 3)
    assembled_images = torch.cat((input, pred), 2)
    img_filename = '%s/%05d.jpg' % (self.image_dir, ep)    
    # logger.info('Save model to: {}'.format(img_filename))
    torchvision.utils.save_image(assembled_images, img_filename, nrow=1)

  # save model
  def write_model(self, ep, total_it, model, optimizer, best_mIoU, device, is_best=True, logger=None):
    model = model.cpu()
    state = {'MTAN':model.state_dict(),
             'optimizer': optimizer.state_dict(),
             'ep':ep,
             'total_it':total_it,
             'best_mIou':best_mIoU}
    # save_path = os.path.join(self.model_dir, '{:0>5d}.pth'.format(ep))
    if is_best:
        save_path = os.path.join(self.model_dir, 'best_model_' + str(ep) + '.pth'.format(ep))
    # else:
    #   save_path = os.path.join(self.model_dir, '{:0>5d}.pth'.format(ep))
    else:
        save_path = os.path.join(self.model_dir, 'fusion.pth'.format(ep))
        if (ep + 1) % 100 == 0:
            save_path = os.path.join(self.model_dir, 'fusion_' + str(ep) + '.pth'.format(ep))
    torch.save(state, save_path)
    # logger.info('Save model to: {}'.format(save_path))
    model.to(device)
    