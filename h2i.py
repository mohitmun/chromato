import numpy as np
from data import colorize_image as CI
from ideepcolor import parse_args
from skimage import color
import os
import datetime
args = parse_args()

colorModel = CI.ColorizeImageCaffe()
colorModel.prep_net(args.gpu, args.color_prototxt, args.color_caffemodel)
model = colorModel
distModel = CI.ColorizeImageCaffeDist()
distModel.prep_net(args.gpu, args.dist_prototxt, args.dist_caffemodel)

import cv2
# distModel.set_image(im_rgb)
def get_input(h, w):
  im = np.zeros((h, w, 3), np.uint8)
  mask = np.zeros((h, w, 1), np.uint8)
  vis_im = np.zeros((h, w, 3), np.uint8)
  userEdits = []
  for ue in userEdits:
      ue.updateInput(im, mask, vis_im)
  im_bgr = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
  return im, mask

from skimage import color
def predict_color():  
  im, mask = get_input()
  im_mask0 = mask > 0.0
  im_mask0 = im_mask0.transpose((2, 0, 1))
  im_lab = color.rgb2lab(im).transpose((2, 0, 1))
  im_ab0 = im_lab[1:3, :, :]
  distModel.net_forward(im_ab0, im_mask0)

# predict_color()

def compute_result(image_file):
  load_size = 256
  win_size = 512

  im_bgr = cv2.imread(image_file)
  im_full = im_bgr.copy()
  # get image for display
  h, w, c = im_full.shape
  max_width = max(h, w)
  r = win_size / float(max_width)
  scale = float(win_size) / load_size
  print('scale = %f' % scale)
  rw = int(round(r * w / 4.0) * 4)
  rh = int(round(r * h / 4.0) * 4)

  im_win = cv2.resize(im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

  dw = int((win_size - rw) / 2)
  dh = int((win_size - rh) / 2)
  win_w = rw
  win_h = rh
  im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
  im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

  gray_win = cv2.resize(im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
  im_bgr = cv2.resize(im_bgr, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
  im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
  lab_win = color.rgb2lab(im_win[:, :, ::-1])

  im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
  im_l = im_lab[:, :, 0]
  l_win = lab_win[:, :, 0]
  im_ab = im_lab[:, :, 1:]
  im_size = im_rgb.shape[0:2]

  im_ab0 = np.zeros((2, load_size, load_size))
  im_mask0 = np.zeros((1, load_size, load_size))
  brushWidth = 2 * scale


  im, mask = get_input(load_size, load_size)
  im_mask0 = mask > 0.0
  im_mask0 = im_mask0.transpose((2, 0, 1))
  im_lab = color.rgb2lab(im).transpose((2, 0, 1))
  im_ab0 = im_lab[1:3, :, :]
  colorModel.load_image(image_file)
  print("before net_forward")
  model.net_forward(im_ab0, im_mask0)
  print("after net_forward")
  ab = model.output_ab.transpose((1, 2, 0))
  ab_win = cv2.resize(ab, (win_w, win_h), interpolation=cv2.INTER_CUBIC)
  pred_lab = np.concatenate((l_win[..., np.newaxis], ab_win), axis=2)
  pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
  result = pred_rgb

  path = os.path.abspath(image_file)
  path, ext = os.path.splitext(path)

  suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
  save_path = "_".join([path, "method"])

  print('saving result to <%s>\n' % save_path)
  if not os.path.exists(save_path):
      os.mkdir(save_path)

  # np.save(os.path.join(save_path, 'im_l.npy'), model.img_l)
  # np.save(os.path.join(save_path, 'im_ab.npy'), im_ab0)
  # np.save(os.path.join(save_path, 'im_mask.npy'), im_mask0)

  result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
  mask = im_mask0.transpose((1, 2, 0)).astype(np.uint8)*255
  # cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
  # cv2.imwrite(os.path.join(save_path, save_path + 'ours.png'), result_bgr)
  cv2.imwrite(os.path.join(save_path, save_path + 'ours_fullres.png'), model.get_img_fullres()[:, :, ::-1])
  # cv2.imwrite(os.path.join(save_path, 'input_fullres.png'), model.get_input_img_fullres()[:, :, ::-1])
  # cv2.imwrite(os.path.join(save_path, 'input.png'), model.get_input_img()[:, :, ::-1])
  # cv2.imwrite(os.path.join(save_path, 'input_ab.png'), model.get_sup_img()[:, :, ::-1])

# image_file = "test1.png"
# compute_result(image_file)


# save_result(res)

# os.system("ffmpeg -i test.mov -vf fps=10 frames/ffout%03d.png")