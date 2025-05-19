import io
import torch 
from einops import rearrange
import matplotlib
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def unnorm_img(x):
    mean = torch.tensor([0.485, 0.456, 0.406])
    std = torch.tensor([0.229, 0.224, 0.225])
    mean = rearrange(mean, 'c -> c 1 1')
    std = rearrange(std, 'c -> c 1 1')
    x = (x[:3]*std) + mean
    return x


def make_figure(img, title=None):
    if len(img.shape) == 3:
        img = img.squeeze()
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    matplotlib.use('Svg')
    h,w = img.shape
    w_ = (w/h)*4
    fig = plt.figure(figsize=(w_+1, 4))
    plt.imshow(img, cmap='jet')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    # plt.axis('off')
    # plt.tight_layout()
    # buf = io.BytesIO()
    # plt.savefig(buf, format='png')
    # buf.seek(0)
    # plt.close()

    # # decode buf to image
    # image = Image.open(buf)
    # image = np.array(image)
    # return image / 255.0
    return fig