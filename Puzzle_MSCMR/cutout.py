import torch
import numpy as np


def Cutout(imgs,labels, n_holes=1, length=16):
    try:
        imgs = imgs.tensors #(12,1,212,212)
    except:
        pass

    labels = [t["masks"] for t in labels] #(12,1,212,212)
    labels = torch.stack(labels)

    h = imgs.shape[2]
    w = imgs.shape[3]
    num = imgs.shape[0]
    labels_list = []
    imgs_list = []

    for i in range(num):
        label = labels[i,:,:,:]
        img = imgs[i,:,:,:]
        mask = np.ones((1, h, w), np.float32)
        for n in range(n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[0, y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask
        label = label * mask

        imgs_list.append(img)
        labels_list.append(label)
    imgs_out = torch.stack(imgs_list)
    labels_out = torch.stack(labels_list)

    return imgs_out, labels_out