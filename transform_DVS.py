import torchvision.transforms as transforms
import numpy as np
import numbers
import random
import torchtoolbox
from torchtoolbox.transform.functional import cutout


class ToList():
    def __call__(self, imgmap):
        A = []
        l = imgmap.shape[0]
        d = imgmap.ndim
        for i in range(l):
            if d == 4:
                pic = imgmap[i, :].transpose(1, 2, 0)
                A.append(pic)
            elif d == 3:
                pic = imgmap[i, :]
                A.append(pic)
        return A


class ToPILImage:
    def __call__(self, imgmap):
        topilimage = transforms.ToPILImage()
        return [topilimage(i) for i in imgmap]


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, imgmap):
        resize = transforms.Resize(self.size)
        return [resize(i) for i in imgmap]


class ToTensor:
    def __call__(self, imgmap):
        totensor = transforms.ToTensor()
        return [totensor(i) for i in imgmap]


class RandomCrop:
    def __init__(self, size, consistent=True):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.consistent = consistent

    def __call__(self, imgmap, flowmap=None):
        img1 = imgmap[0]
        w, h = img1.size
        if self.size is not None:
            th, tw = self.size
            if w == tw and h == th:
                return imgmap
            if not flowmap:
                if self.consistent:
                    x1 = random.randint(0, w - tw)
                    y1 = random.randint(0, h - th)
                    return [i.crop((x1, y1, x1 + tw, y1 + th)) for i in imgmap]
                else:
                    result = []
                    for i in imgmap:
                        x1 = random.randint(0, w - tw)
                        y1 = random.randint(0, h - th)
                        result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                    return result
            elif flowmap is not None:
                assert (not self.consistent)
                result = []
                for idx, i in enumerate(imgmap):
                    proposal = []
                    for j in range(3):  # number of proposal: use the one with largest optical flow
                        x = random.randint(0, w - tw)
                        y = random.randint(0, h - th)
                        proposal.append([x, y, abs(np.mean(flowmap[idx, y:y + th, x:x + tw, :]))])
                    [x1, y1, _] = max(proposal, key=lambda x: x[-1])
                    result.append(i.crop((x1, y1, x1 + tw, y1 + th)))
                return result
            else:
                raise ValueError('wrong case')
        else:
            return imgmap


class Cutout:
    def __call__(self, imgmap):
        if random.random() < 0.5:
            cutout = torchtoolbox.transform.Cutout(p=1)
            return [cutout(i) for i in imgmap]
        else:
            return imgmap
