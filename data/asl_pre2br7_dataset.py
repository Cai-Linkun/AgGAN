from data.base_dataset import BaseDataset, get_transform
import numpy as np
import torch


class AslPre2Br7Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        path = opt.dataroot + "/" + opt.phase + ".npz"
        self._store = np.load(path, allow_pickle=True)
        print("load data from: ", path, " size: ", len(self._store.files))
        self.transform = get_transform(self.opt, convert=True, grayscale=True)

    def __getitem__(self, index):
        key = self._store.files[index]
        imgs = self._store[key].item()
        x = imgs["asl_pre"]
        y = imgs["asl_br7"]

        x[x > 255] = 255
        y[y > 255] = 255
        x = x / 255
        y = y / 255

        path = imgs["asl_pre_path"]
        pix_cls = imgs["pix_cls"]
        x_tensor = torch.FloatTensor(x).T.unsqueeze(0) * 2 - 1
        y_tensor = torch.FloatTensor(y).T.unsqueeze(0) * 2 - 1
        pix_cls = torch.IntTensor(pix_cls).T.unsqueeze(0)

        return {"A": x_tensor, "B": y_tensor, "Path": path, "Uri": key, "pix_cls": pix_cls}

    def __len__(self):
        return len(self._store)
