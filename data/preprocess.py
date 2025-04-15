import os
import re
import numpy as np
import nibabel as nib
from collections import namedtuple

URI = namedtuple("URI", ["testee", "layer"])


def _crop_img_to_size(img, pref_size):
    assert len(img.shape) == 2 and len(pref_size) == 2
    new_img = np.zeros(pref_size)
    w = min(new_img.shape[0], img.shape[0])
    h = min(new_img.shape[1], img.shape[1])
    new_img[:w, 4 : h + 4] = img[:w, :h]
    return new_img


def save_img_with_uri_info(img, nii_path, uri, result_dir, prefix=""):
    uri = URI(uri[0], uri[1])
    # 加载 NIfTI 文件
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()  # 获取图像数据为 NumPy 数组
    print(data.shape)
    header = nii_img.header  # 获取头信息
    affine = nii_img.affine  # 获取仿射变换矩阵

    # 修改图像数据
    img = _crop_img_to_size(img.T, (data.shape[0], data.shape[1]))
    data[:, :, int(uri.layer)] = img

    # 保存修改后的 .nii 文件
    # 使用原始的 affine 和 header 保存新的 NIfTI 文件
    new_img = nib.Nifti1Image(data, affine, header)

    # fname = os.path.basename(nii_path)
    fn = "testee" + str(uri.testee) + "_layer" + str(int(uri.layer) + 1)
    fname = fn + prefix + ".nii"
    out_file = os.path.join(result_dir, fname)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    nib.save(new_img, out_file)

    print(f"Modified NIfTI file saved to: {out_file}")


def _read_nii_as_numpy(path):
    if not os.path.exists(path):
        return None
    nii_img = nib.load(path)
    img = np.array(nii_img.dataobj)
    return img  # float


def _uri2name(uri: URI):
    return "testee" + str(uri.testee) + "_layer" + str(uri.layer)


def get_uri_from_name(name: str):
    pat = "testee(\d+)_layer(\d+)"
    matches = re.match(pattern=pat, string=name)
    if matches is not None:
        return (matches.group(1), matches.group(2))


class NIIStore:
    data_root = "datasets/1217_ASL_T1/"
    asl_pre_dir = data_root + "ASL/PRE_Cor"
    asl_br7_dir = data_root + "ASL/BR7_Cor"
    asl_testee_pat = "wX(\d{2}).*\.nii$"

    t1_pre_dir = data_root + "T1/T1_PRE_RESLICE"
    t1_br7_dir = data_root + "T1/T1_BR7_RESLICE"
    t1_testee_pat = "wsub(\d{1,3}).*\.nii"

    aal_template_file = data_root + "AAL_61x73x61_YCG.nii"

    train_file = data_root + "train.npz"
    test_file = data_root + "test.npz"

    MAX_VALUE = 255
    MEAN = 0.08236331197308984
    STD = 0.1353988744181669

    ORI_SHAPE = (61, 73)
    PROCESSED_SHAPE = (64, 64)

    def __init__(self):
        self.resource = NIIStore.read_all()
        print("resource size: ", len(self.resource))

    def get(self, uri):
        if uri in self.resource:
            return self.resource[uri]
        return None

    def split_and_save_data(self):
        if os.path.exists(NIIStore.train_file) or os.path.exists(NIIStore.test_file):
            assert False, "file exist, please delete them before run this script."

        import random

        random.seed(914)
        indices = list(range(0, len(self.resource)))
        random.shuffle(indices)
        split = int(len(indices) * 0.9)
        train_indices, test_indices = indices[:split], indices[split:]
        train = {}
        test = {}
        for ind in train_indices:
            uri, val = self.__getitem__(ind)
            train[_uri2name(uri)] = val
        for ind in test_indices:
            uri, val = self.__getitem__(ind)
            test[_uri2name(uri)] = val

        print("train len: ", len(train))
        print("test len: ", len(test))
        np.savez(NIIStore.train_file, **train)
        np.savez(NIIStore.test_file, **test)

    def __len__(self):
        return len(self.resource)

    def __getitem__(self, idx):
        uri = list(self.resource.keys())[idx]
        return uri, self.resource[uri]

    def transform_img(img):
        img[img > NIIStore.MAX_VALUE] = NIIStore.MAX_VALUE
        img = img / NIIStore.MAX_VALUE
        return img

    def retransform_img(img):
        img = img * NIIStore.MAX_VALUE
        return img

    def read_all():
        def _read_items(_dir, _pat):
            res = {}
            nil_paths = {}
            for fn in os.listdir(_dir):
                mat_res = re.match(_pat, fn)
                if mat_res:
                    testee = int(mat_res.group(1))
                    nii_path = os.path.join(_dir, fn)
                    img = _read_nii_as_numpy(nii_path)
                    layer_num = img.shape[2]  # 0, 1, 2
                    for l in range(layer_num):
                        if l < 21 or l > 36:
                            continue

                        uri = URI(testee, l)
                        res[uri] = img[:, :, l]
                        nil_paths[uri] = nii_path
            return res, nil_paths

        # ASL/PRE_Cor
        asl_pre_dict, asl_pre_paths = _read_items(NIIStore.asl_pre_dir, NIIStore.asl_testee_pat)

        # ASL/BR7_Cor
        asl_br7_dict, _ = _read_items(NIIStore.asl_br7_dir, NIIStore.asl_testee_pat)

        # T1/T1_PRE_RESLICE
        t1_pre_dict, _ = _read_items(NIIStore.t1_pre_dir, NIIStore.t1_testee_pat)

        # T1/T1_BR7_RESLICE
        t1_br7_dict, _ = _read_items(NIIStore.t1_br7_dir, NIIStore.t1_testee_pat)

        # Combine
        aal = NIIStore.read_aal_template_as_multi_channel_img()
        resources = {}
        for uri in asl_pre_dict:
            testee, layer = uri.testee, uri.layer
            pix_cls = aal[:, :, layer]
            asl_pre_img = NIIStore.pre_process(asl_pre_dict[uri])
            pix_cls = NIIStore.pre_process(pix_cls)
            is_all_zero = np.all(asl_pre_img == 0) or np.all(pix_cls == 0)
            if is_all_zero:
                continue

            if not uri in asl_br7_dict:
                continue
            asl_br7_img = NIIStore.pre_process(asl_br7_dict[uri])
            is_all_zero = np.all(asl_br7_img == 0)
            if is_all_zero:
                continue

            if uri in asl_br7_dict and uri in t1_pre_dict and uri in t1_br7_dict:
                resources[uri] = {
                    "asl_pre": asl_pre_img,
                    "asl_br7": NIIStore.pre_process(asl_br7_dict[uri]),
                    "t1_pre": NIIStore.pre_process(t1_pre_dict[uri]),
                    "t1_br7": NIIStore.pre_process(t1_br7_dict[uri]),
                    "asl_pre_path": asl_pre_paths[uri],
                    "pix_cls": pix_cls,
                }
                assert (
                    resources[uri]["asl_pre"].shape
                    == resources[uri]["asl_br7"].shape
                    == resources[uri]["t1_pre"].shape
                    == resources[uri]["t1_br7"].shape
                    == resources[uri]["pix_cls"].shape
                )
        return resources

    def pre_process(img):
        img = np.nan_to_num(img)
        assert img.shape == NIIStore.ORI_SHAPE, img.shape
        _img = np.zeros(NIIStore.PROCESSED_SHAPE)
        assert _img.shape == NIIStore.PROCESSED_SHAPE
        w = min(NIIStore.ORI_SHAPE[0], NIIStore.PROCESSED_SHAPE[0])
        h = min(NIIStore.ORI_SHAPE[1], NIIStore.PROCESSED_SHAPE[1])
        _img[:w, :h] = img[:w, 4 : h + 4]
        return _img

    def read_aal_template_as_multi_channel_img():
        img = _read_nii_as_numpy(NIIStore.aal_template_file)
        return img


if __name__ == "__main__":
    store = NIIStore()
    store.split_and_save_data()

    train = np.load(NIIStore.train_file, allow_pickle=True)
    print(train)
    print(len(train.files))
    print(train[train.files[0]].dtype)
    print(train[train.files[0]].item()["asl_pre"])
    # print(train.files)
