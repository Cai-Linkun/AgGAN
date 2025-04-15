import os
import csv
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
from util.util import seed_everything
from data.preprocess import save_img_with_uri_info, get_uri_from_name

try:
    import wandb
except ImportError:
    print('Warning: wandb package cannot be found. The option "--use_wandb" will result in error.')

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
from skimage.metrics import normalized_root_mse

def calc_img_metrics(tgt, fake):
    SSIM = structural_similarity(tgt, fake, data_range=255)
    PSNR = peak_signal_noise_ratio(tgt, fake, data_range=255)
    MSE = mean_squared_error(tgt, fake)
    NRMSE = normalized_root_mse(tgt, fake)
    N_MSE = mean_squared_error(tgt / 255, fake / 255)
    N_NRMSE = normalized_root_mse(tgt / 255, fake / 255)
    N1_MSE = mean_squared_error((tgt / 255) * 2 - 1, (fake / 255) * 2 - 1)
    N1_NRMSE = normalized_root_mse((tgt / 255) * 2 - 1, (fake / 255) * 2 - 1)
    metric = {
        "mse": MSE,
        "psnr": PSNR,
        "ssim": SSIM,
        "nrmse": NRMSE,
        "n_mse": N_MSE,
        "n_nrmse": N_NRMSE,
        "n1_mse": N1_MSE,
        "n1_nrmse": N1_NRMSE,
    }
    print(metric)
    return metric


if __name__ == '__main__':
    seed_everything(914)
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    opt.no_dropout = True
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    # initialize logger
    if opt.use_wandb:
        wandb_run = wandb.init(project=opt.wandb_project_name, name=opt.name, config=opt) if not wandb.run else wandb.run
        wandb_run._label(repo='CycleGAN-and-pix2pix')

    # create a website
    web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    if opt.load_iter > 0:  # load_iter is 0 by default
        web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    print('creating web directory', web_dir)
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    metric_result_f = open(os.path.join(web_dir, "metric.csv"), "w")
    metric_result_writer = csv.DictWriter(
        metric_result_f, ["image", "psnr", "mse", "ssim", "nrmse", "n_mse", "n_nrmse", "n1_mse", "n1_nrmse"]
    )
    metric_result_writer.writeheader()

    nii_save_dir = os.path.join(opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch), "nii")
    nii_path = "./datasets/1217_ASL_T1/ASL/BR7_Cor"
    nii_name = "wX{}_corr-CBF.nii_brain.nii"

    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):
        if i >= opt.num_test:  # only apply our model to opt.num_test images.
            break
        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference
        visuals = model.get_current_visuals()  # get image results
        img_path = model.get_image_paths()     # get image paths
        t, l = get_uri_from_name(img_path[0])
        img_path = ["testee_" + t + "_layer_" + str(int(l) + 1)]
        _nii_path = os.path.join(nii_path, nii_name.format(t.zfill(2)))
        if "real_B_vis" in visuals and "fake_B_vis" in visuals:
            metric = calc_img_metrics(visuals["real_B_vis"], visuals["fake_B_vis"])
            metric["image"] = img_path[0]
            metric_result_writer.writerow(metric)

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
        save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, width=opt.display_winsize, use_wandb=opt.use_wandb)

        save_img_with_uri_info(visuals["fake_B_vis"], _nii_path, (t, l), nii_save_dir)
        save_img_with_uri_info(
            np.absolute(visuals["fake_B_vis"] - visuals["real_B_vis"]), _nii_path, (t, l), nii_save_dir, prefix="_diff"
        )
    webpage.save()  # save the HTML
    metric_result_f.close()
