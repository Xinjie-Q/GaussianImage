import math
import os
import time
from pathlib import Path
from typing import Optional
import argparse
import yaml
import numpy as np
import torch
# import tyro
import sys
from PIL import Image
from torch import Tensor, optim
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim
from utils import *
# from optimizer import OptimizerConfig, Optimizers
# from scheduler import SchedulerConfig, MultiStepSchedulerConfig
from tqdm import tqdm
from collections import OrderedDict
import random
import copy

class SimpleTrainer2d:
    """Trains random 2d gaussians to fit an image."""
    def __init__(
        self,
        image_path: Path,
        num_points: int = 2000,
        model_name:str = "GaussianImage_Cholesky",
        iterations:int = 30000,
        model_path = None,
        args = None,
    ):
        self.device = torch.device("cuda:0")
        self.gt_image = image_path_to_tensor(image_path).to(self.device) #gt_image.to(device=self.device)
        self.num_points = num_points
        image_path = Path(image_path)
        image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.log_dir = Path(f"./checkpoints_quant/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{image_name}")

        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=True).to(self.device)
            
        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=True).to(self.device)
            
        self.logwriter = LogWriter(self.log_dir, train=False)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)

    def test(self,):
        self.gaussian_model.eval()
        with torch.no_grad():
            encoding_dict = self.gaussian_model.compress_wo_ec()
            out = self.gaussian_model.decompress_wo_ec(encoding_dict)
            start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.decompress_wo_ec(encoding_dict)
            end_time = (time.time() - start_time)/100
        data_dict = self.gaussian_model.analysis_wo_ec(encoding_dict)
    
        out_img = out["render"].float()
        mse_loss = F.mse_loss(out_img, self.gt_image)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out_img, self.gt_image, data_range=1, size_average=True).item()
        
        data_dict["psnr"] = psnr
        data_dict["ms-ssim"] = ms_ssim_value
        data_dict["rendering_time"] = end_time
        data_dict["rendering_fps"] = 1/end_time
        np.save(self.log_dir / "test.npy", data_dict)
        self.logwriter.write("Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, 1/end_time))
        self.logwriter.write("PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(psnr, ms_ssim_value, data_dict["bpp"]))
        self.logwriter.write("position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))
        return data_dict


def image_path_to_tensor(image_path: Path):
    import torchvision.transforms as transforms
    img = Image.open(image_path)
    transform = transforms.ToTensor()
    img_tensor = transform(img).unsqueeze(0) #[1, C, H, W]
    return img_tensor

def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.")
    parser.add_argument(
        "-d", "--dataset", type=str, default='./dataset/kodak/', help="Training dataset"
    )
    parser.add_argument(
        "--data_name", type=str, default='kodak', help="Training dataset"
    )
    parser.add_argument(
        "--iterations", type=int, default=50000, help="number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--model_name", type=str, default="GaussianImage_Cholesky", help="model selection: GaussianImage_Cholesky, GaussianImage_RS, 3DGS"
    )
    parser.add_argument(
        "--sh_degree", type=int, default=3, help="SH degree (default: %(default)s)"
    )
    parser.add_argument(
        "--num_points",
        type=int,
        default=50000,
        help="2D GS points (default: %(default)s)",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Path to a checkpoint")
    parser.add_argument("--seed", type=float, default=1, help="Set random seed for reproducibility")
    parser.add_argument("--quantize", action="store_true", help="Quantize")
    parser.add_argument("--save_imgs", action="store_true", help="Save image")
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default: %(default)s)",
    )
    parser.add_argument("--pretrained", type=str, help="Path to a checkpoint")
    args = parser.parse_args(argv)
    return args

def main(argv):
    args = parse_args(argv)
    # Cache the args as a text string to save them in the output dir later
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)

    logwriter = LogWriter(Path(f"./checkpoints_quant/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"), train=False)
    psnrs, ms_ssims, eval_times, eval_fpses, bpps = [], [], [], [], []
    position_bpps, cholesky_bpps, feature_dc_bpps = [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        image_length, start = 24, 0
    elif args.data_name == "DIV2K_valid_LRX2":
        image_length, start = 100, 800
    for i in range(start, start+image_length):
        if args.data_name == "kodak":
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
            model_path = Path(args.model_path) / f'kodim{i+1:02}' / 'gaussian_model.best.pth.tar'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'
            model_path = Path(args.model_path) / f'{i+1:04}x2' / 'gaussian_model.best.pth.tar'
        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=model_path)

        data_dict = trainer.test()
        psnrs.append(data_dict["psnr"])
        ms_ssims.append(data_dict["ms-ssim"])
        eval_times.append(data_dict["rendering_time"])
        eval_fpses.append(data_dict["rendering_fps"])
        bpps.append(data_dict["bpp"])
        position_bpps.append(data_dict["position_bpp"])
        cholesky_bpps.append(data_dict["cholesky_bpp"])
        feature_dc_bpps.append(data_dict["feature_dc_bpp"])
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
            image_name, trainer.H, trainer.W, data_dict["psnr"],  data_dict["ms-ssim"], data_dict["bpp"], 
            data_dict["rendering_time"], data_dict["rendering_fps"], 
            data_dict["position_bpp"], data_dict["cholesky_bpp"], data_dict["feature_dc_bpp"]))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_bpp = torch.tensor(bpps).mean().item()
    avg_position_bpp = torch.tensor(position_bpps).mean().item()
    avg_cholesky_bpp = torch.tensor(cholesky_bpps).mean().item()
    avg_feature_dc_bpp = torch.tensor(feature_dc_bpps).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Eval:{:.8f}s, FPS:{:.4f}, position_bpp:{:.4f}, cholesky_bpp:{:.4f}, feature_dc_bpp:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_eval_time, avg_eval_fps, 
        avg_position_bpp, avg_cholesky_bpp, avg_feature_dc_bpp))    
    
if __name__ == "__main__":
    main(sys.argv[1:])
