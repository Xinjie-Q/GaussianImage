import math
import time
from pathlib import Path
import argparse
import yaml
import numpy as np
import torch
import sys
from PIL import Image
import torch.nn.functional as F
from pytorch_msssim import ms_ssim
from utils import *
from tqdm import tqdm
import random
import copy
import torchvision.transforms as transforms

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
        self.image_name = image_path.stem
        BLOCK_H, BLOCK_W = 16, 16
        self.H, self.W = self.gt_image.shape[2], self.gt_image.shape[3]
        self.iterations = iterations
        self.log_dir = Path(f"./checkpoints_quant/{args.data_name}/{model_name}_{args.iterations}_{num_points}/{self.image_name}")
        self.save_imgs = args.save_imgs

        if model_name == "GaussianImage_Cholesky":
            from gaussianimage_cholesky import GaussianImage_Cholesky
            self.gaussian_model = GaussianImage_Cholesky(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=True).to(self.device)
            
        elif model_name == "GaussianImage_RS":
            from gaussianimage_rs import GaussianImage_RS
            self.gaussian_model = GaussianImage_RS(loss_type="L2", opt_type="adan", num_points=self.num_points, H=self.H, W=self.W, BLOCK_H=BLOCK_H, BLOCK_W=BLOCK_W, 
                device=self.device, lr=args.lr, quantize=True).to(self.device)

        self.logwriter = LogWriter(self.log_dir)

        if model_path is not None:
            print(f"loading model path:{model_path}")
            checkpoint = torch.load(model_path, map_location=self.device)
            model_dict = self.gaussian_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            self.gaussian_model.load_state_dict(model_dict)
            self.gaussian_model._init_data()

    def train(self):     
        psnr_list, iter_list = [], []
        progress_bar = tqdm(range(1, self.iterations+1), desc="Training progress")
        best_psnr = 0
        self.gaussian_model.train()
        start_time = time.time()
        best_psnr = 0
        for iter in range(1, self.iterations+1):
            loss, psnr = self.gaussian_model.train_iter_quantize(self.gt_image)
            psnr_list.append(psnr)
            iter_list.append(iter)
            if best_psnr < psnr:
                best_psnr = psnr
                best_model_dict = copy.deepcopy(self.gaussian_model.state_dict())
            with torch.no_grad():
                if iter % 10 == 0:
                    progress_bar.set_postfix({f"Loss":f"{loss.item():.{7}f}", "PSNR":f"{psnr:.{4}f}", "Best PSNR":f"{best_psnr:.{4}f}"})
                    progress_bar.update(10)
        end_time = time.time() - start_time
        progress_bar.close()
        psnr_value, ms_ssim_value, bpp = self.test()
        torch.save(self.gaussian_model.state_dict(), self.log_dir / "gaussian_model.pth.tar")
        self.gaussian_model.load_state_dict(best_model_dict)
        best_psnr_value, best_ms_ssim_value, best_bpp = self.test(True)
        torch.save(best_model_dict, self.log_dir / "gaussian_model.best.pth.tar")
        with torch.no_grad():
            self.gaussian_model.eval()
            test_start_time = time.time()
            for i in range(100):
                _ = self.gaussian_model.forward_quantize()
            test_end_time = (time.time() - test_start_time)/100

        self.logwriter.write("Training Complete in {:.4f}s, Eval time:{:.8f}s, FPS:{:.4f}".format(end_time, test_end_time, 1/test_end_time))
        np.save(self.log_dir / "training.npy", {"iterations": iter_list, "training_psnr": psnr_list, "training_time": end_time, 
        "psnr": psnr_value, "ms-ssim": ms_ssim_value, "rendering_time": test_end_time, "rendering_fps": 1/test_end_time, "bpp":bpp, 
        "best_psnr":best_psnr_value, "best_ms-ssim":best_ms_ssim_value, "best_bpp": best_bpp})
        return psnr_value, ms_ssim_value, end_time, test_end_time, 1/test_end_time, bpp, best_psnr_value, best_ms_ssim_value, best_bpp

    def test(self, best=False):
        self.gaussian_model.eval()
        with torch.no_grad():
            out = self.gaussian_model.forward_quantize()
        out_img = out["render"].float()
        self.gt_image = self.gt_image.float()
        mse_loss = F.mse_loss(out_img, self.gt_image)
        psnr = 10 * math.log10(1.0 / mse_loss.item())
        ms_ssim_value = ms_ssim(out["render"].float(), self.gt_image.float(), data_range=1, size_average=True).item()
        m_bit, s_bit, r_bit, c_bit = out["unit_bit"]
        bpp = (m_bit + s_bit + r_bit + c_bit)/self.H/self.W

        strings = "Best Test" if best else "Test"
        self.logwriter.write("{} PSNR:{:.4f}, MS_SSIM:{:.6f}, bpp:{:.4f}".format(strings, psnr, 
            ms_ssim_value, bpp))
        if self.save_imgs:
            transform = transforms.ToPILImage()
            img = transform(out_img.squeeze(0))
            name = "_codec_best.png" if best else "_codec.png" 
            name = self.image_name + name
            img.save(str(self.log_dir / name))
        return psnr, ms_ssim_value, bpp


def image_path_to_tensor(image_path: Path):
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

    logwriter = LogWriter(Path(f"./checkpoints_quant/{args.data_name}/{args.model_name}_{args.iterations}_{args.num_points}"))
    psnrs, ms_ssims, training_times, eval_times, eval_fpses, bpps = [], [], [], [], [], []
    best_psnrs, best_ms_ssims, best_bpps = [], [], []
    image_h, image_w = 0, 0
    if args.data_name == "kodak":
        image_length, start = 24, 0
    elif args.data_name == "DIV2K_valid_LRX2":
        image_length, start = 100, 800

    for i in range(start, start+image_length):
        if args.data_name == "kodak":
            image_path = Path(args.dataset) / f'kodim{i+1:02}.png'
            model_path = Path(args.model_path) / f'kodim{i+1:02}' / 'gaussian_model.pth.tar'
        elif args.data_name == "DIV2K_valid_LRX2":
            image_path = Path(args.dataset) /  f'{i+1:04}x2.png'
            model_path = Path(args.model_path) / f'{i+1:04}x2' / 'gaussian_model.pth.tar'
        
        trainer = SimpleTrainer2d(image_path=image_path, num_points=args.num_points, 
            iterations=args.iterations, model_name=args.model_name, args=args, model_path=model_path)
        psnr, ms_ssim, training_time, eval_time, eval_fps, bpp, best_psnr, best_ms_ssim, best_bpp = trainer.train()
        best_psnrs.append(best_psnr)
        best_ms_ssims.append(best_ms_ssim)
        best_bpps.append(best_bpp)
        psnrs.append(psnr)
        ms_ssims.append(ms_ssim)
        training_times.append(training_time) 
        eval_times.append(eval_time)
        eval_fpses.append(eval_fps)
        bpps.append(bpp)
        image_h += trainer.H
        image_w += trainer.W
        image_name = image_path.stem
        logwriter.write("{}: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, bpp:{:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f}, Best bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
            image_name, trainer.H, trainer.W, psnr, ms_ssim, bpp, best_psnr, best_ms_ssim, best_bpp, training_time, eval_time, eval_fps))

    avg_psnr = torch.tensor(psnrs).mean().item()
    avg_ms_ssim = torch.tensor(ms_ssims).mean().item()
    avg_training_time = torch.tensor(training_times).mean().item()
    avg_eval_time = torch.tensor(eval_times).mean().item()
    avg_eval_fps = torch.tensor(eval_fpses).mean().item()
    avg_bpp = torch.tensor(bpps).mean().item()

    avg_best_psnr = torch.tensor(best_psnrs).mean().item()
    avg_best_ms_ssim = torch.tensor(best_ms_ssims).mean().item()
    avg_best_bpp = torch.tensor(best_bpps).mean().item()
    avg_h = image_h//image_length
    avg_w = image_w//image_length

    logwriter.write("Average: {}x{}, PSNR:{:.4f}, MS-SSIM:{:.4f}, Bpp:{:.4f}, Best PSNR:{:.4f}, Best MS-SSIM:{:.4f}, Best bpp:{:.4f}, Training:{:.4f}s, Eval:{:.8f}s, FPS:{:.4f}".format(
        avg_h, avg_w, avg_psnr, avg_ms_ssim, avg_bpp, avg_best_psnr, avg_best_ms_ssim, avg_best_bpp, avg_training_time, avg_eval_time, avg_eval_fps))    
if __name__ == "__main__":
    main(sys.argv[1:])
