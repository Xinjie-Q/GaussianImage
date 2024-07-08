from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import num_sh_bases, spherical_harmonics
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from optimizer import Adan

def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N, 1)
    v = torch.rand(N, 1)
    w = torch.rand(N, 1)
    return torch.cat(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )

class Gaussian3D(nn.Module):
    def __init__(self, loss_type="Fusion2", **kwargs):
        super().__init__()
        self.loss_type = loss_type
        self.init_num_points = kwargs["num_points"]
        self.H, self.W = kwargs["H"], kwargs["W"]
        self.BLOCK_W, self.BLOCK_H = kwargs["BLOCK_W"], kwargs["BLOCK_H"]
        self.tile_bounds = (
            (self.W + self.BLOCK_W - 1) // self.BLOCK_W,
            (self.H + self.BLOCK_H - 1) // self.BLOCK_H,
            1,
        ) # 
        self.device = kwargs["device"]

        self._xyz = nn.Parameter((2 * (torch.rand(self.init_num_points, 3) - 0.5)))
        distances, _ = self.k_nearest_sklearn(self._xyz.data, 3)
        distances = torch.from_numpy(distances)
        # find the average of the three nearest neighbors for each point and use that as the scale
        avg_dist = distances.mean(dim=-1, keepdim=True)
        self._scaling = nn.Parameter(torch.log(avg_dist.repeat(1, 3)))
        self._opacity = nn.Parameter(torch.logit(0.1 * torch.ones(self.init_num_points, 1)))
        self._rotation = nn.Parameter(random_quat_tensor(self.init_num_points))

        self.active_sh_degree = kwargs["sh_degree"]
        dim_sh = num_sh_bases(self.active_sh_degree)
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 1, 3))
        self._features_rest = nn.Parameter(torch.zeros((self.init_num_points, dim_sh - 1, 3)))

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        fov_x = math.pi / 2.0
        self.focal = 0.5 * float(self.W) / math.tan(0.5 * fov_x)
        self.viewmat = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 8.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            device=self.device,
        )
        self.translation = torch.tensor([0, 0, -8], device=self.device).view(1, 3)
        self.scaling_activation = torch.exp
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def forward(self):
        colors = self.get_features
        quats = self.get_rotation
        self.xys, depths, self.radii, conics, num_tiles_hit, cov3d = project_gaussians(self.get_xyz, self.get_scaling, 1, quats, 
            self.viewmat, self.viewmat, self.focal, self.focal, self.W / 2, self.H / 2, self.H, self.W, self.tile_bounds)

        if self.active_sh_degree > 0:
            viewdirs = self.get_xyz.detach() - self.translation  # (N, 3)
            viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            n = self.active_sh_degree
            rgbs = spherical_harmonics(n, viewdirs, colors)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)
        else:
            rgbs = torch.sigmoid(colors[:, 0, :])

        out_img, alpha = rasterize_gaussians(self.xys, depths, self.radii, conics, num_tiles_hit,
                rgbs, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=True)
        alpha = alpha[..., None]
        out_img = torch.clamp(out_img, max=1.0)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render": out_img}

    def train_iter(self, gt_image):
        render_pkg = self.forward()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7)
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr

    def k_nearest_sklearn(self, x, k):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()
        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors
        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)
        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)
        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

