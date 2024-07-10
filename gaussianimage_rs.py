from gsplat.project_gaussians_2d_scale_rot import project_gaussians_2d_scale_rot
from gsplat.rasterize_sum import rasterize_gaussians_sum
from pytorch_msssim import SSIM
from utils import *
import torch
import torch.nn as nn
import numpy as np
import math
from quantize import *
from optimizer import Adan

class GaussianImage_RS(nn.Module):
    def __init__(self, loss_type="L2", **kwargs):
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

        self._xyz = nn.Parameter(torch.atanh(2 * (torch.rand(self.init_num_points, 2) - 0.5)))
        self._scaling = nn.Parameter(torch.rand(self.init_num_points, 2))
        self.register_buffer('_opacity', torch.ones((self.init_num_points, 1)))
        self._rotation = nn.Parameter(torch.rand(self.init_num_points, 1))
        self._features_dc = nn.Parameter(torch.rand(self.init_num_points, 3))

        self.last_size = (self.H, self.W)
        self.background = torch.ones(3, device=self.device)
        self.rotation_activation = torch.sigmoid
        self.register_buffer('bound', torch.tensor([0.5, 0.5]).view(1, 2))
        self.quantize = kwargs["quantize"]

        if self.quantize:
            self.xyz_quantizer = FakeQuantizationHalf.apply 
            self.features_dc_quantizer = VectorQuantizer(codebook_dim=3, codebook_size=8, num_quantizers=2, vector_type="vector", kmeans_iters=5) 
            self.scaling_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=2) 
            self.rotation_quantizer = UniformQuantizer(signed=False, bits=6, learned=True, num_channels=1)

        if kwargs["opt_type"] == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=kwargs["lr"])
        else:
            self.optimizer = Adan(self.parameters(), lr=kwargs["lr"])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20000, gamma=0.5)

    def _init_data(self):
        self.scaling_quantizer._init_data(self._scaling)
        self.rotation_quantizer._init_data(self.get_rotation)

    @property
    def get_scaling(self):
        return torch.abs(self._scaling+self.bound)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)*2*math.pi
    
    @property
    def get_xyz(self):
        return torch.tanh(self._xyz)
    
    @property
    def get_features(self):
        return self._features_dc
    
    @property
    def get_opacity(self):
        return self._opacity 
    
    def forward(self):
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(self.get_xyz, self.get_scaling, self.get_rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                self.get_features, self.get_opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1) #[H, W, 3]
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

    def forward_quantize(self):
        l_vqm, m_bit = 0, 16*self.init_num_points*2
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        scaling, l_vqs, s_bit = self.scaling_quantizer(self._scaling)
        scaling = torch.abs(scaling + self.bound)
        rotation, l_vqr, r_bit = self.rotation_quantizer(self.get_rotation)
        colors, l_vqc, c_bit = self.features_dc_quantizer(self.get_features)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        vq_loss = l_vqm + l_vqs + l_vqr + l_vqc 
        return {"render": out_img, "vq_loss": vq_loss, "unit_bit":[m_bit, s_bit, r_bit, c_bit]}

    def train_iter_quantize(self, gt_image):
        render_pkg = self.forward_quantize()
        image = render_pkg["render"]
        loss = loss_fn(image, gt_image, self.loss_type, lambda_value=0.7) + render_pkg["vq_loss"]
        loss.backward()
        with torch.no_grad():
            mse_loss = F.mse_loss(image, gt_image)
            psnr = 10 * math.log10(1.0 / mse_loss.item())
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none = True)
        self.scheduler.step()
        return loss, psnr

    def compress_wo_ec(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        quant_scaling, _ = self.scaling_quantizer.compress(self._scaling)
        quant_rotation, _ = self.rotation_quantizer.compress(self.get_rotation)
        _, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "quant_scaling": quant_scaling, "quant_rotation": quant_rotation}

    def decompress_wo_ec(self, encoding_dict):
        xyz, quant_scaling, quant_rotation = encoding_dict["xyz"], encoding_dict["quant_scaling"], encoding_dict["quant_rotation"]
        feature_dc_index = encoding_dict["feature_dc_index"]
        means = torch.tanh(xyz.float())
        scaling = self.scaling_quantizer.decompress(quant_scaling)
        scaling = torch.abs(scaling + self.bound)
        rotation = self.rotation_quantizer.decompress(quant_rotation)
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}
    
    def analysis_wo_ec(self, encoding_dict):
        quant_scaling, quant_rotation, feature_dc_index = encoding_dict["quant_scaling"], encoding_dict["quant_rotation"], encoding_dict["feature_dc_index"]

        total_bits = 0
        initial_bits, scaling_codebook_bits, rotation_codebook_bits, feature_dc_codebook_bits = 0, 0, 0, 0

        
        scaling_codebook_bits += self.scaling_quantizer.scale.numel()*torch.finfo(self.scaling_quantizer.scale.dtype).bits
        scaling_codebook_bits += self.scaling_quantizer.beta.numel()*torch.finfo(self.scaling_quantizer.beta.dtype).bits
        rotation_codebook_bits += self.rotation_quantizer.scale.numel()*torch.finfo(self.rotation_quantizer.scale.dtype).bits
        rotation_codebook_bits += self.rotation_quantizer.beta.numel()*torch.finfo(self.rotation_quantizer.beta.dtype).bits  

        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            feature_dc_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits

        initial_bits += scaling_codebook_bits
        initial_bits += rotation_codebook_bits
        initial_bits += feature_dc_codebook_bits

        quant_scaling, quant_rotation, feature_dc_index = quant_scaling.cpu().numpy(), quant_rotation.cpu().numpy(), feature_dc_index.cpu().numpy()
        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += quant_scaling.size * 6
        total_bits += quant_rotation.size * 6
        index_max = np.max(feature_dc_index)
        max_bit = np.ceil(np.log2(index_max))
        total_bits += feature_dc_index.size * max_bit

        position_bits = self._xyz.numel()*16
        scaling_bits, rotation_bits, feature_dc_bits = 0, 0, 0
        scaling_bits += scaling_codebook_bits
        scaling_bits += quant_scaling.size * 6
        rotation_bits += rotation_codebook_bits
        rotation_bits += quant_rotation.size * 6
        feature_dc_bits += feature_dc_codebook_bits
        feature_dc_bits += feature_dc_index.size * max_bit

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        scaling_bpp = scaling_bits/self.H/self.W
        rotation_bpp = rotation_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        cholesky_bpp = scaling_bpp+rotation_bpp
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp, "scaling_bpp": scaling_bpp,
            "rotation_bpp": rotation_bpp}

    def compress(self):
        means = torch.tanh(self.xyz_quantizer(self._xyz))
        _, scaling_index = self.scaling_quantizer.compress(self._scaling)
        _, rotation_index = self.rotation_quantizer.compress(self.get_rotation)
        _, feature_dc_index = self.features_dc_quantizer.compress(self.get_features)
        return {"xyz":self._xyz.half(), "feature_dc_index": feature_dc_index, "scaling_index": scaling_index, "rotation_index": rotation_index}

    def decompress(self, encoding_dict):
        xyz, scaling_index, feature_dc_index = encoding_dict["xyz"], encoding_dict["scaling_index"], encoding_dict["feature_dc_index"]
        rotation_index = encoding_dict["rotation_index"]
        means = torch.tanh(xyz.float())
        scaling = self.scaling_quantizer.decompress(scaling_index)
        scaling = torch.abs(scaling + self.bound)
        rotation = self.rotation_quantizer.decompress(rotation_index)
        colors = self.features_dc_quantizer.decompress(feature_dc_index)
        self.xys, depths, self.radii, conics, num_tiles_hit = project_gaussians_2d_scale_rot(means, scaling, rotation, self.H, self.W, self.tile_bounds)
        out_img = rasterize_gaussians_sum(self.xys, depths, self.radii, conics, num_tiles_hit,
                colors, self._opacity, self.H, self.W, self.BLOCK_H, self.BLOCK_W, background=self.background, return_alpha=False)
        out_img = torch.clamp(out_img, 0, 1)
        out_img = out_img.view(-1, self.H, self.W, 3).permute(0, 3, 1, 2).contiguous()
        return {"render":out_img}

    def analysis(self, encoding_dict):
        scaling_index, rotation_index, feature_dc_index = encoding_dict["scaling_index"], encoding_dict["rotation_index"], encoding_dict["feature_dc_index"]
        scaling_compressed, scaling_histogram_table, scaling_unique = compress_matrix_flatten_categorical(scaling_index.int().flatten().tolist())
        rotation_compressed, rotation_histogram_table, rotation_unique = compress_matrix_flatten_categorical(rotation_index.int().flatten().tolist())
        feature_dc_compressed, feature_dc_histogram_table, feature_dc_unique = compress_matrix_flatten_categorical(feature_dc_index.int().flatten().tolist())  
        scaling_lookup = dict(zip(scaling_unique, scaling_histogram_table.astype(np.float64) / np.sum(scaling_histogram_table).astype(np.float64)))
        rotation_lookup = dict(zip(rotation_unique, rotation_histogram_table.astype(np.float64) / np.sum(rotation_histogram_table).astype(np.float64)))
        feature_dc_lookup = dict(zip(feature_dc_unique, feature_dc_histogram_table.astype(np.float64) / np.sum(feature_dc_histogram_table).astype(np.float64)))

        total_bits = 0
        initial_bits, scaling_codebook_bits, rotation_codebook_bits, feature_dc_codebook_bits = 0, 0, 0, 0
        for quantizer_index, layer in enumerate(self.scaling_quantizer.quantizer.layers):
            scaling_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        for quantizer_index, layer in enumerate(self.rotation_quantizer.quantizer.layers):
            rotation_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
        for quantizer_index, layer in enumerate(self.features_dc_quantizer.quantizer.layers):
            feature_dc_codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits

        initial_bits += scaling_codebook_bits
        initial_bits += rotation_codebook_bits
        initial_bits += feature_dc_codebook_bits
        initial_bits += get_np_size(scaling_histogram_table) * 8
        initial_bits += get_np_size(scaling_unique) * 8 
        initial_bits += get_np_size(rotation_histogram_table) * 8
        initial_bits += get_np_size(rotation_unique) * 8 
        initial_bits += get_np_size(feature_dc_histogram_table) * 8
        initial_bits += get_np_size(feature_dc_unique) * 8  

        total_bits += initial_bits
        total_bits += self._xyz.numel()*16
        total_bits += get_np_size(scaling_compressed) * 8
        total_bits += get_np_size(rotation_compressed) * 8
        total_bits += get_np_size(feature_dc_compressed) * 8

        position_bits = self._xyz.numel()*16
        scaling_bits, rotation_bits, feature_dc_bits = 0, 0, 0
        scaling_bits += scaling_codebook_bits
        scaling_bits += get_np_size(scaling_histogram_table) * 8
        scaling_bits += get_np_size(scaling_unique) * 8   
        scaling_bits += get_np_size(scaling_compressed) * 8
        rotation_bits += rotation_codebook_bits
        rotation_bits += get_np_size(rotation_histogram_table) * 8
        rotation_bits += get_np_size(rotation_unique) * 8   
        rotation_bits += get_np_size(rotation_compressed) * 8
        feature_dc_bits += feature_dc_codebook_bits
        feature_dc_bits += get_np_size(feature_dc_histogram_table) * 8
        feature_dc_bits += get_np_size(feature_dc_unique) * 8  
        feature_dc_bits += get_np_size(feature_dc_compressed) * 8

        bpp = total_bits/self.H/self.W
        position_bpp = position_bits/self.H/self.W
        scaling_bpp = scaling_bits/self.H/self.W
        rotation_bpp = rotation_bits/self.H/self.W
        feature_dc_bpp = feature_dc_bits/self.H/self.W
        cholesky_bpp = scaling_bpp+rotation_bpp
        return {"bpp": bpp, "position_bpp": position_bpp, 
            "cholesky_bpp": cholesky_bpp, "feature_dc_bpp": feature_dc_bpp, "scaling_bpp": scaling_bpp,
            "rotation_bpp": rotation_bpp}

