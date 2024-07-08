from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch
from torch import nn
from utils import *
import torch.nn.functional as F
import constriction
import numpy as np

def grad_scale(x, scale):
    return (x - x * scale).detach() + x * scale

def ste(x):
    return (x.round() - x).detach() + x

class FakeQuantizationHalf(torch.autograd.Function):
    """performs fake quantization for half precision"""

    @staticmethod
    def forward(_, x):
        return x.half().float()

    @staticmethod
    def backward(_, grad_output):
        return grad_output

class UniformQuantizer(nn.Module):
    def __init__(self, signed=False, bits=8, learned=False, num_channels=1, entropy_type="none", weight=0.001):
        super().__init__()
        if signed:
            self.qmin = -2**(bits - 1)
            self.qmax = 2 ** (bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2 ** bits - 1

        self.learned = learned
        self.entropy_type = entropy_type
        if self.learned:
            self.scale = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)
            self.beta = nn.Parameter(torch.ones(num_channels)/self.qmax, requires_grad=True)

        self.weight = weight

    def _init_data(self, tensor):
        device = tensor.device
        t_min, t_max = tensor.min(dim=0)[0], tensor.max(dim=0)[0]
        scale = (t_max - t_min) / (self.qmax-self.qmin)
        self.beta.data = t_min.to(device)
        self.scale.data = scale.to(device)

    def forward(self, x):
        if self.learned:
            grad = 1.0 / ((self.qmax * x.numel()) ** 0.5)
            s_scale = grad_scale(self.scale, grad)
            beta_scale = grad_scale(self.beta, grad)
            s_scale, beta_scale = self.scale, self.beta
            code = ((x - beta_scale) / s_scale).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant * s_scale + beta_scale
        else:
            code = (x * self.qmax).clamp(self.qmin, self.qmax)
            quant = ste(code)
            dequant = quant / self.qmax

        bits, entropy_loss = 0, 0
        if not self.training:
            num_points, num_channels = x.shape
            bits = self.size(quant)
            # unit_bit = bits / num_points / num_channels
        return dequant, entropy_loss*self.weight, bits

    def size(self, quant):
        index_bits = 0
        compressed, histogram_table, unique = compress_matrix_flatten_categorical(quant.int().flatten().tolist())
        index_bits += get_np_size(compressed) * 8
        index_bits += get_np_size(histogram_table) * 8
        index_bits += get_np_size(unique) * 8 
        index_bits += self.scale.numel()*torch.finfo(self.scale.dtype).bits
        index_bits += self.beta.numel()*torch.finfo(self.beta.dtype).bits
        return index_bits

    def compress(self, x):
        code = ((x - self.beta) / self.scale).clamp(self.qmin, self.qmax)
        return code.round(), code.round()* self.scale + self.beta

    def decompress(self, x):
        return x * self.scale + self.beta

class VectorQuantizer(nn.Module):
    def __init__(self, num_quantizers=1, codebook_dim=1, codebook_size=64, kmeans_iters=10, vector_type="vector"):
        super().__init__()
        self.num_quantizers = num_quantizers
        self.vector_type = vector_type
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                self.quantizer = VectorQuantize(dim=codebook_dim, codebook_size=codebook_size, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters)#learnable_codebook=True, ema_update = False, orthogonal_reg_weight =1.)
        else:
            if self.vector_type == "vector":
                self.quantizer = ResidualVQ(dim=codebook_dim, codebook_size=codebook_size, num_quantizers=num_quantizers, decay = 0.8, commitment_weight = 1., kmeans_init = True, 
                    kmeans_iters = kmeans_iters) #learnable_codebook=True, ema_update=False, orthogonal_reg_weight=0., in_place_codebook_optimizer=torch.optim.Adam)

    def forward(self, x):
        if self.training:
            x, _, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            return x, l_vq, 0
        else:
            num_points, num_channels = x.shape
            x, embed_index, l_vq = self.quantizer(x)
            l_vq = torch.sum(l_vq)
            bits = self.size(embed_index)
            # unit_bit = bits / num_points / num_channels
            return x, l_vq, bits

    def size(self, embed_index):
        if self.num_quantizers == 1:
            if self.vector_type == "vector":
                codebook_bits = self.quantizer._codebook.embed.numel()*torch.finfo(self.quantizer._codebook.embed.dtype).bits
            elif self.vector_type == "ste":
                codebook_bits = self.quantizer.embedding.weight.data.numel()*torch.finfo(self.quantizer.embedding.weight.data.dtype).bits
            index_bits = 0
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        else:
            codebook_bits, index_bits = 0, 0
            for quantizer_index, layer in enumerate(self.quantizer.layers):
                if self.vector_type == "vector":
                    codebook_bits += layer._codebook.embed.numel()*torch.finfo(layer._codebook.embed.dtype).bits
                elif self.vector_type == "ste":
                    codebook_bits += layer.embedding.weight.data.numel()*torch.finfo(layer.embedding.weight.data.dtype).bits
            compressed, histogram_table, unique = compress_matrix_flatten_categorical(embed_index.int().flatten().tolist())
            index_bits += get_np_size(compressed) * 8
            index_bits += get_np_size(histogram_table) * 8
            index_bits += get_np_size(unique) * 8  
        total_bits = codebook_bits + index_bits
        #print("vq:", embed_index.shape, codebook_bits, index_bits)
        return total_bits

    def compress(self, x):
        x, embed_index, _ = self.quantizer(x)
        return x, embed_index

    def decompress(self, embed_index):
        recon = 0
        for i,layer in enumerate(self.quantizer.layers):
            recon += layer._codebook.embed[0, embed_index[:, i]]
        return recon

def compress_matrix_flatten_categorical(matrix, return_table=False):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    matrix = np.array(matrix) #matrix.flatten()
    unique, unique_indices, unique_inverse, unique_counts = np.unique(matrix, return_index=True, return_inverse=True, return_counts=True, axis=None)
    min_value = np.min(unique)
    max_value = np.max(unique)
    unique = unique.astype(judege_type(min_value, max_value))
    message = unique_inverse.astype(np.int32)
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model)
    compressed = encoder.get_compressed()
    return compressed, unique_counts, unique

def decompress_matrix_flatten_categorical(compressed, unique_counts, quant_symbol, symbol_length, symbol_shape):
    '''
    :param matrix: np.array
    :return compressed, symtable
    '''
    probabilities = unique_counts.astype(np.float64) / np.sum(unique_counts).astype(np.float64)
    entropy_model = constriction.stream.model.Categorical(probabilities)
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, symbol_length)
    decoded = quant_symbol[decoded].reshape(symbol_shape)#.astype(np.int32)
    return decoded


def judege_type(min, max):
    if min>=0:
        if max<=256:
            return np.uint8
        elif max<=65535:
            return np.uint16
        else:
            return np.uint32
    else:
        if max<128 and min>=-128:
            return np.int8
        elif max<32768 and min>=-32768:
            return np.int16
        else:
            return np.int32
        
def get_np_size(x):
    return x.size * x.itemsize
