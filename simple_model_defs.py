import torch
import torch.nn.functional as F
from torch import nn

# --- UNet Parts ---

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int, F_int)
            )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int, F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=kernel_size, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1, 1),
            nn.Sigmoid()
        )

        self.gelu = nn.GELU()

    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.gelu(g1+x1)
        psi = self.psi(psi)

        return x*psi

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, dilation=1, kernel_size=5, padding_mode='zeros'):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        padding = (kernel_size//2)*dilation
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(mid_channels, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False, padding_mode=padding_mode),
            nn.BatchNorm2d(out_channels, out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, dilation=1, kernel_size=5, padding_mode='zeros'):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dilation=dilation, kernel_size=kernel_size, padding_mode=padding_mode)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AttnUp(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.attn = Attention_block(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.attn = Attention_block(F_g=out_channels, F_l=out_channels, F_int=out_channels//2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x1_x2_attn = self.attn(g=x1, x=x2)
        x = torch.cat([x1_x2_attn, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._in_channels = in_channels
        self.conv = nn.Conv2d(self._in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

    def getInChannels(self):
        return self._in_channels

# --- Maskable ---

class Maskable:
    def __init__(self, n_classes, input_channel_mask_index, output_channel_background_index):
        self.n_maskable_classes = n_classes
        self.input_channel_mask_index = input_channel_mask_index
        self.output_channel_background_index = output_channel_background_index
        self.mask_output = input_channel_mask_index >= 0 and output_channel_background_index >= 0

    def mask(self, softmaxed_preds, mask):
        mask_inv = 1-mask
        stack = [mask]*self.n_maskable_classes
        stack[self.output_channel_background_index] = mask_inv
        stacked_mask = torch.stack(stack, dim=1)
        masked_preds = softmaxed_preds.multiply(stacked_mask)
        return masked_preds
    def get_classes_count(self):
        return self.n_maskable_classes
    def get_input_channel_mask_index(self):
        return self.input_channel_mask_index
    def get_output_channel_background_index(self):
        return self.output_channel_background_index

# --- UNet ---

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, hyperparameters):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        # Helper to extract layer config safely
        def get_layer_param(layer_name, param_name, default=None):
            return hyperparameters["input"]["model_parameters"]["layers"][layer_name].get(param_name, default)

        self.inc = DoubleConv(in_channels=n_channels,
                              out_channels=get_layer_param("inc", "out_channels"),
                              dilation=get_layer_param("inc", "dilation"),
                              kernel_size=get_layer_param("inc", "kernel_size"),
                              padding_mode=get_layer_param("inc", "padding_mode"))

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        last_up_out_channels = None
        
        layers_config = hyperparameters["input"]["model_parameters"]["layers"]
        
        # Sort keys to ensure order if python version < 3.7 (not issue here but good practice)
        # Actually standard dict order is insertion order in py3.7+. 
        # The YAML has down_1, down_2... we should iterate in order.
        
        # Explicit iteration for down layers
        down_keys = [k for k in layers_config.keys() if "down_" in k]
        down_keys.sort() # down_1, down_2, ...
        
        for layer in down_keys:
            l = Down(in_channels=layers_config[layer]["in_channels"],
                     out_channels=layers_config[layer]["out_channels"],
                     dilation=layers_config[layer]["dilation"],
                     kernel_size=layers_config[layer]["kernel_size"],
                     padding_mode=layers_config[layer]["padding_mode"])
            self.down_layers.append(l)
            
        # Explicit iteration for up layers
        up_keys = [k for k in layers_config.keys() if "up_" in k]
        up_keys.sort() # up_0, up_1...
        
        for layer in up_keys:
            up_module = AttnUp if layers_config[layer]["attention"] else Up
            factor = 2 if layers_config[layer]["bilinear"] else 1
            l = up_module(in_channels=layers_config[layer]["in_channels"],
                          out_channels=layers_config[layer]["out_channels"] // factor,
                          bilinear=layers_config[layer]["bilinear"])

            last_up_out_channels = layers_config[layer]["out_channels"] // factor
            self.up_layers.append(l)

        #Because this is a unet we assert that there are at least as many down layers as up layers
        # assert(len(self.down_layers) == len(self.up_layers))

        self.outc = OutConv(last_up_out_channels, n_classes)

    def _compute_down_convolutions(self, x):
        down_outs = [x]
        for l in self.down_layers:
            down_outs.append(l(down_outs[-1]))
        return down_outs

    def _compute_up_convolutions(self, down_outs):
        intermediate = down_outs[-1]
        for l, d in zip(self.up_layers, down_outs[-2::-1]):
            intermediate = l(intermediate, d)
        return intermediate

    def forward(self, x):
        x_inc = self.inc(x)
        down_outs = self._compute_down_convolutions(x_inc)
        up_convolved_rep = self._compute_up_convolutions(down_outs)
        preds = self.outc(up_convolved_rep)
        return preds

# --- MaskedUNet ---

class MaskedUNet(UNet, Maskable):
    def __init__(self, n_channels, n_classes, hyperparameters, input_channel_mask_index=-1, output_channel_background_index=-1):
        UNet.__init__(self, n_channels, n_classes, hyperparameters)
        Maskable.__init__(self, n_classes, input_channel_mask_index, output_channel_background_index)

        self.softmax = nn.Softmax(dim=1)

        if output_channel_background_index >= 0 or input_channel_mask_index >= 0:
            if not self.mask_output:
                raise ValueError("Both output_channel_background_index and input_channel_mask_index must be specified when masking.")

    def forward(self, x, do_softmax=True):
        preds = super().forward(x)
        if do_softmax:
            preds = self.softmax(preds)
        if self.mask_output:
            mask = x[:, self.input_channel_mask_index]
            return self.mask(preds, mask)
        return preds

    def load_state_dict_from_ckpt(self, state_dict):
        """Custom loader to handle key prefixes from lightning and renaming"""
        new_state_dict = {}
        for key, v in state_dict.items():
            # Remove "model." prefix from Lightning
            k = key.replace("model.", "")
            
            # Key mapping for backward compatibility as per original code
            k = k.replace("down1", "down_layers.0")
            k = k.replace("down2", "down_layers.1")
            k = k.replace("down3", "down_layers.2")
            k = k.replace("down4", "down_layers.3")
            k = k.replace("down5", "down_layers.4")
            k = k.replace("up0", "up_layers.0")
            k = k.replace("up1", "up_layers.1")
            k = k.replace("up2", "up_layers.2")
            k = k.replace("up3", "up_layers.3")
            k = k.replace("up4", "up_layers.4")

            new_state_dict[k] = v

        super().load_state_dict(new_state_dict, strict=True)
