from torch.nn import functional as F
import torch
from torch import nn
from torch.cuda.amp import autocast as autocast
from timm.layers import DropPath
from utils import StochasticReverseComplement,StochasticShift,SwitchReverse

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None] * x + self.bias[:, None]
        return x
class Attention(nn.Module):
    def __init__(self, dim, ca_num_heads=2,):
        super().__init__()
        self.dim = dim
        self.ca_num_heads = ca_num_heads
        self.lnorm = nn.Sequential(
            LayerNorm(dim, eps=1e-6,),
            nn.GELU(),
        )
        self.norm = nn.Sequential(
            nn.BatchNorm1d(dim,momentum=0.9),
            nn.GELU(),
        )
        self.act = nn.GELU()
        self.proj = nn.Conv1d(dim,dim,1)

        self.split_groups = self.dim // ca_num_heads

        self.v = nn.Conv1d(dim, dim, 1)
        self.s = nn.Conv1d(dim, dim, 1)
        for i in range(self.ca_num_heads):
            local_conv = nn.Conv1d(dim // self.ca_num_heads, dim // self.ca_num_heads, kernel_size=(3 + i * 2),
                                   padding=(1 + i), stride=1, groups=dim // self.ca_num_heads)
            setattr(self, f"local_conv_{i + 1}", local_conv)
        self.proj0 = nn.Conv1d(dim, dim * 2, kernel_size=1, padding=0, stride=1,
                               groups=self.split_groups)
        self.proj1 = nn.Conv1d(dim * 2, dim, kernel_size=1, padding=0, stride=1)

    def forward(self, x):
        B,C,H = x.shape
        x = self.lnorm(x)
        v = self.v(x)
        s = self.s(x).reshape(B, self.ca_num_heads, C // self.ca_num_heads,H).permute(1,0,2,3)
        for i in range(self.ca_num_heads):
            local_conv = getattr(self, f"local_conv_{i + 1}")
            s_i = s[i]
            s_i = local_conv(s_i).reshape(B, self.split_groups, -1, H, )
            if i == 0:
                s_out = s_i
            else:
                s_out = torch.cat([s_out, s_i], 2)
        s_out = s_out.reshape(B, C, H,)
        s_out = self.proj1(self.act(self.proj0(s_out)))
        x = s_out * v
        x = self.norm(x)
        x = self.proj(x)
        return x
class Stem(nn.Module):
    def __init__(self,inc,outc,):
        super(Stem, self).__init__()
        self.act = nn.GELU()
        self.Conv1= nn.Conv1d(inc,outc, kernel_size=17, padding=17 // 2, bias=False)
        self.res_block = nn.Sequential(
            nn.BatchNorm1d(num_features=outc, momentum=0.9),
            nn.GELU(),
            nn.Conv1d(outc, outc, kernel_size=1, stride=1, bias=False, padding="same")
        )
        self.Pool = nn.Sequential(
            nn.MaxPool1d(kernel_size=3)
        )  # [768,896]

    def forward(self, x):
        x = self.act(x)
        y = self.Conv1(x)
        return self.Pool(y + self.res_block(y))
class Patch_embedding(nn.Module):
    def __init__(self,inc,outc):
        super(Patch_embedding, self).__init__()
        self.act_norm = nn.Sequential(
            nn.BatchNorm1d(num_features=inc, momentum=0.9),
            nn.GELU(),
            nn.Conv1d(in_channels=inc, out_channels=outc, kernel_size=5, stride=1, bias=False, padding="same")
        )

    def forward(self, x):
        '''
        :param x:[B,C,H]
        :return: [B,C,H]
        '''
        return self.act_norm(x)
class Mlp(nn.Module):
    def __init__(self, in_features,expand_ratio=1):
        super().__init__()
        expand_filters = in_features * expand_ratio
        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_features,momentum=0.9),
            nn.GELU(),
            nn.Conv1d(in_features, expand_filters, 1, 1, 0, bias=False),)

        self.proj = nn.Sequential(
            nn.BatchNorm1d(expand_filters, momentum=0.9),
            nn.GELU(),
            nn.Conv1d(expand_filters, expand_filters, 3, 1, 1, groups=expand_filters,bias=False))

    def forward(self, x):
        x = self.conv1(x)
        x = self.proj(x) + x
        return x

class block(nn.Module):
    def __init__(self,dim,drop_path,attn=False):
        super(block,self).__init__()
        self.dim = dim
        self.attn_if = attn
        self.LPU = nn.Sequential(nn.BatchNorm1d(num_features=dim, momentum=0.9),
                                 nn.GELU(),
                                 nn.Conv1d(dim, dim, 3, 1, 1, groups=dim))
        if self.attn_if:
            self.attn = Attention(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim)
    def forward(self, x):
        '''
        :param x:[B,C,H]
        :return: [B,C,H]
        '''
        x = self.LPU(x) + x
        if self.attn_if:
            x = x + self.drop_path(self.attn(x))
        else:
            x = x + self.drop_path(self.mlp(x))
        return x
class stage_block(nn.Module):
    def __init__(self, in_c, depth, dim,drop_path,attn=False):
        super(stage_block,self).__init__()

        self.dim = dim
        self.depth = depth
        self.embedding = Patch_embedding(inc=in_c,outc=dim)
        # build blocks
        self.blocks = nn.ModuleList([
            block(dim=dim,
                drop_path=drop_path[i],
                attn = attn)
            for i in range(depth)])
        self.pool = nn.MaxPool1d(kernel_size=2)
    def forward(self, x):
        '''
        :param x:[B,C,H]
        :return: [B,C,H]
        '''
        x = self.embedding(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.pool(x)
        return x

class model(nn.Module):
    def __init__(self,num_cell,):
        super(model, self).__init__()
        depths = [1, 1, 1, 1, 1, 1]
        channels = [768, 864, 968, 1084, 1220, 1368, 1536]
        self.num_stages = len(depths)

        dpr = [0,0,0,0,0,0,]
        self.StochasticReverseComplement = StochasticReverseComplement()
        self.StochasticShift = StochasticShift()

        self.stem = Stem(inc=4,outc=channels[0]) #[768,896]

        self.stages = nn.ModuleList()

        for i in range(self.num_stages):
            layer = stage_block(in_c=channels[i],dim=channels[i+1],depth=depths[i],
                                drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])], 
                                attn = True)
            self.stages.append(layer)

        self.final = nn.Sequential(
            nn.BatchNorm1d(num_features=1536, momentum=0.9, ),
            nn.GELU(),

            nn.Conv1d(in_channels=1536, out_channels=768, kernel_size=1, padding="same"),
            nn.BatchNorm1d(num_features=768, momentum=0.9, ),
            nn.GELU(),  # output: (768,14)
            nn.Flatten(start_dim=1, end_dim=-1),  # output: (batch_size, 1792)
            nn.Linear(in_features=10752, out_features=32),
            nn.BatchNorm1d(32, momentum=0.9),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Linear(32, num_cell, bias=True),
        )
        self.SwitchReverse = SwitchReverse()
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', )

    def forward(self, x):
        with autocast():
            x, bool = self.StochasticReverseComplement(x)
            x = self.StochasticShift(x)
            x = self.stem(x)
            for stage in self.stages:
                x = stage(x)
            x = self.final(x)
            x = self.SwitchReverse([x, bool])
        return x
# input_tensor = torch.randn(2,4,2688)
# model = model(num_cell=2034)
# output =model(input_tensor)
# print(output.size())
