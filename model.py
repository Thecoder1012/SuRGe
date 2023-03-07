import torch
from torch import nn

'''
Pixel Normalization
'''
class PixelNorm(nn.Module):
    def __init__(self):
        super(PixelNorm, self).__init__()
        self.epsilon = 1e-8

    def forward(self, x):
        return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.epsilon)

'''
Implementation of Convblock
-------------------------------------------
Conv + BatchNorm (True/False) + PReLU (True/False) --> Generator
Conv + PixelNorm + LReLU (True/False) - > Discriminator
'''
class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        use_pn = False,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.pn = PixelNorm() if use_pn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))

'''
Upsampleblock (U0,U1)
-----------------
Conv + NN + PReLU
'''

class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1)
        #self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.ub = nn.Upsample(scale_factor = scale_factor, mode='nearest')
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ub(self.conv(x)))

'''
Residual-Block (R0, R1)
	-> ConvBlock (conv + bn + activation) [Sub-Block 1]
	-> ConvBlock (conv + bn) [Sub-Block 2]
	Sub-Block 1 + Sub-Block 2
	
We use Ng residual blocks in each stage of Generator
and
Nd residual blocks in Discriminator
'''

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.block1 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.block2 = ConvBlock(
            in_channels,
            in_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            use_act=False,
        )

    def forward(self, x):
        out = self.block1(x)
        out = self.block2(out)
        return out + x

'''
Implementation of Generator
'''
class Generator(nn.Module):
    def __init__(self, alpha=0.4, beta = 0.1, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.initial_cont = ConvBlock(num_channels, num_channels*2, kernel_size=9, stride=1, padding=4, use_bn=False)
        
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.residuals_cont = nn.Sequential(*[ResidualBlock(num_channels*2) for _ in range(num_blocks)])
        
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.convblock_cont = ConvBlock(num_channels*2, num_channels*2, kernel_size=3, stride=1, padding=1, use_act=False) 
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2))
        self.upsamples_cont = nn.Sequential(UpsampleBlock(num_channels*2, 2))
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        self.final_cont = nn.Conv2d(num_channels*2, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        initial = self.initial(x) 			# Low Level Features C0
        x = self.residuals(initial) 			# High level Features R0
        x = self.convblock(x) + initial 		# F0 = I0 + C0
        x = self.upsamples(x) 				# Upsampling Block - U0
        initial_cont = self.initial_cont(x) 		# C1
        element_init = self.upsamples(initial) 	# U0 - - > C0
        element_init = self.initial_cont(element_init) # C1 - - > U0 -- > C0
        x = self.residuals_cont(initial_cont)		# R1
        #WFMM
        x = (1 - self.alpha)* self.convblock_cont(x)+ self.beta * initial_cont + (self.alpha - self.beta) * element_init				# F1
        x = self.upsamples_cont(x)			# U1
        return torch.tanh(self.final_cont(x))

'''
Implementation of Discriminator
(Nd Residual Blocks)
 -> Conv block [conv + PixelNorm + LReLU]
'''
class Discriminator(nn.Module):
    def __init__(self, in_channels=3, num_channels = 64, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        self.init_conv = ConvBlock(in_channels, num_channels, stride=1, padding=1, use_bn=False, use_pn=False,discriminator=True)
        self.layer1_1 =  ConvBlock(num_channels, num_channels, stride=2, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer1_2 =  ConvBlock(num_channels, num_channels, stride=1, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer2_1 =  ConvBlock(num_channels, num_channels*2, stride=1, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer2_2 =  ConvBlock(num_channels*2, num_channels*2, stride=1, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer3_1 =  ConvBlock(num_channels*2, num_channels*4, stride=2, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer3_2 =  ConvBlock(num_channels*4, num_channels*4, stride=1, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer4_1 =  ConvBlock(num_channels*4, num_channels*8, stride=2, padding=1, use_bn=False, use_pn=True,discriminator=True)
        self.layer4_2 =  ConvBlock(num_channels*8, num_channels*8, stride=1, padding=1, use_bn=False, use_pn=True,discriminator=True)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )
    
    '''
    Multiple Residual Blocks with Skip Connection Followed by Classifier Head (H)
    '''
    def forward(self, x):
        init = self.init_conv(x)
        x1_1 = self.layer1_1(init)
        x1_2 = self.layer1_2(x1_1)
        x = x1_1 + x1_2
        x2_1 = self.layer2_1(x)
        x2_2 = self.layer2_2(x2_1)
        x = x2_1 + x2_2
        x3_1 = self.layer3_1(x)
        x3_2 = self.layer3_2(x3_1)
        x = x3_1 + x3_2
        x4_1 = self.layer4_1(x)
        x4_2 = self.layer4_2(x4_1)
        x = x4_1 + x4_2

        return self.classifier(x)  		#Classification Head - H

