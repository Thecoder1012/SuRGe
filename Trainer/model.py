import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        discriminator=False,
        use_act=True,
        use_bn=True,
        **kwargs,
    ):
        super().__init__()
        self.use_act = use_act
        self.cnn = nn.Conv2d(in_channels, out_channels, **kwargs, bias=not use_bn)
        self.bn = nn.BatchNorm2d(out_channels) if use_bn else nn.Identity()
        self.act = (
            nn.LeakyReLU(0.2, inplace=True)
            if discriminator
            else nn.PReLU(num_parameters=out_channels)
        )

    def forward(self, x):
        return self.act(self.bn(self.cnn(x))) if self.use_act else self.bn(self.cnn(x))


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale_factor):
        super().__init__()
        self.conv = nn.Conv2d(in_c, in_c, 3, 1, 1)
        #self.ps = nn.PixelShuffle(scale_factor)  # in_c * 4, H, W --> in_c, H*2, W*2
        self.ps = nn.Upsample(scale_factor = scale_factor, mode='nearest')
        self.act = nn.PReLU(num_parameters=in_c)

    def forward(self, x):
        return self.act(self.ps(self.conv(x)))


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


class Generator(nn.Module):
    def __init__(self, in_channels=3, num_channels=64, num_blocks=8):
        super().__init__()
        # self.alpha = nn.Parameter(torch.tensor(0.5))
        # self.beta = nn.Parameter(torch.tensor(0.5))
        self.hyp = nn.Parameter(torch.tensor([0.33, 0.33, 0.34]))
        self.hyp_p = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.initial = ConvBlock(in_channels, num_channels, kernel_size=9, stride=1, padding=4, use_bn=False)
        self.initial_cont = ConvBlock(64, 128, kernel_size=9, stride=1, padding=4, use_bn=False)
        
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_blocks)])
        self.residuals_cont = nn.Sequential(*[ResidualBlock(128) for _ in range(num_blocks)])
        
        self.convblock = ConvBlock(num_channels, num_channels, kernel_size=3, stride=1, padding=1, use_act=False)
        self.convblock_cont = ConvBlock(128, 128, kernel_size=3, stride=1, padding=1, use_act=False) 
        self.upsamples = nn.Sequential(UpsampleBlock(num_channels, 2))
        self.upsamples_cont = nn.Sequential(UpsampleBlock(128, 2))
        
        self.final = nn.Conv2d(num_channels, in_channels, kernel_size=9, stride=1, padding=4)
        self.final_cont = nn.Conv2d(128, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        #print(x.shape)
        # alpha = self.alpha
        # beta = self.beta
        alpha, beta, gamma = F.softmax(self.hyp, dim = 0)
        meu, neu = F.softmax(self.hyp_p, dim = 0)
        
        initial = self.initial(x)
        #print(initial.shape)
        x = self.residuals(initial)
        #print(x.shape)
        x = meu * self.convblock(x) + neu * initial
        #print(x.shape)
        x = self.upsamples(x)
        #print("Upsample:",x.shape)
        initial_cont = self.initial_cont(x)
        #print(initial_cont.shape)
        element_init = self.upsamples(initial)
        #print(element_init.shape)
        element_init = self.initial_cont(element_init)
        #print(element_init.shape)
        x = self.residuals_cont(initial_cont)
        #print(x.shape)
        x = alpha * self.convblock_cont(x)+ beta * initial_cont + gamma * element_init
        x = self.upsamples_cont(x)
        #print(x.shape)
        return torch.tanh(self.final_cont(x))


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 64, 128, 128, 256, 256, 512, 512]):
        super().__init__()
        blocks = []
        for idx, feature in enumerate(features):
            blocks.append(
                ConvBlock(
                    in_channels,
                    feature,
                    kernel_size=3,
                    stride=1 + idx % 2,
                    padding=1,
                    discriminator=True,
                    use_act=True,
                    use_bn=False if idx == 0 else True,
                )
            )
            in_channels = feature

        self.blocks = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(512*6*6, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        x = self.blocks(x)
        return self.classifier(x)

def test():
    low_resolution = 64  # 96x96 -> 24x24
    with torch.cuda.amp.autocast():
        x = torch.randn((5, 3, low_resolution, low_resolution))
        gen = Generator()
        gen_out = gen(x)
        disc = Discriminator()
        disc_out = disc(gen_out)

        print(gen_out.shape)
        print(disc_out.shape)


if __name__ == "__main__":
    test()
