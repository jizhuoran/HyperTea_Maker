import torch.nn as nn

class LSTMClassifier(nn.Module):

    def __init__(self):
        super(LSTMClassifier, self).__init__()

        self.conv1      = nn.Conv2d(3, 32, kernel_size=9, stride=1, padding = 4, bias=True)
        self.elu1       = nn.ELU()
        self.bn1        = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.conv2      = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding = 1, bias=True)
        self.elu2       = nn.ELU()
        self.bn2        = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.conv3      = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding = 1, bias=True)
        self.elu3       = nn.ELU()
        self.bn3        = nn.BatchNorm2d(128, eps=1e-05, momentum=0, affine=True, track_running_stats=False)

        self.res1_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res1_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res1_relu1 = nn.ReLU()
        self.res1_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res1_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res2_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res2_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res2_relu1 = nn.ReLU()
        self.res2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res2_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res3_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res3_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res3_relu1 = nn.ReLU()
        self.res3_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res3_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res4_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res4_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res4_relu1 = nn.ReLU()
        self.res4_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res4_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.res5_conv1 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res5_bn1   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)
        self.res5_relu1 = nn.ReLU()
        self.res5_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding = 1, bias=False)
        self.res5_bn2   = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.deconv1    = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=0, groups=1, bias=True)
        self.de_elu1    = nn.ELU()
        self.de_bn1     = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.deconv2    = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1, output_padding=0, groups=1, bias=True)
        self.de_elu2    = nn.ELU()
        self.de_bn2     = nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=False)

        self.deconv3    = nn.ConvTranspose2d(32, 3, 9, stride=1, padding=4, output_padding=0, groups=1, bias=True)
        self.de_tanh3   = nn.Tanh()

        # self.scale_weight = torch.tensor(scale_weight, dtype = torch.float)
        # self.scale_bias = torch.tensor(scale_bias, dtype = torch.float)

    def forward(self, data):

        temp = self.bn1(self.elu1(self.conv1(data)))
        temp = self.bn2(self.elu2(self.conv2(temp)))
        temp = self.bn3(self.elu3(self.conv3(temp)))
        temp += self.res1_bn2(self.res1_conv2(self.res1_relu1(self.res1_bn1(self.res1_conv1(temp)))))
        temp += self.res2_bn2(self.res2_conv2(self.res2_relu1(self.res2_bn1(self.res2_conv1(temp)))))
        temp += self.res3_bn2(self.res3_conv2(self.res3_relu1(self.res3_bn1(self.res3_conv1(temp)))))
        temp += self.res4_bn2(self.res4_conv2(self.res4_relu1(self.res4_bn1(self.res4_conv1(temp)))))
        temp += self.res5_bn2(self.res5_conv2(self.res5_relu1(self.res5_bn1(self.res5_conv1(temp)))))


        temp = self.de_bn1(self.de_elu1(self.deconv1(temp)))
        temp = self.de_bn2(self.de_elu2(self.deconv2(temp)))
        temp = self.de_tanh3(self.deconv3(temp))

        temp = (temp + 1) * 127.5

        return temp