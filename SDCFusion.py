import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

def RGB2YCrCb(rgb_image):
    R = rgb_image[:, 0:1]
    G = rgb_image[:, 1:2]
    B = rgb_image[:, 2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = Y.clamp(0.0,1.0)
    Cr = Cr.clamp(0.0,1.0).detach()
    Cb = Cb.clamp(0.0,1.0).detach()
    return Y, Cb, Cr

def YCbCr2RGB(Y, Cb, Cr):
    ycrcb = torch.cat([Y, Cr, Cb], dim=1)
    B, C, W, H = ycrcb.shape
    im_flat = ycrcb.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3)
    out = out.clamp(0,1.0)
    return out

class SDCFusion(nn.Module):
    def __init__(self, n_classes):
        super(SDCFusion, self).__init__()
        self.num_layers = 30

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            )
        self.down_conv1 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            )
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            )
        self.down_conv3 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            )
        
        self.conv_t1 = nn.Sequential(
            nn.Conv2d(1, self.num_layers, kernel_size=3, stride=1, padding=1),
            #nn.BatchNorm2d(self.num_layers),
            nn.PReLU(),
            )
        self.down_conv_t1 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            )
        self.down_conv_t2 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            )
        self.down_conv_t3 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=2, padding=1),
            nn.PReLU(),
            )
        
        self.GEFM1 = CDIM(self.num_layers, self.num_layers)
        self.GEFM2 = CDIM(self.num_layers, self.num_layers)
        self.GEFM3 = CDIM(self.num_layers, self.num_layers)
        self.GEFM4 = CDIM(self.num_layers, self.num_layers)
        
        self.up_conv3 = nn.Sequential(
            nn.Conv2d(self.num_layers * 2, self.num_layers, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor = 2, mode='bicubic', align_corners=True)
        )
        self.up_conv2 = nn.Sequential(
            nn.Conv2d(self.num_layers * 3, self.num_layers, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor = 2, mode='bicubic', align_corners=True)
        )
        self.up_conv1 = nn.Sequential(
            nn.Conv2d(self.num_layers * 3, self.num_layers, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.Upsample(scale_factor = 2, mode='bicubic', align_corners=True)
        )
        
        self.fusion = nn.Sequential(
            nn.Conv2d(self.num_layers * 3, self.num_layers * 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.num_layers * 2, self.num_layers, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.num_layers, self.num_layers // 2, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
            nn.Conv2d(self.num_layers // 2, self.num_layers // 2, kernel_size=3, stride=1, padding=1),
            nn.PReLU(),
            nn.Conv2d(self.num_layers // 2, 1, kernel_size=1, stride=1, padding=0),
            nn.Tanh(),
            )
        
        self.up_conv_sm3 = nn.Sequential(
            nn.Conv2d(self.num_layers, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_layers),
            nn.Upsample(scale_factor = 2, mode='bicubic', align_corners=True)
        )
        self.up_conv_sm2 = nn.Sequential(
            nn.Conv2d(self.num_layers * 2, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_layers),
            nn.Upsample(scale_factor = 2, mode='bicubic', align_corners=True)
        )
        self.up_conv_sm1 = nn.Sequential(
            nn.Conv2d(self.num_layers * 2, self.num_layers, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.num_layers),
            nn.Upsample(scale_factor = 2, mode='bicubic', align_corners=True)
        )
        
        self.seg_decoder = SegD(2 * self.num_layers, self.num_layers)      
        self.classfier = SegHead(feature=self.num_layers, n_classes=n_classes) 
        
    def encoder(self, thermal, rgb):
        
        rgb1 = self.conv1(rgb)
        rgb2 = self.down_conv1(rgb1)
        rgb3 = self.down_conv2(rgb2)
        rgb4 = self.down_conv3(rgb3)
        
        thermal1 = self.conv_t1(thermal)
        thermal2 = self.down_conv_t1(thermal1)
        thermal3 = self.down_conv_t2(thermal2)
        thermal4 = self.down_conv_t3(thermal3)
        
        return rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4
        
    def cross_modal_fusion(self, rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4):
    
        sem1 = self.GEFM1(rgb1, thermal1)
        sem2 = self.GEFM2(rgb2, thermal2)
        sem3 = self.GEFM3(rgb3, thermal3)
        sem4 = self.GEFM4(rgb4, thermal4)
        
        return sem1, sem2, sem3, sem4
    
    def decoder_fusion(self, sem1, sem2, sem3, sem4, rgb1, rgb2, rgb3, rgb4):
    
        fuse_de3 = self.up_conv3(torch.cat((rgb4, sem4), 1))
        fuse_de2 = self.up_conv2(torch.cat((fuse_de3, rgb3, sem3), 1))
        fuse_de1 = self.up_conv1(torch.cat((fuse_de2, rgb2, sem2), 1))
        fused_img = self.fusion(torch.cat((fuse_de1, rgb1, sem1), 1))  
        
        return fused_img
    
    def segmentation(self, sem1, sem2, sem3, sem4):
    
        sem_de3 = self.up_conv_sm3(sem4)
        sem_de2 = self.up_conv_sm2(torch.cat((sem_de3, sem3), 1))
        sem_de1 = self.up_conv_sm1(torch.cat((sem_de2, sem2), 1)) 
        seg_f = torch.cat((sem_de1, sem1), 1)
        seg_f = self.seg_decoder(seg_f)
        semantic_out = self.classfier(seg_f)
        
        return semantic_out
        
    def forward(self, rgb, depth):
        
        rgb = rgb
        thermal = depth[:, :1, ...]

        #####################Shared encoder#####################

        rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4 = self.encoder(thermal, rgb)
        sem1, sem2, sem3, sem4 = self.cross_modal_fusion(rgb1, rgb2, rgb3, rgb4, thermal1, thermal2, thermal3, thermal4)

        #####################Fusion decoder#####################
        
        fused_img = self.decoder_fusion(sem1, sem2, sem3, sem4, rgb1, rgb2, rgb3, rgb4)
        
        #####################Segmentation decoder#####################
        
        semantic_out = self.segmentation(sem1, sem2, sem3, sem4)
        
        ##################################Validation####################################
        
        Y_vi, Cb_vi, Cr_vi = RGB2YCrCb(rgb)
        fused_img_color = YCbCr2RGB(fused_img, Cb_vi, Cr_vi)
        
        fu_rgb1, fu_rgb2, fu_rgb3, fu_rgb4, fu_thermal1, fu_thermal2, fu_thermal3, fu_thermal4 = self.encoder(thermal, fused_img_color)
        fu_sem1, fu_sem2, fu_sem3, fu_sem4 = self.cross_modal_fusion(fu_rgb1, fu_rgb2, fu_rgb3, fu_rgb4, fu_thermal1, fu_thermal2, fu_thermal3, fu_thermal4)
        fu_semantic_out = self.segmentation(fu_sem1, fu_sem2, fu_sem3, fu_sem4) 

        return fused_img, semantic_out, fu_semantic_out, fused_img_color

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, dilation=1, groups=1, bn=True, relu=True):
        padding = ((kernel_size - 1) * dilation + 1) // 2
        super(ConvBNReLU, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, dilation=dilation, groups=groups,
                              bias=False if bn else True)
        self.bn = bn
        if bn:
            self.bnop = nn.BatchNorm2d(out_planes)
        self.relu = relu
        if relu:
            self.reluop = nn.ReLU6(inplace=True)

    def forward(self, x):
        
        x = self.conv(x)
        if self.bn:
            x = self.bnop(x)
        if self.relu:
            x = self.reluop(x)
            
        return x
    
class BBasicConv2d(nn.Module):
    def __init__(
        self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False,
    ):
        super(BBasicConv2d, self).__init__()

        self.basicconv = nn.Sequential(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=bias,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        
        return self.basicconv(x)
    
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        
        x_source = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x) * x_source + x_source

    
class CDIM(nn.Module):
    def __init__(self, in_C, out_C):
        super(CDIM, self).__init__()
        
        self.RGB_K= BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_V = BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.RGB_Q = BBasicConv2d(in_C, out_C, 3, 1, 1)
        
        self.INF_K= BBasicConv2d(out_C, out_C, 3, 1, 1)
        self.INF_V = BBasicConv2d(out_C, out_C, 3, 1, 1)   
        self.INF_Q = BBasicConv2d(out_C, out_C, 3, 1, 1)
        
        self.REDUCE = BBasicConv2d(out_C * 4, out_C, 3, 1, 1)
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.gamma4 = nn.Parameter(torch.zeros(1))
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.RGB_SPA_ATT = SpatialAttention()
        self.INF_SPA_ATT = SpatialAttention() 
        
        self.SEC_REDUCE = BBasicConv2d(out_C * 3, out_C, 3, 1, 1)
        
    def forward(self, x, y):
        
        m_batchsize, c, h, w = x.shape
        
        x_re = F.interpolate(x, size=(32, 32), scale_factor=None, mode='bicubic', align_corners=None)
        y_re = F.interpolate(y, size=(32, 32), scale_factor=None, mode='bicubic', align_corners=None)
 
        RGB_K = self.RGB_K(x_re)
        RGB_V = self.RGB_V(x_re)
        RGB_Q = self.RGB_Q(x_re)   

        INF_K = self.INF_K(y_re)
        INF_V = self.INF_V(y_re)
        INF_Q = self.INF_Q(y_re)  
        
        DUAL_V = RGB_V + INF_V
        
        RGB_V = RGB_V.view(m_batchsize, -1, 32*32)
        RGB_K = RGB_K.view(m_batchsize, -1, 32*32).permute(0, 2, 1)
        RGB_Q = RGB_Q.view(m_batchsize, -1, 32*32)
        
        INF_V = INF_V.view(m_batchsize, -1, 32*32)
        INF_K = INF_K.view(m_batchsize, -1, 32*32).permute(0, 2, 1)
        INF_Q = INF_Q.view(m_batchsize, -1, 32*32)
        
        DUAL_V = DUAL_V.view(m_batchsize, -1, 32*32)
        
        # rgb_att
        RGB_mask = torch.bmm(RGB_K, RGB_Q)
        RGB_mask = self.softmax(RGB_mask)
        RGB_refine = torch.bmm(DUAL_V, RGB_mask.permute(0, 2, 1))
        RGB_refine = RGB_refine.view(m_batchsize, -1, 32, 32)
        RGB_refine = self.gamma1 * RGB_refine
        RGB_refine_view = RGB_refine
        RGB_refine = F.interpolate(RGB_refine, size=(h, w), scale_factor=None, mode='bicubic', align_corners=None) + x
        
        # inf_att
        INF_mask = torch.bmm(INF_K, INF_Q)
        INF_mask = self.softmax(INF_mask)
        INF_refine = torch.bmm(DUAL_V, INF_mask.permute(0, 2, 1))
        INF_refine = INF_refine.view(m_batchsize, -1, 32, 32)
        INF_refine = self.gamma2 * INF_refine
        INF_refine_view = INF_refine
        INF_refine = F.interpolate(INF_refine, size=(h, w), scale_factor=None, mode='bicubic', align_corners=None) + y
        
        #rgb_inf_att
        RGB_INF_mask = torch.bmm(RGB_K, INF_Q)
        RGB_INF_mask = self.softmax(RGB_INF_mask)
        RGB_INF_refine = torch.bmm(RGB_V, RGB_INF_mask.permute(0, 2, 1))
        RGB_INF_refine = RGB_INF_refine.view(m_batchsize, -1, 32, 32)
        RGB_INF_refine = self.gamma3 * RGB_INF_refine
        RGB_INF_refine_view = RGB_INF_refine
        RGB_INF_refine = F.interpolate(RGB_INF_refine, size=(h, w), scale_factor=None, mode='bicubic', align_corners=None) + y
        
        #inf_rgb_att
        INF_RGB_mask = torch.bmm(INF_K, RGB_Q)
        INF_RGB_mask = self.softmax(INF_RGB_mask)
        INF_RGB_refine = torch.bmm(INF_V, INF_RGB_mask.permute(0, 2, 1))
        INF_RGB_refine = INF_RGB_refine.view(m_batchsize, -1, 32, 32)
        INF_RGB_refine = self.gamma4 * INF_RGB_refine
        INF_RGB_refine_view = INF_RGB_refine
        INF_RGB_refine = F.interpolate(INF_RGB_refine, size=(h, w), scale_factor=None, mode='bicubic', align_corners=None) + x
        
        GLOBAL_ATT = self.REDUCE(torch.cat((RGB_refine, INF_refine, RGB_INF_refine, INF_RGB_refine), 1))
        RGB_SPA_ATT = self.RGB_SPA_ATT(x)
        INF_SPA_ATT = self.INF_SPA_ATT(y)        
        
        out = self.SEC_REDUCE(torch.cat([GLOBAL_ATT, INF_SPA_ATT, RGB_SPA_ATT], dim=1))
        
        return out

class SegD(nn.Module):
    def __init__(self, in_channel=64, out_channel=64):
        super(SegD, self).__init__()
        
        self.block1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            BasicConv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        
        x1 = self.block1(x)
        x2 = self.block2(x1)
        out = self.block3(x2)
        
        return out
    
class SegHead(nn.Module):
    
    def __init__(self, feature=64, n_classes=9):
        super(SegHead, self).__init__()

        self.semantic_conv1 = ConvBNReLU(feature, feature, kernel_size=1)
        self.semantic_conv2 = nn.Conv2d(feature, n_classes, kernel_size=3, padding=1)

    def forward(self, feat):

        feat_sematic = self.semantic_conv1(feat)
        semantic_out = self.semantic_conv2(feat_sematic)

        return semantic_out
    

    
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        
        x = self.conv(x)
        x = self.bn(x)
        
        return x
    
