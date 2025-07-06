import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _L1Conv2dFunction(Function):
    @staticmethod
    def forward(
        ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, l1_lambda=0.01
    ):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.l1_lambda = l1_lambda

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        # Вычисляем градиенты как обычно
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(
                input.shape, weight, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(
                input, weight.shape, grad_output, ctx.stride, ctx.padding, ctx.dilation, ctx.groups
            )
            # Добавляем L1 регуляризацию
            grad_weight += ctx.l1_lambda * torch.sign(weight)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None, None, None, None


class L1Conv2d(nn.Module):
    """Сверточный слой с L1 регуляризацией"""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        l1_lambda=0.01,
    ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels // groups, kernel_size, kernel_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.l1_lambda = l1_lambda

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return _L1Conv2dFunction.apply(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            self.l1_lambda,
        )


class SpatialAttention(nn.Module):
    """Attention механизм для CNN"""

    def __init__(self, in_channels, reduction_ratio=8):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1),
            nn.Sigmoid(),
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, kernel_size=7, padding=3), nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_attention(x)

        # Spatial attention
        sa = self.spatial_attention(x)

        # Комбинируем
        attention = ca * sa
        return x * attention


class _SwishFunction(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input * torch.sigmoid(input)

    @staticmethod
    def backward(ctx, grad_output):
        input = ctx.saved_tensors[0]
        sigmoid = torch.sigmoid(input)
        return grad_output * (sigmoid * (1 + input * (1 - sigmoid)))


class Swish(nn.Module):
    """Функция активации Swish"""

    def forward(self, input):
        return _SwishFunction.apply(input)


class StochasticPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if stride is not None else kernel_size
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.padding = padding

    def forward(self, x):
        # Проверка размерности
        if x.dim() != 4:
            raise RuntimeError("Expected 4D (batch, channel, height, width) tensor")

        # Размеры
        b, c, h, w = x.size()
        kh, kw = self.kernel_size
        sh, sw = self.stride

        # Разбиваем на патчи
        patches = x.unfold(2, kh, sh).unfold(3, kw, sw)
        patches = patches.contiguous().view(b, c, -1, kh, kw)

        # Вычисляем вероятности
        eps = 1e-8
        probs = patches - patches.min(dim=3, keepdim=True)[0].min(dim=4, keepdim=True)[0] + eps
        probs = probs / (probs.sum(dim=(3, 4), keepdim=True) + eps)

        # Выбираем случайно по вероятностям
        dist = torch.distributions.Categorical(probs.view(b, c, -1, kh * kw))
        indices = dist.sample()

        # Собираем результат
        out_h = (h - kh) // sh + 1
        out_w = (w - kw) // sw + 1
        result = patches.view(b, c, -1, kh * kw).gather(3, indices.unsqueeze(-1)).squeeze(-1)
        return result.view(b, c, out_h, out_w)


class BottleneckBlock(nn.Module):
    """Bottleneck Residual блок (1x1 -> 3x3 -> 1x1)"""

    def __init__(self, in_channels, out_channels, stride=1, expansion=4):
        super().__init__()
        mid_channels = out_channels // expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class WideBlock(nn.Module):
    """Wide Residual блок (увеличенная ширина)"""

    def __init__(self, in_channels, out_channels, stride=1, widen_factor=2):
        super().__init__()
        mid_channels = out_channels * widen_factor
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)
