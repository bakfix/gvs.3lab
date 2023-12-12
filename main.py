import unittest
import torch.nn
import torch.random
import numpy as np
import math
from torch.utils.cpp_extension import load

ext = load(
        name='linearlayer',
        sources=['linearlayer.cu'],
        extra_cuda_cflags=['-O4'],
        extra_cflags=['-O4'],
    )

class CudaLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return ext.my_forward_linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_input, grad_weight, grad_bias = ext.my_backward_linear(input, weight, bias, grad_output)

        return grad_input, grad_weight, grad_bias


class PythonLinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.save_for_backward(input, weight, bias)
        return input @ weight.t() + bias

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight
        if ctx.needs_input_grad[1]:
            grad_weight = torch.mm(grad_output.t(), input)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class LabTest(unittest.TestCase):
    @staticmethod
    def reset_parameters(w , b):
        torch.nn.init.kaiming_uniform_(w , a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(b, -bound, bound )

    def test_linear_layer(self):
        # Характеристики тензоров
        tensor_opt = {
            'device': 'cuda',
            'dtype': torch.float32,
            'requires_grad': True
        }

        # Переменные для своей реализации
        x = torch.ones((128, 9216), **tensor_opt)
        w1 = torch.empty((4096, 9216), **tensor_opt)
        b1 = torch.empty((4096, ), **tensor_opt)
        w2 = torch.empty((16, 4096), **tensor_opt)
        b2 = torch.empty((16, ), **tensor_opt)

        # Инициализация переменных
        LabTest.reset_parameters(w1, b1)
        LabTest.reset_parameters(w2, b2)

        # Реализация своей линейной свертки
        y = CudaLinearLayer.apply(x, w1, b1)
        z = CudaLinearLayer.apply(y, w2, b2)

        # Переменные для фреймворка
        x_ = x.detach().clone().requires_grad_()
        w1_ = w1.detach().clone().requires_grad_()
        b1_ = b1.detach().clone().requires_grad_()
        w2_ = w2.detach().clone().requires_grad_()
        b2_ = b2.detach().clone().requires_grad_()

        # Реализация линейной свертки фреймворка
        y_ = PythonLinearLayer.apply(x_, w1_, b1_)
        z_ = PythonLinearLayer.apply(y_, w2_, b2_)

        # Проверка результатов своей и фреймфорка
        self.assertTrue(torch.allclose(z, z_, atol=1e-4, rtol=1e-3))

        # Используем градиент
        z.backward(torch.ones_like(z))
        z_.backward(torch.ones_like(z_))

        # print(b1.grad, '\n', b1_.grad)

        # Проверка значений
        with torch.no_grad():
            self.assertTrue(
                torch.allclose(x.grad, x_.grad, atol=1e-4, rtol=1e-3)
            )
            self.assertTrue(
                torch.allclose(w1.grad, w1_.grad, atol=1e-4, rtol=1e-3)
            )
            self.assertTrue(
                torch.allclose(w2.grad, w2_.grad, atol=1e-4, rtol=1e-3)
            )
            self.assertTrue(
                torch.allclose(b1.grad, b1_.grad, atol=1e-4, rtol=1e-3)
            )
            self.assertTrue(
                torch.allclose(b2.grad, b2_.grad, atol=1e-4, rtol=1e-3)
            )


if __name__ == '__main__':
    unittest.main()