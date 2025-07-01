import torch
from utils import my_print


# Задание 1
my_print("Задание 1", "-" * 100)
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)
z = torch.tensor(4.0, requires_grad=True)

f = x**2 + y**2 + z**2 + 2 * x * y * z  # Вычисляем функцию
f.backward()  # Вычисляем градиенты

print(f"Градиент по x: {x.grad}")  # Аналитически: 2x + 2yz = 4 + 24 = 28
print(f"Градиент по y: {y.grad}")  # 2y + 2xz = 6 + 16 = 22
print(f"Градиент по z: {z.grad}")  # 2z + 2xy = 8 + 12 = 20


# Задание 2
my_print("Задание 2", "-" * 50)

# Данные
x_data = torch.tensor([1.0, 2.0, 3.0])
y_true = torch.tensor([2.0, 4.0, 6.0])

# Веса
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Forward
y_pred = w * x_data + b

# Вычисление ошибки
mse = torch.mean((y_pred - y_true) ** 2)

# Вычисление градиентов
mse.backward()

print(f"Градиент по w: {w.grad}")
print(f"Градиент по b: {b.grad}")


# Задание 3
my_print("Задание 3", "-" * 50)
x = torch.tensor(1.5, requires_grad=True)
f = torch.sin(x**2 + 1)  # Функция
f.backward(retain_graph=True)  # Градиент

print("Градиент f(x) = sin(x^2 + 1): ", x.grad.item())
print("Проверка через autograd.grad: ", torch.autograd.grad(f, x, retain_graph=True)[0].item())
