import torch
from utils import my_print


# Задание 1
my_print("Задание 1", "-" * 100)
random_tensor = torch.rand(3, 4)  # Тензор 3x4 со случайными числами от 0 до 1
zeros_tensor = torch.zeros(2, 3, 4)  # Тензор 2x3x4, заполненный нулями
ones_tensor = torch.ones(5, 5)  # Тензор 5x5, заполненный единицами
range_tensor = torch.arange(0, 16).reshape(4, 4)  # Тензор 4x4 с числами от 0 до 15

my_print("Тензор 3x4, заполненный случайными числами", random_tensor)
my_print("Тензор 2x3x4, заполненный нулями", zeros_tensor)
my_print("Тензор 5x5, заполненный единицами", ones_tensor)
my_print("Тензор 4x4 с числами от 0 до 15", range_tensor)


# Задание 2
my_print("Задание 2", "-" * 100)
a = torch.rand(3, 4)  # тензор A размером 3x4
b = torch.rand(4, 3)  # тензор B размером 4x3

my_print("Тензор A", a)
my_print("Тензор B", b)
my_print("Транспонированный A", a.T)
my_print("Матричное умножение A на B", a @ b)
my_print("Поэлементное умножение A на A транспонированное", a * b.T)
my_print("Сумма всех элементов A", a.sum())


# Задание 3
my_print("Задание 3", "-" * 100)
tensor = torch.rand(5, 5, 5)  # тензор размером 5x5x5

my_print("Тензор", tensor)
my_print("Первая строка", tensor[0, :, :])
my_print("Последний столбец", tensor[:, :, -1])
my_print("Подматрица 2x2 из центра тензора", tensor[2, 1:3, 1:3])
my_print("Элементы с четными индексами", tensor[::2, ::2, ::2])


# Задание 4
my_print("Задание 4", "-" * 100)
tensor = torch.arange(24)  # тензор размером 24 элемента

my_print("Тензор", tensor)
my_print("Преобразование в форму 2x12", tensor.reshape(2, 12))
my_print("Форма 3x8", tensor.reshape(3, 8))
my_print("Форма 4x6", tensor.reshape(4, 6))
my_print("Форма 2x3x4", tensor.reshape(2, 3, 4))
my_print("Форма 2x2x2x3", tensor.reshape(2, 2, 2, 3))
