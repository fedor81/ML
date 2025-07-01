import torch
import time

# Проверка доступности CUDA
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Создаем большие матрицы
matrices = [torch.rand(64, 1024, 1024), torch.rand(128, 512, 512), torch.rand(256, 256, 256)]


def measure_time(operation, *args):
    """Измеряет время выполнения операции на CPU и GPU"""
    count = 10
    avg_cpu = 0
    avg_gpu = 0

    for i in range(count):
        avg_cpu += measure_time_cpu(operation, *args)

    if torch.cuda.is_available():
        args = [arg.to(device) for arg in args]
        for i in range(count):
            avg_gpu += measure_time_gpu(operation, *args)

    cpu_time = avg_cpu / count
    gpu_time = avg_gpu / count
    speedup = cpu_time / gpu_time if gpu_time > 0 else 0

    return cpu_time, gpu_time, speedup


def measure_time_gpu(operation, *args):
    """Измеряет время выполнения операции на GPU"""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start.record()
    operation(*args)
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end)


def measure_time_cpu(operation, *args):
    "Измеряет время выполнения операции на CPU"
    start_time = time.time()
    operation(*args)
    return time.time() - start_time


def log_results(operation, matrix, cpu_time, gpu_time, speedup):
    """Выводит результаты в консоль"""
    print(f"{operation} \t | {matrix.shape} | {cpu_time:.6f} | {gpu_time:.6f} | {speedup:.2f}x")


# Собираем результаты
print(f"{'Операция':<24} | {'Размер':<28} | {'CPU (мс)':<7} | {'GPU (мс)':<7} | {'Ускорение':<7}")

for matrix in matrices:
    log_results(
        "Матричное умножение",
        matrix,
        *measure_time(
            torch.matmul, matrix, torch.rand(matrix.shape[0], matrix.shape[2], matrix.shape[1])
        ),
    )
    log_results("Поэлементное сложение", matrix, *measure_time(torch.add, matrix, matrix))
    log_results("Поэлементное умножение", matrix, *measure_time(torch.mul, matrix, matrix))
    log_results("Транспонирование", matrix, *measure_time(lambda x: x.transpose(1, 2), matrix))
    log_results("Сумма всех элементов", matrix, *measure_time(torch.sum, matrix))
