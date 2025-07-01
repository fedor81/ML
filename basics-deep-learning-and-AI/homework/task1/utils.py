def my_print(message: str, obj, add_new_line=True):
    """Функция для красивого вывода сообщений"""
    if add_new_line:
        print("")

    print(f"{message}:")
    print(obj)
