from markupsafe import functools


def array_equals(func):
    def inner(arr_1, arr_2):
        if len(arr_1) != len(arr_2):
            raise IndexError("Array lengths do not match")
        
        return func(arr_1, arr_2)

# def is_valid_arr(num):
#     def decorator(func):
#         @functools.wraps(func)
#         def wrapper():