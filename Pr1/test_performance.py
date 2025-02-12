import timeit
from nested_iterator import NestedIterator

nested_list = [1, [2, [3, 4], 5], 6, [[7, 8], 9], 10]


def run_iterator():
    iterator = NestedIterator(nested_list)
    for _ in iterator:
        pass


execution_time = timeit.timeit(run_iterator, number=1000)

iterator = NestedIterator(nested_list)
flattened_list = list(iterator)

print("Розгорнутий список:", flattened_list)
print(f"Час виконання: {execution_time:.6f} секунд на 1000 запусків")
