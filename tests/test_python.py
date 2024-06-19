import inspect


def test_get_function_name():
    def a():
        pass

    assert a.__name__ == "a"


def test_get_number_of_function_arguments():
    def f(a, b):
        pass

    assert len(inspect.signature(f).parameters) == 2
