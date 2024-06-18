def test_get_function_name():
    def a():
        pass

    assert a.__name__ == "a"
