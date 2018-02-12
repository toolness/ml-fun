import stupid_example


def test_stupid_example():
    loss, acc = stupid_example.main(lambda msg: None)
    assert acc > 0.99
