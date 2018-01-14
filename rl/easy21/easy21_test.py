from easy21 import *


def test_algorithms_do_not_smoke():
    run_monte_carlo(30)
    run_lfa(10, 0.5, 0.05, 0.1)
    run_sarsa(1000, 0.5)
    run_q_learning(1000, 0.5)


def test_diffs_and_mse_work():
    big = run_monte_carlo(30_000)
    small = run_monte_carlo(30)
    assert big.get_max_diff(small) > 0
    assert big.get_mean_squared_err(small) > 0


def test_callbacks_work():
    def callback(obj):
        nonlocal times_called
        assert isinstance(obj, ExpectedRewardMatrix)
        times_called += 1

    times_called = 0
    run_monte_carlo(3, callback)
    assert times_called == 3

    times_called = 0
    run_sarsa(7, 0.5, callback)
    assert times_called == 7

    times_called = 0
    run_lfa(5, 0.5, 0.05, 0.1, callback)
    assert times_called == 5


def test_alg_names_work():
    assert run_monte_carlo.alg_name == "Monte Carlo"


def test_describe_params_works():
    assert describe_params({'lambda_val': 0.5, 'boop': 1}) == "boop=1, λ=0.5"
