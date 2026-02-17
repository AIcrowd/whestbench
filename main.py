from estimators import combined_estimator

from evaluate import score_estimator, ContestParams

print(score_estimator(
    combined_estimator,
    n_circuits=10,
    n_samples=10000,
    contest_params=ContestParams(
        width=100,
        max_depth=30,
        budgets=[10, 100, 1000, 10000],
        time_tolerance=0.1,
    )
))