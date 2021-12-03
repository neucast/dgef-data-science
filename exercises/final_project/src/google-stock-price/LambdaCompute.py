from numpy import arange
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV


# Compute best lambda value for Ridge regressor.
def ridgeLambdaCompute(alphas, X, y):
    print("Ridge lambda compute by sklearn.linear_model Ridge cross validation.")
    ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
    ridgecv.fit(X, y)
    print("Mean squared error: %3f" % ridgecv.best_score_)
    print("Best lambda value: %3f" % ridgecv.alpha_)

    print("Comparing computed lambda...")
    range = arange(0, 1, 0.1)
    regressor = Ridge(fit_intercept=True, normalize=True)
    verifyLambdaValue(regressor, range, X, y)

    return ridgecv.alpha_


# Compute best lambda value for Lasso regressor.
def lassoLambdaCompute(alphas, X, y):
    print("Lasso lambda compute by sklearn.linear_model Ridge cross validation.")
    lassocv = LassoCV(alphas=alphas, normalize=True)
    lassocv.fit(X, y)
    print("Best lambda value: %3f" % lassocv.alpha_)

    print("Comparing computed lambda...")
    range = arange(0, 1, 0.1)
    regressor = Lasso(fit_intercept=True, normalize=True)
    verifyLambdaValue(regressor, range, X, y)

    return lassocv.alpha_


# Scikit-learn offers a function for time-series validation, TimeSeriesSplit. The function splits training data into
# multiple segments. We use the first segment to train the model with a set of hyper-parameters, to test it with the second.
# Then we train the model with first two chunks and measure it with the third part of the data. In this way we
# do k-1 times of cross-validation.
def verifyLambdaValue(regressor, range, X, y):
    print("Compute and verify lambda value by Time series split with Grid-Search cross validation.")
    tscv = TimeSeriesSplit(n_splits=5)
    model = regressor
    # cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    grid = dict()
    grid["alpha"] = range
    search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=tscv, n_jobs=1)
    results = search.fit(X, y)
    print("Mean absolute error: %.3f" % results.best_score_)
    print("Best lambda value: %s" % results.best_params_)
