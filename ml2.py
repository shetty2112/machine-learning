from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

# xs = np.array([1,2,3,4,5,6], dtype=float)
# ys = np.array([5,4,6,5,6,7], dtype=float)

def create_dataset(hm, variance, step, correlation=True):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    xs = [x for x in range(len(ys))]
    return np.array(xs, dtype=float), np.array(ys, dtype=float)

def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs) * mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)

    return m,b

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig)**2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return(1 - (squared_error_regr/squared_error_y_mean))


xs, ys = create_dataset(40, 40, 2, 'pos')

m,b = best_fit_slope_and_intercept(xs,ys)

regression_line = [(m*x)+b for x in xs]

predict_X = 8
predict_y = [m*predict_X + b]

r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_X, predict_y, s = 100, color = "g")
plt.plot(xs,regression_line)
plt.show()