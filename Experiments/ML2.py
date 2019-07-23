#   Linear Regression From Scratch

from statistics import mean
import numpy as np
import matplotlib.pyplot as plot
from matplotlib import style

x_vals = np.array([3,4,5,6,7,8], dtype = np.float64)
y_vals = np.array([5,3,6,5,8,6], dtype = np.float64)

style.use('dark_background')

def lin_regression(xs,ys):
    xm = mean(xs)
    ym = mean(ys)
    xym = mean(xs*ys)
    xm2 = xm*xm
    x2m = mean(xs*xs)
    slope = (xm*ym - xym)/(xm2 - x2m)
    y_int = ym - slope*xm
    return slope, y_int


def sq_err(yvals, ys):
    return sum((ys-yvals)**2)

def r_sq_val(yvals, ys):
    y_mean = [mean(yvals) for y in yvals]
    sq_err_estimate = sq_err(yvals, ys)
    sq_err_mean = sq_err(yvals, y_mean)
    return 1 - (sq_err_estimate/sq_err_mean)

m,b = lin_regression(x_vals,y_vals)
line = [ m*x + b for x in x_vals]
r_squared = r_sq_val(y_vals,line)
print(m,b,line,r_squared)

# plot.scatter(x_vals,y_vals)
# plot.plot(line)
# plot.show()







