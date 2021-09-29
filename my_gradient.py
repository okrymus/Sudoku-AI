import numpy as np

np.random.seed(789)

def rosenbrock(x, a=1, b=100):
    '''
    Compute and return the value of the Rosenbrock function at input x
    :param x: numpy vector of shape (2,)
    :param a: scalar for the variable a in the Rosenbrock function
    :param b: scalar for the variable b in the Rosenbrock function
    :return: scalar 
    '''
    #print(sum(b*(x[1:]-x[:-1]**2.0)**2.0 + (a-x[:-1])**2.0))
    print((a-x[0])**2.0 + b*(x[1]-x[0]**2.0)**2.0)
    return (a-x[0])**2.0 + b*(x[1]-x[0]**2.0)**2.0

def rosenbrock_grad(x, a=1, b=100):
    '''
    Compute the gradient of the Rosenbrock function at point x
    :param x: numpy vector of shape (2,)
    :param a: scalar for the variable a in the Rosenbrock function
    :param b: scalar for the variable b in the Rosenbrock function
    :return: ndarray vector of shape (2,)
    '''

    value1 = -4 * b * x[0] * (x[1] - x[0] ** 2) - 2 * (a - x[0])
    value2 = 2*b*(x[1]-x[0]**2)
    return np.array([value1,value2])

def rosenbrock_hessian(x, a=1, b=100):
    '''
    Compute the hessian of the Rosenbrock function at point x
    :param x: numpy vector of shape (2,)
    :param a: scalar for the variable a in the Rosenbrock function
    :param b: scalar for the variable b in the Rosenbrock function
    :return: ndarray vector of shape (2,2)
    '''
    dict = {(0,0): 2-4*b*(x[1]-3*x[0]**2), (0,1): -4*x[0]*b, (1,0): -4*x[0]*b, (1,1): 2*b }
    return np.array([[dict[(0,0)], dict[(0,1)]], [dict[(1,0)], dict[(1,1)]]])

def gradient_descent(fn, grad_fn, x0, lr, threshold=1e-10, max_steps=100000):
    '''
    compute the gradient descent of a function until the minimum threshold is obtained or max_steps is reached
    :param fn: function that takes vector of size x0 as input and outputs a scalar
    :param grad_fn: gradient of the fn function 
    :param x0: the initial starting location of x0
    :param lr: the step size 
    :param threshold: minimum threshold to check for convergence of the function output
    :param max_steps: maximum number of steps to take
    :return: tuple of (scalar: ending fn value, ndarray: final x, int: number of iterations)
    '''
    end_move = False
    step = 0
    x = x0

    while not end_move:
        prev_x = x
        x = x - lr * grad_fn(x)
        step+=1

        if abs(fn(prev_x)-fn(x)) < threshold:
            end_move = True
        if step == max_steps:
            end_move = True

    fx = fn(x)
    return fx, x, step


def newton_method(fn, grad_fn, hessian_fn, x0, lr, threshold=1e-10, max_steps=100000):
    '''
    find the parameters that minimize a function fn using Newton's Method. 
    To invert the hessian use the function np.linalg.
    :param fn: function that takes vector of size x0 as input and outputs a scalar
    :param grad_fn: gradient of the fn function 
    :param hessian_fn: 
    :param x0: the initial starting location of x0
    :param lr: the step size 
    :param threshold: minimum threshold to check for convergence of the function output
    :param max_steps: maximum number of steps to take
    :return: tuple of (scalar: ending fn value, ndarray: final x, int: number of iterations) 
    '''
    # x = x0
    # fx = fn(x)
    # step = 0

    end_move = False
    step = 0
    x = x0
    df = lambda x: 1 + x ** 2

    while not end_move:
        prev_x = x
        x = x - lr * np.dot(np.linalg.inv(hessian_fn(x)), grad_fn(x))
        step += 1
        if abs(fn(prev_x) - fn(x)) < threshold:
            end_move = True
        if step == max_steps:
            end_move = True

    fx = fn(x)
    return fx, x, step



if __name__ == '__main__':
    # these lambda functions define the Rosenbrock functions with fixed 'a' and 'b' values
    rosen_fn = lambda x: rosenbrock(x, a=1, b=100)
    rosen_grad_fn = lambda x: rosenbrock_grad(x, a=1, b=100)
    rosen_hess_fn = lambda x: rosenbrock_hessian(x, a=1, b=100)

    # This runs gradient descent on on the Rosenbrock function and prints out the result
    x0 = np.zeros(2)
    val, x, itrs = gradient_descent(rosen_fn, rosen_grad_fn, x0=x0, lr=0.001)
    print(val, itrs, x)

    # This runs Newton's method on the Rosebrock function
    x0 = np.zeros(2)
    val, x, itrs = newton_method(rosen_fn, rosen_grad_fn, rosen_hess_fn, x0=x0, lr=1.0)
    print(val, itrs, x)

    # run gradient descent from random points on [(-1, -1), (1, 1)] with different learning rates
    num_trials = 10
    lrs = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.0025]
    results = []
    for lr in lrs:
        lr_res = []
        print('Running trials of gradient descent with lr {0}'.format(lr))
        for trial in range(num_trials):
            x0 = np.random.random(2) * 2 -1
            val, x, itrs = gradient_descent(rosen_fn, rosen_grad_fn, x0=x0, lr=lr)
            lr_res.append([np.linalg.norm(x-np.ones_like(x)), itrs])
        lr_mn = np.array(lr_res).mean(axis=0)
        lr_std = np.array(lr_res).std(axis=0)
        results.append([lr, lr_mn[0], lr_std[0], lr_mn[1], lr_std[1]])

    print("\n")
    print("Learning rate | mean solution error +- std | mean iterations +- std")
    for i in range(len(results)):
        print("{0:.4f}\t{1:.5f}+-{2:.5f}\t{3:>#08.1f}+-{4:.2f}".format(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]))

    # run newton's method from random points on [(-1, -1), (1, 1)] with different learning rates
    num_trials = 10
    lrs = [0.0001, 0.0003, 0.0005, 0.0007, 0.001, 0.002, 0.0025, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    results = []
    for lr in lrs:
        lr_res = []
        print("Running trials of Newtown's Method with lr {0}".format(lr))
        for trial in range(num_trials):
            x0 = np.random.random(2) * 2 - 1
            val, x, itrs = newton_method(rosen_fn, rosen_grad_fn, rosen_hess_fn, x0=x0, lr=lr)
            lr_res.append([np.linalg.norm(x - np.ones_like(x)), itrs])
        lr_mn = np.array(lr_res).mean(axis=0)
        lr_std = np.array(lr_res).std(axis=0)
        results.append([lr, lr_mn[0], lr_std[0], lr_mn[1], lr_std[1]])

    print("\n")
    print("Learning rate | mean solution error +- std | mean iterations +- std")
    for i in range(len(results)):
        print("{0:.4f}\t{1:.5f}+-{2:.5f}\t{3:>#08.1f}+-{4:.2f}".format(results[i][0], results[i][1], results[i][2], results[i][3], results[i][4]))
