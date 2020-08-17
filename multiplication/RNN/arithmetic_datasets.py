import random
import numpy as np

def gen_data_add(num_examples, range_start, range_end, integers_only=False):
    X = []
    y = []
    for ex in range(num_examples):
        num1 = int(random.random()*(range_end - range_start) + range_start)
        num2 = int(random.random()*(range_end - range_start) + range_start)
        
        if integers_only:
            num1 = int(num1)
            num2 = int(num2)

        X.append([num1,num2,43])        
        y.append(num1+num2)
    return np.array(X), np.array(y)

def gen_data_sub(num_examples, range_start, range_end, integers_only=False):
    X = []
    y = []
    for ex in range(num_examples):
        num1 = int(random.random()*(range_end - range_start) + range_start)
        num2 = int(random.random()*(range_end - range_start) + range_start)
        
        if integers_only:
            num1 = int(num1)
            num2 = int(num2)

        X.append([num1,num2,145])        
        y.append(num1-num2)
    return np.array(X), np.array(y)

def gen_data_mult(num_examples, range_start, range_end, integers_only=False):
    X = []
    y = []
    for ex in range(num_examples):
        num1 = int(random.random()*(range_end - range_start) + range_start)
        num2 = int(random.random()*(range_end - range_start) + range_start)
        
        if integers_only:
            num1 = int(num1)
            num2 = int(num2)

        X.append([num1,num2,242])        
        y.append(num1*num2)
    return np.array(X), np.array(y)

def gen_data_div(num_examples, range_start, range_end, integers_only=False):
    X = []
    y = []
    for ex in range(num_examples):
        num1 = int(random.random()*(range_end - range_start) + range_start)
        num2 = int(random.random()*(range_end - range_start) + range_start)
        
        if integers_only:
            num1 = int(num1)
            num2 = int(num2)

        X.append([num1,num2,347])        
        y.append(num1/num2)
    return np.array(X), np.array(y)

def gen_data(num_examples, range_start, range_end, integers_only=False):
    num_examples = int(num_examples/4)
    
    ax, ay = gen_data_add(num_examples, range_start, range_end,integers_only=integers_only)
    mx, my = gen_data_sub(num_examples, range_start, range_end,integers_only=integers_only)
    mux, muy = gen_data_mult(num_examples, range_start, range_end,integers_only=integers_only)
    dx, dy = gen_data_div(num_examples, range_start, range_end,integers_only=integers_only)
    
    X = np.concatenate((ax, mx, mux, dx))
    y = np.concatenate((ay, my, muy, dy))
    
    return X, y