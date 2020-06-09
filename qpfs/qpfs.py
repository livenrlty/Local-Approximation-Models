import numpy as np
import scipy as sc
import pandas as pd

from cvxpy import *

def create_opt_problem(X, y, sim, rel, verbose=False):
    """
    % Function generates matrix Q and vector b
    % which represent feature similarities and feature relevances
    %
    % Input:
    % X - [m, n] - design matrix
    % y - [m, 1] - target vector
    % sim - string - indicator of the way to compute feature similarities,
    % support values are 'correl' and 'mi'
    % rel - string - indicator of the way to compute feature significance,
    % support values are 'correl', 'mi' and 'signif'
    %
    % Output:
    % Q - [n ,n] - matrix of features similarities
    % b - [n, 1] - vector of feature relevances
    """
    
    if verbose == True:
        print ("Constructing the problem...")
        print ('Similarity measure: {}, feature relevance measure: {}'.format(sim, rel))
    if len(y.shape) == 1:
        y_mat = y[:, np.newaxis]
    else:
        y_mat = y[:]
        
    df = pd.DataFrame(np.hstack([X, y_mat]))
    cor = np.array(df.corr())

    if sim == 'correl':
        Q = cor[:-1, :-1]
    else:
        print ("Wrong similarity measure")
        
    if rel == 'correl':
        b = cor[:-1, [-1]]
    elif rel == 'log-reg':
        lr = LogisticRegression()
        b = np.zeros((X.shape[1], 1))
        for i in range(X.shape[1]):
            lr.fit(X[:, [i]], y)
            y_pred = lr.predict(X[:, [i]])
            b[i] = np.corrcoef(y_pred, y)[0, 1]
        b = np.nan_to_num(b)
    else:
        print ("Wrong relevance measure")
        
    if verbose == True:
        print ("Problem has been constructed.")
    return Q, b


def solve_opt_problem(Q, b, verbose=False):
    """
     Function solves the quadratic optimization problem stated to select
     significance and noncollinear features

     Input:
     Q - [n, n] - matrix of features similarities
     b - [n, 1] - vector of feature relevances

     Output:
     x - [n, 1] - solution of the quadratic optimization problem
    """
    
    n = Q.shape[0]
    x = Variable(n)
    
    objective = Minimize(quad_form(x, Q) - b.T*x)
    constraints = [x >= 0, norm(x, 1) <= 1]
    prob = Problem(objective, constraints)
    
    if verbose == True:
        print ("Solving the QP problem...")
    
    prob.solve()
    
    if verbose == True:
        print ("The problem has been solved!")
        print ("Problem status:", prob.status)
        print

    return np.array(x.value).flatten()
    
def quadratic_programming(X, y, sim='correl', rel='correl', verbose=False):
    Q, b = create_opt_problem(X, y, sim, rel, verbose)
    print()
    qp_score = solve_opt_problem(Q, b, verbose)
    return qp_score