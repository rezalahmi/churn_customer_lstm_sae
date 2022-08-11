import numpy as np
from lstm_sae import lstm_sae


# error rate
def error_rate(X_train, x, opts):
    # parameters
    fold = opts['fold']
    node_layer1 = opts['node_layer1']
    node_layer2 = opts['node_layer2']
    node_layer3 = opts['node_layer3']
    xt = fold['xt']
    xv = fold['xv']

    # Number of instances
    num_train = np.size(xt, 0)
    num_valid = np.size(xv, 0)
    # Define selected features
    X_train = xt[:, x == 1]
    X_valid = xv[:, x == 1]
    # Training
    model = lstm_sae(node_layer1, node_layer2, node_layer3)
    model.compile()
    model.fit(X_train)
    # Prediction
    X_prediction = model.predict(X_valid)
    acc = np.sum(X_valid == X_prediction) / num_valid
    error = 1 - acc

    return error


# Error rate & Feature size
def Fun(X_train, x, opts):
    # Parameters
    alpha = 0.99
    beta = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost = 1
    else:
        # Get error rate
        error = error_rate(X_train, x, opts)
        # Objective function
        cost = alpha * error + beta * (num_feat / max_feat)

    return cost
