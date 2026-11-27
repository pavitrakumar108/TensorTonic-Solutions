import numpy as np

def adam_step(
    param: list,
    grad: list,
    m: list,
    v: list,
    t: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (param_new, m_new, v_new) as NumPy arrays.
    """
    # Write code here
    parameter = np.array(param, dtype = float);
    gradient = np.array(grad, dtype = float);
    m = np.array(m, dtype = float);
    v = np.array(v, dtype = float);
    newM = beta1* m + (1 - beta1) * gradient;
    newV = beta2 * v + (1 - beta2) * (gradient ** 2);
    mCorrection = newM / (1 - beta1 ** t);
    vCorrection = newV / (1 - beta2 ** t);
    parameterNew = parameter - (lr * mCorrection)/ (np.sqrt(vCorrection) + eps);
    return parameterNew, newM, newV
    
    pass