import numpy as np

def least_squares_normal(x, y):
    """
    Fits a straight line y = mx + c to the data.
    Returns the slope (m), intercept (c), and the standard error of the slope (m_error).
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    
    if n <= 2:
        raise ValueError("At least 3 data points are required to calculate the error.")
        
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate sums of squares
    S_xx = np.sum((x - x_mean)**2)
    S_xy = np.sum((x - x_mean) * (y - y_mean))
    
    # Calculate slope (m) and intercept (c)
    m = S_xy / S_xx
    c = y_mean - m * x_mean
    
    # Calculate error in slope
    y_pred = m * x + c
    rss = np.sum((y - y_pred)**2)          # Residual sum of squares
    s_yx = np.sqrt(rss / (n - 2))          # Standard error of estimate (n-2 degrees of freedom)
    m_error = s_yx / np.sqrt(S_xx)
    
    return m, c, m_error

def least_squares_zero_intercept(x, y):
    """
    Fits a straight line y = mx to the data (intercept forced to 0).
    Returns the slope (m) and the standard error of the slope (m_error).
    """
    x = np.array(x)
    y = np.array(y)
    n = len(x)
    
    if n <= 1:
        raise ValueError("At least 2 data points are required to calculate the error.")
    
    # Calculate sums of squares
    sum_xx = np.sum(x**2)
    sum_xy = np.sum(x * y)
    
    # Calculate slope (m)
    m = sum_xy / sum_xx
    
    # Calculate error in slope
    y_pred = m * x
    rss = np.sum((y - y_pred)**2)          # Residual sum of squares
    s_yx = np.sqrt(rss / (n - 1))          # Standard error of estimate (n-1 degrees of freedom)
    m_error = s_yx / np.sqrt(sum_xx)
    
    return m, m_error