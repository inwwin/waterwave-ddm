"""
A collection of models of ISF functions to test

x[0] refers to q_r (radial component of wavevector)
x[1] refers to q_theta (angular component of wavevector in radians between -pi/2 to pi/2)
x[2] refers to tau (lag time)
"""
import numpy as np


def _background_v1(x, *params):
    C1, _, v1, _, _, _, _ = params
    exponent = -v1 * x[0] * x[2]
    return -C1 * np.expm1(exponent)


def _background_v1_jac(x, *params):
    C1, _, v1, _, _, _, _ = params
    exponent = -v1 * x[0] * x[2]

    jac = np.zeros((x.shape[1], len(params)))
    # C1
    jac[:, 0] = -np.expm1(exponent)
    # v1
    jac[:, 2] = +C1 * x[0] * x[2] * np.exp(exponent)
    return jac


def _background_v2(x, *params):
    C1, _, tau1, _, _, _, _ = params
    exponent = -x[2] / tau1
    return -C1 * np.expm1(exponent)


def _background_v2_jac(x, *params):
    C1, _, tau1, _, _, _, _ = params
    exponent = -x[2] / tau1

    jac = np.zeros((x.shape[1], len(params)))
    # C1
    jac[:, 0] = -np.expm1(exponent)
    # tau1
    jac[:, 2] = +C1 / np.square(tau1) * np.exp(exponent)
    return jac


def _foreground_v1(x, *params):
    _, C2, _, v2, lambda_, tau2, phi = params

    e1e2 = np.exp(-x[0] / lambda_ - x[2] / tau2) * x[0] * x[2]
    C2e1e2 = C2 * e1e2
    phase1 = v2 * x[0] * x[2]
    cos1 = np.cos(phase1)
    phase2 = x[1] + phi
    cos2 = np.cos(phase2)

    return C2e1e2 * cos1 * cos2


def _foreground_v1_jac(x, *params):
    _, C2, _, v2, lambda_, tau2, phi = params

    e1e2 = np.exp(-x[0] / lambda_ - x[2] / tau2) * x[0] * x[2]
    C2e1e2 = C2 * e1e2
    phase1 = v2 * x[0] * x[2]
    cos1 = np.cos(phase1)
    sin1 = np.sin(phase1)
    phase2 = x[1] + phi
    cos2 = np.cos(phase2)
    sin2 = np.sin(phase2)

    jac = np.zeros((x.shape[1], len(params)))
    # C2
    jac[:, 1] =   e1e2 * cos1 * cos2
    # v2
    jac[:, 3] = C2e1e2 * sin1 * cos2 * (-x[0] * x[2])
    # lambda_
    jac[:, 4] = C2e1e2 * cos1 * cos2 * (+x[0] / np.square(lambda_))
    # tau2
    jac[:, 5] = C2e1e2 * cos1 * cos2 * (+x[2] / np.square(tau2))
    # phi
    jac[:, 6] = C2e1e2 * cos1 * -sin2
    return jac


def _foreground_v2(x, *params):
    _, C2, _, v2, lambda_, tau2, phi = params

    e1e2 = np.exp(-x[0] / lambda_ - x[2] / tau2)
    C2e1e2 = C2 * e1e2
    phase1 = v2 * x[0] * x[2]
    sin1 = np.sin(phase1)
    phase2 = x[1] + phi
    cos2 = np.cos(phase2)

    return C2e1e2 * sin1 * cos2


def _foreground_v2_jac(x, *params):
    _, C2, _, v2, lambda_, tau2, phi = params

    e1e2 = np.exp(-x[0] / lambda_ - x[2] / tau2)
    C2e1e2 = C2 * e1e2
    phase1 = v2 * x[0] * x[2]
    cos1 = np.cos(phase1)
    sin1 = np.sin(phase1)
    phase2 = x[1] + phi
    cos2 = np.cos(phase2)
    sin2 = np.sin(phase2)

    jac = np.zeros((x.shape[1], len(params)))
    # C2
    jac[:, 1] =   e1e2 * sin1 * cos2
    # v2
    jac[:, 3] = C2e1e2 * cos1 * cos2 * (+x[0] * x[2])
    # lambda_
    jac[:, 4] = C2e1e2 * sin1 * cos2 * (+x[0] / np.square(lambda_))
    # tau2
    jac[:, 5] = C2e1e2 * sin1 * cos2 * (+x[2] / np.square(tau2))
    # phi
    jac[:, 6] = C2e1e2 * sin1 * -sin2
    return jac


def model1(x, *params):
    """
    The model defined as I_1 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, v1, v2, lambda_, tau2, phi = params
    """
    return _background_v1(x, *params) + _foreground_v1(x, *params)


def model1_jac(x, *params):
    """
    The Jacobian of the model defined as I_1 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, v1, v2, lambda_, tau2, phi = params
    """
    return _background_v1_jac(x, *params) + _foreground_v1_jac(x, *params)


def model2(x, *params):
    """
    The model defined as I_2 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, tau1, v2, lambda_, tau2, phi = params
    """
    return _background_v2(x, *params) + _foreground_v1(x, *params)


def model2_jac(x, *params):
    """
    The Jacobian of the model defined as I_2 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, tau1, v2, lambda_, tau2, phi = params
    """
    return _background_v2_jac(x, *params) + _foreground_v1_jac(x, *params)


def model3(x, *params):
    """
    The model defined as I_3 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, v1, v2, lambda_, tau2, phi = params
    """
    return _background_v1(x, *params) + _foreground_v2(x, *params)


def model3_jac(x, *params):
    """
    The Jacobian of the model defined as I_3 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, v1, v2, lambda_, tau2, phi = params
    """
    return _background_v1_jac(x, *params) + _foreground_v2_jac(x, *params)


def model4(x, *params):
    """
    The model defined as I_4 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, tau1, v2, lambda_, tau2, phi = params
    """
    return _background_v2(x, *params) + _foreground_v2(x, *params)


def model4_jac(x, *params):
    """
    The Jacobian of the model defined as I_4 as given in
    [this image](https://raw.githubusercontent.com/inwwin/waterwave-ddm/master/assets/modelv1.png)

    x[0] refers to q_r
    x[1] refers to q_theta
    x[2] refers to tau

    C1, C2, tau1, v2, lambda_, tau2, phi = params
    """
    return _background_v2_jac(x, *params) + _foreground_v2_jac(x, *params)


# Default params for scipy.optimize.curve_fit
modelv1_params_initial_guess = (1, 1, 1, 1, 1, 1, 0)
modelv1_params_lower_bound = (0, 0, 0, 0, 0, 0, -np.pi / 2)
modelv1_params_upper_bound = (np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, +np.pi / 2)
model1_params_initial_guess = modelv1_params_initial_guess
model2_params_initial_guess = modelv1_params_initial_guess
model3_params_initial_guess = modelv1_params_initial_guess
model4_params_initial_guess = modelv1_params_initial_guess
model1_params_lower_bound = modelv1_params_lower_bound
model2_params_lower_bound = modelv1_params_lower_bound
model3_params_lower_bound = modelv1_params_lower_bound
model4_params_lower_bound = modelv1_params_lower_bound
model1_params_upper_bound = modelv1_params_upper_bound
model2_params_upper_bound = modelv1_params_upper_bound
model3_params_upper_bound = modelv1_params_upper_bound
model4_params_upper_bound = modelv1_params_upper_bound
models = (model1, model2, model3, model4)
models_jac = (model1_jac, model2_jac, model3_jac, model4_jac)
models_initial_guess = (model1_params_initial_guess,
                        model2_params_initial_guess,
                        model3_params_initial_guess,
                        model4_params_initial_guess)
models_params_lower_bound = (model1_params_lower_bound,
                             model2_params_lower_bound,
                             model3_params_lower_bound,
                             model4_params_lower_bound)
models_params_upper_bound = (model1_params_upper_bound,
                             model2_params_upper_bound,
                             model3_params_upper_bound,
                             model4_params_upper_bound)
