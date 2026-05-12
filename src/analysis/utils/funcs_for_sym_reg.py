import os
import pickle

import numpy as np
from pysr import PySRRegressor, sympy2jax
from sympy import symbols
from sklearn.linear_model import LinearRegression

import ec.spline_responses as sr

def whiten_y_data(y_data, y_train_data):
    mean, std = y_train_data.mean(), y_train_data.std()
    y = (y_data - mean) / std
    return y

def load_all_responses(save_path, seed):
    with open(os.path.join(save_path, f'single_feature_spline_responses-{seed}.pkl'),
              'rb') as f:
        single_feature_spline_responses = pickle.load(f)

    with open(os.path.join(save_path, f'double_feature_spline_responses-{seed}.pkl'),
              'rb') as f:
        double_feature_spline_responses = pickle.load(f)

    with open(os.path.join(save_path, f'just_inputs_splines-{seed}.pkl'),
              'rb') as f:
        just_inputs_splines = pickle.load(f)

    with open(os.path.join(save_path, f'input_x_input_splines-{seed}.pkl'),
              'rb') as f:
        input_x_input_splines = pickle.load(f)

    return just_inputs_splines, single_feature_spline_responses, input_x_input_splines, double_feature_spline_responses

def save_model(what, who, where):
    with open(os.path.join(where, f'{who}_sr_model.pkl'), 'wb') as f:
        pickle.dump(what, f)

def load_model(who, where):
    with open(os.path.join(where, f'{who}_sr_model.pkl'), 'rb') as f:
        model = pickle.load(f)
    return model

def get_feature_names_from_file_info(info):
    num_of_features =info['n_num_features']
    try:
        feature_names = list(info['num_feature_intro'].values())
    except:
        feature_names = [f'X_{i}' for i in np.arange(num_of_features)]

    return feature_names

def get_feature_name_combinations_from_info(info):
    feature_name_combinations = []
    feature_names = get_feature_names_from_file_info(info)
    for f1, feature_name_1 in enumerate(feature_names):
        for f2, feature_name_2 in enumerate(feature_names):
            if f1 < f2:
                feature_name_combinations.append(f'{feature_name_1}_{feature_name_2}')

    return feature_name_combinations

def get_data_indices_from_feature_names(f_names, info):
    """
    If f_name is a single name it returns the index of the feature in the data. If it is a double name (f1_f2) iot returns
    a tuple of a tuple of the individual indices of each name in the data and of the index of their combination.
    :param f_names: Either a single name (e.g. 'X1') or a combination of two names (e.g. 'X1_X2') seperated by an underscore
    :param info: The info dict of the data
    :return: Either an int index of the name or a tuple of a tuple of indices and an index of the combination.
    """
    feature_names = get_feature_names_from_file_info(info)

    if f_names in feature_names:
        return np.argwhere(np.array(feature_names) == f_names)[0][0]
    else:
        feature_name_combinations_indices = {}
        feature_name_combinations = get_feature_name_combinations_from_info(info)
        i = 0
        for f1, feature_name_1 in enumerate(feature_names):
            for f2, feature_name_2 in enumerate(feature_names):
                if f1 < f2:
                    feature_name_combinations_indices[feature_name_combinations[i]] = \
                        f1 * len(feature_names) + f2 - np.sum(range(f1 + 2))
                    i += 1

        f1 = f_names.split('_')[0]
        f2 = f_names.split('_')[1]
        f1_i = np.argwhere(np.array(feature_names) == f1)[0][0]
        f2_i = np.argwhere(np.array(feature_names) == f2)[0][0]

        return (f1_i, f2_i), feature_name_combinations_indices[f_names]

def fit_sr_models(save_path, info, x_train_data, model):
    feature_names = get_feature_names_from_file_info(info)

    just_inputs_splines, single_feature_spline_components, single_feature_spline_responses = \
        sr.get_single_feature_response_to_existing_data(model, x_train_data, is_data_scaled=False)

    input_x_input_splines, double_feature_spline_components, double_feature_spline_responses = \
        sr.get_double_feature_response_to_existing_data(model, x_train_data, is_data_scaled=False)

    sr_models_path = os.path.join(save_path, 'sr_models')
    if not os.path.isdir(sr_models_path):
        os.mkdir(sr_models_path)

    default_pysr_params = dict(
        binary_operators=['+', '*'],
        unary_operators=['exp', 'cos'],
        populations=30,
        maxdepth=8,
        model_selection="score",
        nested_constraints={"cos": {"cos": 0}, "exp":{"exp": 0}},
        input_stream='devnull',
        verbosity=0)

    single_feature_sr_models = {}
    for i, f in enumerate(single_feature_spline_responses.T):
        feature_name = feature_names[i]
        if not os.path.isfile(os.path.join(sr_models_path, f'{feature_name}_sr_model.pkl')):
            print(f'Fitting new {feature_name} sr model')
            sr_reg_model = PySRRegressor(**default_pysr_params)
            sr_reg_model.fit(X=x_train_data[:, i].reshape(-1, 1), y=f)
            save_model(sr_reg_model, feature_name, sr_models_path)
        else:
            sr_reg_model = load_model(feature_name, sr_models_path)
        single_feature_sr_models[feature_name] = sr_reg_model

    double_feature_sr_models = {}
    for f1, feature_name_1 in enumerate(feature_names):
        for f2, feature_name_2 in enumerate(feature_names):
            if f1 < f2:
                feature_name_combination = f'{feature_name_1}_{feature_name_2}'
                _, combination_index = get_data_indices_from_feature_names(feature_name_combination, info)
                double_f_response = double_feature_spline_responses[:, combination_index]
                x_data = x_train_data[:, [f1, f2]]

                if not os.path.isfile(os.path.join(sr_models_path, f'{ feature_name_combination}_sr_model.pkl')):
                    print(f'Fitting new {feature_name_combination} sr model')
                    sr_reg_model = PySRRegressor(**default_pysr_params)
                    sr_reg_model.fit(X=x_data, y=double_f_response)
                    save_model(sr_reg_model, feature_name_combination, sr_models_path)
                else:
                    sr_reg_model = load_model(feature_name_combination, sr_models_path)
                double_feature_sr_models[feature_name_combination] = sr_reg_model

    return single_feature_sr_models, double_feature_sr_models

def fit_just_inputs(x_train_data, just_input_splines):
    fits = []
    for i in range(just_input_splines.shape[1]):
        lr = LinearRegression()
        fit = lr.fit(X=x_train_data[:, i].reshape(-1, 1), y=just_input_splines[:, i])
        fits.append(fit)

    return fits

def get_index_combinations(num_of_features):
    index_combinations = []
    for k in range(num_of_features):
        for j in range(num_of_features):
            if k<j:
                index_combinations.append([k, j])
    return index_combinations

def fit_inputs_x_inputs(x_train_data, input_x_input_spines):
    x_data_index_combinations = get_index_combinations(x_train_data.shape[1])
    fits = []
    for i in range(input_x_input_spines.shape[1]):
        lr = LinearRegression()
        fit = lr.fit(X=x_train_data[:, x_data_index_combinations[i]], y=input_x_input_spines[:, i])
        fits.append(fit)

    return fits

def get_jax_equation_from_pysr_model_df(sr_model, order_from_best=0):
    equations_df = sr_model.equations_
    top_best = equations_df.iloc[equations_df['score'].nlargest(len(equations_df)).index].reset_index()
    x0, x1 = symbols('x0 x1')
    symbols_in = [x0, x1] if 'x1' in top_best['equation'].iloc[order_from_best] else [x0]
    f, params = sympy2jax(top_best['sympy_format'].iloc[order_from_best], symbols_in=symbols_in)

    return f, params

def fitted_equation_response(function, parameters, x_data):
    if len(x_data.shape) == 1:
        x_data = x_data.reshape(-1, 1)

    return np.array(function(x_data, parameters))

def single_model_predict(sr_model, x_data, order_from_best=0):
    f, params = get_jax_equation_from_pysr_model_df(sr_model, order_from_best=order_from_best)
    y_predict = fitted_equation_response(f, params, x_data)
    return y_predict

def predict_from_symbolic_reg(model, save_path, info, x_data, x_train_data):
    ridge = model[-1]
    spline_coefficients = ridge.coef_
    num_of_resolutions = len(model[1].spline_resolutions)

    feature_names = get_feature_names_from_file_info(info)
    feature_names_combinations = get_feature_name_combinations_from_info(info)

    just_inputs_splines, single_feature_spline_components, single_feature_spline_responses = \
        sr.get_single_feature_response_to_existing_data(model, x_data, is_data_scaled=False)

    input_x_input_splines, double_feature_spline_components, double_feature_spline_responses = \
        sr.get_double_feature_response_to_existing_data(model, x_data, is_data_scaled=False)

    single_feature_sr_models, double_feature_sr_models = fit_sr_models(save_path, info, x_train_data, None)
    single_feature_linear_fits = fit_just_inputs(x_data, just_inputs_splines)
    double_feature_linear_fits = fit_inputs_x_inputs(x_data, input_x_input_splines)

    y_predict = np.zeros((len(x_data)))
    for i in range(just_inputs_splines.shape[1]):
        y_predict = y_predict + single_feature_linear_fits[i].predict(x_data[:, i].reshape(-1, 1))  * spline_coefficients[i]
        f, params = get_jax_equation_from_pysr_model_df(single_feature_sr_models[feature_names[i]], 0)
        y_predict = y_predict + fitted_equation_response(f, params, x_data[:, i])

    num_of_single_feature_components = x_data.shape[1] * (1 + 2 * num_of_resolutions)
    x_data_index_combinations = get_index_combinations(x_data.shape[1])
    for i in range(input_x_input_splines.shape[1]):
        y_predict = y_predict + double_feature_linear_fits[i].predict(x_data[:, x_data_index_combinations[i]]) \
                    * spline_coefficients[i + num_of_single_feature_components]
        f, params = get_jax_equation_from_pysr_model_df(double_feature_sr_models[feature_names_combinations[i]], 0)
        y_predict = y_predict + fitted_equation_response(f, params, x_data[:, x_data_index_combinations[i]])

    y_predict = y_predict + ridge.intercept_

    return y_predict
