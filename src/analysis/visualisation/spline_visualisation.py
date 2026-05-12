

from matplotlib import pyplot as plt
import numpy as np

import ec.spline_responses as spl_res
from src.analysis.utils import funcs_for_sym_reg as sym_reg
import analysis.utils.funcs_on_strings as str_utils

SUBPLOT_ADJUST = {'left': 0.03, 'bottom': 0.03, 'right': 0.97, 'top': 0.94, 'wspace': 0.29, 'hspace': 0.26}
FIG_SIZE = (12, 6)
FIG_DPI = 150

def find_closest_indices(a, b):
    a = np.array(a)
    b = np.array(b)

    indices = [np.argmin(np.abs(a - val)) for val in b]

    return np.array(indices)

def generate_single_feature_subplot(sp_loc, colspan, rowspan, fig, sp_grid, x_data, y_data, real_x_data, title=None,
                                    from_data=False, linewidth=2):
    subplot = plt.subplot2grid(shape=sp_grid, loc=sp_loc, colspan=colspan, rowspan=rowspan, fig=fig)
    subplot_ax2 = subplot.twinx()
    subplot_ax2.hist(x=real_x_data, bins=200, alpha = 0.5)
    if from_data:
        subplot.scatter(x_data, y_data, c='r', s=5)
    else:
        subplot.plot(x_data, y_data, c='r', linewidth=linewidth)

    if title is not None:
        subplot.set_title(title)

    return subplot

def generate_double_feature_subplot(sp_loc, colspan, rowspan, fig, sp_grid, feature_combo, feature_names, x_train_data,
                                    x_data, y_data, vis_dim='3d', title=None, from_data=False, data_size=15):

    combination_index = feature_combo[0] * len(feature_names) + feature_combo[1] - np.sum(range(feature_combo[0] + 2))

    data = y_data[:, combination_index]
    X = x_data[:, feature_combo[0]]
    Y = x_data[:, feature_combo[1]]
    if not from_data:
        X, Y = np.meshgrid(X, Y)
        data = y_data[:, :, combination_index]

    if vis_dim == '3d':
        subplot = plt.subplot2grid(shape=sp_grid, loc=sp_loc, colspan=colspan, rowspan=rowspan, fig=fig, projection='3d')
        surf = subplot.plot_surface(X, Y, data, linewidth=1, antialiased=False, cmap='coolwarm')

    elif vis_dim == '2d':
        subplot = plt.subplot2grid(shape=sp_grid, loc=sp_loc, colspan=colspan, rowspan=rowspan, fig=fig)
        if from_data:
            subplot.scatter(x=X, y=Y, s=20, c=data, cmap='coolwarm')
        else:
            subplot.pcolor(X, Y, data, cmap='rainbow', vmin=np.min(y_data[:, :, 7:]), vmax=np.max(y_data[:, :, 7:]))
            subplot.scatter(x=x_train_data[:, feature_combo[0]], y=x_train_data[:, feature_combo[1]], c='w', s=data_size, alpha=0.5)

    #subplot.set_xlabel(feature_names[0])
    #subplot.set_ylabel(feature_names[1])

    if title is not None:
        subplot.set_title(title)

    return subplot

def plot_single_feature_splines(model, feature_names, x_train_data, unscaled_feature_inputs, just_inputs_splines,
                                single_feature_spline_components, single_feature_composite_spline, from_data=False,
                                save=True, linewidth=2):
    spline_resolutions = model.steps[1][1].spline_resolutions
    spline_coefficients = spl_res.get_spline_summation_coefficients(model)
    all_figures = {}
    for f, feature_name in enumerate(feature_names):
        all_feature_plots = {}
        all_figures[feature_name] = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
        all_figures[feature_name].suptitle(feature_name)
        all_figures[feature_name].subplots_adjust(**SUBPLOT_ADJUST)
        for r, res in enumerate(spline_resolutions):
            coef = np.round(single_feature_spline_components[f][f'l{res}'][0], 3)
            all_feature_plots[f'c{res}'] = generate_single_feature_subplot(sp_loc=(0, r), colspan=1, rowspan=1,
                                                                           fig=all_figures[feature_name],
                                                                           sp_grid=(3, 3),
                                                                           x_data=unscaled_feature_inputs[:, f], y_data=
                                                                           single_feature_spline_components[f][
                                                                               f'l{res}'][1],
                                                                           real_x_data=x_train_data[:, f],
                                                                           title=f'Linear_{res}. Coef = {coef:.3f}',
                                                                           from_data=from_data, linewidth=linewidth)

            coef = np.round(single_feature_spline_components[f][f'c{res}'][0], 3)
            all_feature_plots[f'c{res}'] = generate_single_feature_subplot(sp_loc=(1, r), colspan=1, rowspan=1,
                                                                           fig=all_figures[feature_name],
                                                                           sp_grid=(3, 3),
                                                                           x_data=unscaled_feature_inputs[:, f], y_data=
                                                                           single_feature_spline_components[f][
                                                                               f'c{res}'][1],
                                                                           real_x_data=x_train_data[:, f],
                                                                           title=f'Cubic_{res}. Coef = {coef:.3f}',
                                                                           from_data=from_data, linewidth=linewidth)

        all_feature_plots['input'] = generate_single_feature_subplot(sp_loc=(2, 0), colspan=1, rowspan=1,
                                                                     fig=all_figures[feature_name], sp_grid=(3, 3),
                                                                     x_data=unscaled_feature_inputs[:, f],
                                                                     y_data=just_inputs_splines[:, f],
                                                                     real_x_data=x_train_data[:, f],
                                                                     title=f'Input. Coef = {spline_coefficients[f]}',
                                                                     from_data=from_data, linewidth=linewidth)

        all_feature_plots['full'] = generate_single_feature_subplot(sp_loc=(2, 1), colspan=2, rowspan=1,
                                                                    fig=all_figures[feature_name], sp_grid=(3, 3),
                                                                    x_data=unscaled_feature_inputs[:, f],
                                                                    y_data=single_feature_composite_spline[:, f],
                                                                    real_x_data=x_train_data[:, f], title='Full Spline',
                                                                    from_data=from_data, linewidth=linewidth)
        if save:
            plt.savefig(f'{save} 1D spline of {feature_name}.pdf')

def plot_all_features_and_combos(feature_names, x_train_data, x_data, y_data_single_feature, y_data_double_feature,
                                 vis_dim='3d', from_data=False, save=True, linewidth=2, data_size=15):
    fig = plt.figure(figsize=FIG_SIZE, dpi=FIG_DPI)
    fig.subplots_adjust(**SUBPLOT_ADJUST)
    num_of_features = len(feature_names)
    shape = (num_of_features, num_of_features)
    for f1, feature_name_1 in enumerate(feature_names):
        for f2, feature_name_2 in enumerate(feature_names):
            if f1 == f2:
                _ = generate_single_feature_subplot(sp_loc=(f1, f2), colspan=1, rowspan=1, fig=fig, sp_grid=shape,
                                                    x_data=x_data[:, f1], y_data=y_data_single_feature[:, f1],
                                                    real_x_data=x_train_data[:, f1], title=feature_name_1,
                                                    from_data=from_data, linewidth=linewidth)
            if f1 > f2:
                _ = generate_double_feature_subplot(sp_loc=(f1, f2), colspan=1, rowspan=1, fig=fig, sp_grid=shape,
                                                    feature_combo=(f1, f2),
                                                    feature_names=(feature_name_1, feature_name_2),
                                                    x_train_data=x_train_data, x_data=x_data,
                                                    y_data=y_data_double_feature, vis_dim=vis_dim, title=None,
                                                    from_data=from_data, data_size=data_size)
    plt.subplots_adjust(left=0.03, bottom=0.04, right=0.97, top=0.97, wspace=0.3, hspace=0.4)
    if save:
        plt.savefig(f'{save} 2D splines.pdf')


def plot_sr_and_spline_single_feature_responses_on_data(keras_model, dataset_info, x_data, x_train_data,
                                                        sr_models_save_path):

    feature_names = sym_reg.get_feature_names_from_file_info(dataset_info)
    num_of_features = len(feature_names)

    just_inputs_splines, single_feature_spline_components, single_feature_spline_responses = \
        spl_res.get_single_feature_response_to_existing_data(keras_model, x_data, is_data_scaled=False)


    single_feature_sr_models, double_feature_sr_models = sym_reg.fit_sr_models(sr_models_save_path,
                                                                               dataset_info, x_train_data)

    nrows = int(np.floor(num_of_features / 4 ))
    ncols = int(np.min([num_of_features, 4]))
    fig, axes = plt.subplots(nrows, ncols)
    for i in range(x_data.shape[1]):
        feature_name = feature_names[i]
        sr_model = single_feature_sr_models[feature_name]
        equation = str_utils.format_floats_to_sig_figs(str(sr_model.sympy()), 3)
        func, params = sym_reg.get_jax_equation_from_pysr_model_df(sr_model, 0)
        single_feature_response = sym_reg.fitted_equation_response(func, params, x_data[:, i])

        axes[int(np.floor(i / 4)), i % 4].scatter(x_data[:, i], single_feature_spline_responses[:, i])
        axes[int(np.floor(i / 4)), i % 4].scatter(x_data[:, i], single_feature_response)
        axes[int(np.floor(i / 4)), i % 4].set_title(f'{feature_name}\n{equation}')