

import os
os.environ['KERAS_BACKEND'] = 'jax'
import pickle
import jax
import numpy as np
import json

data_dir = os.path.join('talent_benchmark', 'data')
datasets = os.listdir(data_dir)
all_models_folder = os.path.join('talent_benchmark', 'results_model')
always_there_folder = 'Norm-standard-Nan-mean-new-Cat-catboost'
model_file_name = 'best-val-0.pkl'


#dataset = 'concrete_compressive_strength'
def get_features(dataset, model_name = 'ecmac', exp_models_folder='ecmac_2025_11_04_fast_splines_ar2',
                 feature_points_in_span=100):
    arity = int(exp_models_folder.split('ar')[1])
    model_path = os.path.join(all_models_folder, exp_models_folder, f'{dataset}-{model_name}', always_there_folder, model_file_name)
    N_train_path = os.path.join(data_dir, dataset, 'N_train.npy')
    info_path = os.path.join(data_dir, dataset, 'info.json')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(info_path, 'rb') as f:
        info = json.load(f)

    N_train = np.load(N_train_path)
    num_of_features = N_train.shape[1]

    try:
        feature_names = list(info['num_feature_intro'].values())
    except:
        feature_names =[f'X_{i}' for i in np.arange(num_of_features)]

    num_of_points_in_feature = feature_points_in_span

    if arity ==1 :
        feature_inputs = np.tile(np.arange(0, 1, 1/num_of_points_in_feature), (num_of_features, 1)).T

        all_features = model.steps[1][1].transform(feature_inputs)

        just_inputs_splines = all_features[:, :num_of_features]
        single_func_splines = all_features[:, num_of_features:]

        input_x_input_names = None
        f_x_f_names = None
        f_x_f_splines = None
        input_x_input_splines = None

    elif arity ==2:
        # Create the inputs
        num_of_combinations = len(np.triu_indices(num_of_features, 1)[0])
        combination_indices = list(zip(np.triu_indices(num_of_features, 1)[0], np.triu_indices(num_of_features, 1)[1]))
        input_x_input_names = [f'{feature_names[c[0]]} x {feature_names[c[1]]}' for c in combination_indices]
        f_x_f_names = [f'F({feature_names[c[0]]}) x F({feature_names[c[1]]})' for c in combination_indices]

        feature_inputs = np.zeros((num_of_points_in_feature**2 * num_of_combinations, num_of_features))
        nx, ny = (num_of_points_in_feature, num_of_points_in_feature)
        x = np.linspace(0, 1, nx)
        y = np.linspace(0, 1, ny)
        combo_x_inputs, combo_y_inputs = np.meshgrid(x, y)
        combo_x_inputs = np.reshape(combo_x_inputs, (num_of_points_in_feature**2, 1)).squeeze()
        combo_y_inputs = np.reshape(combo_y_inputs, (num_of_points_in_feature ** 2, 1)).squeeze()

        for i in np.arange(num_of_combinations):
            feature_inputs[num_of_points_in_feature**2 * i : num_of_points_in_feature**2 * (i + 1),
                           combination_indices[i][0]] = combo_x_inputs
            feature_inputs[num_of_points_in_feature ** 2 * i: num_of_points_in_feature ** 2 * (i + 1),
                           combination_indices[i][1]] = combo_y_inputs

        # Calculate all the features
        all_features = model.steps[1][1].transform(feature_inputs)

        # Extract the correct values for each group of features
        just_inputs_splines = np.zeros((num_of_points_in_feature, num_of_features))
        single_func_splines = np.zeros((num_of_points_in_feature, num_of_features))
        for i in np.arange(num_of_features):
            _, indices = np.unique(feature_inputs[:, i], return_index=True)
            just_inputs_splines[:, i] = all_features[indices, i]
            single_func_splines[:, i] = all_features[indices, i + num_of_features]

        f_x_f_splines = np.zeros((num_of_points_in_feature, num_of_points_in_feature, num_of_combinations))
        input_x_input_splines = np.zeros((num_of_points_in_feature, num_of_points_in_feature, num_of_combinations))
        x_start_indices = [(num_of_features - i) * num_of_points_in_feature**2 for i in np.arange(1, num_of_features)]
        x_start_indices.insert(0, 0)
        for i in np.arange(num_of_combinations):
            x_start_index = i * num_of_points_in_feature**2
            for j in np.arange(num_of_points_in_feature):
                f_x_f_splines[:, j, i] = all_features[num_of_points_in_feature * j + x_start_index : num_of_points_in_feature * (j + 1) + x_start_index,
                                         i + num_of_features * 2]
                input_x_input_splines[:, j, i] = all_features[num_of_points_in_feature * j + x_start_index : num_of_points_in_feature * (j + 1) + x_start_index,
                                         i + num_of_features * 2 + num_of_combinations]

    return (feature_names, input_x_input_names, f_x_f_names,
            all_features, just_inputs_splines, single_func_splines, f_x_f_splines, input_x_input_splines)


'''
rows = arity
f = plt.figure()
f.set_label(dataset)
plot_simple = plt.subplot2grid((rows, rows), (0, 0), fig=f, colspan=1, rowspan=1)
if arity ==2 :
    plot_f_combos = plt.subplot2grid((rows, rows), (0, 1), fig=f, rowspan=1, colspan=1)
    plot_input_combos = plt.subplot2grid((2, rows), (1, 0), fig=f, rowspan=1, colspan=1)

    hf = plt.figure()
    ha = hf.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(x, y)
    ha.plot_surface(X, Y, f_x_f_splines[:, :, 0])


plot_simple.plot(np.arange(0, 1, 0.01), single_func_splines)
plot_simple.legend(feature_names, bbox_to_anchor=(-0.06, 1.0), fontsize=9)
if arity == 2:
    plot_f_combos.plot(np.arange(0, 1, 0.01), f_x_f_splines)
    #axs[0,1].legend(f_x_f_names, ncol=3)
    plot_input_combos.plot(np.arange(0, 1, 0.01), input_x_input_splines)
'''