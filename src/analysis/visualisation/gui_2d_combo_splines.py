

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import flet as ft
from flet.matplotlib_chart import MatplotlibChart
from analysis.spline_generation import spline_generation_for_upto_v2_models as sg

matplotlib.use("svg")
matplotlib.rcParams.update({'font.size': 12})
num_of_x_points = 100
model_name = 'ecmac'
exp_models_folder = 'ecmac_2025_11_07_fast_v3_splines_ar2'

def true_data_visualiser(page: ft.Page):

    page.vertical_alignment = ft.MainAxisAlignment.SPACE_BETWEEN

    fig = plt.figure(figsize=(12, 8))
    ax_3d = plt.subplot2grid((6, 8), (0, 0), rowspan=6, colspan=6, projection='3d')
    ax_3d.set_aspect(aspect='auto', adjustable='box')
    ax_fx = plt.subplot2grid((6, 8), (0, 6), rowspan=2, colspan=2)
    ax_x = plt.subplot2grid((6, 8), (3, 6), rowspan=2, colspan=2)


    def load_dataset(e):
        global f_x_f_names
        global f_x_f_splines

        print(f'LOADING {dataset_dropdown.value}')
        chart_3d.opacity=0.2
        page.update()
        feature_names, input_x_input_names, f_x_f_names,\
        all_features, just_inputs_splines, single_func_splines, f_x_f_splines, input_x_input_splines = \
            sg.get_features(dataset_dropdown.value, exp_models_folder=exp_models_folder,
                            model_name=model_name, feature_points_in_span=num_of_x_points)
        print(f'FINISHED LOADING {dataset_dropdown.value}')
        chart_3d.opacity = 1
        page.update()

        ax_fx.clear()
        ax_x.clear()
        ax_fx.plot(np.arange(0, 1, 1 / num_of_x_points), single_func_splines)
        ax_x.plot(np.arange(0, 1, 1 / num_of_x_points), just_inputs_splines)

        combo_dropdown.options = [ft.DropdownOption(name) for name in f_x_f_names]
        combo_dropdown.value = f_x_f_names[0]
        page.update()

        redraw(None)

    def redraw(e):
        ax_3d.clear()
        combo = 0
        for i, n in enumerate(f_x_f_names):
            if n == combo_dropdown.value:
                combo = i

        spline_surface = f_x_f_splines[:, :, combo]
        x = y = np.arange(0, 1, 1 / num_of_x_points)
        X, Y = np.meshgrid(x, y)
        Z = spline_surface

        ax_3d.set_xlabel(f_x_f_names[combo].split('x')[0])
        ax_3d.set_ylabel(f_x_f_names[combo].split('x')[1])
        ax_3d.set_zlabel('Feature value')
        ax_3d.plot_wireframe(Y, X, Z, linewidth=0.2, rcount=num_of_x_points, ccount=num_of_x_points)

        ax_3d.view_init(elev=elevation_slider.value, azim=azimuth_slider.value)

        fig.tight_layout()

        page.update()

    dataset_dropdown = ft.Dropdown(options=[ft.DropdownOption(ds) for ds in sg.datasets],
                                   value='concrete_compressive_strength', on_change=load_dataset, width=500)


    combo_dropdown = ft.Dropdown(width=500, on_change=redraw)
    chart_3d = MatplotlibChart(fig, expand=True)

    elevation_slider = ft.Slider(label='Elevation {value}', value=428, on_change_end=redraw, min=180, max=540, divisions=4 * 36)
    azimuth_slider = ft.Slider(label='Azimuth {value}', value=360, on_change_end=redraw, min=180, max=540, divisions=4 * 36)


    load_dataset(dataset_dropdown.value)

    page.add(dataset_dropdown)
    page.add(combo_dropdown)
    page.add(chart_3d)
    page.add(ft.Divider(height=10))
    page.add(elevation_slider)
    page.add(azimuth_slider)

    page.update()

ft.app(true_data_visualiser)
