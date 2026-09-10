
import pandas as pd
import os

dir = 'src/exps/2026_09_all_data_dlinear_reproduction'
dir = 'ptsbenchmark/log_dir'
ltsf_datasets = ['etth1', 'etth2', 'ettm1', 'ettm2', 'traffic_ltsf', 'electricity_ltsf',
    'exchange_ltsf', 'weather_ltsf']
stsf_datasets=(
    'exchange_rate_nips', 'solar_nips', 'electricity_nips', 'traffic_nips', 'wiki2000_nips'
)

all_datasets = os.listdir(dir)
pred_len = [96, 132, 336, 720]
index = pd.MultiIndex.from_tuples([(i, k) for i in ['NMAE', 'CRPS'] for k in pred_len], names=['Metric', 'Horizon'])
ltsf_results = pd.DataFrame(columns=ltsf_datasets, index=index)
stsf_results = pd.DataFrame(columns=stsf_datasets, index=pd.Index(['NMAE', 'CRPS'], name='Metric'))


for dataset in all_datasets:
    try:
        results = pd.read_csv(os.path.join(dir, dataset, 'horizons_results.csv'))
        if dataset.split('_')[1] == 'ltsf':
            ds = dataset.split('_')[0] + '_' + dataset.split('_')[1]
            pred = int(dataset.split('_')[4][4:])
        elif dataset.split('_')[1] == 'nips':
            ds = dataset.split('_')[0] + '_' + dataset.split('_')[1]
            pred = None
        elif dataset.split('_')[2] == 'nips':
            ds = dataset.split('_')[0] + '_' + dataset.split('_')[1] + '_' + dataset.split('_')[2]
            pred = None
        else:
            ds = dataset.split('_')[0]
            pred = int(dataset.split('_')[3][4:])

        if ds in ltsf_datasets:
            ltsf_results.loc[('NMAE', pred), ds] = results['test_ND'][0]
            ltsf_results.loc[('CRPS', pred), ds] = results['test_CRPS'][0]
        if ds in stsf_datasets:
            stsf_results.loc['NMAE', ds] = results['test_ND'][0]
            stsf_results.loc['CRPS', ds] = results['test_ND'][0]
    except:
        pass
ltsf_results = ltsf_results.rename(index={132: 192}, level='Horizon')
paper_ltsf_results = pd.read_csv(os.path.join(dir, 'paper_dlinear_long_term.csv')).set_index(['Metric', 'Horizon'])
paper_stsf_results = pd.read_csv(os.path.join(dir, 'paper_dlinear_short_term.csv')).set_index(['Metric'])


# The results are pretty much the same. The worst diff is the solar_nips (stsf) at 24% difference.
dif_ltsf = (paper_ltsf_results - ltsf_results) / paper_ltsf_results
dif_stsf = (paper_stsf_results - stsf_results) / paper_stsf_results
