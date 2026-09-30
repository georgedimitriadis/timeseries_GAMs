PYTHONPATH := /home/gd25222/george/repos/timeseries_GAMs/src:/home/gd25222/george/repos/timeseries_GAMs/ptsbenchmark

MODEL ?= tsfresh_bayesian_ridge
DATASET ?= electricity_nips
CORE ?= 0

.PHONY: run_one_dataset

run_one_pts_dataset_one_cpu:
	PYTHONPATH=$(PYTHONPATH) taskset -c $(CORE) python ./ptsbenchmark/run.py \
		--config ptsbenchmark/config/default/$(MODEL).yaml \
		--data.data_manager.init_args.dataset $(DATASET) \
		--data.data_manager.init_args.path ./ptsbenchmark/datasets \
		--trainer.default_root_dir ./src/exps/$(MODEL)

run_one_pts_dataset_many_cpu:
	PYTHONPATH=$(PYTHONPATH) python ./ptsbenchmark/run.py \
		--config ptsbenchmark/config/default/$(MODEL).yaml \
		--data.data_manager.init_args.dataset $(DATASET) \
		--data.data_manager.init_args.path ./ptsbenchmark/datasets \
		--trainer.default_root_dir ./src/exps/$(MODEL)

run_all_pts_datasets_dlinear_multistep:
	export CUDA_VISIBLE_DEVICES=0
	./run_probts_all_datasets.sh dlinear

run_all_pts_datasets_dlinear_autoreg:
	export CUDA_VISIBLE_DEVICES=0
	./run_probts_all_datasets.sh CUDA_VISIBLE_DEVICES=1 dlinear_autoreg

run_nbeats_on_dysts_data_for_paper_lorenz:
	export CUDA_VISIBLE_DEVICES=0 python run_nbeats_on_dysts_data_for_paper.py --systems Lorenz --compare

run_nbeats_on_dysts_data_for_paper_all_systems:
	export CUDA_VISIBLE_DEVICES=0 python run_nbeats_on_dysts_data_for_paper.py --systems all --compare

run_nbeats_on_dysts_data_for_paper_all_systems_multiple_times:
	export CUDA_VISIBLE_DEVICES=0 ./run_nbeats_on_dysts_data_for_paper_repeat.sh 10 --systems all


