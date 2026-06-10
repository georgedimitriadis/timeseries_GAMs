PYTHONPATH := /home/gd25222/george/repos/timeseries_GAMs/src:/home/gd25222/george/repos/timeseries_GAMs/ptsbenchmark

MODEL ?= tsfresh_bayesian_ridge
DATASET ?= electricity_nips
CORE ?= 0

.PHONY: run_one_dataset

run_one_dataset_one_cpu:
	PYTHONPATH=$(PYTHONPATH) taskset -c $(CORE) python ./ptsbenchmark/run.py \
		--config ptsbenchmark/config/default/$(MODEL).yaml \
		--data.data_manager.init_args.dataset $(DATASET) \
		--data.data_manager.init_args.path ./ptsbenchmark/datasets \
		--trainer.default_root_dir ./src/exps/$(MODEL)

run_one_dataset_many_cpu:
	PYTHONPATH=$(PYTHONPATH) python ./ptsbenchmark/run.py \
		--config ptsbenchmark/config/default/$(MODEL).yaml \
		--data.data_manager.init_args.dataset $(DATASET) \
		--data.data_manager.init_args.path ./ptsbenchmark/datasets \
		--trainer.default_root_dir ./src/exps/$(MODEL)