python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m ConditionalReversedVISEM --data_dir ./assets/data/morphomnist/ --default_root_dir ./SVIExperimentReversedArrowsFinal --gpus 0 --terminate_on_nan --pgm_lr 1e-5

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m ConditionalDecoderReversedVISEM --data_dir ./assets/data/morphomnist/ --default_root_dir SVIExperimentReversedArrowsFinal --gpus 0 --terminate_on_nan --pgm_lr 1e-5

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m IndependentReversedVISEM --data_dir ./assets/data/morphomnist/ --default_root_dir SVIExperimentReversedArrowsFinal --gpus 0 --terminate_on_nan --pgm_lr 1e-5

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m ConditionalReversedVISEM --data_dir ./assets/data/morphomnist/intensity_causes_thickness/ --default_root_dir ./SVIExperimentReversedArrowsFinal --gpus 0 --terminate_on_nan --max_epochs 400

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m ConditionalDecoderReversedVISEM --data_dir ./assets/data/morphomnist/intensity_causes_thickness/ --default_root_dir SVIExperimentReversedArrowsFinal --gpus 0 --terminate_on_nan

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m IndependentReversedVISEM --data_dir ./assets/data/morphomnist/intensity_causes_thickness/ --default_root_dir SVIExperimentReversedArrowsFinal --gpus 0 --max_epochs 200 --terminate_on_nan 
