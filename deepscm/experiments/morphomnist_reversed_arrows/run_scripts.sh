python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m ConditionalReversedVISEM --data_dir ./assets/data/morphomnist/intensity_causes_thickness --log-dir SVIExperimentReversedArrows --default_root_dir . --decoder_type fixed_var --gpus 0 --terminate_on_nan True

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m ConditionalDecoderReversedVISEM --data_dir ./assets/data/morphomnist/intensity_causes_thickness --log-dir SVIExperimentReversedArrows --default_root_dir . --decoder_type fixed_var --gpus 0

python3 -m deepscm.experiments.morphomnist_reversed_arrows.trainer -e SVIExperiment -m IndependentReversedVISEM --data_dir ./assets/data/morphomnist/intensity_causes_thickness --log-dir SVIExperimentReversedArrows --default_root_dir . --decoder_type fixed_var --gpus 0
