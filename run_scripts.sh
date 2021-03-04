python3 -m deepscm.experiments.morphomnist.trainer -e SVIExperiment -m ConditionalReversedVISEM --data_dir ./assets/data/morphomnist --default_root_dir . --decoder_type fixed_var --gpus 1


pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
python3 -m deepscm.experiments.morphomnist.trainer -e SVIExtensionExperiment -m ConditionalClassReversedVISEM --data_dir ./assets/data/morphomnist