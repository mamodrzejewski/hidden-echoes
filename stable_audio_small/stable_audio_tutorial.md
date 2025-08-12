# Stable Audio Small configuration tutorial

After many efforts, I was able to launch the model. Here is the step-by-step instruction.

## Instruction

1. Create virtual env (Python 3.10 recommended)
2. Clone stable-audio-tools repository: https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools
3. install dependencies:
* go to *stable-audio-tools* directory
* type: *pip install -e .*
4. Log in to huggingface account and go to: https://huggingface.co/stabilityai/stable-audio-open-small
5. Create the directory structure for training:
* in main directory create subdirectory for training data
* create dataset_config.json file
* Download two files: *model.cpkt* and *base_model_config.json*
6. From https://github.com/Stability-AI/stable-audio-tools/tree/main/stable_audio_tools/configs/dataset_configs download *local_training_example.json* and *custom_md_example.py*. Change paths in *local_training_example.json*, set path to your subdirectory for training data.
7. You should be able to start training the model: *python3 train.py --dataset-config ../small_model/dataset_config.json --model-config ../small_model/base_model_config.json --name stable_audio_small_finetune*

