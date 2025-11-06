import torch
import os
import argparse
import json
from stable_audio_tools.inference.generation import generate_diffusion_cond
from stable_audio_tools.models.pretrained import get_pretrained_model
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import copy_state_dict, load_ckpt_state_dict
from pathlib import Path
from einops import rearrange
from itertools import product
import random
import torchaudio
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
import metrics


def load_model(model_config=None, model_ckpt_path=None, pretrained_name=None, pretransform_ckpt_path=None, device="cuda", model_half=False):
    global model, sample_rate, sample_size, model_type
    
    if pretrained_name is not None:
        print(f"Loading pretrained model {pretrained_name}")
        model, model_config = get_pretrained_model(pretrained_name)

    elif model_config is not None and model_ckpt_path is not None:
        print(f"Creating model from config")
        model = create_model_from_config(model_config)

        print(f"Loading model checkpoint from {model_ckpt_path}")
        copy_state_dict(model, load_ckpt_state_dict(model_ckpt_path))

    sample_rate = model_config["sample_rate"]
    sample_size = model_config["sample_size"]
    model_type = model_config["model_type"]

    if pretransform_ckpt_path is not None:
        print(f"Loading pretransform checkpoint from {pretransform_ckpt_path}")
        model.pretransform.load_state_dict(load_ckpt_state_dict(pretransform_ckpt_path), strict=False)
        print(f"Done loading pretransform")
        
    model.to(device).eval().requires_grad_(False)

    if model_half:
        model.to(torch.float16)
        
    print(f"Done loading model")

    return model, model_config
def run(model_path, model_cfg, output_dir, init_audio, prompts, steps, cfg_scales, init_seed, num_seeds):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_dir, exist_ok=True)
    with open(model_cfg) as f:
        model_config = json.load(f)
    model, model_config = load_model(model_config, model_path)
    print(f'model type: {model_config["model_type"]}')
    print(f'model diff objective: {model.diffusion_objective}')

    if init_audio is not None:
        print(f"Loading init audio from {init_audio} with forced mono")
        audio_np, sr = librosa.load(init_audio, sr=model_config["sample_rate"], mono=True)  # force mono
        audio_tensor = torch.from_numpy(audio_np).float()
        audio_tensor = audio_tensor.unsqueeze(0)
        init_audio = (sr, audio_tensor)
    else:
        init_audio = None

    tensors = []
    seeds = np.arange(init_seed, num_seeds, 1.0, dtype=int)
    metrics_list = []
    
    for seed in seeds:
        generations = []
        print(f"seed: {seed}")
        for prompt, step, cfg_scale in product(prompts, steps, cfg_scales):
            print(f"\n=== Generating for: prompt='{prompt}', steps={step}, cfg_scale={cfg_scale} ===\n")
            #torch.manual_seed(random.randint(0, 1000))

            conditioning = [{
                "prompt": prompt,
                "seconds_start": 0,
                "seconds_total": 11
            }]

            audio = generate_diffusion_cond(
                model=model,
                conditioning=conditioning,
                sample_size=model_config["sample_size"],
                steps=step,
                seed=seed,
                cfg_scale=cfg_scale,
                sigma_min=0.3,
                sigma_max=500,
                sampler_type="euler",
                return_latents=False,
                device=device,
                init_audio=init_audio
            )

            audio = rearrange(audio, "b d n -> d (b n)")
            #tensors.append(audio)

            #audio_out = audio.to(torch.float32).div(torch.max(torch.abs(audio))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
            audio_out = (
                audio.to(torch.float32)
                .div(torch.max(torch.abs(audio)))  # do [-1,1]
                .clamp(-1, 1)
                .cpu()
                .numpy()
            )
            if audio_out.shape[0] == 2:
                audio_out = np.mean(audio_out, axis=0)
            safe_prompt = prompt.replace(" ", "_").replace("/", "_")
            filename = f"output_prompt-{safe_prompt}_steps-{step}_cfg-{cfg_scale}_seed-{seed}.npz"
            out_path = os.path.join(output_dir, filename)
            #np.savez_compressed(out_path, y=audio_out)
            delay = 75
            cep = metrics.get_cepstrum(audio_out)
            z75 = metrics.get_z_score(cep[0:250+1], delay, start_buff=25)
            z75_blurred = metrics.get_local_peak_z(cep[0:250+1], delay, start_buff=25)
            max75 = cep[delay:delay+1]
            max75e = max(cep[delay-3:delay+3])
            ceps_mean = np.mean(cep)
            ceps_med = np.median(cep)
            z75_mad = metrics.get_median_z_score(cep[0:250+1], delay, start_buff=25)
            row = (seed, step, cfg_scale, safe_prompt, z75, z75_blurred, max75, max75e, ceps_mean, ceps_med, z75_mad)
            metrics_list.append(row)
            #cep_mov = metrics.moving_average(cep)
    df = pd.DataFrame(metrics_list)
    df.to_csv(Path(output_dir) / 'metrics.csv', index=False)


    # Średnia z tensorów
    #stacked = torch.stack(tensors, dim=0)
    #mean_tensor = stacked.mean(dim=0)
    #mean_audio = mean_tensor.to(torch.float32).div(torch.max(torch.abs(mean_tensor))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    #torchaudio.save(os.path.join(output_dir, f"zzz_output_mean.wav"), mean_audio, sample_rate=model_config["sample_rate"])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio generation from model with grid search")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model .ckpt")
    parser.add_argument("--model_cfg", type=str, required=True, help="Path to model config json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated audio")
    parser.add_argument("--init_audio", type=str, default=None, help="Optional init audio path")    
    parser.add_argument("--prompt", nargs="+", required=True, help="List of prompts (use quotes)")
    parser.add_argument("--steps", type=int, nargs="+", required=True, help="List of diffusion step counts")
    parser.add_argument("--cfg_scale", type=float, nargs="+", required=True, help="List of cfg scales")
    parser.add_argument("--init_seed", type=int, required=True, help="Initial seed")
    parser.add_argument("--num_seeds", type=int, required=True, help="Number of next seeds")

    args = parser.parse_args()

    run(model_path=args.model_path, model_cfg=args.model_cfg, output_dir=args.output_dir, init_audio=args.init_audio, prompts=args.prompt, steps=args.steps, cfg_scales=args.cfg_scale, init_seed=args.init_seed, num_seeds=args.num_seeds)
