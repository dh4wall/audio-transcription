import os
from TTS.utils.trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.layers.xtts.trainer.gpt_trainer import GPTArgs, GPTTrainer, GPTTrainerConfig, XttsAudioConfig
from TTS.utils.manage import ModelManager
import wandb
import sys
from TTS.utils.logging.wandb_logger import WandbLogger

os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"
os.environ["WANDB_CONSOLE"] = "off"

def add_artifact(self, file_or_dir, name, artifact_type, aliases=None):
    print(f"========Ignoring artifact: {name} {file_or_dir}========")
    return


WandbLogger.add_artifact = add_artifact

RUN_NAME = "kaggletest"
PROJECT_NAME = "gore" 
DASHBOARD_LOGGER = "wandb" 
LOGGER_URI = None


import os

# Base output folder - ADJUST THIS PATH AS NEEDED
OUT_PATH = "/home/dh4wall/Documents/abcdef" 
os.makedirs(OUT_PATH, exist_ok=True)
print("Output path:", OUT_PATH)

# Checkpoints folder
CHECKPOINTS_OUT_PATH = os.path.join(OUT_PATH, "XTTS_v2.0_original_model_files")
os.makedirs(CHECKPOINTS_OUT_PATH, exist_ok=True)

# DVAE files
DVAE_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/dvae.pth"
MEL_NORM_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/mel_stats.pth"

DVAE_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(DVAE_CHECKPOINT_LINK))
MEL_NORM_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(MEL_NORM_LINK))

# Download DVAE files if needed
if not os.path.isfile(DVAE_CHECKPOINT) or not os.path.isfile(MEL_NORM_FILE):
    print(" > Downloading DVAE files!")
    ModelManager._download_model_files([MEL_NORM_LINK, DVAE_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True)

# XTTS v2.0 files
TOKENIZER_FILE_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/vocab.json"
XTTS_CHECKPOINT_LINK = "https://coqui.gateway.scarf.sh/hf-coqui/XTTS-v2/main/model.pth"

TOKENIZER_FILE = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(TOKENIZER_FILE_LINK))
XTTS_CHECKPOINT = os.path.join(CHECKPOINTS_OUT_PATH, os.path.basename(XTTS_CHECKPOINT_LINK))

# Download XTTS v2.0 files if needed
if not os.path.isfile(TOKENIZER_FILE) or not os.path.isfile(XTTS_CHECKPOINT):
    print(" > Downloading XTTS v2.0 files!")
    ModelManager._download_model_files(
        [TOKENIZER_FILE_LINK, XTTS_CHECKPOINT_LINK], CHECKPOINTS_OUT_PATH, progress_bar=True
    )

# ADJUST THIS PATH AS NEEDED
training_dir = "/home/dh4wall/Documents/abcdef"

# Make sure the folder exists
if not os.path.isdir(training_dir):
    raise FileNotFoundError(f"Training directory does not exist: {training_dir}")

print("Training directory:", training_dir)


OPTIMIZER_WD_ONLY_ON_WEIGHTS = True  
START_WITH_EVAL = True  
BATCH_SIZE = 1
GRAD_ACUMM_STEPS = 252
LANGUAGE = "var"


model_args = GPTArgs(
    max_conditioning_length=143677,
    min_conditioning_length=66150,
    debug_loading_failures=True,
    max_wav_length=223997, 
    max_text_length=200, 
    mel_norm_file=MEL_NORM_FILE,
    dvae_checkpoint=DVAE_CHECKPOINT,
    xtts_checkpoint=XTTS_CHECKPOINT,  
    tokenizer_file=TOKENIZER_FILE,
    gpt_num_audio_tokens=1026, 
    gpt_start_audio_token=1024,
    gpt_stop_audio_token=1025,
    gpt_use_masking_gt_prompt_approach=True,
    gpt_use_perceiver_resampler=True,
)

audio_config = XttsAudioConfig(sample_rate=16000, dvae_sample_rate=16000, output_sample_rate=24000) 

# ADJUST THIS PATH AS NEEDED
SPEAKER_REFERENCE = "/home/dh4wall/Documents/abcdef/datasets/wavs/varhadi_0001.wav"


config = GPTTrainerConfig(
    run_eval=True,
    epochs = 1000, 
    output_path=OUT_PATH,
    model_args=model_args,
    run_name=RUN_NAME,
    project_name=PROJECT_NAME,
    run_description="""
        GPT XTTS training
        """,
    dashboard_logger=DASHBOARD_LOGGER,
    wandb_entity=None,
    logger_uri=LOGGER_URI,
    audio=audio_config,
    batch_size=BATCH_SIZE,
    batch_group_size=48,
    eval_batch_size=BATCH_SIZE,
    num_loader_workers=8,
    eval_split_max_size=256, 
    print_step=50, 
    plot_step=100, 
    log_model_step=1000, 
    save_step=9999999999, 
    save_n_checkpoints=1,
    save_checkpoints=False,
    print_eval=False,
    optimizer="AdamW",
    optimizer_wd_only_on_weights=OPTIMIZER_WD_ONLY_ON_WEIGHTS,
    optimizer_params={"betas": [0.9, 0.96], "eps": 1e-8, "weight_decay": 1e-2},
    lr=5e-06,  
    lr_scheduler="MultiStepLR",
    lr_scheduler_params={"milestones": [50000 * 18, 150000 * 18, 300000 * 18], "gamma": 0.5, "last_epoch": -1},
) 

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.models.xtts import XttsArgs

with torch.serialization.safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs]):
    model = GPTTrainer.init_from_config(config)


dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    # ADJUST THIS PATH AS NEEDED
    meta_file_train="/home/dh4wall/Documents/abcdef/datasets/metadata.csv",
    language=LANGUAGE,
    path=training_dir
)

train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=False)

if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method('spawn', force=True)

    if not eval_samples:
        print("No eval samples found — training will run without evaluation.")
        trainer = Trainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=False,
                grad_accum_steps=GRAD_ACUMM_STEPS,
            ),
            config,
            output_path=OUT_PATH,
            model=model,
            train_samples=train_samples,
            eval_samples=None,
        )
    else:
        trainer = Trainer(
            TrainerArgs(
                restore_path=None,
                skip_train_epoch=False,
                start_with_eval=START_WITH_EVAL,
                grad_accum_steps=GRAD_ACUMM_STEPS,
            ),
            config,
            output_path=OUT_PATH,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )

    trainer.fit()