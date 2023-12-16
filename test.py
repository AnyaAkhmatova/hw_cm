import argparse
import json
import os
from pathlib import Path
import glob
from glob import glob

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchaudio

from tqdm import tqdm
import pandas as pd

import wandb

import hw_cm.model as module_arch
from hw_cm.utils.parse_config import ConfigParser




class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.test_dir = test_dir
        self.filenames = []
        for format in ["mp3", "wav", "flac"]:
            for filename in glob(test_dir + "/*." + format):
                self.filenames.append(filename)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio, sr = torchaudio.load(self.filenames[idx])
        audio = audio[0:1, :]
        if sr != 16000:
            audio = torchaudio.functional.resample(audio, sr, 16000)
        audio = audio[0]
        padded_audio = torch.zeros(64000)
        for left_border in range(0, padded_audio.shape[0], audio.shape[0]):
            right_border = min(padded_audio.shape[0], left_border + audio.shape[0])
            padded_audio[left_border: right_border] = audio[: (right_border - left_border)]
        return padded_audio


def main(config):
    logger = config.get_logger("test")

    dataset = TestDataset(config["test_dir"])
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = config.init_obj(config["arch"], module_arch)
    logger.info(model)

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, map_location=device)
    state_dict = checkpoint["state_dict"]
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()


    wandb.init(
            project=config['trainer'].get('wandb_project'),
            config=config,
            name="test"
        )
    df = pd.DataFrame(columns=["audio", "bonafide proba", "spoof proba"])
        
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            batch.to(device)
            logits = model(batch)

            cur_audio = batch[0]
            probs = nn.functional.softmax(logits[0, :], dim=-1)
            bonafide_proba = probs[1].item()
            spoof_proba = probs[0].item()
            df.loc[len(df)] = [wandb.Audio(cur_audio.reshape(-1, 1).detach().cpu().numpy(), sample_rate=16000), 
                               bonafide_proba, spoof_proba]
            
    wandb.log({"test audios and probas": wandb.Table(dataframe=df)})
    logger.info("Table is added to wandb.")


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="PyTorch Template")
    args.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    args.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    args.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )

    args = args.parse_args()

    # set GPUs
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    # first, we need to obtain config with model parameters
    # we assume it is located with checkpoint in the same folder
    model_config = Path(args.resume).parent / "config.json"
    with model_config.open() as f:
        config = ConfigParser(json.load(f), resume=args.resume)

    # update with addition configs from `args.config` if provided
    if args.config is not None:
        with Path(args.config).open() as f:
            config.config.update(json.load(f))

    main(config)

