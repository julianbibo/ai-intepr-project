from hear21passt.base import load_model
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm
from pathlib import Path
from audiocraft.models import MusicGen
import argparse
import random
import numpy as np
import csv
from torch.utils.data import DataLoader

from activations_dataset import ActivationsDataset

from mlp_inferencing import MLP

mlp_model = MLP(
    input_dim=1024,
    hidden_dim=2048,
    output_dim=1024,
    dropout=0.2
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


music_classifier = load_model(mode="logits").to(device)


SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)


instrument = 'piano'
BATCH_SIZE = 128

logit_indices = {'piano': 153, 'guitar': 140, 'trumpet': 187, 'violin': 191}
instrument_logit_index = logit_indices[instrument]

NUM_LAYERS = 23

avg_baseline_logits = {layer: 0 for layer in range(NUM_LAYERS)}
avg_our_logits = {layer: 0 for layer in range(NUM_LAYERS)}
avg_baseline_probs = {layer: 0 for layer in range(NUM_LAYERS)}
avg_our_probs = {layer: 0 for layer in range(NUM_LAYERS)}


for layer in range(NUM_LAYERS):

    print(f"Testing layer {layer} for instrument {instrument}")

    # Create the dataset for the current layer and instrument
    test_set = ActivationsDataset(
        data_dir='/home/scur1188/ai-intepr-project/data',
        instruments=[instrument],
        seeds=[1, 2, 3],
        split="test",
        layer=layer,
    )

    test_dataloader = DataLoader(
        test_set,
        batch_size=BATCH_SIZE,
        shuffle=False
    )


    # create stock musicgen decoderlens
    baseline_music_model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
    baseline_music_model.set_generation_params(duration=4)
    baseline_music_model.compression_model.eval()
    baseline_music_model.lm.eval()
    up_to_layer = baseline_music_model.lm.transformer.layers[:layer]
    baseline_music_model.lm.transformer.layers = nn.ModuleList([*up_to_layer])



    # create our model
    pt_file = torch.load(
        f"/home/scur1188/ai-intepr-project/weights/layer_{layer:02d}_{instrument}_seeds-1,2,3_mlp.pt", map_location=device)
    mlp_model.load_state_dict(pt_file['model_state_dict'])
    mlp_model.eval()
    mlp_model.to(device)

    our_music_model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
    our_music_model.set_generation_params(duration=4)
    our_music_model.compression_model.eval()
    our_music_model.lm.eval()
    up_to_layer = our_music_model.lm.transformer.layers[:layer]
    our_music_model.lm.transformer.layers = nn.ModuleList([*up_to_layer, mlp_model])



    # counters (reset for each layer)
    baseline_instrument_logits_sum = 0
    our_instrument_logits_sum = 0
    baseline_instrument_probs_sum = 0
    our_instrument_probs_sum = 0
    num_test_samples = 0

    for idx, (x_batch, y_batch, prompts) in tqdm(enumerate(test_dataloader)):

        # print(f"\n{prompts=}")

        current_batch_size = x_batch.size(0)
        num_test_samples += current_batch_size

        with torch.no_grad():
            # baseline results
            baseline_audio_waveforms = baseline_music_model.generate(list(prompts))
            baseline_class_logits = music_classifier(baseline_audio_waveforms.squeeze().to(device))
            baseline_instrument_logits = baseline_class_logits[:, instrument_logit_index].sum()
            baseline_instrument_logits_sum += baseline_instrument_logits
            
            # Calculate softmax probabilities for baseline
            baseline_class_probs = F.softmax(baseline_class_logits, dim=1)
            baseline_instrument_probs = baseline_class_probs[:, instrument_logit_index].sum()
            baseline_instrument_probs_sum += baseline_instrument_probs

            # our results
            our_audio_waveforms = our_music_model.generate(list(prompts))
            our_class_logits = music_classifier(our_audio_waveforms.squeeze().to(device))
            our_instrument_logits = our_class_logits[:, instrument_logit_index].sum()
            our_instrument_logits_sum += our_instrument_logits
            
            # Calculate softmax probabilities for our model
            our_class_probs = F.softmax(our_class_logits, dim=1)
            our_instrument_probs = our_class_probs[:, instrument_logit_index].sum()
            our_instrument_probs_sum += our_instrument_probs

        torch.cuda.empty_cache()

    avg_baseline_logits[layer] = baseline_instrument_logits_sum / num_test_samples
    avg_our_logits[layer] = our_instrument_logits_sum / num_test_samples
    avg_baseline_probs[layer] = baseline_instrument_probs_sum / num_test_samples
    avg_our_probs[layer] = our_instrument_probs_sum / num_test_samples

    print("Layer info")
    for layer_idx in range(len(avg_baseline_logits)):
        print(f" - layer {layer_idx}:")
        print(f"     - baseline logits: {avg_baseline_logits[layer_idx]}")
        print(f"     -     ours logits: {avg_our_logits[layer_idx]}")
        print(f"     - baseline probs:  {avg_baseline_probs[layer_idx]}")
        print(f"     -     ours probs:  {avg_our_probs[layer_idx]}")


# save to CSV
with open(f'/home/scur1188/ai-intepr-project/results/{instrument}_logits_and_softmax_comparison.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    # Write header
    writer.writerow(['layer', 'avg_baseline_logits', 'avg_our_logits', 'avg_baseline_probs', 'avg_our_probs'])

    # Write data rows
    for layer in avg_baseline_logits.keys():
        writer.writerow([layer, avg_baseline_logits[layer], avg_our_logits[layer], 
                        avg_baseline_probs[layer], avg_our_probs[layer]])


full_model_logits_sum = 0
full_model_probs_sum = 0
full_model_run_test_samples = 0

# create stock musicgen decoderlens
full_music_model = MusicGen.get_pretrained("facebook/musicgen-small", device=device)
full_music_model.set_generation_params(duration=4)
full_music_model.compression_model.eval()
full_music_model.lm.eval()

for idx, (x_batch, y_batch, prompts) in tqdm(enumerate(test_dataloader)):

    current_batch_size = x_batch.size(0)
    full_model_run_test_samples += current_batch_size

    with torch.no_grad():
        # full model results
        audio_waveforms = full_music_model.generate(list(prompts))
        full_class_logits = music_classifier(audio_waveforms.squeeze().to(device))
        full_instrument_logits = full_class_logits[:, instrument_logit_index].sum()
        full_model_logits_sum += full_instrument_logits
        
        # Calculate softmax probabilities for full model
        full_class_probs = F.softmax(full_class_logits, dim=1)
        full_instrument_probs = full_class_probs[:, instrument_logit_index].sum()
        full_model_probs_sum += full_instrument_probs


    torch.cuda.empty_cache()



full_model_avg_logits = full_model_logits_sum / full_model_run_test_samples
full_model_avg_probs = full_model_probs_sum / full_model_run_test_samples

print(f"{full_model_avg_logits=}")
print(f"{full_model_avg_probs=}")