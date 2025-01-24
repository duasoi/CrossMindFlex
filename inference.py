import os
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import cosine_similarity
import clip
import torch.nn as nn
import re
import numpy as np
from dataset_test import EEGDataset
from model import ATMS
from diffusion_prior import *
from custom_pipeline import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="EEG-based cross-subject image generation")
    parser.add_argument('--data_path', type=str, default="/root/EEG/datasets/")
    parser.add_argument('--src_subj', type=str, default='sub-01')
    parser.add_argument('--tgt_subj', type=str, default='sub-08')
    parser.add_argument('--model_name', type=str, default='FL')
    parser.add_argument('--domain_model_path', type=str, default="/root/EEG/logs_eeg/eeg_federated.pt")
    parser.add_argument('--eeg_model_path', type=str, default="/root/EEG/models/contrast/ATMS/sub-08/11-12_19-56/40.pth")
    parser.add_argument('--prior_path', type=str, default="/root/EEG/models/ATMS/")
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--guidance_scale', type=float, default=5.0)
    return parser.parse_args()

def load_clip_model():
    return clip.load("/root/EEG/pkl/ViT-H-14/open_clip_pytorch_model.bin", device=device)

def load_data():
    texts = []
    text_directory = "/root/EEG/images/test_images/"
    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()
    for dir in dirnames:
        try:
            idx = dir.index('_')
            description = dir[idx + 1:]
        except ValueError:
            continue
        texts.append(description)
    return texts

texts = load_data()

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def visualize_features(source, mapped, target):
    combined = np.concatenate([source, mapped, target], axis=0)
    labels = np.concatenate([
        np.zeros(len(source)),
        np.ones(len(mapped)) * 1,
        np.ones(len(target)) * 2
    ])
    tsne = TSNE(n_components=2, perplexity=8, learning_rate=300, n_iter=5000, random_state=42)
    reduced = tsne.fit_transform(combined)
    plt.figure(figsize=(6, 4.8))
    plt.scatter(reduced[labels == 0, 0], reduced[labels == 0, 1], c='blue', alpha=0.6, s=50, marker='o', label='Source (Original)')
    plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1], c='lime', alpha=0.8, s=80, marker='^', edgecolors='black', linewidths=0.5, label='Source (Adaption)')
    plt.scatter(reduced[labels == 2, 0], reduced[labels == 2, 1], c='red', alpha=0.6, s=50, marker='s', edgecolors='black', linewidths=0.5, label='Target')
    plt.legend()
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    save_path = "/home/yyl/data/hxf/EEG/tsne_plot_1.png"
    plt.savefig(save_path)
    plt.show()

def extract_final_features(src_loader, tgt_loader, domain_model, eeg_model, src_idx, tgt_idx, tgt_subj, batch_size):
    all_final_features = []
    src_originals, src_mapped, tgt_features = [], [], []
    with torch.no_grad():
        for batch in tgt_loader:
            tgt_data, _ = batch
            tgt_data = tgt_data.view(tgt_data.size(0), -1).to(device)
            tgt_features.append(tgt_data.cpu().numpy())
        for batch in src_loader:
            eeg_data, _ = batch
            eeg_data = eeg_data.to(device).view(eeg_data.size(0), -1)
            src_originals.append(eeg_data.cpu().numpy())
            mapped = domain_model(eeg_data, src_idx, tgt_idx)
            src_mapped.append(mapped.detach().cpu().numpy())
            mapped_reshaped = mapped.view(mapped.size(0), 63, 250)
            subject_id = extract_id_from_string(tgt_subj)
            subject_ids = torch.full((mapped_reshaped.size(0),), subject_id, dtype=torch.long, device=device)
            final_features = eeg_model(mapped_reshaped, subject_ids)
            all_final_features.append(final_features)
        tgt_features = np.concatenate(tgt_features, axis=0)
        src_originals = np.concatenate(src_originals, axis=0)
        src_mapped = np.concatenate(src_mapped, axis=0)
        visualize_features(src_originals, src_mapped, tgt_features)
        concatenated_features = torch.cat(all_final_features, dim=0)
    return concatenated_features

class SharedLinearMLP(nn.Module):
    def __init__(self, input_dims, shared_dim, output_dims):
        super(SharedLinearMLP, self).__init__()
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, 2048),
                nn.ReLU(),
                nn.Linear(2048, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048)
            ) for input_dim in input_dims
        ])
        self.output_fc = nn.ModuleList([nn.Linear(shared_dim, output_dim) for output_dim in output_dims])
        self.activation = nn.LeakyReLU(0.01)
        self.shared_fc = nn.Linear(shared_dim, shared_dim)

    def forward(self, x, src_label, tgt_label):
        input_layer = self.feature_extractor[src_label]
        output_layer = self.output_fc[tgt_label]
        shared = input_layer(x)
        shared = self.activation(shared)
        shared = self.shared_fc(shared)
        shared = self.activation(shared)
        final_output = output_layer(shared)
        return final_output

def main(args):
    model, preprocess = load_clip_model()
    input_dim, shared_dim, output_dim = 63 * 250, 2048, 63 * 250
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04', 'sub-05', 'sub-06', 'sub-07', 'sub-08']
    src_idx = subjects.index(args.src_subj)
    tgt_idx = subjects.index(args.tgt_subj)
    num_subjects = len(subjects)
    domain_model = SharedLinearMLP([input_dim] * num_subjects, shared_dim, [output_dim] * num_subjects).to(device)
    domain_model.load_state_dict(torch.load(args.domain_model_path, map_location=device))
    domain_model.eval()
    eeg_model = ATMS(63, 250).to(device)
    eeg_model.load_state_dict(torch.load(args.eeg_model_path, map_location=device), strict=False)
    eeg_model.eval()
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    pipe = Pipe(diffusion_prior, device=device)
    prior_weight_path = os.path.join(args.prior_path, f"{args.tgt_subj}/diffusion_prior.pt")
    pipe.diffusion_prior.load_state_dict(torch.load(prior_weight_path, map_location=device))
    generator = Generator4Embeds(num_inference_steps=4, device=device)
    tgt_dataset = EEGDataset(args.data_path, subjects=[args.tgt_subj])
    src_dataset = EEGDataset(args.data_path, subjects=[args.src_subj])
    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    final_features = extract_final_features(src_loader, tgt_loader, domain_model, eeg_model, src_idx, tgt_idx, args.tgt_subj, args.batch_size)
    output_dir = f"generated_imgs/GT_8/{args.src_subj}_to_{args.tgt_subj}_{args.model_name}_test"
    os.makedirs(output_dir, exist_ok=True)
    for k in tqdm(range(200)):
        eeg_embeds = final_features[k:k + 1].to(device)
        h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
        max_similarity = -1.0
        best_image = None
        for j in range(10):
            image = generator.generate(h.to(dtype=torch.float16))
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = model.encode_image(image_input).float()
            similarity = cosine_similarity(image_embedding, eeg_embeds).item()
            if similarity > max_similarity:
                max_similarity = similarity
                best_image = image
        path = f'{output_dir}/{k:03}.png'
        best_image.save(path)
if __name__ == "__main__":
    args = parse_args()
    main(args)
