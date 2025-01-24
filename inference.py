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

    # 添加所有参数并设置默认值
    parser.add_argument('--data_path', type=str, default="/data2/xyh1/EEG/datasets/",
                        help='Path to the EEG dataset')
    parser.add_argument('--src_subj', type=str, default='sub-01', help='Source subject (e.g., sub-01)')
    parser.add_argument('--tgt_subj', type=str, default='sub-08', help='Target subject (e.g., sub-02)')
    parser.add_argument('--model_name', type=str, default='FL', help='name ')
    parser.add_argument('--domain_model_path', type=str,
                        # default="/home/yyl/data/hxf/EEG/logs_eeg/eeg_domain_transfer_model_2048.pt",
                        default="/home/yyl/data/hxf/EEG/logs_eeg/eeg_model_2048_federated.pt",
                        help='Path to the pretrained EEG domain transfer model')
    parser.add_argument('--eeg_model_path', type=str,
                        # default="/home/yyl/data/hxf/EEG/models/contrast/ATMS/sub-08/11-13_20-32/40.pth",
                        # default="/home/yyl/data/hxf/EEG/models/contrast/ATMS/sub-08/11-23_10-54/40.pth",
                        default="/home/yyl/data/hxf/EEG/models/contrast/ATMS/sub-08/11-12_19-56/40.pth",
                        help='Path to the pretrained EEG feature extraction model')
    parser.add_argument('--prior_path', type=str,
                        default="/home/yyl/data/hxf/EEG/models/ATMS/",
                        help='Path to the prior weights of the diffusion model')
    parser.add_argument('--batch_size', type=int, default=200, help='Batch size for DataLoader')
    parser.add_argument('--num_inference_steps', type=int, default=50, help='Number of inference steps')
    parser.add_argument('--guidance_scale', type=float, default=5.0, help='Guidance scale for image generation')

    # 尝试解析命令行参数，如果没有传入则使用默认值
    return parser.parse_args()


# 加载 CLIP 模型
def load_clip_model():
    return clip.load("/data2/xyh1/EEG/pkl/ViT-H-14/open_clip_pytorch_model.bin", device=device)

def load_data():
    texts = []
    text_directory = "/data2/xyh1/EEG/images/test_images/"

    dirnames = [d for d in os.listdir(text_directory) if os.path.isdir(os.path.join(text_directory, d))]
    dirnames.sort()

    for dir in dirnames:
        try:
            idx = dir.index('_')
            description = dir[idx + 1:]
        except ValueError:
            print(f"Skipped: {dir} due to no '_' found.")
            continue

        texts.append(description)

    return texts

# 调用 load_data() 获取文本描述
texts = load_data()

def extract_id_from_string(s):
    match = re.search(r'\d+$', s)
    if match:
        return int(match.group())
    return None
import seaborn as sns
import matplotlib.pyplot as plt

# def plot_heatmap(src_data, tgt_data, title=""):
#     """
#     可视化源域和目标域的热图。
#     src_data 和 tgt_data 形状应该为 (batch_size, channels, time_steps)
#     """
#     # 转换为 NumPy 数组，并取第一个样本进行展示
#     src_data = src_data.cpu().numpy()
#     tgt_data = tgt_data.cpu().numpy()
#
#     # 绘制源域热图
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(src_data[0], cmap='viridis', cbar=True, xticklabels=10, yticklabels=10)
#     plt.title(f"Source Domain {title}")
#     plt.xlabel("Time Steps")
#     plt.ylabel("Channels")
#     plt.show()
#
#     # 绘制目标域热图
#     plt.figure(figsize=(12, 6))
#     sns.heatmap(tgt_data[0], cmap='viridis', cbar=True, xticklabels=10, yticklabels=10)
#     plt.title(f"Target Domain {title}")
#     plt.xlabel("Time Steps")
#     plt.ylabel("Channels")
#     plt.show()
from sklearn.manifold import TSNE
#
def visualize_features(source, mapped, target):
    # 合并所有特征
    combined = np.concatenate([source, mapped, target], axis=0)
    labels = np.concatenate([
        np.zeros(len(source)),  # 源域原始特征标签 0
        np.ones(len(mapped)) * 1,  # 映射特征标签 1
        np.ones(len(target)) * 2  # 目标域特征标签 2
    ])
    print(f"Source: {len(source)}, Mapped: {len(mapped)}, Target: {len(target)}")
    # 使用 t-SNE 降维
    tsne = TSNE(
        n_components=2,
        perplexity=8,  # 降低以凸显局部结构
        learning_rate=300,  # 增大以增强分离
        n_iter=5000,  # 增加迭代次数
        random_state=42
    )
    reduced = tsne.fit_transform(combined)

    # 绘制散点图
    plt.figure(figsize=(6, 4.8))

    # 源域原始特征：蓝色圆形（较小）
    plt.scatter(reduced[labels == 2, 0], reduced[labels == 2, 1],
                c='blue', alpha=0.6, s=50, marker='o', label='Source (Original)')

    # # 映射特征：绿色三角形（中等大小）
    # plt.scatter(reduced[labels == 1, 0], reduced[labels == 1, 1],
    #             c='lime', alpha=0.8, s=80, marker='^', edgecolors='black',
    #             linewidths=0.5, label='Source (Adaption)')

    # 目标域特征：红色方块（较大）
    plt.scatter(reduced[labels == 2, 0], reduced[labels == 2, 1],
                c='red', alpha=0.6, s=50, marker='s', edgecolors='black',
                linewidths=0.5, label='Target')

    plt.legend()
    # plt.title("Domain adaption via t-SNE")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    save_path = "/home/yyl/data/hxf/EEG/tsne_plot_1.png"

    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")
    plt.show()

# 使用跨域模型和 EEG 模型提取最终特征
def extract_final_features(src_loader, tgt_loader, domain_model, eeg_model, src_idx, tgt_idx, tgt_subj, batch_size):
    all_final_features = []

    # 初始化数据收集容器
    src_originals, src_mapped, tgt_features = [], [], []

    with torch.no_grad():

        for batch in tgt_loader:
            tgt_data, _ = batch
            tgt_data = tgt_data.view(tgt_data.size(0), -1).to(device)
            tgt_features.append(tgt_data.cpu().numpy())

        # 处理源域数据（同时收集原始特征、映射特征和生成最终特征）
        for batch in src_loader:
            eeg_data, _ = batch
            eeg_data = eeg_data.to(device).view(eeg_data.size(0), -1)


            src_originals.append(eeg_data.cpu().numpy())


            mapped = domain_model(eeg_data, src_idx, tgt_idx)
            src_mapped.append(mapped.detach().cpu().numpy())


            mapped_reshaped = mapped.view(mapped.size(0), 63, 250)

            # 生成 subject_ids
            subject_id = extract_id_from_string(tgt_subj)  # 确保该函数已定义
            subject_ids = torch.full(
                (mapped_reshaped.size(0),),
                subject_id,
                dtype=torch.long,
                device=device
            )

            # 通过 EEG 模型
            final_features = eeg_model(mapped_reshaped, subject_ids)
            all_final_features.append(final_features)


        tgt_features = np.concatenate(tgt_features, axis=0)
        src_originals = np.concatenate(src_originals, axis=0)
        src_mapped = np.concatenate(src_mapped, axis=0)

        print("Source original shape:", src_originals.shape)  # 例如 (N, 15750)
        print("Mapped features shape:", src_mapped.shape)  # 应与源域一致
        print("Target features shape:", tgt_features.shape)  # 应与源域一致
        # 执行可视化（确保 visualize_features 函数已定义）
        # reduced_mapped = TSNE(n_components=2).fit_transform(src_mapped)
        # plt.scatter(reduced_mapped[:, 0], reduced_mapped[:, 1], c="green", alpha=0.8)
        # plt.title("仅映射特征")
        # plt.show()
        visualize_features(src_originals, src_mapped, tgt_features)

        # 合并最终特征
        concatenated_features = torch.cat(all_final_features, dim=0)
        print(f"Final features shape: {concatenated_features.shape}")

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

        # 定义输出层
        self.output_fc = nn.ModuleList([nn.Linear(shared_dim, output_dim) for output_dim in output_dims])

        # 激活函数和共享层
        self.activation = nn.LeakyReLU(0.01)
        self.shared_fc = nn.Linear(shared_dim, shared_dim)


    def forward(self, x, src_label, tgt_label):
        # 输入层
        input_layer = self.feature_extractor[src_label]
        output_layer = self.output_fc[tgt_label]
        # print(src_label)

        # 执行特征提取（通过输入的线性层）
        shared = input_layer(x)

        # 激活函数
        shared = self.activation(shared)

        # 共享层（共享的FC层 + 批归一化）
        shared = self.shared_fc(shared)
        # shared = self.batch_norm(shared)
        shared = self.activation(shared)

        # Dropout（如果需要）
        # shared = self.dropout(shared)

        # 输出层
        final_output = output_layer(shared)

        return final_output
def main(args):
    # 加载 CLIP 模型
    model, preprocess = load_clip_model()

    # 加载 EEG 跨域模型
    input_dim, shared_dim, output_dim = 63 * 250, 2048, 63 * 250
    subjects = ['sub-01', 'sub-02', 'sub-03', 'sub-04',
                'sub-05', 'sub-06', 'sub-07', 'sub-08']

    src_idx = subjects.index(args.src_subj)
    tgt_idx = subjects.index(args.tgt_subj)

    num_subjects = len(subjects)
    domain_model = SharedLinearMLP([input_dim] * num_subjects, shared_dim, [output_dim] * num_subjects).to(device)
    domain_model.load_state_dict(torch.load(args.domain_model_path, map_location=device))
    print("domain model", args.domain_model_path)

    domain_model.eval()

    # 加载 EEG 特征提取模型
    eeg_model = ATMS(63, 250).to(device)
    print("eeg_model_path", args.eeg_model_path)
    # eeg_model.load_state_dict(torch.load(args.eeg_model_path, map_location=device))
    eeg_model.load_state_dict(torch.load(args.eeg_model_path, map_location=device), strict=False)
    eeg_model.eval()

    # 初始化扩散模型并加载权重
    diffusion_prior = DiffusionPriorUNet(cond_dim=1024, dropout=0.1).to(device)
    pipe = Pipe(diffusion_prior, device=device)

    prior_weight_path = os.path.join(args.prior_path, f"{args.tgt_subj}/diffusion_prior.pt")
    pipe.diffusion_prior.load_state_dict(torch.load(prior_weight_path, map_location=device))
    print(f"Loaded diffusion prior weights from {prior_weight_path}")

    # 加载生成器
    generator = Generator4Embeds(num_inference_steps=4, device=device)

    # 加载源受试者的数据集
    tgt_dataset = EEGDataset(args.data_path, subjects=[args.tgt_subj])
    src_dataset = EEGDataset(args.data_path, subjects=[args.src_subj])
    src_loader = DataLoader(src_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    tgt_loader = DataLoader(tgt_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 提取最终的 EEG 特征
    final_features = extract_final_features(
        src_loader, tgt_loader, domain_model, eeg_model, src_idx, tgt_idx, args.tgt_subj, args.batch_size
    )

    print(f"Final EEG features shape: {final_features.shape}")

    # 创建保存目录
    output_dir = f"generated_imgs/GT_8/{args.src_subj}_to_{args.tgt_subj}_{args.model_name}_test"
    # output_dir = f"generated_imgs/f2_to_8"
    os.makedirs(output_dir, exist_ok=True)

    # emb_eeg_test = torch.load("/data2/xyh1/EEG/feature_PT/ATM_S_eeg_features_sub-02_test.pt")
    # 生成并保存最佳图像
    for k in tqdm(range(200)):  # 200
        eeg_embeds = final_features[k:k + 1].to(device)  # ori embeding
        h = pipe.generate(c_embeds=eeg_embeds, num_inference_steps=50, guidance_scale=5.0)
        max_similarity = -1.0  # 用于追踪最高相似度
        best_image = None  # 用于存储最相似的图像
        for j in range(10):
            image = generator.generate(h.to(dtype=torch.float16))
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_embedding = model.encode_image(image_input).float()

            # 计算与 GT 嵌入 (eeg_embeds) 的相似度
            similarity = cosine_similarity(image_embedding, eeg_embeds).item()
            print(f'Similarity for image {j}: {similarity}')
            # 更新最相似的图像
            if similarity > max_similarity:
                max_similarity = similarity
                best_image = image
        path = f'{output_dir}/{texts[k]}/best.png'
        os.makedirs(os.path.dirname(path), exist_ok=True)  # 确保目录存在
        best_image.save(path)  # 保存最佳图像 200 (200,3,500,500)
        print(f'Best image saved to {path}')


if __name__ == "__main__":
    args = parse_args()
    main(args)
