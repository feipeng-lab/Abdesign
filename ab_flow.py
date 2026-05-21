import os
import csv
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. 词表与基础配置（引入独立的 <PAD> 标记）
# ==========================================
AA_O_VALS = "ACDEFGHIKLMNPQRSTVWY"
PAD_TOKEN = "_"  # 独立填充符
AA = AA_O_VALS + PAD_TOKEN  # 总共 21 个 Token

AA2ID = {a: i for i, a in enumerate(AA)}
PAD_ID = AA2ID[PAD_TOKEN]
VOCAB_SIZE = len(AA)  # 21


# ==========================================
# 2. 数据集读取（支持动态长度与 Padding 掩码）
# ==========================================
class AIRRDataset(Dataset):
    def __init__(self, path, max_n=50000, L=30):
        self.L = L
        df = pd.read_csv(path)
        self.samples = []

        for i in range(len(df)):
            if "cdr3_aa" not in df.columns:
                continue

            seq = str(df["cdr3_aa"].iloc[i])
            if len(seq) < 5 or len(seq) > L:  # 过滤过短或超过最大长度的序列
                continue

            # 确保序列中没有异常字符
            if any(c not in AA_O_VALS for c in seq):
                continue

            v = str(df["v_call"].iloc[i]) if "v_call" in df.columns else "UNK"
            j = str(df["j_call"].iloc[i]) if "j_call" in df.columns else "UNK"

            self.samples.append((seq, v, j))
            if len(self.samples) >= max_n:
                break

        # 动态生成 V/J 基因的编码词表
        self.v2id = {v: i for i, v in enumerate(set([s[1] for s in self.samples]))}
        self.j2id = {j: i for i, j in enumerate(set([s[2] for s in self.samples]))}

    def encode(self, seq):
        # 使用独立的 PAD_TOKEN 填充到固定长度 L
        padded_seq = seq[:self.L].ljust(self.L, PAD_TOKEN)
        x0 = torch.tensor([AA2ID[c] for c in padded_seq], dtype=torch.long)
        
        # 构造有效长度掩码：真实氨基酸为 False，PAD 区域为 True
        # 对应 PyTorch Transformer 的 src_key_padding_mask 格式
        padding_mask = torch.tensor([c == PAD_TOKEN for c in padded_seq], dtype=torch.bool)
        return x0, padding_mask

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, v, j = self.samples[idx]
        x0, padding_mask = self.encode(seq)
        return {
            "x0": x0,
            "padding_mask": padding_mask,
            "v": torch.tensor(self.v2id[v], dtype=torch.long),
            "j": torch.tensor(self.j2id[j], dtype=torch.long)
        }


# ==========================================
# 3. 升级版模型架构（位置编码 + Mask 支撑）
# ==========================================
class AIRRBFN(nn.Module):
    def __init__(self, vocab=VOCAB_SIZE, dim=256, n_v=500, n_j=500, max_len=30):
        super().__init__()
        # 能够无缝接收连续概率向量 [B, L, 21] 
        self.token_emb = nn.Linear(vocab, dim) 
        
        # 可学习的位置编码向量 [1, max_len, dim]
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

        self.v_emb = nn.Embedding(n_v, dim)
        self.j_emb = nn.Embedding(n_j, dim)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=8,
                batch_first=True,
                activation="gelu"
            ),
            num_layers=4
        )
        self.out = nn.Linear(dim, vocab)

    # def forward(self, xt_prob, t, v, j, src_key_padding_mask=None):
    #     # xt_prob: [B, L, 21]
    #     x = self.token_emb(xt_prob)
    #     B, L, _ = x.shape

    #     # 注入位置信息
    #     x = x + self.pos_emb[:, :L, :]

    #     # 注入时间、V基因、J基因等条件特征
    #     t_emb = self.time_mlp(t.view(B, 1)).unsqueeze(1)
    #     v_emb = self.v_emb(v).unsqueeze(1)
    #     j_emb = self.j_emb(j).unsqueeze(1)
    #     cond = t_emb + v_emb + j_emb
        
    #     x = x + cond

    #     # 传入 src_key_padding_mask，使 Attention 机制自动忽略 PAD 部分
    #     h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        
    #     return self.out(h) # 输出未归一化的 Logits [B, L, 21]
    def forward(self, xt_prob, t, v=None, j=None, src_key_padding_mask=None):
        x = self.token_emb(xt_prob)
        B, L, _ = x.shape
        x = x + self.pos_emb[:, :L, :]

        # 时间编码是必须的
        cond = self.time_mlp(t.view(B, 1)).unsqueeze(1)

        # 只有当传入了 v 和 j 时，才叠加基因条件
        if v is not None:
            cond = cond + self.v_emb(v).unsqueeze(1)
        if j is not None:
            cond = cond + self.j_emb(j).unsqueeze(1)
            
        x = x + cond
        h = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.out(h)


# ==========================================
# 4. 加噪流与损失函数
# ==========================================
def sample_xt_soft(x0, t, vocab=VOCAB_SIZE):
    """
    t=0 时为纯噪声（均匀分布），t=1 时为绝对干净的 x0 概率。
    """
    x0_oh = F.one_hot(x0, vocab).float()
    uniform = torch.ones_like(x0_oh) / vocab
    probs = t[:, None, None] * x0_oh + (1 - t[:, None, None]) * uniform
    return probs


# def loss_fn(model, x0, padding_mask, t, v, j):
#     # 1. 构造连续空间的加噪特征
#     xt_soft = sample_xt_soft(x0, t)
    
#     # 2. 前向传播，带上 padding 掩码避免模型学习到无意义的 padding 预测
#     logits = model(xt_soft, t, v, j, src_key_padding_mask=padding_mask)
    
#     # 3. 计算交叉熵损失（利用 reduction="none" 过滤掉 pad 的损失贡献）
#     loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), x0.view(-1), reduction='none')
    
#     # 排除 padding 区域的损失
#     loss_mask = ~padding_mask.view(-1)
#     masked_loss = loss * loss_mask.float()
    
#     return masked_loss.sum() / loss_mask.sum()
def flow_matching_loss(model, x0, padding_mask, t, v, j):
    """
    标准的连续单形流匹配损失函数
    x0: [B, L] 目标离散标签
    """
    B, L = x0.shape
    x0_oh = F.one_hot(x0, VOCAB_SIZE).float() # [B, L, 21]
    uniform = torch.ones_like(x0_oh) / VOCAB_SIZE
    
    # 1. 构造当前时间步 t 的混合状态 xt
    t_col = t[:, None, None] # [B, 1, 1]
    xt = t_col * x0_oh + (1 - t_col) * uniform
    
    # 2. 计算理论上的真实流场 (Target Velocity Field)
    # 根据线性插值：d(xt)/dt = x0_oh - uniform
    target_flow = x0_oh - uniform
    
    # 3. 模型预测当前的流场
    # 模型接收当前的概率分布 xt，并预测演化方向
    # pred_flow = model(xt, t, v, j, src_key_padding_mask=padding_mask) # [B, L, 21]
    # 训练时，有 10% 的概率让模型在没有 V/J 条件下学习
    if torch.rand(1) > 0.1:
        logits = model(xt, t, v, j, src_key_padding_mask=padding_mask)
    else:
        # 传入 None，强迫模型进行无条件重构练习
        logits = model(xt, t, None, None, src_key_padding_mask=padding_mask)
    
    # 4. 计算流场的均方误差（MSE Loss）
    loss = F.mse_loss(logits, target_flow, reduction='none')
    
    # 同样，排除 padding 区域的损失
    loss_mask = ~padding_mask.unsqueeze(-1).expand_as(loss)
    masked_loss = loss * loss_mask.float()
    
    return masked_loss.sum() / loss_mask.sum()

import torch
import torch.nn as nn
import torch.nn.functional as F

# ========================================================
# 2. 彻底打破 Mode Collapse 的全新随机流场采样函数
# ========================================================
@torch.no_grad()
def sample_flow_matching(model, v=None, j=None, B=1, L=30, steps=50, temperature=0.7, device="cuda"):
    """
    自适应条件/无条件的流匹配采样函数
    - 如果传入 v 和 j: 自动根据 v 的属性识别 B 和 device (有条件生成)
    - 如果 v 和 j 为 None: 使用参数传入的 B 和 device (纯盲盒无条件生成)
    """
    # 1. 动态识别运行设备 (Device) 和 样本数量 (Batch Size)
    if v is not None:
        device = v.device
        B = v.shape[0]
    else:
        # 如果是无条件生成，device 默认使用外部传入的参数
        device = torch.device(device if torch.cuda.is_available() else "cpu")
        B = B

    # 2. 初始化单形（Simplex）空间中的随机狄利克雷噪声
    alpha = torch.ones(B, L, VOCAB_SIZE, device=device) * 100.0
    p = torch.distributions.Dirichlet(alpha).sample() 
    
    dt = 1.0 / steps

    for i in range(steps):
        t_val = i / steps
        t = torch.full((B,), t_val, device=device)

        # 3. 前向传播（将 v 和 j 原封不动传给模型，模型内部已支持 None）
        pred_logits = model(p, t, v, j, src_key_padding_mask=None)
        
        # 4. 温度缩放与概率场流动
        x0_pred = F.softmax(pred_logits / temperature, dim=-1)
        flow = (x0_pred - p) / (1.0 - t_val + 1e-5)
        
        # 欧拉步进与边界投影
        p = p + flow * dt
        p = torch.clamp(p, min=0.0, max=1.0)
        p = p / p.sum(dim=-1, keepdim=True)

    # 5. 离散化生成最终 Token
    x_final = torch.zeros(B, L, dtype=torch.long, device=device)
    for b in range(B):
        for l in range(L):
            x_final[b, l] = torch.multinomial(p[b, l], 1)

    seqs = decode(x_final)
    return x_final, seqs

def decode(x):
    """解码张量为氨基酸字符串，并自动裁剪掉尾部的 PAD 字符"""
    results = []
    for seq in x:
        seq_str = "".join([AA[i] for i in seq.tolist()])
        # 遇到第一个填充符则进行截断，还原真实长度
        if PAD_TOKEN in seq_str:
            seq_str = seq_str.split(PAD_TOKEN)[0]
        results.append(seq_str)
    return results


# ==========================================
# 6. 运行主入口
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="ERR220397_Heavy_Bulk.csv")
    parser.add_argument("--mode", type=str, default="sample")  # train / sample
    parser.add_argument("--model_path", type=str, default="airr_flow_v2.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 设定最大序列长度为 30
    MAX_L = 30
    EPOCHS = 100
    ds = AIRRDataset(args.path, L=MAX_L)
    print(f'训练样本数：{len(ds)}')

    model = AIRRBFN(n_v=len(ds.v2id), n_j=len(ds.j2id), max_len=MAX_L).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if args.mode == "train":
        loader = DataLoader(ds, batch_size=64, shuffle=True)
        print("开始训练模型...")
        for epoch in range(EPOCHS):
            total = 0
            model.train()
            for batch in loader:
                x0 = batch["x0"].to(device)
                padding_mask = batch["padding_mask"].to(device)
                v = batch["v"].to(device)
                j = batch["j"].to(device)

                # 随机生成时间步 t (1e-3 到 1-1e-3)
                t = torch.rand(x0.size(0), device=device).clamp(1e-3, 1 - 1e-3)

                loss = flow_matching_loss(model, x0, padding_mask, t, v, j)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total += loss.item()

            print(f"[Epoch {epoch + 1}/ {EPOCHS}] loss = {total / len(loader):.4f}")

        torch.save({"model": model.state_dict(), "v2id": ds.v2id, "j2id": ds.j2id}, args.model_path)
        print("✔ 权重保存成功至", args.model_path)

    elif args.mode == "sample":
        if not os.path.exists(args.model_path):
            print(f"找不到权重文件 {args.model_path}，请先使用 --mode train 进行训练。")
            exit()
            
        ckpt = torch.load(args.model_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        # ===================== 统计模型参数 =====================
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("=" * 60)
        print(f"模型总参数：{total_params:,}")
        print(f"可训练参数：{trainable_params:,}")
        print(f"参数量（百万）：{total_params / 1e6:.2f} M")
        print("=" * 60)
        exit()
        v2id, j2id = ckpt["v2id"], ckpt["j2id"]

        # 模拟 3 个特定 V/J 基因的条件抗体生成任务
        v = torch.tensor([list(v2id.values())[0]] * 100, device=device)
        j = torch.tensor([list(j2id.values())[0]] * 100, device=device)

        print("\n===== AIRR-BFN (V2) 生成的抗体序列 =====")
        # x, seqs = sample_flow_matching(model, v, j, L=MAX_L, steps=50)
        x, seqs = sample_flow_matching(model, None, None, B=100, L=MAX_L, steps=50)
        save_file = "/home/fpk/project/IVD/Abdesign/flow_generated_CDR_sequences.csv"
        # 文件不存在则写入表头
        file_exists = os.path.exists(save_file)

        with open(save_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # 写入表头
            if not file_exists:
                writer.writerow(["抗体编号", "序列长度", "CDR序列"])
            
            # 保存每条序列
            for i, s in enumerate(seqs):
                seq_len = len(s)
                writer.writerow([i+1, seq_len, s])
                
                # 控制台输出
                print(f"[抗体 {i + 1}] 长: {seq_len} | 序列: {s}")

        print(f"\n 全部序列已保存到：{save_file}")
        