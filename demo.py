import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

AA = "ACDEFGHIKLMNPQRSTVWY"
AA2ID = {a:i for i,a in enumerate(AA)}


class AIRRDataset(Dataset):

    def __init__(self, path, max_n=50000):

        df = pd.read_csv(path)

        self.samples = []

        for i in range(len(df)):

            if "cdr3_aa" not in df.columns:
                continue

            seq = str(df["cdr3_aa"].iloc[i])

            if len(seq) < 5:
                continue

            if any(c not in AA for c in seq):
                continue

            v = str(df["v_call"].iloc[i]) if "v_call" in df.columns else "UNK"
            j = str(df["j_call"].iloc[i]) if "j_call" in df.columns else "UNK"

            self.samples.append((seq, v, j))

            if len(self.samples) >= max_n:
                break

        # dynamic vocab
        self.v2id = {v:i for i,v in enumerate(set([s[1] for s in self.samples]))}
        self.j2id = {j:i for i,j in enumerate(set([s[2] for s in self.samples]))}

    def encode(self, seq, L=30):

        seq = seq[:L].ljust(L, "A")

        return torch.tensor([AA2ID[c] for c in seq])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        seq, v, j = self.samples[idx]

        return {
            "x0": self.encode(seq),
            "v": torch.tensor(self.v2id[v]),
            "j": torch.tensor(self.j2id[j])
        }

def sample_xt(x0, t, vocab=20):

    x0_oh = F.one_hot(x0, vocab).float()
    uniform = torch.ones_like(x0_oh) / vocab

    probs = (1 - t[:,None,None]) * x0_oh + t[:,None,None] * uniform

    return torch.distributions.Categorical(probs=probs).sample()

class AIRRBFN(nn.Module):

    def __init__(self, vocab=20, dim=256, n_v=500, n_j=500):

        super().__init__()

        self.token_emb = nn.Embedding(vocab, dim)

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

    def forward(self, xt, t, v, j):

        x = self.token_emb(xt)

        B, L, _ = x.shape

        t_emb = self.time_mlp(t.view(B,1)).unsqueeze(1)

        v_emb = self.v_emb(v).unsqueeze(1)
        j_emb = self.j_emb(j).unsqueeze(1)

        cond = t_emb + v_emb + j_emb

        x = x + cond

        h = self.encoder(x)

        return self.out(h)
def sample_xt_soft(x0, t, vocab=20):

    x0_oh = F.one_hot(x0, vocab).float()
    uniform = torch.ones_like(x0_oh) / vocab

    return (1 - t[:,None,None]) * x0_oh + t[:,None,None] * uniform

def loss_fn(model, x0, t, v, j):

    xt_soft = sample_xt_soft(x0, t)

    logits = model(xt_soft.argmax(-1), t, v, j)

    target = x0

    return F.cross_entropy(
        logits.view(-1, 20),
        target.view(-1)
    )

def entropy_loss(logits):

    p = F.softmax(logits, -1)
    return -(p * torch.log(p + 1e-8)).mean()

def train(path):

    device = "cuda"

    ds = AIRRDataset(path)
    loader = DataLoader(ds, batch_size=64, shuffle=True)

    model = AIRRBFN(
        n_v=len(ds.v2id),
        n_j=len(ds.j2id)
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(10):

        total = 0

        for batch in loader:

            x0 = batch["x0"].to(device)
            v = batch["v"].to(device)
            j = batch["j"].to(device)

            t = torch.rand(x0.size(0), device=device).clamp(1e-3, 1-1e-3)

            loss = loss_fn(model, x0, t, v, j)
            # loss = loss + 0.01 * entropy_loss(logits)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total += loss.item()

        print(f"epoch {epoch}: {total/len(loader):.4f}")

@torch.no_grad()
def sample(model, v, j, L=30, T=50):

    B = v.shape[0]
    device = v.device

    x = torch.randint(0,20,(B,L), device=device)

    for t_step in range(T):

        t = torch.full((B,), t_step / T, device=device)

        logits = model(x, t, v, j)

        p = F.softmax(logits, -1)

        x = torch.multinomial(p.view(-1,20),1).view(B,L)

    return x

AA = "ACDEFGHIKLMNPQRSTVWY"

def decode(x):
    return "".join([AA[i] for i in x[0].tolist()])

@torch.no_grad()
def sample_bfn_flow(model, v, j, L=30, T=50, steps=50):

    device = v.device
    B = v.shape[0]

    # p(x) 初始化为 uniform
    p = torch.ones(B, L, 20, device=device) / 20

    dt = 1.0 / steps

    for i in range(steps):

        t = torch.full((B,), i / steps, device=device)

        # 当前 token（soft → hard projection）
        x = torch.multinomial(p.view(-1,20), 1).view(B, L)

        # flow field
        flow = model(x, t, v, j)  # [B,L,20]

        # softmax stabilize flow
        flow = torch.tanh(flow)

        # Euler update
        p = p + flow * dt

        # normalize
        p = F.softmax(p, dim=-1)

    # final sample
    x_final = torch.multinomial(p.view(-1,20), 1).view(B, L)


    seqs = decode(x)

    return x, seqs

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/home/fpk/project/IVD/Abdesign/ERR220397_Heavy_Bulk.csv")
    parser.add_argument("--mode", type=str, default="sample")  # train / sample
    parser.add_argument("--model_path", type=str, default="airr_bfn.pt")

    args = parser.parse_args()

    device = "cuda"

    # ======================
    # load dataset
    # ======================
    ds = AIRRDataset(args.path)

    # ======================
    # build model
    # ======================
    model = AIRRBFN(
        n_v=len(ds.v2id),
        n_j=len(ds.j2id)
    ).to(device)

    # ======================
    # optimizer
    # ======================
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # ======================
    # TRAIN MODE
    # ======================
    if args.mode == "train":

        loader = DataLoader(ds, batch_size=64, shuffle=True)

        for epoch in range(100):

            total = 0

            for batch in loader:

                x0 = batch["x0"].to(device)
                v = batch["v"].to(device)
                j = batch["j"].to(device)

                t = torch.rand(x0.size(0), device=device).clamp(1e-3, 1-1e-3)

                loss = loss_fn(model, x0, t, v, j)

                opt.zero_grad()
                loss.backward()
                opt.step()

                total += loss.item()

            print(f"[Epoch {epoch}] loss = {total/len(loader):.4f}")

        torch.save({
            "model": model.state_dict(),
            "v2id": ds.v2id,
            "j2id": ds.j2id
        }, args.model_path)

        print("✔ model saved to", args.model_path)

    # ======================
    # SAMPLE MODE
    # ======================
    elif args.mode == "sample":

        ckpt = torch.load(args.model_path, map_location=device)

        model.load_state_dict(ckpt["model"])
        model.eval()

        v2id = ckpt["v2id"]
        j2id = ckpt["j2id"]

        # pick V/J
        v = torch.tensor([list(v2id.values())[0]], device=device)
        j = torch.tensor([list(j2id.values())[0]], device=device)

        print("\n===== AIRR-BFN GENERATED ANTIBODIES =====\n")

        x, seqs = sample_bfn_flow(model, v, j, L=25, T=50)

        # for i, s in enumerate(seqs):
        #     print(f"[Seq {i+1}] {s}")
        print(seqs)