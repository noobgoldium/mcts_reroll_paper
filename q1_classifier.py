"""
Q1: 언제 빛도우로 전환하는 게 최적인가?
- 1줄 완성 후 vs 2줄 완성 후 vs 더 기다리기
- 입력: resource_tier, goal_id, progress, budget_g, budget_l, tier_counts(5), opt_max_tier(12)
- 출력: best_action (분류)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import glob
import os

# ── 설정 ─────────────────────────────────────────────────────────────────────
CSV_DIR   = "."          # CSV 파일 위치 (현재 디렉터리)
BATCH     = 512
EPOCHS    = 50
LR        = 1e-3
HIDDEN    = [256, 128, 64]
SEED      = 42
DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── CSV 로드 ──────────────────────────────────────────────────────────────────
def load_csvs(csv_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(csv_dir, "biscuit_tier_*.csv"))
    if not files:
        raise FileNotFoundError(f"CSV 없음: {csv_dir}/biscuit_tier_*.csv")
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_csv(f))
        except Exception as e:
            print(f"  skip {f}: {e}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"총 {len(df):,}행 로드 ({len(files)}개 파일)")
    return df

# ── 전처리 ────────────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "resource_tier",   # 자원 구간 0~4
    "goal_id",         # 목표 인덱스 0~9
    "progress",        # 현재 진행도
    "matched_count",   # 현재 매칭된 목표 슬롯 수
    "budget_g",        # 남은 일반 비스킷 예산
    "budget_l",        # 남은 빛도우 예산
    "tier_0", "tier_1", "tier_2", "tier_3", "tier_4",  # 현재 슬롯 티어 분포
]
# opt0~11_tier 중 최대값을 feature로 추가
OPT_COLS = [f"opt{i}_tier" for i in range(12)]

TARGET_COL = "best_action"

def preprocess(df: pd.DataFrame):
    # opt 최대티어
    existing_opt = [c for c in OPT_COLS if c in df.columns]
    if existing_opt:
        df["opt_max_tier"] = df[existing_opt].max(axis=1)
    else:
        df["opt_max_tier"] = 0.0

    feat_cols = FEATURE_COLS + ["opt_max_tier"]
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 없음: {missing}")

    required = feat_cols + [TARGET_COL, "goal_id", "resource_tier", "biscuit_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 없음: {missing}")

    # 종료 레코드는 의사결정 행동 분류의 학습 대상에서 제외한다.
    df = df.dropna(subset=required).copy()
    if "terminal_reason" in df.columns:
        df = df[df["terminal_reason"] == 0]
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].between(0, 5)].dropna(subset=feat_cols + [TARGET_COL]).copy()

    # 동일 biscuit episode의 decision row가 양쪽 split에 섞이는 누수를 차단한다.
    groups = (df["resource_tier"].astype(str) + ":" +
              df["goal_id"].astype(str) + ":" +
              df["biscuit_id"].astype(str)).values

    X = df[feat_cols].astype(np.float32).values
    y_raw = df[TARGET_COL].astype(int).values

    # 레이블 인코딩
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print(f"클래스: {le.classes_} ({len(le.classes_)}개)")
    print(f"샘플 수: {len(X):,}  피처: {X.shape[1]}")
    return X, y, le, feat_cols, groups

# ── Dataset ───────────────────────────────────────────────────────────────────
class BiscuitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ── 모델 ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: list, n_class: int, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_class))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train(model, loader, opt, criterion):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * len(yb)
        correct += (pred.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * len(yb)
        correct += (pred.argmax(1) == yb).sum().item()
        n += len(yb)
    return total_loss / n, correct / n

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # 1. 데이터
    df = load_csvs(CSV_DIR)
    X, y, le, feat_cols, groups = preprocess(df)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]
    tr_loader = DataLoader(BiscuitDataset(X_tr, y_tr), batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(BiscuitDataset(X_te, y_te), batch_size=BATCH)

    # 2. 모델
    model = MLP(X.shape[1], HIDDEN, len(le.classes_)).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    # 3. 학습
    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train(model, tr_loader, opt, criterion)
        va_loss, va_acc = evaluate(model, te_loader, criterion)
        scheduler.step()

        if va_acc > best_acc:
            best_acc = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f}")

    # 4. 최종 평가
    model.load_state_dict(best_state)
    _, test_acc = evaluate(model, te_loader, criterion)
    print(f"\n✅ Best val acc: {best_acc:.4f}  |  Test acc: {test_acc:.4f}")

    # 5. 피처 중요도 (permutation importance 간이)
    print("\n── 피처 중요도 (permutation) ──")
    model.eval()
    X_te_t = torch.tensor(X_te, dtype=torch.float32).to(DEVICE)
    y_te_t = torch.tensor(y_te, dtype=torch.long).to(DEVICE)
    with torch.no_grad():
        base_acc = (model(X_te_t).argmax(1) == y_te_t).float().mean().item()

    importances = []
    for i, col in enumerate(feat_cols):
        X_perm = X_te_t.clone()
        idx = torch.randperm(X_perm.shape[0])
        X_perm[:, i] = X_perm[idx, i]
        with torch.no_grad():
            perm_acc = (model(X_perm).argmax(1) == y_te_t).float().mean().item()
        importances.append((col, base_acc - perm_acc))

    importances.sort(key=lambda x: -x[1])
    for col, drop in importances:
        print(f"  {col:<20s} {drop:+.4f}")

    # 6. 저장
    torch.save({
        "model_state": best_state,
        "label_encoder_classes": le.classes_,
        "feat_cols": feat_cols,
        "hidden": HIDDEN,
        "test_acc": test_acc,
    }, "q1_classifier.pt")
    print("\n모델 저장: q1_classifier.pt")

if __name__ == "__main__":
    main()