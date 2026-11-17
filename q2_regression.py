"""
Q2: 빛도우 세션에서 잔여 자원량에 따른 만족 기준치가 파인만 공식과 수렴하는가?
- 파인만 맛집 공식: threshold ∝ √n / √(n+1)  (PNAS 2026)
- 입력: resource_tier, goal_id, bl_remaining_at_switch, matched_at_switch, progress
- 출력: q_light_avg (빛도우 세션 평균 Q값, 연속값) → 회귀
- 추가: 예측값을 파인만 공식과 비교해 수렴 여부 검증
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import glob
import os

# ── 설정 ─────────────────────────────────────────────────────────────────────
CSV_DIR  = "."
BATCH    = 512
EPOCHS   = 60
LR       = 1e-3
HIDDEN   = [256, 128, 64]
SEED     = 42
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── 파인만 공식 ───────────────────────────────────────────────────────────────
def feynman_threshold(n: np.ndarray) -> np.ndarray:
    """√n / √(n+1) — 잔여 자원 n에 대한 만족 기준치"""
    n = np.asarray(n, dtype=np.float64)
    return np.sqrt(n) / np.sqrt(n + 1)

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
    "resource_tier",
    "goal_id",
    "bl_remaining_at_switch",  # 빛도우 전환 시 잔여량 — 핵심 변수
    "matched_at_switch",       # 전환 시점에 이미 매칭된 수
    "progress",
    "budget_g",
    "budget_l",
]
TARGET_COL = "q_light_avg"   # 빛도우 세션 평균 Q값

def preprocess(df: pd.DataFrame):
    missing = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(f"컬럼 없음: {missing}")

    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])

    # bl_remaining이 0인 행 제외 (전환 안 한 케이스)
    df = df[df["bl_remaining_at_switch"] > 0].copy()

    X = df[FEATURE_COLS].astype(np.float32).values
    y = df[TARGET_COL].astype(np.float32).values

    # 파인만 기준치 계산 (비교용)
    n_remaining = df["bl_remaining_at_switch"].values
    df["feynman_ref"] = feynman_threshold(n_remaining)

    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    print(f"샘플 수: {len(X):,}  |  y 범위: [{y.min():.4f}, {y.max():.4f}]")
    return X, y, scaler, df

# ── Dataset ───────────────────────────────────────────────────────────────────
class BiscuitDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# ── 모델 ──────────────────────────────────────────────────────────────────────
class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden: list, dropout=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ── 학습 루프 ─────────────────────────────────────────────────────────────────
def train(model, loader, opt, criterion):
    model.train()
    total_loss, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        loss = criterion(pred, yb)
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * len(yb)
        n += len(yb)
    return total_loss / n

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, n = 0.0, 0
    preds, targets = [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred = model(xb)
        total_loss += criterion(pred, yb).item() * len(yb)
        preds.append(pred.cpu()); targets.append(yb.cpu())
        n += len(yb)
    preds = torch.cat(preds).numpy().flatten()
    targets = torch.cat(targets).numpy().flatten()
    ss_res = ((targets - preds) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return total_loss / n, r2, preds, targets

# ── 파인만 수렴 검증 ──────────────────────────────────────────────────────────
def feynman_convergence_test(df_full: pd.DataFrame, preds: np.ndarray, te_idx: np.ndarray):
    """
    예측된 q_light_avg vs 파인만 기준치 상관관계 분석
    수렴 기준: Pearson r > 0.7, p < 0.05
    """
    from scipy import stats

    df_te = df_full.iloc[te_idx].copy()
    df_te["pred_q"] = preds
    df_te["feynman_ref"] = feynman_threshold(df_te["bl_remaining_at_switch"].values)

    r, p = stats.pearsonr(df_te["pred_q"], df_te["feynman_ref"])
    print(f"\n── 파인만 수렴 검증 ──")
    print(f"  Pearson r  = {r:.4f}")
    print(f"  p-value    = {p:.4e}")

    if r > 0.7 and p < 0.05:
        print("  ✅ 수렴: MLP 예측값이 파인만 공식과 유의하게 상관")
    elif r > 0.4 and p < 0.05:
        print("  ⚠️  부분 수렴: 약한 상관, 추가 분석 필요")
    else:
        print("  ❌ 비수렴: 파인만 공식과 상관 없음")

    # 구간별 평균 비교
    df_te["tier_bin"] = pd.cut(df_te["bl_remaining_at_switch"], bins=5)
    summary = df_te.groupby("tier_bin", observed=True).agg(
        pred_q_mean=("pred_q", "mean"),
        feynman_mean=("feynman_ref", "mean"),
        count=("pred_q", "count"),
    )
    print("\n  잔여량 구간별 비교:")
    print(summary.to_string())

    return r, p

# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    print(f"Device: {DEVICE}")

    # 1. 데이터
    df = load_csvs(CSV_DIR)
    X, y, scaler, df_processed = preprocess(df)

    idx = np.arange(len(X))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=SEED)

    tr_loader = DataLoader(BiscuitDataset(X[tr_idx], y[tr_idx]), batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(BiscuitDataset(X[te_idx], y[te_idx]), batch_size=BATCH)

    # 2. 모델
    model = MLPRegressor(X.shape[1], HIDDEN).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    criterion = nn.MSELoss()

    # 3. 학습
    best_r2, best_state = -999, None
    for epoch in range(1, EPOCHS + 1):
        tr_loss = train(model, tr_loader, opt, criterion)
        va_loss, va_r2, _, _ = evaluate(model, te_loader, criterion)
        scheduler.step()

        if va_r2 > best_r2:
            best_r2 = va_r2
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d} | train MSE {tr_loss:.6f} | val MSE {va_loss:.6f} R² {va_r2:.4f}")

    # 4. 최종 평가
    model.load_state_dict(best_state)
    _, test_r2, preds, targets = evaluate(model, te_loader, criterion)
    rmse = np.sqrt(((targets - preds) ** 2).mean())
    print(f"\n✅ Best val R²: {best_r2:.4f}  |  Test R²: {test_r2:.4f}  |  RMSE: {rmse:.6f}")

    # 5. 파인만 수렴 검증
    try:
        feynman_convergence_test(df_processed, preds, te_idx)
    except ImportError:
        print("scipy 없음 — pip install scipy 후 재실행")

    # 6. 저장
    torch.save({
        "model_state": best_state,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "feat_cols": FEATURE_COLS,
        "hidden": HIDDEN,
        "test_r2": test_r2,
        "rmse": rmse,
    }, "q2_regression.pt")
    print("\n모델 저장: q2_regression.pt")

if __name__ == "__main__":
    main()