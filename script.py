"""
Comparação de YOLOv8s, YOLOv10s e YOLO26s com validação cruzada k-fold
no dataset D-Fire (Ultralytics).

Requer ultralytics recente (YOLO26 exige versão >= 8.4):
    pip install -U ultralytics

Etapas:
  1. Gera os folds (uma vez; reaproveita se já existirem).
  2. Treina cada modelo em cada fold com os MESMOS hiperparâmetros.
  3. Salva resultados por fold (retomável), resumo (média ± desvio) e gráficos.
"""

import pandas as pd
import yaml
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from sklearn.model_selection import KFold

from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.utils import check_det_dataset


# ==========================================
# CONFIGURAÇÃO
# ==========================================

DATA_YAML = "D-Fire/data.yaml"

MODELS = ["yolov8s.pt", "yolov10s.pt", "yolo26s.pt"]

KSPLIT = 5
SPLIT_SEED = 20

# Hiperparâmetros idênticos para todos os modelos
EPOCHS = 100
BATCH = 16
IMGSZ = 640
WORKERS = 0
TRAIN_SEED = 0

PROJECT = "kfold_yolo_comparacao"

# Teste rápido do pipeline: True = só 1 fold e 3 épocas
QUICK_TEST = False


# ==========================================
# 1. GERAÇÃO DOS FOLDS
# ==========================================

def build_folds(data, dataset_path, save_path):
    """Cria os .txt/.yaml de cada fold. Retorna a lista de yamls."""

    folds = [f"split_{n}" for n in range(1, KSPLIT + 1)]
    ds_yamls = [save_path / f"{s}.yaml" for s in folds]

    if all(y.exists() for y in ds_yamls):
        print("Folds já existem, reaproveitando:", save_path)
        return ds_yamls

    sources = []
    for split in ("train", "val"):
        source = data.get(split)
        if source:
            sources.extend(source if isinstance(source, list) else [source])

    dataset = YOLODataset(sources, data=data, augment=False)
    images = [Path(p).resolve() for p in dataset.im_files]
    print("Total de imagens:", len(images))

    classes = data["names"]
    cls_idx = sorted(classes.keys())

    # Matriz imagem x classe (contagem de objetos)
    rows = []
    for label in dataset.labels:
        counter = Counter(label["cls"].flatten().astype(int))
        rows.append([counter.get(c, 0) for c in cls_idx])
    labels_df = pd.DataFrame(rows, columns=cls_idx, index=images, dtype=float)

    kf = KFold(n_splits=KSPLIT, shuffle=True, random_state=SPLIT_SEED)
    kfolds = list(kf.split(labels_df))

    folds_df = pd.DataFrame(index=labels_df.index, columns=folds)
    fold_lbl_distrb = pd.DataFrame(index=folds, columns=cls_idx)

    for i, (train_idx, val_idx) in enumerate(kfolds, start=1):
        folds_df.loc[labels_df.index[train_idx], f"split_{i}"] = "train"
        folds_df.loc[labels_df.index[val_idx], f"split_{i}"] = "val"

        train_totals = labels_df.iloc[train_idx].sum()
        val_totals = labels_df.iloc[val_idx].sum()
        fold_lbl_distrb.loc[f"split_{i}"] = val_totals / (train_totals + 1e-7)

    save_path.mkdir(parents=True, exist_ok=True)

    for split in folds:
        for partition in ("train", "val"):
            split_images = folds_df.index[folds_df[split] == partition]
            txt_file = save_path / f"{split}_{partition}.txt"
            txt_file.write_text(
                "\n".join(str(p) for p in split_images) + "\n",
                encoding="utf-8",
            )

        fold_data = {
            "path": str(dataset_path),
            "train": str((save_path / f"{split}_train.txt").resolve()),
            "val": str((save_path / f"{split}_val.txt").resolve()),
            "names": classes,
            "nc": len(classes),
        }
        with open(save_path / f"{split}.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(fold_data, f, sort_keys=False)

    folds_df.to_csv(save_path / "kfold_datasplit.csv")
    fold_lbl_distrb.to_csv(save_path / "kfold_label_distribution.csv")

    print("Arquivos dos folds salvos em:", save_path)
    print("\nProporção val/train por classe (ideal ~0.25):")
    print(fold_lbl_distrb)

    return ds_yamls


# ==========================================
# 2. TREINO E AVALIAÇÃO
# ==========================================

def run_experiments(ds_yamls, results_csv):
    """Treina cada modelo em cada fold. Retoma de onde parou."""

    if results_csv.exists():
        df = pd.read_csv(results_csv)
        rows = df.to_dict("records")
        done = {(r["model"], r["fold"]) for r in rows}
    else:
        rows, done = [], set()

    epochs = 3 if QUICK_TEST else EPOCHS
    yamls = ds_yamls[:1] if QUICK_TEST else ds_yamls

    for model_name in MODELS:
        stem = Path(model_name).stem

        for k, dataset_yaml in enumerate(yamls, start=1):

            if (stem, k) in done:
                print(f"Pulando {stem} fold {k} (já concluído)")
                continue

            print(f"\n{'=' * 60}")
            print(f"Modelo: {stem} | Fold {k}/{len(yamls)}")
            print(f"{'=' * 60}")

            model = YOLO(model_name)

            model.train(
                data=str(dataset_yaml),
                epochs=epochs,
                batch=BATCH,
                imgsz=IMGSZ,
                workers=WORKERS,
                project=PROJECT,
                name=f"{stem}_fold{k}",
                seed=TRAIN_SEED,
                deterministic=True,
                exist_ok=True,
            )

            # Avaliação final com os melhores pesos no conjunto de validação do fold
            m = model.val(data=str(dataset_yaml), imgsz=IMGSZ, workers=WORKERS)

            row = {
                "model": stem,
                "fold": k,
                "precision": float(m.box.mp),
                "recall": float(m.box.mr),
                "mAP50": float(m.box.map50),
                "mAP50-95": float(m.box.map),
                "inference_ms": float(m.speed["inference"]),
            }

            # mAP por classe (ex.: fumaça / fogo)
            for idx, ap in zip(m.box.ap_class_index, m.box.ap50):
                row[f"AP50_{m.names[int(idx)]}"] = float(ap)

            rows.append(row)
            pd.DataFrame(rows).to_csv(results_csv, index=False)

    return pd.DataFrame(rows)


# ==========================================
# 3. RESUMO E GRÁFICOS
# ==========================================

def summarize(df, out_dir):
    metrics = ["precision", "recall", "mAP50", "mAP50-95", "inference_ms"]
    metrics += [c for c in df.columns if c.startswith("AP50_")]

    resumo = df.groupby("model")[metrics].agg(["mean", "std"])
    resumo.to_csv(out_dir / "resumo_modelos.csv")

    print("\n" + "=" * 60)
    print("RESUMO (média ± desvio padrão entre os folds)")
    print("=" * 60)
    for model_name, g in df.groupby("model"):
        print(f"\n{model_name}")
        for m in metrics:
            print(f"  {m:>14}: {g[m].mean():.4f} ± {g[m].std():.4f}")

    # Boxplots por métrica
    plot_metrics = ["mAP50", "mAP50-95", "precision", "recall"]
    fig, axes = plt.subplots(1, len(plot_metrics), figsize=(4.5 * len(plot_metrics), 4.5))

    order = [Path(m).stem for m in MODELS]
    for ax, metric in zip(axes, plot_metrics):
        data = [df.loc[df["model"] == m, metric].values for m in order]
        ax.boxplot(data, labels=order, showmeans=True)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=20)

    fig.suptitle(f"Comparação de modelos - {KSPLIT}-fold cross-validation")
    fig.tight_layout()
    fig.savefig(out_dir / "boxplot_metricas.png", dpi=150)
    plt.close(fig)

    # Trade-off precisão x velocidade
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model_name, g in df.groupby("model"):
        ax.errorbar(
            g["inference_ms"].mean(),
            g["mAP50-95"].mean(),
            xerr=g["inference_ms"].std(),
            yerr=g["mAP50-95"].std(),
            fmt="o",
            capsize=4,
            label=model_name,
        )
    ax.set_xlabel("Tempo de inferência (ms/img)")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Precisão x velocidade")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tradeoff_velocidade.png", dpi=150)
    plt.close(fig)

    print("\nArquivos salvos em:", out_dir)


def main():
    yaml_file = Path(DATA_YAML).resolve()
    data = check_det_dataset(str(yaml_file))
    dataset_path = Path(data["path"]).resolve()

    print("Arquivo YAML:", yaml_file)
    print("Raiz do dataset:", dataset_path)

    save_path = (dataset_path.parent / f"{KSPLIT}-Fold_Cross-val").resolve()
    ds_yamls = build_folds(data, dataset_path, save_path)

    out_dir = Path(PROJECT)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = run_experiments(ds_yamls, out_dir / "resultados_por_fold.csv")
    summarize(df, out_dir)

    print("\nComparação K-Fold concluída!")


if __name__ == "__main__":
    main()