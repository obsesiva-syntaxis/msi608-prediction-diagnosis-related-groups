"""Generate publication figures for the top-20 GRD dataset (spanish `es` or english `en`).

Usage:  python generar_figuras.py [es|en]
Spanish figures go to dataset/figs; English ones to paper_ieee_access/figs.
"""
import json
import os
import sys
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.metrics import roc_auc_score, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping

SEED = 42
TS = "20260905-203334"
BASE = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE, "dataset")
DATA_PATH = os.path.join(DATASET_DIR, "dataset_elpino.csv")
IR_GRD = os.path.join(DATASET_DIR, "IR-GRD V3.1 CON PRECIOS FONASA 2016.xlsx")
CIE10 = os.path.join(DATASET_DIR, "CIE-10.xlsx")
CIE9 = os.path.join(DATASET_DIR, "CIE-9.xlsx")
MODEL_DIR = os.path.join(BASE, "models")
ES_FIGS = os.path.join(BASE, "figs")

LANG = "en" if (len(sys.argv) > 1 and sys.argv[1] == "en") else "es"
FIG_DIR = ES_FIGS if LANG == "es" else os.path.join(BASE, "..", "paper_ieee_access", "figs")
HIST_PATH = os.path.join(ES_FIGS, "historia_lstm_fig5.json")


L = {
    "es": {
        "fig1_title": "Figura 1 — Distribución de los 20 GRD más frecuentes",
        "discharges": "Egresos",
        "fig2_title": "Figura 2 — Edad y sexo por GRD",
        "male": "Hombre", "female": "Mujer",
        "fig3_title": "Figura 3 — Códigos clínicos más frecuentes",
        "fig3a": "a) Diagnósticos (CIE-10) más frecuentes",
        "fig3b": "b) Procedimientos (CIE-9-CM) más frecuentes",
        "frequency": "Frecuencia",
        "fig4_title": "Figura 4 — Arquitectura del modelo ganador (LSTM, 178 692 parámetros)",
        "fig4_in": "Entrada\ncódigos clínicos\n(secuencia ≤ 65 tokens)",
        "fig4_meta": "Meta: Edad, Sexo\n(2 variables, estandarizadas)",
        "fig4_lstm": "LSTM\n(64 unidades)",
        "fig4_note": ("Entrenamiento: sparse_categorical_crossentropy + Adam, batch 64, máx. 60 épocas, "
                      "early stopping sobre val_loss (paciencia 8)."),
        "train": "Entrenamiento", "val": "Validación", "epoch": "Época", "loss": "Pérdida", "acc": "Exactitud",
        "fig5_title": "Figura 5 — Entrenamiento de la LSTM (mejor val_loss {vl:.4f} en época {e})",
        "fig6_title": "Figura 6 — Matriz de confusión en test — LSTM (acc {a:.3f})",
        "fig7_title": "Figura 7 — AUC por GRD en test — LSTM",
        "fig8_title": "Figura 8 — Importancia de las 20 características principales (Random Forest)",
        "fig8_sub": "Códigos sin etiqueta = procedimientos CIE-9-CM",
        "importance": "Importancia",
        "extra_title": "Extra — Cantidad de códigos por egreso",
        "extra_a": "a) Diagnósticos por egreso", "extra_b": "b) Procedimientos por egreso",
        "n_diag": "Nº de diagnósticos", "n_proc": "Nº de procedimientos",
    },
    "en": {
        "fig1_title": "Distribution of the 20 most frequent GRDs",
        "discharges": "Discharges",
        "fig2_title": "Age and sex by GRD",
        "male": "Male", "female": "Female",
        "fig3_title": "Most frequent clinical codes",
        "fig3a": "a) Most frequent diagnoses (ICD-10)",
        "fig3b": "b) Most frequent procedures (ICD-9-CM)",
        "frequency": "Frequency",
        "fig4_title": "Architecture of the winning model (LSTM, 178,692 parameters)",
        "fig4_in": "Input\nclinical codes\n(sequence \u2264 65 tokens)",
        "fig4_meta": "Meta: Age, Sex\n(2 standardized variables)",
        "fig4_lstm": "LSTM\n(64 units)",
        "fig4_note": ("Training: sparse_categorical_crossentropy + Adam, batch 64, up to 60 epochs, "
                      "early stopping on val_loss (patience 8)."),
        "train": "Training", "val": "Validation", "epoch": "Epoch", "loss": "Loss", "acc": "Accuracy",
        "fig5_title": "LSTM training (best val_loss {vl:.4f} at epoch {e})",
        "fig6_title": "Confusion matrix on test set \u2014 LSTM (acc {a:.3f})",
        "fig7_title": "AUC per GRD on test set \u2014 LSTM",
        "fig8_title": "Importance of the top-20 features (Random Forest)",
        "fig8_sub": "Unlabeled codes = ICD-9-CM procedures",
        "importance": "Importance",
        "extra_title": "Extra \u2014 Number of codes per discharge",
        "extra_a": "a) Diagnoses per discharge", "extra_b": "b) Procedures per discharge",
        "n_diag": "No. of diagnoses", "n_proc": "No. of procedures",
    },
}[LANG]

sns.set_theme(style="whitegrid")
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200, "font.size": 10})


def load_data():
    with open(DATA_PATH, encoding="utf-8") as f:
        header = f.readline().strip().split(";")

    def clean_colname(col):
        col = col.split("-")[0].strip()
        if col.startswith("Diag") or col.startswith("Proc"):
            partes = col.split(" ")
            col = partes[0] + partes[1]
        return col

    features = [clean_colname(c) for c in header]
    diag_cols = [c for c in features if c.startswith("Diag")]
    proc_cols = [c for c in features if c.startswith("Proced")]

    df = pd.read_csv(DATA_PATH, sep=";", encoding="utf-8", names=features, skiprows=1, dtype=str)

    def clean_codes(s):
        return s.str.split("-", n=1).str[0].str.strip()

    for c in diag_cols + proc_cols:
        df[c] = clean_codes(df[c])

    df["Edad"] = pd.to_numeric(df["Edad en años"], errors="coerce")
    df["Sexo"] = df["Sexo (Desc)"].map({"Mujer": 1, "Hombre": 0})
    df["GRD"] = clean_codes(df["GRD"])
    df = df[diag_cols + proc_cols + ["Edad", "Sexo", "GRD"]]
    df = df.dropna(subset=["Edad"]).copy()
    df["Edad"] = df["Edad"].astype(int)

    mapper = {}
    try:
        grd_tabla = pd.read_excel(IR_GRD)
        mapper = {str(int(c)).zfill(6): n for c, n in zip(grd_tabla["IR-GRD CÓDIGO"], grd_tabla["NOMBRE DEL GRUPO GRD"])}
    except Exception:
        pass

    top_grd = df["GRD"].value_counts().head(20)
    top_names = top_grd.index.tolist()
    data = df[df["GRD"].isin(top_names)].copy()

    desc_short = {}
    for c in top_names:
        d = mapper.get(c, "?")
        desc_short[c] = (d[:48] + "…") if len(d) > 48 else d
    return data, diag_cols, proc_cols, mapper, top_grd, desc_short


def build_features(data, diag_cols, proc_cols):
    data[diag_cols + proc_cols] = data[diag_cols + proc_cols].fillna("")

    freq_d = data[diag_cols].apply(pd.Series.value_counts).sum(axis=1).drop(index="", errors="ignore")
    freq_p = data[proc_cols].apply(pd.Series.value_counts).sum(axis=1).drop(index="", errors="ignore")
    diag_sorted = freq_d.sort_values(ascending=False).index.tolist()
    proc_sorted = freq_p.sort_values(ascending=False).index.tolist()
    n_diag, n_proc = len(diag_sorted), len(proc_sorted)
    VOCAB = n_diag + n_proc + 1

    tok_d = {c: i + 1 for i, c in enumerate(diag_sorted)}
    tok_p = {c: n_diag + i + 1 for i, c in enumerate(proc_sorted)}

    token_rows = []
    for _, r in data.iterrows():
        toks = [tok_d[r[c]] for c in diag_cols if r[c] in tok_d]
        toks += [tok_p[r[c]] for c in proc_cols if r[c] in tok_p]
        token_rows.append(toks)

    def multi_hot(tokens, size=VOCAB):
        v = np.zeros(size, dtype=np.float32)
        if tokens:
            v[np.asarray(tokens, dtype=int)] = 1.0
        return v

    X_seq = pad_sequences(token_rows, padding="post")
    X_codes = np.stack([multi_hot(t) for t in token_rows])
    meta = np.column_stack([data["Edad"].to_numpy(dtype=float), data["Sexo"].to_numpy(dtype=float)])

    le = LabelEncoder()
    y = le.fit_transform(data["GRD"])

    token_names = [""] + diag_sorted + proc_sorted

    idx = np.arange(len(data))
    idx_train, idx_tmp, y_train, y_tmp = train_test_split(idx, y, test_size=0.30, stratify=y, random_state=SEED)
    idx_val, idx_test, _, _ = train_test_split(idx_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)

    sc = StandardScaler().fit(meta[idx_train])
    meta_s = sc.transform(meta)

    Xclas_train = np.hstack([X_codes[idx_train], meta_s[idx_train]])
    Xclas_test = np.hstack([X_codes[idx_test], meta_s[idx_test]])

    Xseq = {"train": X_seq[idx_train], "val": X_seq[idx_val], "test": X_seq[idx_test]}
    meta_p = {"train": meta_s[idx_train], "val": meta_s[idx_val], "test": meta_s[idx_test]}
    splits = {"train": idx_train, "val": idx_val, "test": idx_test}

    return Xseq, meta_p, Xclas_train, Xclas_test, y, le, token_names, splits, data


def build_lstm(vocab_size, seq_len, n_classes, n_meta=2):
    seq_in = Input(shape=(seq_len,), name="secuencia")
    meta_in = Input(shape=(n_meta,), name="meta")
    emb = Embedding(vocab_size, 64, mask_zero=True, name="embedding")(seq_in)
    lstm = LSTM(64, return_sequences=False, name="lstm")(emb)
    mi = Dense(16, activation="relu", name="meta_dense")(meta_in)
    x = Concatenate(name="concat")([lstm, mi])
    x = Dropout(0.3)(x)
    out = Dense(n_classes, activation="softmax", name="salida")(x)
    return Model(inputs=[seq_in, meta_in], outputs=out)


def fig1_distribucion(data, desc_short):
    vc = data["GRD"].value_counts()
    labels = [f"{c} — {desc_short[c]}" for c in vc.index[::-1]]
    plt.figure(figsize=(13, 6))
    bars = plt.barh(labels, vc.values[::-1], color=plt.cm.viridis(np.linspace(0.15, 0.9, len(vc))))
    plt.xlabel(L["discharges"]); plt.title(L["fig1_title"])
    for b, v in zip(bars, vc.values[::-1]):
        plt.text(v + 4, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8)
    plt.xlim(0, vc.max() * 1.08)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig1_distribucion.png")); plt.close()


def fig2_edad_sexo(data, desc_short):
    order = data.groupby("GRD")["Edad"].median().sort_values().index
    fig, ax = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={"width_ratios": [3, 1]})
    sns.boxplot(data=data, x="Edad", y="GRD", order=order, hue="GRD", palette="viridis", legend=False, ax=ax[0])
    ax[0].set_yticks(range(len(order)))
    ax[0].set_yticklabels([f"{c} — {desc_short[c]}" for c in order], fontsize=7.5)
    pct_f = data["Sexo"].mean() * 100
    counts = data["Sexo"].value_counts().sort_index()
    ax[1].bar([L["male"], L["female"]], [counts.get(0, 0), counts.get(1, 0)], color=["#457b9d", "#e76f51"])
    ax[1].set_ylabel(L["discharges"])
    ax[1].set_title(f"Sex ({(100 - pct_f):.1f}% / {pct_f:.1f}%)")
    fig.suptitle(L["fig2_title"], y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig2_edad_sexo.png")); plt.close()


def fig3_codigos(data, diag_cols, proc_cols):
    cnt_diag = Counter(data[diag_cols].to_numpy().flatten().tolist())
    cnt_proc = Counter(data[proc_cols].to_numpy().flatten().tolist())
    d10 = {}
    d9 = {}
    try:
        cie10 = pd.read_excel(CIE10)
        d10 = dict(zip(cie10["Código"].astype(str), cie10["Descripción"]))
    except Exception:
        pass
    try:
        cie9 = pd.read_excel(CIE9)
        d9 = dict(zip(cie9["Código"].astype(str), cie9["Descripción"]))
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))
    for ax, cnt, dct, title, color in [
        (axes[0], cnt_diag, d10, L["fig3a"], "#2a9d8f"),
        (axes[1], cnt_proc, d9, L["fig3b"], "#e76f51"),
    ]:
        top = [t for t in cnt.most_common(12) if t[0]]
        labels = [f"{c} — {dct.get(c, '')[:44]}" for c, _ in top[::-1]]
        bars = ax.barh(labels, [v for _, v in top[::-1]], color=color)
        for b, (c, v) in zip(bars, top[::-1]):
            ax.text(v + 20, b.get_y() + b.get_height() / 2, str(v), va="center", fontsize=8)
        ax.set_xlabel(L["frequency"]); ax.set_title(title)
        ax.set_xlim(0, max(v for _, v in top) * 1.08)
    fig.suptitle(L["fig3_title"], y=1.0)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig3_codigos.png")); plt.close()


def fig4_arquitectura(vocab_size, seq_len, n_classes):
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.axis("off")
    ax.set_xlim(0, 19)
    ax.set_ylim(-3.7, 8.6)

    def box(x, y, w, h, text, fc="#f0f0f0", ec="#333333", fs=9, bold=False):
        ax.add_patch(matplotlib.patches.Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=1.5, zorder=3))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", zorder=4)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.6, mutation_scale=16))

    y0 = 1.2
    h = 0.8
    ax.text(0.02, y0 + 6.6, L["fig4_title"], fontsize=12, fontweight="bold")

    box(0.2, y0, 2.1, h, L["fig4_in"], fc="#dbe7f3", fs=8.5)
    box(0.2, y0 - 2.0, 2.1, h, L["fig4_meta"], fc="#dbe7f3", fs=8.5)
    box(3.2, y0, 2.4, h, "Embedding\n(2 250 \u2192 64, mask_zero)", fc="#ffe8cc", fs=8.5)
    box(6.5, y0, 2.1, h, L["fig4_lstm"], fc="#fff3bf", fs=8.5)
    box(3.2, y0 - 2.0, 2.4, h, "Dense(16, ReLU)", fc="#ffe8cc", fs=8.5)
    box(9.5, y0, 2.3, h, "Concatenate\n(64 + 16 = 80)", fc="#d8f5e3", fs=8.5)
    box(12.7, y0, 2.1, h, "Dropout(0.3)", fc="#e9ecef", fs=8.5)
    box(15.4, y0, 2.6, h, "Dense(20, softmax)\np(GRD)", fc="#f8d7da", bold=True, fs=8.5)

    arrow(2.3, y0 + h / 2, 3.2, y0 + h / 2)
    arrow(5.7, y0 + h / 2, 6.5, y0 + h / 2)
    arrow(2.3, y0 - 2.0 + h / 2, 3.2, y0 - 2.0 + h / 2)
    arrow(5.6, y0 - 2.0 + h / 2, 9.5, y0 + h / 2)
    arrow(8.7, y0 + h / 2, 9.5, y0 + h / 2)
    arrow(11.9, y0 + h / 2, 12.7, y0 + h / 2)
    arrow(14.9, y0 + h / 2, 15.4, y0 + h / 2)

    ax.text(0.2, y0 - 4.5, L["fig4_note"], fontsize=8.5, style="italic")
    plt.savefig(os.path.join(FIG_DIR, "fig4_arquitectura.png"), bbox_inches="tight", pad_inches=0.15)
    plt.close()


def fig5_entrenamiento(Xseq, meta_p, y, n_classes, vocab_size):
    if os.path.exists(HIST_PATH):
        with open(HIST_PATH) as f:
            hist_json = json.load(f)
    else:
        seq_len = Xseq["train"].shape[1]
        model = build_lstm(vocab_size, seq_len, n_classes)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        es = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True)
        hist = model.fit(
            {"secuencia": Xseq["train"], "meta": meta_p["train"]}, y[X["train"]],
            validation_data=({"secuencia": Xseq["val"], "meta": meta_p["val"]}, y[X["val"]]),
            epochs=60, batch_size=64, callbacks=[es], verbose=0)
        hist_json = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        with open(HIST_PATH, "w") as f:
            json.dump(hist_json, f)

    best_e = int(np.argmin(hist_json["val_loss"])) + 1
    best_vl = float(np.min(hist_json["val_loss"]))
    eps = np.arange(1, len(hist_json["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.5))
    axes[0].plot(eps, hist_json["loss"], label=L["train"], color="#2a9d8f")
    axes[0].plot(eps, hist_json["val_loss"], label=L["val"], color="#e76f51")
    axes[0].set_xlabel(L["epoch"]); axes[0].set_ylabel(L["loss"])
    axes[0].set_title("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(eps, hist_json["accuracy"], label=L["train"], color="#2a9d8f")
    axes[1].plot(eps, hist_json["val_accuracy"], label=L["val"], color="#e76f51")
    axes[1].set_xlabel(L["epoch"]); axes[1].set_ylabel(L["acc"])
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.suptitle(L["fig5_title"].format(vl=best_vl, e=best_e), y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig5_entrenamiento.png")); plt.close()
    print(f"Fig 5: best val_loss {best_vl:.4f} at epoch {best_e}")


def fig6_confusion(y_test, y_pred, le):
    plt.figure(figsize=(11.5, 9.5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=le.classes_, cmap="Blues", values_format="d",
        colorbar=False, xticks_rotation=45)
    plt.title(L["fig6_title"].format(a=np.mean(y_pred == y_test)))
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig6_confusion.png")); plt.close()


def fig7_auc(y_test, y_prob, le):
    aucs = [roc_auc_score((y_test == c).astype(int), y_prob[:, c]) for c in range(len(le.classes_))]
    order = np.argsort(aucs)
    labels = [le.classes_[c] for c in order]
    vals = [aucs[c] for c in order]
    plt.figure(figsize=(8.5, 6.5))
    bars = plt.barh(labels, vals, color=plt.cm.Blues(np.linspace(0.45, 0.9, len(vals))))
    for b, v in zip(bars, vals):
        plt.text(v - 0.004, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center", ha="right", fontsize=8)
    plt.axvline(np.mean(aucs), color="#e63946", linestyle="--", lw=1.5)
    plt.text(np.mean(aucs) + 0.001, 0.5, f"AUC macro = {np.mean(aucs):.3f}", rotation=90, va="center", color="#e63946", fontsize=9)
    plt.xlabel("AUC (one-vs-rest)"); plt.xlim(0.9, 1.005)
    plt.title(L["fig7_title"])
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig7_auc.png")); plt.close()


def fig8_importancias(rf, token_names):
    imp = rf.feature_importances_
    cols = token_names + ["Edad", "Sexo"]
    fi = pd.DataFrame({"feature": cols, "importancia": imp}).sort_values("importancia", ascending=False).head(20)
    fi = fi.iloc[::-1]
    plt.figure(figsize=(9, 6.5))
    bars = plt.barh(fi["feature"], fi["importancia"], color=plt.cm.viridis(np.linspace(0.15, 0.9, len(fi))))
    for b, v in zip(bars, fi["importancia"]):
        plt.text(v + 0.0005, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center", fontsize=8)
    plt.xlabel(L["importance"])
    plt.suptitle(L["fig8_title"], y=1.0, fontweight="bold")
    plt.title(L["fig8_sub"])
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig8_importancias.png")); plt.close()


def fig_extra_ncodigos(data, diag_cols, proc_cols):
    n_diag = data[diag_cols].apply(lambda s: (s != "").sum(), axis=1)
    n_proc = data[proc_cols].apply(lambda s: (s != "").sum(), axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(n_diag, bins=np.arange(-0.5, 36.5, 1), edgecolor="white", color="#2a9d8f")
    axes[0].set_xlabel(L["n_diag"]); axes[0].set_ylabel(L["discharges"])
    axes[0].set_title(L["extra_a"])
    axes[1].hist(n_proc, bins=np.arange(-0.5, 30.5, 1), edgecolor="white", color="#e76f51")
    axes[1].set_xlabel(L["n_proc"]); axes[1].set_ylabel(L["discharges"])
    axes[1].set_title(L["extra_b"])
    fig.suptitle(L["extra_title"], y=1.02)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "fig_extra_ncodigos.png")); plt.close()


if __name__ == "__main__":
    os.makedirs(FIG_DIR, exist_ok=True)
    print("LANG:", LANG, "->", os.path.abspath(FIG_DIR))
    data, diag_cols, proc_cols, mapper, top_grd, desc_short = load_data()
    print("Discharges (top-20):", len(data))

    fig1_distribucion(data, desc_short)
    fig2_edad_sexo(data, desc_short)
    fig3_codigos(data, diag_cols, proc_cols)
    fig_extra_ncodigos(data, diag_cols, proc_cols)

    Xseq, meta_p, Xclas_train, Xclas_test, y, le, token_names, X, data = build_features(data, diag_cols, proc_cols)
    print("Vocab:", len(token_names), "| seq_len:", Xseq["train"].shape[1], "| split:", [len(v) for v in X.values()])

    modelo_lstm = build_lstm(len(token_names), Xseq["train"].shape[1], len(le.classes_))
    modelo_lstm.load_weights(os.path.join(MODEL_DIR, f"multiclase_lstm_{TS}.keras"))
    rf = joblib.load(os.path.join(MODEL_DIR, f"multiclase_rf_{TS}.joblib"))

    y_test = y[X["test"]]
    y_pred = np.argmax(modelo_lstm.predict({"secuencia": Xseq["test"], "meta": meta_p["test"]}, verbose=0), axis=1)
    y_prob = modelo_lstm.predict({"secuencia": Xseq["test"], "meta": meta_p["test"]}, verbose=0)

    fig4_arquitectura(len(token_names), Xseq["train"].shape[1], len(le.classes_))
    fig5_entrenamiento(Xseq, meta_p, y, len(le.classes_), len(token_names))
    fig6_confusion(y_test, y_pred, le)
    fig7_auc(y_test, y_prob, le)
    fig8_importancias(rf, token_names)

    print("Figures written to", os.path.abspath(FIG_DIR))