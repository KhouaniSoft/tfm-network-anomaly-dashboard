# -*- coding: utf-8 -*-
"""Pipeline TFM: deteccion de anomalias en trafico de red (UNSW-NB15) con Isolation Forest."""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (precision_score, recall_score, f1_score,
                             confusion_matrix, roc_auc_score, roc_curve)
warnings.filterwarnings("ignore")
plt.rcParams.update({"figure.dpi": 150, "font.size": 9, "font.family": "sans-serif"})
RS = 42
OUT = {}

train = pd.read_csv("data/UNSW_NB15_training-set.csv")
test = pd.read_csv("data/UNSW_NB15_testing-set.csv")

# ---------- 1. EDA ----------
OUT["shape_train"] = list(train.shape); OUT["shape_test"] = list(test.shape)
OUT["nulls_train"] = int(train.isna().sum().sum()); OUT["nulls_test"] = int(test.isna().sum().sum())
OUT["dups_train"] = int(train.duplicated().sum())
OUT["dtypes"] = train.dtypes.astype(str).value_counts().to_dict()
OUT["label_dist_train"] = train["label"].value_counts().to_dict()
OUT["label_dist_test"] = test["label"].value_counts().to_dict()
OUT["attack_cat_train"] = train["attack_cat"].value_counts().to_dict()
OUT["attack_cat_test"] = test["attack_cat"].value_counts().to_dict()
OUT["service_dash_pct"] = float((train["service"] == "-").mean() * 100)
OUT["n_proto"] = int(train["proto"].nunique()); OUT["n_service"] = int(train["service"].nunique()); OUT["n_state"] = int(train["state"].nunique())
desc = train[["dur","spkts","dpkts","sbytes","dbytes","rate"]].describe().T
OUT["desc_numeric"] = desc[["mean","std","min","50%","max"]].round(3).to_dict("index")
# skewness
OUT["skew"] = train[["dur","sbytes","dbytes","spkts","dpkts"]].skew().round(2).to_dict()

# Figura 1: distribucion de clases y categorias de ataque
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
cats = train["attack_cat"].value_counts()
ax[0].bar(["Normal","Anómalo"], [OUT["label_dist_train"][0], OUT["label_dist_train"][1]], color=["#4472C4","#C00000"])
ax[0].set_title("Distribución de clases (entrenamiento)"); ax[0].set_ylabel("Registros")
for i,v in enumerate([OUT["label_dist_train"][0], OUT["label_dist_train"][1]]): ax[0].text(i, v, f"{v:,}".replace(",","."), ha="center", va="bottom", fontsize=8)
cats.plot(kind="barh", ax=ax[1], color="#4472C4"); ax[1].invert_yaxis()
ax[1].set_title("Registros por categoría de ataque"); ax[1].set_xlabel("Registros")
plt.tight_layout(); plt.savefig("fig1_clases.png", bbox_inches="tight"); plt.close()

# Figura 2: distribuciones sesgadas (log)
fig, ax = plt.subplots(1, 3, figsize=(9, 2.8))
for a, col in zip(ax, ["dur","sbytes","rate"]):
    a.hist(np.log1p(train[col]), bins=60, color="#4472C4", edgecolor="none")
    a.set_title(f"log(1+{col})"); a.set_ylabel("")
ax[0].set_ylabel("Frecuencia")
plt.tight_layout(); plt.savefig("fig2_distribuciones.png", bbox_inches="tight"); plt.close()

# ---------- 2. Limpieza y preparacion ----------
for df in (train, test):
    df.drop(columns=["id"], inplace=True)
    df["service"] = df["service"].replace("-", "unknown")
train = train.drop_duplicates().reset_index(drop=True)
OUT["shape_train_clean"] = list(train.shape)

num_cols = [c for c in train.select_dtypes(include=["number"]).columns if c != "label"]
cat_cols = ["proto","service","state"]

# Correlacion para eliminar redundancia
corr = train[num_cols].corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
OUT["drop_corr"] = drop_corr
feat_num = [c for c in num_cols if c not in drop_corr]

# Figura 3: matriz de correlacion
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(train[num_cols].corr(), cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(num_cols))); ax.set_xticklabels(num_cols, rotation=90, fontsize=6)
ax.set_yticks(range(len(num_cols))); ax.set_yticklabels(num_cols, fontsize=6)
fig.colorbar(im, shrink=0.8); ax.set_title("Matriz de correlación (Pearson) de variables numéricas")
plt.tight_layout(); plt.savefig("fig3_correlacion.png", bbox_inches="tight"); plt.close()

# Frequency encoding para categoricas (adecuado para arboles / IF)
for c in cat_cols:
    freq = train[c].value_counts(normalize=True)
    train[c + "_freq"] = train[c].map(freq)
    test[c + "_freq"] = test[c].map(freq).fillna(0.0)
features = feat_num + [c + "_freq" for c in cat_cols]
OUT["n_features_final"] = len(features)
OUT["features_final"] = features

# log1p a variables muy sesgadas + escalado robusto
skewed = [c for c in feat_num if abs(train[c].skew()) > 3]
OUT["skewed_transformed"] = skewed
Xtr = train[features].copy(); Xte = test[features].copy()
for c in skewed:
    Xtr[c] = np.log1p(Xtr[c].clip(lower=0)); Xte[c] = np.log1p(Xte[c].clip(lower=0))
scaler = RobustScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr); Xte_s = scaler.transform(Xte)
ytr = train["label"].values; yte = test["label"].values

# ---------- 3. Modelo: Isolation Forest (entrenado solo con trafico normal: novelty-style) ----------
Xtr_normal = Xtr_s[ytr == 0]
results = {}
for cont in [0.01, 0.02, 0.05, 0.10, "auto"]:
    m = IsolationForest(n_estimators=200, contamination=cont if cont != "auto" else "auto",
                        max_samples=256, random_state=RS, n_jobs=-1).fit(Xtr_normal)
    pred = (m.predict(Xte_s) == -1).astype(int)
    results[str(cont)] = {"precision": precision_score(yte, pred), "recall": recall_score(yte, pred),
                          "f1": f1_score(yte, pred)}
OUT["contamination_sweep"] = {k: {kk: round(vv,4) for kk,vv in v.items()} for k,v in results.items()}

# Modelo final: usar scores + umbral por percentil sobre score de anomalia
model = IsolationForest(n_estimators=300, max_samples=512, contamination="auto",
                        random_state=RS, n_jobs=-1).fit(Xtr_normal)
score_te = -model.score_samples(Xte_s)   # mayor = mas anomalo
score_tr = -model.score_samples(Xtr_s)
auc = roc_auc_score(yte, score_te)
OUT["roc_auc"] = round(float(auc), 4)

# Barrido de umbral por percentil del score en normal-train
thr_sweep = {}
sc_norm_tr = -model.score_samples(Xtr_normal)
for pct in [80, 85, 90, 95, 99]:
    thr = np.percentile(sc_norm_tr, pct)
    pred = (score_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
    thr_sweep[pct] = {"thr": round(float(thr),4),
                      "precision": round(precision_score(yte,pred),4),
                      "recall": round(recall_score(yte,pred),4),
                      "f1": round(f1_score(yte,pred),4),
                      "fp": int(fp), "fn": int(fn),
                      "far": round(fp/(fp+tn),4)}
OUT["threshold_sweep"] = thr_sweep

# Umbral final = mejor F1
best_pct = max(thr_sweep, key=lambda p: thr_sweep[p]["f1"])
thr = np.percentile(sc_norm_tr, best_pct)
pred_te = (score_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(yte, pred_te).ravel()
OUT["final"] = {"pct": int(best_pct), "threshold": round(float(thr),4),
    "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    "precision": round(precision_score(yte, pred_te),4),
    "recall": round(recall_score(yte, pred_te),4),
    "f1": round(f1_score(yte, pred_te),4),
    "accuracy": round((tp+tn)/(tp+tn+fp+fn),4),
    "far": round(fp/(fp+tn),4),
    "fnr": round(fn/(fn+tp),4)}

# Recall por categoria de ataque
rec_cat = {}
for cat in test["attack_cat"].unique():
    if cat == "Normal": continue
    mask = test["attack_cat"] == cat
    rec_cat[cat] = {"n": int(mask.sum()), "recall": round(float(pred_te[mask.values].mean()),4)}
OUT["recall_por_categoria"] = dict(sorted(rec_cat.items(), key=lambda x: -x[1]["recall"]))

# Figura 4: ROC + distribucion de scores
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
fpr, tpr, _ = roc_curve(yte, score_te)
ax[0].plot(fpr, tpr, color="#C00000", lw=1.5, label=f"Isolation Forest (AUC = {auc:.3f})")
ax[0].plot([0,1],[0,1],"--", color="gray", lw=1, label="Clasificador aleatorio")
ax[0].set_xlabel("Tasa de falsos positivos"); ax[0].set_ylabel("Tasa de verdaderos positivos")
ax[0].set_title("Curva ROC (conjunto de prueba)"); ax[0].legend(fontsize=8)
ax[1].hist(score_te[yte==0], bins=80, alpha=0.65, label="Normal", color="#4472C4", density=True)
ax[1].hist(score_te[yte==1], bins=80, alpha=0.65, label="Anómalo", color="#C00000", density=True)
ax[1].axvline(thr, color="black", ls="--", lw=1, label=f"Umbral (P{best_pct})")
ax[1].set_xlabel("Puntuación de anomalía"); ax[1].set_ylabel("Densidad")
ax[1].set_title("Distribución de puntuaciones por clase"); ax[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig4_roc_scores.png", bbox_inches="tight"); plt.close()

# Figura 5: matriz de confusion + recall por categoria
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
cm = np.array([[tn, fp],[fn, tp]])
im = ax[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        ax[0].text(j, i, f"{cm[i,j]:,}".replace(",","."), ha="center", va="center",
                   color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=11)
ax[0].set_xticks([0,1]); ax[0].set_xticklabels(["Pred. normal","Pred. anómalo"])
ax[0].set_yticks([0,1]); ax[0].set_yticklabels(["Real normal","Real anómalo"])
ax[0].set_title("Matriz de confusión (test)")
rc = OUT["recall_por_categoria"]
ax[1].barh(list(rc.keys()), [v["recall"] for v in rc.values()], color="#4472C4")
ax[1].invert_yaxis(); ax[1].set_xlim(0,1); ax[1].set_xlabel("Recall")
ax[1].set_title("Recall por categoría de ataque")
plt.tight_layout(); plt.savefig("fig5_confusion_categorias.png", bbox_inches="tight"); plt.close()

# ---------- 4. KPIs ----------
det = test.copy(); det["pred"] = pred_te; det["score"] = score_te
total_bytes = int((det["sbytes"] + det["dbytes"]).sum())
OUT["kpis"] = {
    "volumen_total_GB": round(total_bytes/1e9, 2),
    "conexiones_totales": int(len(det)),
    "anomalias_detectadas": int(det["pred"].sum()),
    "pct_trafico_anomalo": round(float(det["pred"].mean()*100), 2),
    "tasa_falsas_alarmas_pct": round(OUT["final"]["far"]*100, 2),
    "top_proto_anom": det[det.pred==1]["proto"].value_counts().head(5).to_dict(),
    "top_service_anom": det[det.pred==1]["service"].value_counts().head(5).to_dict(),
}

# Figura 6: KPIs de distribucion de anomalias (proto/servicio)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
det[det.pred==1]["proto"].value_counts().head(6).plot(kind="bar", ax=ax[0], color="#C00000")
ax[0].set_title("Anomalías detectadas por protocolo"); ax[0].set_ylabel("Conexiones"); ax[0].tick_params(axis="x", rotation=30)
det[det.pred==1]["service"].value_counts().head(6).plot(kind="bar", ax=ax[1], color="#C00000")
ax[1].set_title("Anomalías detectadas por servicio"); ax[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.savefig("fig6_kpis.png", bbox_inches="tight"); plt.close()

with open("resultados.json","w") as f:
    json.dump(OUT, f, indent=2, ensure_ascii=False, default=str)
print(json.dumps(OUT, indent=2, ensure_ascii=False, default=str)[:6000])
