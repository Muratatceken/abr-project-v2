#!/usr/bin/env python3
"""
ABR Dataset — Kapsamlı Veri Analizi ve Görselleştirme

Bu script abr_dataset.xlsx dosyasını detaylıca analiz eder ve
publication-ready görsellerle birlikte bir markdown raporu üretir.

Kullanım:
    python scripts/analyze_dataset.py
    python scripts/analyze_dataset.py --excel data/abr_dataset.xlsx --output analysis_results
"""

import os
import sys
import argparse
import warnings
from pathlib import Path
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from scipy import signal as sp_signal
from scipy import stats

warnings.filterwarnings("ignore")

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
})

# Colorblind-friendly palette (Wong 2011)
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#F0E442",
           "#56B4E9", "#E69F00", "#000000"]


def load_data(excel_path: str) -> pd.DataFrame:
    """Excel dosyasını yükle ve temel bilgileri yazdır."""
    print(f"Yükleniyor: {excel_path}")
    df = pd.read_excel(excel_path)
    print(f"  Satır: {len(df):,}  |  Sütun: {len(df.columns)}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GENEL BAKIŞ
# ═══════════════════════════════════════════════════════════════════════════════

def general_overview(df: pd.DataFrame, out: Path) -> dict:
    """Genel veri seti istatistikleri."""
    info = {}

    # Temel boyutlar
    info["total_rows"] = len(df)
    info["total_columns"] = len(df.columns)

    # Sütun türleri
    ts_cols = [str(i) for i in range(1, 201)]
    ts_present = [c for c in ts_cols if c in df.columns]
    info["ts_columns"] = len(ts_present)

    # Eksik veri
    missing = df.isnull().sum()
    info["columns_with_missing"] = int((missing > 0).sum())
    info["total_missing_cells"] = int(missing.sum())
    info["missing_pct"] = float(missing.sum() / df.size * 100)

    # Hasta sayısı
    if "Patient_ID" in df.columns:
        info["unique_patients"] = int(df["Patient_ID"].nunique())
        info["samples_per_patient_mean"] = round(len(df) / df["Patient_ID"].nunique(), 1)
        info["samples_per_patient_median"] = int(df.groupby("Patient_ID").size().median())
    else:
        info["unique_patients"] = "N/A"

    # Filtreleme istatistikleri
    if "Stimulus Polarity" in df.columns:
        polarity_counts = df["Stimulus Polarity"].value_counts()
        info["polarity_distribution"] = polarity_counts.to_dict()
        info["alternate_count"] = int(polarity_counts.get("Alternate", 0))
    if "Sweeps Rejected" in df.columns:
        info["sweeps_lt100"] = int((df["Sweeps Rejected"] < 100).sum())
        info["sweeps_gte100"] = int((df["Sweeps Rejected"] >= 100).sum())

    # Hedef dağılımı
    if "Hear_Loss" in df.columns:
        info["target_distribution"] = df["Hear_Loss"].value_counts().to_dict()
        info["num_classes"] = int(df["Hear_Loss"].nunique())

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 2. FİLTRELEME ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def filtering_analysis(df: pd.DataFrame, out: Path) -> dict:
    """Filtreleme adımlarının etkisini analiz et."""
    info = {}
    n0 = len(df)
    info["original"] = n0

    # Adım 1: Alternate polarity
    if "Stimulus Polarity" in df.columns:
        df_alt = df[df["Stimulus Polarity"] == "Alternate"]
        info["after_alternate"] = len(df_alt)
        info["alternate_removed"] = n0 - len(df_alt)
    else:
        df_alt = df

    # Adım 2: Sweeps < 100
    if "Sweeps Rejected" in df.columns:
        df_filt = df_alt[df_alt["Sweeps Rejected"] < 100]
        info["after_sweeps"] = len(df_filt)
        info["sweeps_removed"] = len(df_alt) - len(df_filt)
    else:
        df_filt = df_alt

    info["final"] = len(df_filt)
    info["total_removed"] = n0 - len(df_filt)
    info["retention_pct"] = round(len(df_filt) / n0 * 100, 1)

    # Görsel: filtreleme funnel
    fig, ax = plt.subplots(figsize=(8, 4))
    stages = ["Ham Veri", "Alternate\nPolarity", "Sweeps < 100"]
    counts = [n0, info.get("after_alternate", n0), info["final"]]
    bars = ax.barh(stages[::-1], counts[::-1], color=[COLORS[0], COLORS[1], COLORS[2]])
    for bar, count in zip(bars, counts[::-1]):
        ax.text(bar.get_width() + n0 * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{count:,} ({count/n0*100:.0f}%)", va="center", fontsize=11)
    ax.set_xlabel("Örnek Sayısı")
    ax.set_title("Filtreleme Aşamaları")
    ax.set_xlim(0, n0 * 1.25)
    fig.savefig(out / "01_filtering_funnel.png")
    plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 3. HEDEF DEĞİŞKEN ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def target_analysis(df: pd.DataFrame, out: Path) -> dict:
    """Hear_Loss hedef değişkenini analiz et."""
    info = {}
    if "Hear_Loss" not in df.columns:
        return info

    vc = df["Hear_Loss"].value_counts()
    info["class_counts"] = vc.to_dict()
    info["imbalance_ratio"] = round(vc.max() / vc.min(), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    ax = axes[0]
    colors = COLORS[: len(vc)]
    bars = ax.bar(range(len(vc)), vc.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(vc)))
    ax.set_xticklabels(vc.index, rotation=30, ha="right")
    ax.set_ylabel("Örnek Sayısı")
    ax.set_title("Sınıf Dağılımı (Hear_Loss)")
    for bar, v in zip(bars, vc.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + len(df) * 0.005,
                f"{v:,}\n({v/len(df)*100:.1f}%)", ha="center", va="bottom", fontsize=9)

    # Pie chart
    ax = axes[1]
    wedges, texts, autotexts = ax.pie(
        vc.values, labels=vc.index, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.75)
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title("Sınıf Oranları")

    fig.suptitle("Hearing Loss Sınıf Dağılımı", fontsize=14, y=1.02)
    fig.savefig(out / "02_target_distribution.png")
    plt.close(fig)

    # Hasta başına sınıf dağılımı
    if "Patient_ID" in df.columns:
        patient_classes = df.groupby("Patient_ID")["Hear_Loss"].nunique()
        info["patients_single_class"] = int((patient_classes == 1).sum())
        info["patients_multi_class"] = int((patient_classes > 1).sum())

        fig, ax = plt.subplots(figsize=(8, 4))
        patient_cls_counts = df.groupby(["Patient_ID", "Hear_Loss"]).size().unstack(fill_value=0)
        patient_primary = df.groupby("Patient_ID")["Hear_Loss"].agg(lambda x: x.mode().iloc[0])
        pc_vc = patient_primary.value_counts()
        ax.bar(range(len(pc_vc)), pc_vc.values, color=colors[:len(pc_vc)], edgecolor="white")
        ax.set_xticks(range(len(pc_vc)))
        ax.set_xticklabels(pc_vc.index, rotation=30, ha="right")
        ax.set_ylabel("Hasta Sayısı")
        ax.set_title("Hasta Düzeyinde Sınıf Dağılımı (Birincil Tanı)")
        fig.savefig(out / "03_patient_level_classes.png")
        plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 4. STATİK PARAMETRE ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def static_params_analysis(df: pd.DataFrame, out: Path) -> dict:
    """Age, Intensity, Stimulus Rate, FMP analizi."""
    params = ["Age", "Intensity", "Stimulus Rate", "FMP"]
    present = [p for p in params if p in df.columns]
    if not present:
        return {}

    info = {}
    for p in present:
        s = df[p].dropna()
        info[p] = {
            "count": int(len(s)),
            "missing": int(df[p].isnull().sum()),
            "mean": round(float(s.mean()), 2),
            "std": round(float(s.std()), 2),
            "min": round(float(s.min()), 2),
            "max": round(float(s.max()), 2),
            "median": round(float(s.median()), 2),
            "q25": round(float(s.quantile(0.25)), 2),
            "q75": round(float(s.quantile(0.75)), 2),
            "unique": int(s.nunique()),
        }

    # Histogramlar
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    for i, p in enumerate(present):
        ax = axes[i]
        s = df[p].dropna()
        ax.hist(s, bins=50, color=COLORS[i], edgecolor="white", linewidth=0.3, alpha=0.85)
        ax.axvline(s.mean(), color="red", ls="--", lw=1.5, label=f"Mean={s.mean():.1f}")
        ax.axvline(s.median(), color="black", ls=":", lw=1.5, label=f"Median={s.median():.1f}")
        ax.set_title(p)
        ax.set_ylabel("Frekans")
        ax.legend(fontsize=9)
    for j in range(len(present), 4):
        axes[j].set_visible(False)
    fig.suptitle("Statik Parametre Dağılımları", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "04_static_params_histograms.png")
    plt.close(fig)

    # Box plots — sınıflara göre
    if "Hear_Loss" in df.columns and len(present) >= 2:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.ravel()
        classes = sorted(df["Hear_Loss"].dropna().unique())
        for i, p in enumerate(present):
            ax = axes[i]
            data_by_class = [df[df["Hear_Loss"] == c][p].dropna() for c in classes]
            bp = ax.boxplot(data_by_class, labels=[str(c)[:15] for c in classes],
                           patch_artist=True, widths=0.6)
            for j, box in enumerate(bp["boxes"]):
                box.set_facecolor(COLORS[j % len(COLORS)])
                box.set_alpha(0.7)
            ax.set_title(f"{p} — Sınıflara Göre")
            ax.tick_params(axis="x", rotation=30)
        for j in range(len(present), 4):
            axes[j].set_visible(False)
        fig.suptitle("Statik Parametreler × Hearing Loss Sınıfı", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(out / "05_static_params_by_class.png")
        plt.close(fig)

    # Korelasyon matrisi
    if len(present) >= 2:
        fig, ax = plt.subplots(figsize=(7, 6))
        corr = df[present].corr()
        info["correlation_matrix"] = {f"{a}-{b}": round(corr.loc[a, b], 3)
                                      for a in present for b in present if a != b}
        im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=45, ha="right")
        ax.set_yticks(range(len(present)))
        ax.set_yticklabels(present)
        for ii in range(len(present)):
            for jj in range(len(present)):
                ax.text(jj, ii, f"{corr.iloc[ii, jj]:.2f}", ha="center", va="center",
                        fontsize=11, color="white" if abs(corr.iloc[ii, jj]) > 0.5 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title("Statik Parametre Korelasyonları")
        fig.savefig(out / "06_static_correlation.png")
        plt.close(fig)

    # Scatter matrix (pair plot)
    if len(present) >= 2 and "Hear_Loss" in df.columns:
        classes = sorted(df["Hear_Loss"].dropna().unique())
        n = len(present)
        fig, axes = plt.subplots(n, n, figsize=(3.5 * n, 3.5 * n))
        for ii in range(n):
            for jj in range(n):
                ax = axes[ii][jj] if n > 1 else axes
                if ii == jj:
                    for k, c in enumerate(classes):
                        subset = df[df["Hear_Loss"] == c][present[ii]].dropna()
                        ax.hist(subset, bins=30, alpha=0.5, color=COLORS[k % len(COLORS)],
                                label=str(c)[:12])
                    if ii == 0:
                        ax.legend(fontsize=6, loc="upper right")
                else:
                    for k, c in enumerate(classes):
                        subset = df[df["Hear_Loss"] == c]
                        ax.scatter(subset[present[jj]], subset[present[ii]],
                                   s=3, alpha=0.3, color=COLORS[k % len(COLORS)])
                if ii == n - 1:
                    ax.set_xlabel(present[jj], fontsize=9)
                if jj == 0:
                    ax.set_ylabel(present[ii], fontsize=9)
        fig.suptitle("Statik Parametreler — Pair Plot (Sınıf Renkli)", fontsize=14, y=1.01)
        fig.tight_layout()
        fig.savefig(out / "07_static_pairplot.png")
        plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ABR SİNYAL ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def signal_analysis(df: pd.DataFrame, out: Path) -> dict:
    """ABR zaman serisi sinyallerinin detaylı analizi."""
    ts_cols = [str(i) for i in range(1, 201)]
    ts_present = [c for c in ts_cols if c in df.columns]
    if len(ts_present) < 200:
        return {"error": f"Only {len(ts_present)} of 200 time columns found"}

    ts_data = df[ts_present].values.astype(float)
    info = {}

    # Temel istatistikler
    row_means = np.nanmean(ts_data, axis=1)
    row_stds = np.nanstd(ts_data, axis=1)
    row_ranges = np.nanmax(ts_data, axis=1) - np.nanmin(ts_data, axis=1)

    info["signal_stats"] = {
        "global_mean": round(float(np.nanmean(ts_data)), 6),
        "global_std": round(float(np.nanstd(ts_data)), 6),
        "per_sample_mean_range": [round(float(row_means.min()), 4), round(float(row_means.max()), 4)],
        "per_sample_std_range": [round(float(row_stds.min()), 6), round(float(row_stds.max()), 4)],
        "nan_count": int(np.isnan(ts_data).sum()),
        "inf_count": int(np.isinf(ts_data).sum()),
        "zero_variance_signals": int((row_stds < 1e-8).sum()),
        "low_variance_signals": int((row_stds < 1e-6).sum()),
    }

    # ── Görsel 1: Örnek sinyaller ────────────────────────────────────────────
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    axes = axes.ravel()
    np.random.seed(42)
    sample_idx = np.random.choice(len(ts_data), min(9, len(ts_data)), replace=False)
    t_axis = np.arange(200)
    for i, idx in enumerate(sample_idx):
        ax = axes[i]
        sig = ts_data[idx]
        ax.plot(t_axis, sig, color=COLORS[0], lw=0.8)
        ax.fill_between(t_axis, sig, alpha=0.15, color=COLORS[0])
        label = df["Hear_Loss"].iloc[idx] if "Hear_Loss" in df.columns else "?"
        ax.set_title(f"Örnek #{idx} — {label}", fontsize=10)
        ax.set_xlabel("Zaman Noktası")
        ax.set_ylabel("Amplitüd")
    fig.suptitle("Rastgele Seçilmiş ABR Sinyalleri", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "08_sample_signals.png")
    plt.close(fig)

    # ── Görsel 2: Sınıf bazlı ortalama sinyaller ─────────────────────────────
    if "Hear_Loss" in df.columns:
        classes = sorted(df["Hear_Loss"].dropna().unique())
        fig, ax = plt.subplots(figsize=(14, 6))
        for k, c in enumerate(classes):
            mask = df["Hear_Loss"] == c
            mean_sig = np.nanmean(ts_data[mask], axis=0)
            std_sig = np.nanstd(ts_data[mask], axis=0)
            ax.plot(t_axis, mean_sig, color=COLORS[k % len(COLORS)], lw=1.5,
                    label=f"{c} (n={mask.sum():,})")
            ax.fill_between(t_axis, mean_sig - std_sig, mean_sig + std_sig,
                           color=COLORS[k % len(COLORS)], alpha=0.12)
        ax.set_xlabel("Zaman Noktası")
        ax.set_ylabel("Amplitüd")
        ax.set_title("Sınıf Bazlı Ortalama ABR Sinyalleri (±1 SD)")
        ax.legend(loc="upper right")
        fig.savefig(out / "09_class_average_signals.png")
        plt.close(fig)

        # Sınıf bazında overlay
        fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4), sharey=True)
        if len(classes) == 1:
            axes = [axes]
        for k, c in enumerate(classes):
            ax = axes[k]
            mask = df["Hear_Loss"] == c
            subset = ts_data[mask]
            # 20 rastgele örnek
            n_plot = min(20, len(subset))
            for j in range(n_plot):
                ax.plot(t_axis, subset[j], alpha=0.25, color=COLORS[k % len(COLORS)], lw=0.5)
            mean_sig = np.nanmean(subset, axis=0)
            ax.plot(t_axis, mean_sig, color="black", lw=2, label="Ortalama")
            ax.set_title(f"{c}\n(n={mask.sum():,})", fontsize=10)
            ax.set_xlabel("Zaman")
            if k == 0:
                ax.set_ylabel("Amplitüd")
        fig.suptitle("Sınıf İçi Sinyal Varyasyonu", fontsize=13, y=1.03)
        fig.tight_layout()
        fig.savefig(out / "10_class_signal_variation.png")
        plt.close(fig)

    # ── Görsel 3: Sinyal kalitesi dağılımı ─────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    ax = axes[0]
    ax.hist(row_means, bins=80, color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.set_title("Per-Sample Ortalama Dağılımı")
    ax.set_xlabel("Ortalama Amplitüd")
    ax.set_ylabel("Frekans")

    ax = axes[1]
    ax.hist(row_stds, bins=80, color=COLORS[1], edgecolor="white", linewidth=0.3)
    ax.set_title("Per-Sample Std Dağılımı")
    ax.set_xlabel("Standart Sapma")

    ax = axes[2]
    ax.hist(row_ranges, bins=80, color=COLORS[2], edgecolor="white", linewidth=0.3)
    ax.set_title("Per-Sample Dinamik Aralık")
    ax.set_xlabel("Max - Min")

    fig.suptitle("Sinyal Kalite Metrikleri", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "11_signal_quality.png")
    plt.close(fig)

    # ── Görsel 4: Zaman noktası bazlı istatistikler ──────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    timepoint_mean = np.nanmean(ts_data, axis=0)
    timepoint_std = np.nanstd(ts_data, axis=0)
    timepoint_median = np.nanmedian(ts_data, axis=0)

    ax = axes[0]
    ax.plot(t_axis, timepoint_mean, color=COLORS[0], lw=1.5, label="Ortalama")
    ax.plot(t_axis, timepoint_median, color=COLORS[1], lw=1.2, ls="--", label="Medyan")
    ax.fill_between(t_axis, timepoint_mean - timepoint_std, timepoint_mean + timepoint_std,
                   color=COLORS[0], alpha=0.15)
    ax.set_ylabel("Amplitüd")
    ax.set_title("Zaman Noktası Bazlı İstatistikler")
    ax.legend()

    ax = axes[1]
    ax.plot(t_axis, timepoint_std, color=COLORS[2], lw=1.5)
    ax.set_xlabel("Zaman Noktası")
    ax.set_ylabel("Standart Sapma")
    ax.set_title("Zaman Noktası Bazlı Varyans")

    fig.tight_layout()
    fig.savefig(out / "12_timepoint_statistics.png")
    plt.close(fig)

    # ── Görsel 5: Frekans analizi ─────────────────────────────────────────────
    fs = 20000  # 200 samples / 10 ms → 20 kHz
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Ortalama güç spektrumu
    ax = axes[0]
    n_fft = 256
    all_psds = []
    sample_for_psd = np.random.choice(len(ts_data), min(500, len(ts_data)), replace=False)
    for idx in sample_for_psd:
        sig = ts_data[idx]
        if np.std(sig) < 1e-8:
            continue
        f, psd = sp_signal.welch(sig, fs=fs, nperseg=min(64, len(sig)), noverlap=32)
        all_psds.append(psd)
    if all_psds:
        all_psds = np.array(all_psds)
        mean_psd = np.mean(all_psds, axis=0)
        std_psd = np.std(all_psds, axis=0)
        ax.semilogy(f, mean_psd, color=COLORS[0], lw=1.5)
        ax.fill_between(f, np.maximum(mean_psd - std_psd, 1e-15),
                       mean_psd + std_psd, color=COLORS[0], alpha=0.15)
        ax.set_xlabel("Frekans (Hz)")
        ax.set_ylabel("Güç Spektral Yoğunluğu")
        ax.set_title("Ortalama Güç Spektrumu (Welch)")

    # Sınıf bazlı güç spektrumu
    ax = axes[1]
    if "Hear_Loss" in df.columns:
        classes = sorted(df["Hear_Loss"].dropna().unique())
        for k, c in enumerate(classes):
            mask = (df["Hear_Loss"] == c).values
            class_psds = []
            class_indices = np.where(mask)[0]
            sample_c = np.random.choice(class_indices, min(200, len(class_indices)), replace=False)
            for idx in sample_c:
                sig = ts_data[idx]
                if np.std(sig) < 1e-8:
                    continue
                f, psd = sp_signal.welch(sig, fs=fs, nperseg=min(64, len(sig)), noverlap=32)
                class_psds.append(psd)
            if class_psds:
                mean_c = np.mean(class_psds, axis=0)
                ax.semilogy(f, mean_c, color=COLORS[k % len(COLORS)], lw=1.5,
                           label=str(c)[:15])
        ax.set_xlabel("Frekans (Hz)")
        ax.set_ylabel("PSD")
        ax.set_title("Sınıf Bazlı Güç Spektrumu")
        ax.legend(fontsize=8)

    fig.suptitle("Frekans Analizi", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "13_frequency_analysis.png")
    plt.close(fig)

    # ── Görsel 6: SNR analizi ─────────────────────────────────────────────────
    snr_values = []
    for idx in range(len(ts_data)):
        sig = ts_data[idx]
        baseline = sig[:50]  # İlk 50 örnek baseline
        response = sig[50:]
        noise_power = np.var(baseline)
        signal_power = np.var(response)
        if noise_power > 1e-12:
            snr = 10 * np.log10(signal_power / noise_power)
            snr_values.append(snr)
        else:
            snr_values.append(np.nan)
    snr_values = np.array(snr_values)
    valid_snr = snr_values[~np.isnan(snr_values)]

    if len(valid_snr) > 0:
        info["snr_stats"] = {
            "mean_dB": round(float(np.mean(valid_snr)), 2),
            "std_dB": round(float(np.std(valid_snr)), 2),
            "min_dB": round(float(np.min(valid_snr)), 2),
            "max_dB": round(float(np.max(valid_snr)), 2),
            "median_dB": round(float(np.median(valid_snr)), 2),
        }

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
        ax.hist(valid_snr, bins=60, color=COLORS[0], edgecolor="white", linewidth=0.3)
        ax.axvline(np.median(valid_snr), color="red", ls="--", label=f"Medyan={np.median(valid_snr):.1f} dB")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Frekans")
        ax.set_title("SNR Dağılımı")
        ax.legend()

        if "Hear_Loss" in df.columns:
            ax = axes[1]
            classes = sorted(df["Hear_Loss"].dropna().unique())
            snr_by_class = []
            labels = []
            for c in classes:
                mask = (df["Hear_Loss"] == c).values
                class_snr = snr_values[mask]
                class_snr = class_snr[~np.isnan(class_snr)]
                if len(class_snr) > 0:
                    snr_by_class.append(class_snr)
                    labels.append(str(c)[:15])
            bp = ax.boxplot(snr_by_class, labels=labels, patch_artist=True, widths=0.6)
            for j, box in enumerate(bp["boxes"]):
                box.set_facecolor(COLORS[j % len(COLORS)])
                box.set_alpha(0.7)
            ax.set_ylabel("SNR (dB)")
            ax.set_title("Sınıf Bazlı SNR")
            ax.tick_params(axis="x", rotation=30)

        fig.suptitle("Sinyal-Gürültü Oranı Analizi", fontsize=13, y=1.02)
        fig.tight_layout()
        fig.savefig(out / "14_snr_analysis.png")
        plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 6. V-PEAK ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def vpeak_analysis(df: pd.DataFrame, out: Path) -> dict:
    """V (5th) peak latency ve amplitude analizi."""
    lat_col = "V Latancy"  # orijinal veriden gelen typo
    amp_col = "V Amplitude"
    info = {}

    has_lat = lat_col in df.columns
    has_amp = amp_col in df.columns
    if not (has_lat or has_amp):
        return info

    if has_lat:
        lat = df[lat_col]
        info["latency_present"] = int(lat.notna().sum())
        info["latency_missing"] = int(lat.isna().sum())
        info["latency_pct_present"] = round(lat.notna().sum() / len(df) * 100, 1)
        lat_valid = lat.dropna()
        if len(lat_valid) > 0:
            info["latency_stats"] = {
                "mean": round(float(lat_valid.mean()), 3),
                "std": round(float(lat_valid.std()), 3),
                "min": round(float(lat_valid.min()), 3),
                "max": round(float(lat_valid.max()), 3),
                "median": round(float(lat_valid.median()), 3),
            }

    if has_amp:
        amp = df[amp_col]
        info["amplitude_present"] = int(amp.notna().sum())
        info["amplitude_missing"] = int(amp.isna().sum())
        info["amplitude_pct_present"] = round(amp.notna().sum() / len(df) * 100, 1)
        amp_valid = amp.dropna()
        if len(amp_valid) > 0:
            info["amplitude_stats"] = {
                "mean": round(float(amp_valid.mean()), 4),
                "std": round(float(amp_valid.std()), 4),
                "min": round(float(amp_valid.min()), 4),
                "max": round(float(amp_valid.max()), 4),
                "median": round(float(amp_valid.median()), 4),
            }

    # Both present
    both_present = df[[lat_col, amp_col]].notna().all(axis=1).sum() if has_lat and has_amp else 0
    info["both_present"] = int(both_present)
    info["both_present_pct"] = round(both_present / len(df) * 100, 1)

    # Görseller
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    if has_lat:
        ax = axes[0, 0]
        lat_v = df[lat_col].dropna()
        ax.hist(lat_v, bins=60, color=COLORS[0], edgecolor="white", linewidth=0.3)
        ax.set_title(f"V Peak Latency Dağılımı (n={len(lat_v):,})")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Frekans")

    if has_amp:
        ax = axes[0, 1]
        amp_v = df[amp_col].dropna()
        ax.hist(amp_v, bins=60, color=COLORS[1], edgecolor="white", linewidth=0.3)
        ax.set_title(f"V Peak Amplitude Dağılımı (n={len(amp_v):,})")
        ax.set_xlabel("Amplitude (µV)")

    if has_lat and has_amp:
        ax = axes[1, 0]
        both_mask = df[[lat_col, amp_col]].notna().all(axis=1)
        ax.scatter(df.loc[both_mask, lat_col], df.loc[both_mask, amp_col],
                  s=3, alpha=0.3, color=COLORS[2])
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("Latency vs Amplitude")

        # Sınıf bazlı
        ax = axes[1, 1]
        if "Hear_Loss" in df.columns:
            classes = sorted(df["Hear_Loss"].dropna().unique())
            for k, c in enumerate(classes):
                mask = (df["Hear_Loss"] == c) & both_mask
                ax.scatter(df.loc[mask, lat_col], df.loc[mask, amp_col],
                          s=5, alpha=0.4, color=COLORS[k % len(COLORS)], label=str(c)[:12])
            ax.set_xlabel("Latency (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title("Latency vs Amplitude (Sınıf Renkli)")
            ax.legend(fontsize=8, markerscale=3)
        else:
            axes[1, 1].set_visible(False)
    else:
        axes[1, 0].set_visible(False)
        axes[1, 1].set_visible(False)

    fig.suptitle("V Peak (5. Dalga) Analizi", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out / "15_vpeak_analysis.png")
    plt.close(fig)

    # Sınıf bazlı V peak box plot
    if "Hear_Loss" in df.columns and has_lat and has_amp:
        classes = sorted(df["Hear_Loss"].dropna().unique())
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax = axes[0]
        data = [df[df["Hear_Loss"] == c][lat_col].dropna() for c in classes]
        bp = ax.boxplot(data, labels=[str(c)[:12] for c in classes], patch_artist=True, widths=0.6)
        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor(COLORS[j % len(COLORS)])
            box.set_alpha(0.7)
        ax.set_ylabel("Latency (ms)")
        ax.set_title("V Peak Latency — Sınıflara Göre")
        ax.tick_params(axis="x", rotation=30)

        ax = axes[1]
        data = [df[df["Hear_Loss"] == c][amp_col].dropna() for c in classes]
        bp = ax.boxplot(data, labels=[str(c)[:12] for c in classes], patch_artist=True, widths=0.6)
        for j, box in enumerate(bp["boxes"]):
            box.set_facecolor(COLORS[j % len(COLORS)])
            box.set_alpha(0.7)
        ax.set_ylabel("Amplitude (µV)")
        ax.set_title("V Peak Amplitude — Sınıflara Göre")
        ax.tick_params(axis="x", rotation=30)

        fig.tight_layout()
        fig.savefig(out / "16_vpeak_by_class.png")
        plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 7. HASTA BAZLI ANALİZ
# ═══════════════════════════════════════════════════════════════════════════════

def patient_analysis(df: pd.DataFrame, out: Path) -> dict:
    """Hasta bazlı istatistikler ve veri sızıntısı kontrolleri."""
    if "Patient_ID" not in df.columns:
        return {}

    info = {}
    pg = df.groupby("Patient_ID")
    sizes = pg.size()

    info["patient_count"] = int(len(sizes))
    info["samples_per_patient"] = {
        "mean": round(float(sizes.mean()), 1),
        "std": round(float(sizes.std()), 1),
        "min": int(sizes.min()),
        "max": int(sizes.max()),
        "median": int(sizes.median()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(sizes, bins=min(50, sizes.nunique()), color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.axvline(sizes.mean(), color="red", ls="--", label=f"Ortalama={sizes.mean():.0f}")
    ax.set_xlabel("Örnek Sayısı / Hasta")
    ax.set_ylabel("Hasta Sayısı")
    ax.set_title("Hasta Başına Örnek Dağılımı")
    ax.legend()

    ax = axes[1]
    top20 = sizes.nlargest(20)
    ax.barh(range(len(top20)), top20.values, color=COLORS[1])
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels([f"P-{pid}" for pid in top20.index], fontsize=8)
    ax.set_xlabel("Örnek Sayısı")
    ax.set_title("En Çok Örnekli 20 Hasta")

    fig.suptitle("Hasta Bazlı Analiz", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out / "17_patient_analysis.png")
    plt.close(fig)

    # Intensity dağılımı hasta bazında
    if "Intensity" in df.columns:
        intensity_per_patient = pg["Intensity"].nunique()
        info["intensity_levels_per_patient"] = {
            "mean": round(float(intensity_per_patient.mean()), 1),
            "min": int(intensity_per_patient.min()),
            "max": int(intensity_per_patient.max()),
        }

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 8. EKSİK VERİ ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def missing_data_analysis(df: pd.DataFrame, out: Path) -> dict:
    """Eksik veri pattern analizi."""
    info = {}
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if len(missing) == 0:
        info["status"] = "No missing data"
        return info

    info["columns_with_missing"] = len(missing)
    info["top_missing"] = {col: {"count": int(v), "pct": round(v / len(df) * 100, 1)}
                           for col, v in missing.head(15).items()}

    fig, ax = plt.subplots(figsize=(12, max(4, len(missing) * 0.35)))
    cols = missing.head(20)
    y_pos = range(len(cols))
    bars = ax.barh(y_pos, cols.values / len(df) * 100, color=COLORS[3], edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cols.index, fontsize=9)
    ax.set_xlabel("Eksik Oran (%)")
    ax.set_title("Eksik Veri Oranları (En Yüksek 20)")
    for bar, v in zip(bars, cols.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{v:,} ({v/len(df)*100:.1f}%)", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "18_missing_data.png")
    plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# 9. INTENSITY ANALİZİ
# ═══════════════════════════════════════════════════════════════════════════════

def intensity_analysis(df: pd.DataFrame, out: Path) -> dict:
    """Stimulus intensity detaylı analizi — ABR'de kritik parametre."""
    if "Intensity" not in df.columns:
        return {}

    info = {}
    intensity = df["Intensity"].dropna()
    info["unique_levels"] = sorted(intensity.unique().tolist())
    info["level_counts"] = intensity.value_counts().sort_index().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    vc = intensity.value_counts().sort_index()
    ax.bar(vc.index, vc.values, color=COLORS[0], width=3, edgecolor="white")
    ax.set_xlabel("Intensity (dB)")
    ax.set_ylabel("Örnek Sayısı")
    ax.set_title("Intensity Seviye Dağılımı")

    # Sınıf × Intensity heatmap
    if "Hear_Loss" in df.columns:
        ax = axes[1]
        ct = pd.crosstab(df["Hear_Loss"], pd.cut(df["Intensity"], bins=10))
        im = ax.imshow(ct.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(ct.index)))
        ax.set_yticklabels(ct.index, fontsize=9)
        ax.set_xticks(range(len(ct.columns)))
        ax.set_xticklabels([str(c)[:8] for c in ct.columns], rotation=45, ha="right", fontsize=8)
        ax.set_title("Sınıf × Intensity Heatmap")
        ax.set_xlabel("Intensity Aralığı")
        ax.set_ylabel("Hearing Loss Sınıfı")
        plt.colorbar(im, ax=ax, shrink=0.8)
    else:
        axes[1].set_visible(False)

    fig.tight_layout()
    fig.savefig(out / "19_intensity_analysis.png")
    plt.close(fig)

    # Intensity'e göre sinyal morfolojisi
    ts_cols = [str(i) for i in range(1, 201)]
    ts_present = [c for c in ts_cols if c in df.columns]
    if len(ts_present) == 200:
        fig, ax = plt.subplots(figsize=(14, 6))
        t_axis = np.arange(200)
        unique_int = sorted(intensity.unique())
        # En yaygın 6 intensity seviyesi
        top_levels = intensity.value_counts().nlargest(6).index.sort_values()
        for k, level in enumerate(top_levels):
            mask = df["Intensity"] == level
            ts_data = df.loc[mask, ts_present].values.astype(float)
            mean_sig = np.nanmean(ts_data, axis=0)
            ax.plot(t_axis, mean_sig, color=COLORS[k % len(COLORS)], lw=1.5,
                   label=f"{level} dB (n={mask.sum():,})")
        ax.set_xlabel("Zaman Noktası")
        ax.set_ylabel("Amplitüd")
        ax.set_title("Intensity Seviyesine Göre Ortalama ABR Sinyalleri")
        ax.legend()
        fig.savefig(out / "20_intensity_signals.png")
        plt.close(fig)

    return info


# ═══════════════════════════════════════════════════════════════════════════════
# RAPOR OLUŞTURMA
# ═══════════════════════════════════════════════════════════════════════════════

def generate_report(results: dict, out: Path) -> str:
    """Markdown raporu oluştur."""
    r = results
    lines = []
    a = lines.append

    a("# ABR Dataset — Kapsamlı Veri Analizi Raporu")
    a("")
    a(f"> Oluşturulma tarihi: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    a(f"> Kaynak: `data/abr_dataset.xlsx`")
    a("")

    # 1. Genel bakış
    g = r.get("general", {})
    a("## 1. Genel Bakış")
    a("")
    a(f"| Metrik | Değer |")
    a(f"|--------|-------|")
    a(f"| Toplam satır | {g.get('total_rows', 'N/A'):,} |")
    a(f"| Toplam sütun | {g.get('total_columns', 'N/A')} |")
    a(f"| Zaman serisi sütunları | {g.get('ts_columns', 'N/A')} |")
    a(f"| Benzersiz hasta | {g.get('unique_patients', 'N/A')} |")
    a(f"| Hasta başına ort. örnek | {g.get('samples_per_patient_mean', 'N/A')} |")
    a(f"| Eksik hücre (%) | {g.get('missing_pct', 'N/A'):.2f}% |")
    a(f"| Sınıf sayısı | {g.get('num_classes', 'N/A')} |")
    a("")

    # 2. Filtreleme
    f_info = r.get("filtering", {})
    a("## 2. Filtreleme Analizi")
    a("")
    a(f"| Aşama | Örnek Sayısı | Kaldırılan |")
    a(f"|-------|-------------|------------|")
    a(f"| Ham veri | {f_info.get('original', 'N/A'):,} | — |")
    a(f"| Alternate polarity | {f_info.get('after_alternate', 'N/A'):,} | {f_info.get('alternate_removed', 'N/A'):,} |")
    a(f"| Sweeps < 100 | {f_info.get('after_sweeps', 'N/A'):,} | {f_info.get('sweeps_removed', 'N/A'):,} |")
    a(f"| **Final** | **{f_info.get('final', 'N/A'):,}** | {f_info.get('total_removed', 'N/A'):,} ({100 - f_info.get('retention_pct', 0):.1f}%) |")
    a("")
    a("![Filtreleme](01_filtering_funnel.png)")
    a("")

    # 3. Hedef dağılımı
    t_info = r.get("target", {})
    a("## 3. Hearing Loss Sınıf Dağılımı")
    a("")
    if "class_counts" in t_info:
        a("| Sınıf | Sayı | Oran |")
        a("|-------|------|------|")
        total = sum(t_info["class_counts"].values())
        for cls, cnt in sorted(t_info["class_counts"].items(), key=lambda x: -x[1]):
            a(f"| {cls} | {cnt:,} | {cnt/total*100:.1f}% |")
        a(f"| **Toplam** | **{total:,}** | **100%** |")
        a("")
        a(f"Dengesizlik oranı (max/min): **{t_info.get('imbalance_ratio', 'N/A')}x**")
    a("")
    a("![Sınıf Dağılımı](02_target_distribution.png)")
    a("")
    if "patients_single_class" in t_info:
        a(f"- Tek sınıflı hastalar: {t_info['patients_single_class']}")
        a(f"- Çok sınıflı hastalar: {t_info['patients_multi_class']}")
        a("")
        a("![Hasta Sınıfları](03_patient_level_classes.png)")
        a("")

    # 4. Statik parametreler
    s_info = r.get("static", {})
    a("## 4. Statik Parametre Analizi")
    a("")
    params = ["Age", "Intensity", "Stimulus Rate", "FMP"]
    present_params = [p for p in params if p in s_info]
    if present_params:
        a("| Parametre | Ort | Std | Min | Medyan | Max | Eksik |")
        a("|-----------|-----|-----|-----|--------|-----|-------|")
        for p in present_params:
            ps = s_info[p]
            a(f"| {p} | {ps['mean']} | {ps['std']} | {ps['min']} | {ps['median']} | {ps['max']} | {ps['missing']} |")
        a("")
    a("![Histogramlar](04_static_params_histograms.png)")
    a("")
    a("![Sınıf Bazlı](05_static_params_by_class.png)")
    a("")
    a("![Korelasyon](06_static_correlation.png)")
    a("")
    a("![Pair Plot](07_static_pairplot.png)")
    a("")

    # 5. Sinyal analizi
    sig = r.get("signal", {})
    a("## 5. ABR Sinyal Analizi")
    a("")
    ss = sig.get("signal_stats", {})
    if ss:
        a(f"| Metrik | Değer |")
        a(f"|--------|-------|")
        a(f"| Global ortalama | {ss.get('global_mean', 'N/A')} |")
        a(f"| Global std | {ss.get('global_std', 'N/A')} |")
        a(f"| NaN sayısı | {ss.get('nan_count', 0):,} |")
        a(f"| Inf sayısı | {ss.get('inf_count', 0):,} |")
        a(f"| Sıfır varyanslı sinyaller | {ss.get('zero_variance_signals', 0):,} |")
        a(f"| Düşük varyanslı sinyaller | {ss.get('low_variance_signals', 0):,} |")
        a("")

    a("![Örnek Sinyaller](08_sample_signals.png)")
    a("")
    a("![Sınıf Ortalama](09_class_average_signals.png)")
    a("")
    a("![Sinyal Varyasyonu](10_class_signal_variation.png)")
    a("")
    a("![Sinyal Kalitesi](11_signal_quality.png)")
    a("")
    a("![Zaman Noktası İstatistikleri](12_timepoint_statistics.png)")
    a("")
    a("![Frekans Analizi](13_frequency_analysis.png)")
    a("")

    snr = sig.get("snr_stats", {})
    if snr:
        a("### SNR Analizi")
        a("")
        a(f"| Metrik | Değer |")
        a(f"|--------|-------|")
        a(f"| Ortalama SNR | {snr['mean_dB']} dB |")
        a(f"| Medyan SNR | {snr['median_dB']} dB |")
        a(f"| Min SNR | {snr['min_dB']} dB |")
        a(f"| Max SNR | {snr['max_dB']} dB |")
        a("")
        a("![SNR](14_snr_analysis.png)")
        a("")

    # 6. V Peak
    vp = r.get("vpeak", {})
    a("## 6. V Peak (5. Dalga) Analizi")
    a("")
    if vp:
        a(f"| Metrik | Latency | Amplitude |")
        a(f"|--------|---------|-----------|")
        a(f"| Mevcut | {vp.get('latency_present', 'N/A'):,} ({vp.get('latency_pct_present', 'N/A')}%) | {vp.get('amplitude_present', 'N/A'):,} ({vp.get('amplitude_pct_present', 'N/A')}%) |")
        a(f"| Eksik | {vp.get('latency_missing', 'N/A'):,} | {vp.get('amplitude_missing', 'N/A'):,} |")
        a(f"| İkisi birden mevcut | {vp.get('both_present', 'N/A'):,} ({vp.get('both_present_pct', 'N/A')}%) | |")
        a("")
        ls = vp.get("latency_stats", {})
        ams = vp.get("amplitude_stats", {})
        if ls and ams:
            a(f"| İstatistik | Latency | Amplitude |")
            a(f"|------------|---------|-----------|")
            a(f"| Ortalama | {ls.get('mean', 'N/A')} ms | {ams.get('mean', 'N/A')} µV |")
            a(f"| Std | {ls.get('std', 'N/A')} | {ams.get('std', 'N/A')} |")
            a(f"| Min | {ls.get('min', 'N/A')} | {ams.get('min', 'N/A')} |")
            a(f"| Max | {ls.get('max', 'N/A')} | {ams.get('max', 'N/A')} |")
            a("")
    a("![V Peak](15_vpeak_analysis.png)")
    a("")
    a("![V Peak Sınıf](16_vpeak_by_class.png)")
    a("")

    # 7. Hasta analizi
    pa = r.get("patient", {})
    a("## 7. Hasta Bazlı Analiz")
    a("")
    spp = pa.get("samples_per_patient", {})
    if spp:
        a(f"| Metrik | Değer |")
        a(f"|--------|-------|")
        a(f"| Hasta sayısı | {pa.get('patient_count', 'N/A'):,} |")
        a(f"| Hasta başına ort. örnek | {spp.get('mean', 'N/A')} |")
        a(f"| Min / Max | {spp.get('min', 'N/A')} / {spp.get('max', 'N/A')} |")
        a(f"| Medyan | {spp.get('median', 'N/A')} |")
        a("")
    a("![Hasta Analizi](17_patient_analysis.png)")
    a("")

    # 8. Eksik veri
    mv = r.get("missing", {})
    a("## 8. Eksik Veri Analizi")
    a("")
    if mv.get("status") == "No missing data":
        a("Veri setinde eksik veri bulunmamaktadır.")
    else:
        a(f"Eksik veri içeren sütun sayısı: **{mv.get('columns_with_missing', 0)}**")
        a("")
        top = mv.get("top_missing", {})
        if top:
            a("| Sütun | Eksik | Oran |")
            a("|-------|-------|------|")
            for col, v in list(top.items())[:10]:
                a(f"| {col} | {v['count']:,} | {v['pct']}% |")
            a("")
        a("![Eksik Veri](18_missing_data.png)")
    a("")

    # 9. Intensity
    ia = r.get("intensity", {})
    a("## 9. Intensity Analizi")
    a("")
    if ia:
        levels = ia.get("unique_levels", [])
        a(f"Benzersiz intensity seviyeleri: **{len(levels)}**")
        a(f"Aralık: **{min(levels) if levels else 'N/A'} – {max(levels) if levels else 'N/A'} dB**")
        a("")
        a("![Intensity](19_intensity_analysis.png)")
        a("")
        a("![Intensity Sinyalleri](20_intensity_signals.png)")
    a("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="ABR Dataset Analiz Scripti")
    parser.add_argument("--excel", default="data/abr_dataset.xlsx", help="Excel dosya yolu")
    parser.add_argument("--output", default="analysis_results", help="Çıktı klasörü")
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ABR DATASET — KAPSAMLI ANALİZ")
    print("=" * 60)

    # Veri yükle
    df = load_data(args.excel)

    # Filtrelenmiş veri seti (analiz için)
    df_filtered = df.copy()
    if "Stimulus Polarity" in df.columns:
        df_filtered = df_filtered[df_filtered["Stimulus Polarity"] == "Alternate"]
    if "Sweeps Rejected" in df.columns:
        df_filtered = df_filtered[df_filtered["Sweeps Rejected"] < 100]

    results = {}

    print("\n[1/9] Genel bakış...")
    results["general"] = general_overview(df, out)

    print("[2/9] Filtreleme analizi...")
    results["filtering"] = filtering_analysis(df, out)

    print("[3/9] Hedef değişken analizi...")
    results["target"] = target_analysis(df_filtered, out)

    print("[4/9] Statik parametre analizi...")
    results["static"] = static_params_analysis(df_filtered, out)

    print("[5/9] ABR sinyal analizi...")
    results["signal"] = signal_analysis(df_filtered, out)

    print("[6/9] V peak analizi...")
    results["vpeak"] = vpeak_analysis(df_filtered, out)

    print("[7/9] Hasta bazlı analiz...")
    results["patient"] = patient_analysis(df_filtered, out)

    print("[8/9] Eksik veri analizi...")
    results["missing"] = missing_data_analysis(df, out)

    print("[9/9] Intensity analizi...")
    results["intensity"] = intensity_analysis(df_filtered, out)

    # Rapor oluştur
    print("\nRapor oluşturuluyor...")
    report = generate_report(results, out)
    report_path = out / "ABR_Dataset_Analysis_Report.md"
    report_path.write_text(report, encoding="utf-8")

    print(f"\n{'=' * 60}")
    print(f"✓ Analiz tamamlandı!")
    print(f"  Görseller: {out}/")
    print(f"  Rapor:     {report_path}")
    print(f"  Toplam görsel: {len(list(out.glob('*.png')))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
