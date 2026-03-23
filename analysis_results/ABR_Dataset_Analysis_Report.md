# ABR Dataset — Kapsamlı Veri Analizi Raporu

> Oluşturulma tarihi: 2026-03-23 19:22
> Kaynak: `data/abr_dataset.xlsx`

## 1. Genel Bakış

| Metrik | Değer |
|--------|-------|
| Toplam satır | 55,237 |
| Toplam sütun | 496 |
| Zaman serisi sütunları | 200 |
| Benzersiz hasta | 2043 |
| Hasta başına ort. örnek | 27.0 |
| Eksik hücre (%) | 2.03% |
| Sınıf sayısı | 5 |

## 2. Filtreleme Analizi

| Aşama | Örnek Sayısı | Kaldırılan |
|-------|-------------|------------|
| Ham veri | 55,237 | — |
| Alternate polarity | 53,280 | 1,957 |
| Sweeps < 100 | 51,961 | 1,319 |
| **Final** | **51,961** | 3,276 (5.9%) |

![Filtreleme](01_filtering_funnel.png)

## 3. Hearing Loss Sınıf Dağılımı

| Sınıf | Sayı | Oran |
|-------|------|------|
| NORMAL | 41,391 | 79.7% |
| SNİK | 5,513 | 10.6% |
| İTİK | 2,925 | 5.6% |
| TOTAL | 1,715 | 3.3% |
| NÖROPATİ | 417 | 0.8% |
| **Toplam** | **51,961** | **100%** |

Dengesizlik oranı (max/min): **99.3x**

![Sınıf Dağılımı](02_target_distribution.png)

- Tek sınıflı hastalar: 1893
- Çok sınıflı hastalar: 145

![Hasta Sınıfları](03_patient_level_classes.png)

## 4. Statik Parametre Analizi

| Parametre | Ort | Std | Min | Medyan | Max | Eksik |
|-----------|-----|-----|-----|--------|-----|-------|
| Age | 2.64 | 7.39 | 0.0 | 0.0 | 75.0 | 1611 |
| Intensity | 43.97 | 29.76 | 0.0 | 40.0 | 100.0 | 0 |
| Stimulus Rate | 33.68 | 3.62 | 11.1 | 33.1 | 49.1 | 0 |
| FMP | 2.4 | 11.72 | 0.0 | 0.76 | 1515.17 | 16 |

![Histogramlar](04_static_params_histograms.png)

![Sınıf Bazlı](05_static_params_by_class.png)

![Korelasyon](06_static_correlation.png)

![Pair Plot](07_static_pairplot.png)

## 5. ABR Sinyal Analizi

| Metrik | Değer |
|--------|-------|
| Global ortalama | 0.007912 |
| Global std | 0.152598 |
| NaN sayısı | 0 |
| Inf sayısı | 0 |
| Sıfır varyanslı sinyaller | 2 |
| Düşük varyanslı sinyaller | 2 |

![Örnek Sinyaller](08_sample_signals.png)

![Sınıf Ortalama](09_class_average_signals.png)

![Sinyal Varyasyonu](10_class_signal_variation.png)

![Sinyal Kalitesi](11_signal_quality.png)

![Zaman Noktası İstatistikleri](12_timepoint_statistics.png)

![Frekans Analizi](13_frequency_analysis.png)

### SNR Analizi

| Metrik | Değer |
|--------|-------|
| Ortalama SNR | 9.05 dB |
| Medyan SNR | 9.04 dB |
| Min SNR | -19.07 dB |
| Max SNR | 38.05 dB |

![SNR](14_snr_analysis.png)

## 6. V Peak (5. Dalga) Analizi

| Metrik | Latency | Amplitude |
|--------|---------|-----------|
| Mevcut | 38,754 (74.6%) | 38,754 (74.6%) |
| Eksik | 13,207 | 13,207 |
| İkisi birden mevcut | 38,754 (74.6%) | |

| İstatistik | Latency | Amplitude |
|------------|---------|-----------|
| Ortalama | 7.834 ms | 0.174 µV |
| Std | 1.71 | 0.1566 |
| Min | 4.8 | -0.688 |
| Max | 18.47 | 1.249 |

![V Peak](15_vpeak_analysis.png)

![V Peak Sınıf](16_vpeak_by_class.png)

## 7. Hasta Bazlı Analiz

| Metrik | Değer |
|--------|-------|
| Hasta sayısı | 2,038 |
| Hasta başına ort. örnek | 25.5 |
| Min / Max | 1 / 125 |
| Medyan | 24 |

![Hasta Analizi](17_patient_analysis.png)

## 8. Eksik Veri Analizi

Eksik veri içeren sütun sayısı: **28**

| Sütun | Eksik | Oran |
|-------|-------|------|
| I Latancy | 49,196 | 89.1% |
| I Amplitude | 49,196 | 89.1% |
| III Latancy | 48,671 | 88.1% |
| III Amplitude | 48,671 | 88.1% |
| Hear Loss - Left | 36,211 | 65.6% |
| Hear Loss - Right | 36,166 | 65.5% |
| 456 | 16,963 | 30.7% |
| 457 | 16,963 | 30.7% |
| 463 | 16,963 | 30.7% |
| 462 | 16,963 | 30.7% |

![Eksik Veri](18_missing_data.png)

## 9. Intensity Analizi

Benzersiz intensity seviyeleri: **21**
Aralık: **0 – 100 dB**

![Intensity](19_intensity_analysis.png)

![Intensity Sinyalleri](20_intensity_signals.png)
