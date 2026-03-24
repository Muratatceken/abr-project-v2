# ============================================================================
# ABR Transformer V2 — Makefile
# ============================================================================
# Kullanım:
#   make install          — Bağımlılıkları kur
#   make train            — Eğitimi başlat
#   make eval             — Değerlendirme çalıştır
#   make sample           — Sentetik ABR sinyalleri üret
#   make all              — Eğitim → değerlendirme → örnekleme (sıralı)
#   make test             — Testleri çalıştır
#   make lint             — Kod kalitesi kontrolü
#   make clean            — Geçici dosyaları temizle
#   make tensorboard      — TensorBoard'u başlat
#
# Özelleştirme (komut satırından):
#   make train CONFIG=configs/train_hpo_optimized.yaml
#   make train OVERRIDE="optim.lr: 5e-5, trainer.max_epochs: 200"
#   make eval  CHECKPOINT=checkpoints/my_model_best.pt
#   make sample NUM_SAMPLES=100 STEPS=100
# ============================================================================

SHELL := /bin/bash
PYTHON ?= python3

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_CONFIG   ?= configs/train.yaml
EVAL_CONFIG    ?= configs/eval.yaml
HPO_CONFIG     ?= configs/hpo_search_space.yaml
CHECKPOINT     ?= checkpoints/abr_transformer/abr_intensity_cond_v2_best.pt
DATA_PATH      ?= data/processed/ultimate_dataset_with_clinical_thresholds.pkl
OVERRIDE       ?=

# ── Sampling defaults ──────────────────────────────────────────────────────
NUM_SAMPLES    ?= 16
SAMPLE_STEPS   ?= 50
CFG_SCALE      ?= 1.0
ETA            ?= 0.0
OUTPUT_DIR     ?= outputs/samples

# ── TensorBoard ───────────────────────────────────────────────────────────
TB_LOGDIR      ?= runs/abr_transformer
TB_PORT        ?= 6006

# ============================================================================
# Ana hedefler
# ============================================================================

.PHONY: all train eval sample test lint format clean install preprocess \
        tensorboard hpo resume help check-data

## Tüm pipeline'ı çalıştır: eğitim → değerlendirme
all: train eval
	@echo "✓ Pipeline tamamlandı."

## Bağımlılıkları kur
install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	@echo "✓ Bağımlılıklar kuruldu."

# ============================================================================
# Veri
# ============================================================================

## Veri analizi (görseller + rapor)
analyze:
	$(PYTHON) scripts/analyze_dataset.py --excel data/abr_dataset.xlsx --output analysis_results

## Veri ön-işleme (Excel → pkl)
preprocess:
	$(PYTHON) -m data.preprocessing
	@echo "✓ Veri ön-işleme tamamlandı."

## Veri dosyasının varlığını kontrol et
check-data:
	@test -f $(DATA_PATH) || (echo "HATA: $(DATA_PATH) bulunamadı. Önce 'make preprocess' çalıştırın." && exit 1)

# ============================================================================
# Eğitim
# ============================================================================

## Modeli eğit
train: check-data
	$(PYTHON) train.py \
		--config $(TRAIN_CONFIG) \
		$(if $(OVERRIDE),--override "$(OVERRIDE)",)
	@echo "✓ Eğitim tamamlandı."

## Eğitime kaldığı yerden devam et
resume: check-data
	@test -f $(CHECKPOINT) || (echo "HATA: $(CHECKPOINT) bulunamadı." && exit 1)
	$(PYTHON) train.py \
		--config $(TRAIN_CONFIG) \
		--resume $(CHECKPOINT) \
		$(if $(OVERRIDE),--override "$(OVERRIDE)",)
	@echo "✓ Eğitim devam etti."

## HPO-optimize edilmiş ayarlarla eğit
train-hpo: check-data
	$(PYTHON) train.py \
		--config configs/train_hpo_optimized.yaml \
		$(if $(OVERRIDE),--override "$(OVERRIDE)",)

## Hyperparameter optimizasyonu çalıştır
hpo: check-data
	$(PYTHON) scripts/train_with_hpo.py \
		--config $(HPO_CONFIG)

# ============================================================================
# Değerlendirme
# ============================================================================

## Modeli değerlendir (reconstruction mode)
eval: check-data
	@test -f $(CHECKPOINT) || (echo "HATA: $(CHECKPOINT) bulunamadı. Önce 'make train' çalıştırın." && exit 1)
	$(PYTHON) eval.py \
		--config $(EVAL_CONFIG) \
		$(if $(OVERRIDE),--override "$(OVERRIDE)",)
	@echo "✓ Değerlendirme tamamlandı."

# ============================================================================
# Sentetik sinyal üretimi (Inference)
# ============================================================================

## Sentetik ABR sinyalleri üret
sample: check-data
	@test -f $(CHECKPOINT) || (echo "HATA: $(CHECKPOINT) bulunamadı." && exit 1)
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) -c "\
import torch, numpy as np; \
from models import ABRTransformerGenerator; \
from inference import DDIMSampler; \
\
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); \
print(f'Device: {device}'); \
\
ckpt = torch.load('$(CHECKPOINT)', map_location=device, weights_only=False); \
cfg = ckpt['config']; \
model = ABRTransformerGenerator(**cfg['model']).to(device); \
model.load_state_dict(ckpt['model_state_dict']); \
model.eval(); \
\
sampler = DDIMSampler(model, cfg['diffusion']['num_train_steps'], device); \
samples = sampler.sample(batch_size=$(NUM_SAMPLES), steps=$(SAMPLE_STEPS), eta=$(ETA), cfg_scale=$(CFG_SCALE)); \
\
np.save('$(OUTPUT_DIR)/synthetic_abr_signals.npy', samples.cpu().numpy()); \
print(f'Shape: {samples.shape}'); \
print(f'Saved to $(OUTPUT_DIR)/synthetic_abr_signals.npy'); \
"

## Sınıf-koşullu sinyal üret (belirli hearing loss tipi ve intensity ile)
sample-conditioned: check-data
	@test -f $(CHECKPOINT) || (echo "HATA: $(CHECKPOINT) bulunamadı." && exit 1)
	@mkdir -p $(OUTPUT_DIR)
	$(PYTHON) -c "\
import torch, numpy as np; \
from models import ABRTransformerGenerator; \
from inference import DDIMSampler; \
\
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); \
ckpt = torch.load('$(CHECKPOINT)', map_location=device, weights_only=False); \
cfg = ckpt['config']; \
model = ABRTransformerGenerator(**cfg['model']).to(device); \
model.load_state_dict(ckpt['model_state_dict']); \
model.eval(); \
\
sampler = DDIMSampler(model, cfg['diffusion']['num_train_steps'], device); \
\
# Generate for each class at 80 dB intensity \
classes = ['NORMAL', 'SNIK', 'ITIK', 'TOTAL', 'NOROPATI']; \
all_samples = []; \
for cls_id in range(5): \
    intensity = torch.tensor([0.8] * 4, device=device); \
    aux = torch.zeros(4, 3, device=device); \
    cls = torch.tensor([cls_id] * 4, device=device); \
    s = sampler.sample_class_conditioned(cls, intensity=intensity, aux_static=aux, \
                                         steps=$(SAMPLE_STEPS), cfg_scale=$(CFG_SCALE)); \
    all_samples.append(s.cpu().numpy()); \
    print(f'  {classes[cls_id]}: {s.shape}'); \
all_samples = np.concatenate(all_samples, axis=0); \
np.save('$(OUTPUT_DIR)/synthetic_abr_class_conditioned.npy', all_samples); \
print(f'Saved {all_samples.shape[0]} samples to $(OUTPUT_DIR)/'); \
"

# ============================================================================
# Test & Kalite
# ============================================================================

## Tüm testleri çalıştır
test:
	$(PYTHON) -m pytest tests/ -v --tb=short

## Tek test dosyası çalıştır (TEST=tests/test_abr_transformer.py)
TEST ?= tests/test_abr_transformer.py
test-one:
	$(PYTHON) -m pytest $(TEST) -v --tb=short

## Testleri coverage ile çalıştır
test-cov:
	$(PYTHON) -m pytest tests/ -v --cov=. --cov-report=term-missing --tb=short

## Kod formatlama
format:
	$(PYTHON) -m black .
	$(PYTHON) -m isort .
	@echo "✓ Kod formatlandı."

## Lint kontrolü
lint:
	$(PYTHON) -m flake8 . --max-line-length=120 --exclude=.git,__pycache__,*.egg-info
	@echo "✓ Lint kontrolü tamamlandı."

# ============================================================================
# Araçlar
# ============================================================================

## TensorBoard başlat
tensorboard:
	@echo "TensorBoard başlatılıyor: http://localhost:$(TB_PORT)"
	$(PYTHON) -m tensorboard.main --logdir=$(TB_LOGDIR) --port=$(TB_PORT) --bind_all

## Geçici dosyaları temizle
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache
	@echo "✓ Geçici dosyalar temizlendi."

## Eğitim çıktılarını temizle (DİKKAT: checkpoint ve logları siler!)
clean-runs:
	@echo "DİKKAT: Bu komut checkpoint ve TensorBoard loglarını silecek!"
	@read -p "Devam etmek istiyor musunuz? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	rm -rf runs/ checkpoints/ monitoring/ $(OUTPUT_DIR)
	@echo "✓ Eğitim çıktıları temizlendi."

## Kullanılabilir komutları listele
help:
	@echo ""
	@echo "ABR Transformer V2 — Komutlar"
	@echo "════════════════════════════════════════════════════════"
	@echo ""
	@echo "  Kurulum & Veri:"
	@echo "    make install          Bağımlılıkları kur"
	@echo "    make preprocess       Excel → pkl veri dönüşümü"
	@echo ""
	@echo "  Eğitim:"
	@echo "    make train            Modeli eğit"
	@echo "    make resume           Checkpoint'tan devam et"
	@echo "    make train-hpo        HPO-optimize ayarlarla eğit"
	@echo "    make hpo              Hyperparameter optimizasyonu"
	@echo ""
	@echo "  Değerlendirme & Üretim:"
	@echo "    make eval             Modeli değerlendir"
	@echo "    make sample           Sentetik sinyal üret"
	@echo "    make sample-conditioned  Koşullu sinyal üret"
	@echo ""
	@echo "  Test & Kalite:"
	@echo "    make test             Testleri çalıştır"
	@echo "    make test-cov         Coverage ile test"
	@echo "    make lint             Flake8 kontrolü"
	@echo "    make format           Black + isort"
	@echo ""
	@echo "  Araçlar:"
	@echo "    make tensorboard      TensorBoard başlat"
	@echo "    make clean            Geçici dosyaları temizle"
	@echo "    make all              Eğitim → değerlendirme"
	@echo ""
	@echo "  Özelleştirme örnekleri:"
	@echo '    make train CONFIG=configs/train_hpo_optimized.yaml'
	@echo '    make train OVERRIDE="optim.lr: 5e-5"'
	@echo '    make eval  CHECKPOINT=checkpoints/my_best.pt'
	@echo '    make sample NUM_SAMPLES=100 STEPS=100 CFG_SCALE=1.5'
	@echo ""
