.PHONY: help setup setup-dev lint format test prepare-data train-clf train-seg run-experiments app clean

PYTHON     := uv run python
CONFIG_CLF := configs/train_clf.yaml
CONFIG_SEG := configs/train_seg.yaml

help:
	@echo "  setup           Instala dependencias de producción"
	@echo "  setup-dev       Instala dependencias + desarrollo"
	@echo "  lint            Verifica estilo con Ruff"
	@echo "  format          Formatea con Black + Ruff"
	@echo "  test            Ejecuta tests con pytest"
	@echo "  prepare-data    Organiza imágenes raw → classification/"
	@echo "  train-clf       Entrena EfficientNet-B2 (Fase 1)"
	@echo "  train-clf-cmp   Entrena ConvNeXt-Tiny (comparativo)"
	@echo "  train-seg       Entrena YOLOv11-seg (Fase 2)"
	@echo "  convert-ann     JSON X-AnyLabeling → YOLO format"
	@echo "  run-experiments Ejecuta pipeline de experimentos de segmentación"
	@echo "  app             Lanza Streamlit"
	@echo "  clean           Limpia artefactos"

setup:
	uv sync

setup-dev:
	uv sync --extra dev
	uv run pre-commit install

lint:
	uv run ruff check src/ app/ tests/

format:
	uv run black src/ app/ tests/
	uv run ruff check --fix src/ app/ tests/

test:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing

prepare-data:
	uv run prepare-data --src-dir data/raw/ --dest-dir data/processed/classification/

train-clf:
	uv run train-clf --config $(CONFIG_CLF)

train-clf-cmp:
	uv run train-clf --config $(CONFIG_CLF) --model convnext_tiny

train-seg:
	uv run train-seg --config $(CONFIG_SEG)

run-experiments:
	uv run python src/training/run_experiments.py

convert-ann:
	uv run convert-ann data/raw/annotations/ \
		data/processed/segmentation/labels/ \
		--images-dir data/raw/images/

app:
	uv run streamlit run app/app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .coverage htmlcov/ dist/ build/ 2>/dev/null; true