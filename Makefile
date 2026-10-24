PYTHON ?= .venv/bin/python
# Python 3.11 is required (pandas 1.5.3 has no wheels for newer versions).
PYTHON311 ?= python3.11


.PHONY: help setup setup-verify data audit test semisynthetic obd study figures paper slides demo demo-adaptive app all

help:
	@printf '%s\n' 'setup          Create .venv with pinned dependencies (needs python3.11)'
	@printf '%s\n' 'setup-verify   Also install obp (pulls PyTorch, ~1 GB) for make test'
	@printf '%s\n' 'data           Download the Open Bandit Dataset (~413 MB zip, 11 GB CSV) and convert to Parquet'
	@printf '%s\n' 'audit          Summarise rows, dates, click rates and propensities'
	@printf '%s\n' 'test           Check our estimators against obp (needs make setup-verify)'
	@printf '%s\n' 'semisynthetic  Semi-synthetic experiments with exact ground truth (~15 min)'
	@printf '%s\n' 'obd            Real Open Bandit launch experiments, 12-hour windows (~40 min)'
	@printf '%s\n' 'study          Compare launch rules on both experiment sets'
	@printf '%s\n' 'figures        Paper figures'
	@printf '%s\n' 'paper / slides Compile the IEEE paper / Beamer slides'
	@printf '%s\n' 'demo           Live launch decision on one real window'
	@printf '%s\n' 'demo-adaptive  Same, also showing the adaptive (ML difficulty model) margin'
	@printf '%s\n' 'app            Interactive web demo in the browser (http://localhost:8501)'

setup:
	$(PYTHON311) -m venv .venv
	.venv/bin/python -m pip install --upgrade pip
	.venv/bin/python -m pip install -r requirements.txt

setup-verify:
	.venv/bin/python -m pip install -r requirements-verify.txt

data:
	mkdir -p data/raw
	curl -L -o data/raw/open_bandit_dataset.zip https://research.zozo.com/data_release/open_bandit_dataset.zip
	cd data/raw && unzip -q open_bandit_dataset.zip
	$(PYTHON) scripts/convert_obd.py

audit:
	$(PYTHON) scripts/audit_obd.py

test:
	$(PYTHON) tests/test_against_obp.py

semisynthetic:
	$(PYTHON) scripts/semisynthetic_experiments.py

obd:
	$(PYTHON) scripts/obd_experiments.py --window-hours 12

study:
	$(PYTHON) scripts/launch_study.py --permutations 100

figures:
	$(PYTHON) scripts/make_tables.py
	$(PYTHON) scripts/make_figures.py

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

slides:
	cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

demo:
	$(PYTHON) scripts/demo_launch.py --campaign men --window 12 --direction random-to-bts

demo-adaptive:
	$(PYTHON) scripts/demo_launch.py --campaign men --window 12 --direction random-to-bts --adaptive

app:
	$(PYTHON) -m streamlit run app/app.py

all: test semisynthetic obd study figures paper slides
