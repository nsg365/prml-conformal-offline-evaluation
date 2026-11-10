PYTHON ?= .venv/bin/python
# Python 3.11 is required (pandas 1.5.3 has no wheels for newer versions).
PYTHON311 ?= python3.11


.PHONY: help setup setup-verify data audit test semisynthetic obd study figures paper slides demo demo-adaptive app overleaf all clean

help:
	@printf '%s\n' 'setup          Create .venv with pinned dependencies (needs python3.11)'
	@printf '%s\n' 'setup-verify   Also install obp (pulls PyTorch, ~1 GB) for make test'
	@printf '%s\n' 'data           Download the Open Bandit Dataset (~413 MB zip, 11 GB CSV) and convert to Parquet'
	@printf '%s\n' 'audit          Summarise rows, dates, click rates and propensities -> results/obd_audit.json'
	@printf '%s\n' 'test           Check our estimators against obp (needs make setup-verify)'
	@printf '%s\n' 'semisynthetic  Semi-synthetic experiments with exact ground truth (~15 min)'
	@printf '%s\n' 'obd            Real Open Bandit launch experiments, 12-hour windows (~40 min)'
	@printf '%s\n' 'study          Compare launch rules on both experiment sets -> results/launch_study/'
	@printf '%s\n' 'figures        LaTeX tables (paper/tables, slides/tables) and figures (figures/)'
	@printf '%s\n' 'paper / slides Compile the IEEE paper / Beamer slides'
	@printf '%s\n' 'demo           Live launch decision on one real window'
	@printf '%s\n' 'demo-adaptive  Same, also showing the adaptive (ML difficulty model) margin'
	@printf '%s\n' 'app            Interactive web demo in the browser (http://localhost:8501)'
	@printf '%s\n' 'overleaf       Self-contained zips for Overleaf -> build/paper_overleaf.zip, build/slides_overleaf.zip'
	@printf '%s\n' 'clean          Delete Python caches and LaTeX helper files'

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

# LaTeX helper files are deleted after each build; only main.pdf is kept.
LATEX_LEFTOVERS = main.aux main.bbl main.blg main.fdb_latexmk main.fls main.log main.out main.nav main.snm main.toc main.vrb

paper:
	cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && rm -f $(LATEX_LEFTOVERS)

slides:
	cd slides && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex && rm -f $(LATEX_LEFTOVERS)

demo:
	$(PYTHON) scripts/demo_launch.py --campaign men --window 12 --direction random-to-bts

demo-adaptive:
	$(PYTHON) scripts/demo_launch.py --campaign men --window 12 --direction random-to-bts --adaptive

app:
	$(PYTHON) -m streamlit run app/app.py

# Overleaf wants each project in one folder, so copy the shared figures in and fix the paths.
overleaf:
	rm -rf build/overleaf && mkdir -p build/overleaf/paper build/overleaf/slides
	cp -R paper/main.tex paper/references.bib paper/tables figures build/overleaf/paper/
	cp -R slides/main.tex slides/tables figures build/overleaf/slides/
	rm -f build/overleaf/*/figures/*_dark.png
	sed -i.bak 's#{../figures/#{figures/#g' build/overleaf/paper/main.tex build/overleaf/slides/main.tex
	rm -f build/overleaf/*/main.tex.bak build/paper_overleaf.zip build/slides_overleaf.zip
	cd build/overleaf/paper && zip -qr ../../paper_overleaf.zip .
	cd build/overleaf/slides && zip -qr ../../slides_overleaf.zip .
	rm -rf build/overleaf

all: test semisynthetic obd study figures paper slides

clean:
	find . -path ./.venv -prune -o -name __pycache__ -type d -exec rm -rf {} +
	cd paper && rm -f $(LATEX_LEFTOVERS)
	cd slides && rm -f $(LATEX_LEFTOVERS)
