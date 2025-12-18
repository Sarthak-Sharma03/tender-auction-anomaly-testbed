SHELL := /bin/bash

.PHONY: help install test quickstart quickstart-synth report lint

help:
	@echo "Targets:"
	@echo "  install         Install editable with dev extras"
	@echo "  test            Run pytest"
	@echo "  quickstart      Run quickstart on sample xlsx"
	@echo "  quickstart-synth Run quickstart on synthetic data"
	@echo "  report          Build the technical report PDF"
	@echo "  lint            Run ruff"

install:
	python -m pip install -e ".[dev]"

test:
	pytest -q

quickstart:
	tad quickstart --out-dir reports/quickstart

quickstart-synth:
	tad quickstart --synthetic --out-dir reports/quickstart_synthetic

report:
	python tools/build_technical_report_pdf.py

lint:
	ruff check .
