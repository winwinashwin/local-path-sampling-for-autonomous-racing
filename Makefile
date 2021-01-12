SHELL            := /bin/bash
.DEFAULT_GOAL    := all

.PHONY: all
all: _vis

.PHONY: _vis
_vis:
	source venv/bin/activate && \
	python -m visualisation.main