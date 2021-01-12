SHELL            := /bin/bash
.DEFAULT_GOAL    := all

.PHONY: all
all: _vis

.PHONY: _vis
_vis:
	python -m visualisation.main