SHELL := /bin/bash
UV    := uv run

.PHONY: docs-verify docs-generate docs-federate docs-build docs-serve
docs-verify:  ## Fail if any public symbol/CLI command is undocumented
	$(UV) python scripts/generate_docs.py --verify

docs-generate:  ## Generate API + CLI MDX
	$(UV) python scripts/generate_docs.py

docs-federate:  ## Materialize pinned starter-kit docs
	$(UV) python scripts/sync_starterkit_docs.py

docs-build: docs-generate docs-verify docs-federate  ## Full local docs build
	cd website && npm run build && npm run check:gh-pages

docs-serve: docs-generate docs-federate  ## Live-preview the docs locally
	cd website && npm run dev
