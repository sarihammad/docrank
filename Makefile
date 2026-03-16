.PHONY: install index serve test evaluate docker-up docker-down clean

# --------------------------------------------------------------------------
# Development
# --------------------------------------------------------------------------

install:
	pip install --upgrade pip
	pip install -r requirements.txt

# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------

index:
	python scripts/index_corpus.py --source scifact --output-dir indexes

index-dir:
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Usage: make index-dir DATA_PATH=/path/to/docs"; \
		exit 1; \
	fi
	python scripts/index_corpus.py --source directory --data-path $(DATA_PATH) --output-dir indexes

index-jsonl:
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Usage: make index-jsonl DATA_PATH=/path/to/corpus.jsonl"; \
		exit 1; \
	fi
	python scripts/index_corpus.py --source jsonl --data-path $(DATA_PATH) --output-dir indexes

# --------------------------------------------------------------------------
# Serving
# --------------------------------------------------------------------------

serve:
	uvicorn src.serving.app:app \
		--host 0.0.0.0 \
		--port 8000 \
		--reload \
		--log-level info

# --------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------

test:
	pytest tests/ -v --tb=short

test-fast:
	pytest tests/test_pipeline.py -v --tb=short -k "not TestCrossEncoder"

# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

evaluate:
	python scripts/evaluate.py \
		--index-dir indexes \
		--k-values 5 10 20 100 \
		--output evaluation_results.json

evaluate-with-judge:
	python scripts/evaluate.py \
		--index-dir indexes \
		--k-values 5 10 20 100 \
		--with-llm-judge \
		--output evaluation_results.json

# --------------------------------------------------------------------------
# Docker
# --------------------------------------------------------------------------

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f api

# --------------------------------------------------------------------------
# Cleanup
# --------------------------------------------------------------------------

clean:
	rm -rf indexes/*.pkl indexes/*.index indexes/*.json
	rm -f evaluation_results.json
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
