.PHONY: install test demo api clean

install:
	pip install -e ".[all,dev]"

test:
	pytest -q

demo:
	bash examples/run_demo.sh

api:
	uvicorn api.main:app --reload

clean:
	rm -rf runs demo_runs .rageval_cache.sqlite .pytest_cache
