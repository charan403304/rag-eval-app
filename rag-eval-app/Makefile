.PHONY: install test eval eval-judge run index docker

install:
	pip install -r requirements.txt

index:
	python scripts/build_index.py

test:
	pytest tests/ -v

eval:
	python eval/run_eval.py --json eval_report.json

eval-judge:
	python eval/run_eval.py --llm-judge --json eval_report.json

run:
	streamlit run app.py

docker:
	docker compose up --build
