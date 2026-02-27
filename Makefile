.PHONY: setup-models run

setup-models:
	bash scripts/setup_models.sh

run:
	python3 -m streamlit run app.py --server.address localhost --server.port 8501 --browser.gatherUsageStats false
