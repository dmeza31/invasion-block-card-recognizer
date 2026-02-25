.PHONY: install download-cards build-index run-api run-ui augment-dataset composite-cards composite-all-cards gst gpush

install:
	poetry install

download-cards:
	poetry run python -m dataset.downloader

build-index:
	poetry run python -m recognizer.build_index

run-api:
	poetry run uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

run-ui:
	poetry run streamlit run src/ui/app.py --server.port 8501

augment-dataset:
	poetry run python -m dataset.augmentor --input-dir data/reference_images --output-dir data/augmented_images --num-variants 30

composite-cards:
	poetry run python -c "from pathlib import Path; from dataset.compositor import download_sample_backgrounds, composite_on_backgrounds; backgrounds=Path('data/backgrounds'); download_sample_backgrounds(backgrounds, count=20); candidates=sorted(Path('data/reference_images').rglob('*Absorb*.jpg')) or sorted(Path('data/reference_images').rglob('*.jpg')); card=candidates[0] if candidates else (_ for _ in ()).throw(FileNotFoundError('No reference card images found under data/reference_images')); print(f'Using card image: {card}'); composite_on_backgrounds(card_image_path=card, backgrounds_dir=backgrounds, output_path=Path('data/composited'), count=10)"

composite-all-cards:
	poetry run python -m dataset.compositor --reference-dir data/reference_images --backgrounds-dir data/backgrounds --output-dir data/composited --count-per-image 10 --background-count 20

gst:
	git status

gpush:
	@if [ -z "$(MSG)" ]; then \
		echo "Usage: make gpush MSG=\"your commit message\""; \
		exit 1; \
	fi
	git add .
	git commit -m "$(MSG)"
	git push