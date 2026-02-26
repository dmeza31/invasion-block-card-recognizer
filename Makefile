.PHONY: install download-cards build-index build-faiss-index run-api run-ui augment-dataset composite-cards composite-all-cards embed-cards test-recognizer test-recognizer-reference test-recognizer-augmented test-recognizer-composite-no-detector test-recognizer-reference-no-detector test-recognizer-augmented-no-detector test-detector gst gpush

install:
	poetry install

download-cards:
	poetry run python -m dataset.downloader

build-index:
	poetry run python -m recognizer.index_builder

build-faiss-index:
	poetry run python -m recognizer.index_builder

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

embed-cards:
	poetry run python -c "import json, numpy as np; from pathlib import Path; from recognizer.embedder import CardEmbedder; image_dir=Path('data/reference_images'); out_embeddings=Path('data/card_embeddings.npy'); out_metadata=Path('data/embedding_metadata.json'); out_embeddings.parent.mkdir(parents=True, exist_ok=True); embedder=CardEmbedder(); embeddings, metadata=embedder.embed_directory(str(image_dir)); np.save(out_embeddings, embeddings); out_metadata.write_text(json.dumps(metadata, indent=2), encoding='utf-8'); print(f'Saved embeddings: {out_embeddings} shape={embeddings.shape}'); print(f'Saved metadata: {out_metadata} entries={len(metadata)}')"

test-recognizer-composite:
	poetry run pytest tests/test_recognizer_composite.py -s

test-recognizer-reference:
	poetry run pytest tests/test_recognizer_reference.py -s

test-recognizer-augmented:
	poetry run pytest tests/test_recognizer_augmented.py -s

test-recognizer-composite-no-detector:
	poetry run pytest tests/test_recognizer_composite_no_detector.py -s

test-recognizer-reference-no-detector:
	poetry run pytest tests/test_recognizer_reference_no_detector.py -s

test-recognizer-augmented-no-detector:
	poetry run pytest tests/test_recognizer_augmented_no_detector.py -s

test-detector:
	poetry run pytest tests/test_detector.py -q

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