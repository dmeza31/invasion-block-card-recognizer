from __future__ import annotations

import base64
import html
import os
from io import BytesIO
from typing import Any

import requests
import streamlit as st
from PIL import Image


def get_api_url() -> str:
	default_url = "http://localhost:8000"
	secret_url = None

	try:
		secret_url = st.secrets.get("API_URL")
	except Exception:
		secret_url = None

	return os.getenv("API_URL", secret_url or default_url).rstrip("/")


def _infer_mime_and_filename(image_bytes: bytes) -> tuple[str, str]:
	try:
		with Image.open(BytesIO(image_bytes)) as image:
			fmt = (image.format or "JPEG").upper()
	except Exception:
		fmt = "JPEG"

	if fmt == "PNG":
		return "image/png", "upload.png"

	return "image/jpeg", "upload.jpg"


def call_recognize_api(image_bytes: bytes) -> dict[str, Any]:
	mime_type, file_name = _infer_mime_and_filename(image_bytes)
	response = requests.post(
		f"{API_URL}/recognize",
		files={"file": (file_name, image_bytes, mime_type)},
		timeout=60,
	)
	response.raise_for_status()
	return response.json()


def call_detect_api(image_bytes: bytes) -> bytes:
	mime_type, file_name = _infer_mime_and_filename(image_bytes)
	response = requests.post(
		f"{API_URL}/detect",
		files={"file": (file_name, image_bytes, mime_type)},
		timeout=60,
	)
	response.raise_for_status()
	return response.content


def _get_error_detail(exc: requests.HTTPError) -> tuple[int, str]:
	status_code = exc.response.status_code if exc.response is not None else 500
	detail = "Unexpected error"

	if exc.response is not None:
		try:
			payload = exc.response.json()
			detail = payload.get("detail", detail)
		except Exception:
			detail = exc.response.text or detail

	return status_code, detail


def get_reference_image(set_code: str, collector_number: str) -> bytes | None:
	response = requests.get(
		f"{API_URL}/cards/{set_code}/{collector_number}/image",
		timeout=30,
	)

	if response.status_code == 404:
		return None

	response.raise_for_status()
	return response.content


def _thumbnail_data_uri(image_bytes: bytes, max_size: tuple[int, int] = (96, 96)) -> str:
	with Image.open(BytesIO(image_bytes)) as image:
		thumb = image.convert("RGB")
		thumb.thumbnail(max_size)
		buf = BytesIO()
		thumb.save(buf, format="JPEG", quality=85)

	encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
	return f"data:image/jpeg;base64,{encoded}"


def render_batch_results_table(rows: list[dict[str, Any]]) -> None:
	if not rows:
		st.info("No recognized cards found in batch.")
		return

	table_rows: list[str] = []
	for row in rows:
		thumb = row.get("thumbnail", "")
		name = html.escape(str(row.get("card_name", "")))
		set_code = html.escape(str(row.get("set_code", "")).upper())
		collector_number = html.escape(str(row.get("collector_number", "")))
		confidence = f"{float(row.get('confidence', 0.0)):.2f}"

		table_rows.append(
			"<tr>"
			f"<td><img src='{thumb}' width='72' style='border-radius:6px;border:1px solid #ddd;'/></td>"
			f"<td>{name}</td>"
			f"<td>{set_code}</td>"
			f"<td>{collector_number}</td>"
			f"<td>{confidence}</td>"
			"</tr>"
		)

	html_table = f"""
	<div style="max-height:420px; overflow-y:auto; border:1px solid #e6e6e6; border-radius:8px;">
	  <table style="width:100%; border-collapse:collapse;">
	    <thead style="position:sticky; top:0; background:#f7f7f7;">
	      <tr>
	        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Thumbnail</th>
	        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Card Name</th>
	        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Set</th>
	        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Collector #</th>
	        <th style="text-align:left; padding:8px; border-bottom:1px solid #ddd;">Confidence</th>
	      </tr>
	    </thead>
	    <tbody>
	      {''.join(table_rows)}
	    </tbody>
	  </table>
	</div>
	"""

	st.markdown(html_table, unsafe_allow_html=True)


def render_card_header(name: str, set_code: str, collector_number: str, confidence: float) -> None:
	st.markdown(
		f"""
		<div style="border:1px solid #e6e6e6;border-radius:10px;padding:12px;margin-bottom:12px;">
		  <h4 style="margin:0 0 6px 0;">{name}</h4>
		  <p style="margin:0; font-size:0.95rem;"><strong>Set:</strong> {set_code.upper()}</p>
		  <p style="margin:0; font-size:0.95rem;"><strong>Collector #:</strong> {collector_number}</p>
		  <p style="margin:0; font-size:0.95rem;"><strong>Confidence:</strong> {confidence:.2f}</p>
		</div>
		""",
		unsafe_allow_html=True,
	)


st.set_page_config(page_title="MTG Invasion Block Card Recognizer", layout="wide")

API_URL = get_api_url()

st.title("MTG Invasion Block Card Recognizer")
st.write(
	"Upload a card photo and run recognition against the Invasion block reference index. "
	"The app shows detected card output and top match candidates."
)
st.caption(f"Resolved API URL: {API_URL}")

with st.sidebar:
	st.header("About")
	st.write(
		"This app sends your uploaded card image to a FastAPI backend, "
		"runs card detection + recognition, and displays the most likely matches "
		"from the MTG Invasion block catalog."
	)
	st.caption(f"API URL: {API_URL}")


def render_single_image_results(uploaded_bytes: bytes, uploaded_label: str) -> None:
	uploaded_image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")

	try:
		with st.spinner("Running detection and recognition..."):
			detect_bytes = call_detect_api(uploaded_bytes)
			recognize_response = call_recognize_api(uploaded_bytes)
	except requests.HTTPError as exc:
		status_code, detail = _get_error_detail(exc)
		st.error(f"API error ({status_code}): {detail}")
		return
	except requests.RequestException as exc:
		st.error(f"Failed to connect to API at {API_URL}: {exc}")
		return
	except Exception as exc:
		st.error(f"Unexpected error: {exc}")
		return

	cards = recognize_response.get("cards", [])
	if not cards:
		st.info("No cards were recognized in the submitted image.")
		return

	st.subheader("Recognition Results")
	detected_image = Image.open(BytesIO(detect_bytes)).convert("RGB")

	for index, card in enumerate(cards, start=1):
		st.markdown(f"### Detected Card {index}")
		render_card_header(
			name=str(card.get("name", "")),
			set_code=str(card.get("set_code", "")),
			collector_number=str(card.get("collector_number", "")),
			confidence=float(card.get("confidence", 0.0)),
		)

		left_col, middle_col, right_col = st.columns(3)

		with left_col:
			st.caption(uploaded_label)
			st.image(uploaded_image, use_container_width=True)

		with middle_col:
			st.caption("Detected Card Output")
			st.image(detected_image, use_container_width=True)

		with right_col:
			st.caption("Matched Reference Image")
			ref_set = str(card.get("set_code", "")).lower()
			ref_number = str(card.get("collector_number", ""))

			try:
				ref_bytes = get_reference_image(ref_set, ref_number)
				if ref_bytes:
					st.image(Image.open(BytesIO(ref_bytes)).convert("RGB"), use_container_width=True)
				else:
					st.info("Reference image not found.")
			except requests.RequestException:
				st.info("Could not load reference image.")

		top_matches = card.get("top_matches", [])
		with st.expander(f"Top-5 Matches for {card.get('name', 'Card')}"):
			if not top_matches:
				st.write("No top matches returned.")
			else:
				for rank, match in enumerate(top_matches, start=1):
					st.write(
						f"{rank}. {match.get('name', '')} "
						f"({str(match.get('set_code', '')).upper()}) - "
						f"confidence: {float(match.get('confidence', 0.0)):.2f}"
					)


upload_tab, camera_tab, batch_tab = st.tabs(["Upload", "Camera", "Batch"])

with upload_tab:
	uploaded_file = st.file_uploader(
		"Upload a card photo",
		type=["jpg", "jpeg", "png"],
		accept_multiple_files=False,
		key="upload_single",
	)

	if uploaded_file is not None:
		uploaded_bytes = uploaded_file.getvalue()
		uploaded_image = Image.open(BytesIO(uploaded_bytes)).convert("RGB")
		st.subheader("Uploaded Image")
		st.image(uploaded_image, caption=uploaded_file.name, use_container_width=True)

	if st.button("Recognize Card", type="primary", key="btn_upload_recognize"):
		if uploaded_file is None:
			st.warning("Please upload an image first.")
		else:
			render_single_image_results(uploaded_file.getvalue(), "Uploaded Photo")

with camera_tab:
	st.write("Take a photo directly from your camera and run recognition.")
	camera_file = st.camera_input("Capture card photo", key="camera_capture")

	if camera_file is not None:
		camera_bytes = camera_file.getvalue()
		camera_image = Image.open(BytesIO(camera_bytes)).convert("RGB")
		st.subheader("Captured Image")
		st.image(camera_image, use_container_width=True)

	if st.button("Recognize Captured Card", type="primary", key="btn_camera_recognize"):
		if camera_file is None:
			st.warning("Please capture a photo first.")
		else:
			render_single_image_results(camera_file.getvalue(), "Captured Photo")

with batch_tab:
	st.write("Upload multiple images and process them in batch.")
	batch_files = st.file_uploader(
		"Upload card photos",
		type=["jpg", "jpeg", "png"],
		accept_multiple_files=True,
		key="upload_batch",
	)

	if st.button("Run Batch Recognition", type="primary", key="btn_batch_recognize"):
		if not batch_files:
			st.warning("Please upload one or more images for batch recognition.")
		else:
			rows: list[dict[str, Any]] = []
			progress = st.progress(0.0)

			for idx, batch_file in enumerate(batch_files, start=1):
				image_bytes = batch_file.getvalue()
				thumbnail = _thumbnail_data_uri(image_bytes)

				try:
					response = call_recognize_api(image_bytes)
				except requests.HTTPError as exc:
					status_code, detail = _get_error_detail(exc)
					st.warning(f"{batch_file.name}: API error ({status_code}) - {detail}")
					progress.progress(idx / len(batch_files))
					continue
				except requests.RequestException as exc:
					st.warning(f"{batch_file.name}: request failed - {exc}")
					progress.progress(idx / len(batch_files))
					continue

				cards = response.get("cards", [])
				if not cards:
					rows.append(
						{
							"thumbnail": thumbnail,
							"card_name": "No match",
							"set_code": "-",
							"collector_number": "-",
							"confidence": 0.0,
						}
					)
				else:
					for card in cards:
						rows.append(
							{
								"thumbnail": thumbnail,
								"card_name": str(card.get("name", "")),
								"set_code": str(card.get("set_code", "")),
								"collector_number": str(card.get("collector_number", "")),
								"confidence": float(card.get("confidence", 0.0)),
							}
						)

				progress.progress(idx / len(batch_files))

			st.subheader("Batch Results")
			render_batch_results_table(rows)
