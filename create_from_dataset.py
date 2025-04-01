import os
import base64
import soundfile as sf

import sphn
import argilla as rg
from datasets import load_dataset

ds = load_dataset("speech-uk/test-ml-datahoarding")

# Config
ds_source = "speech-uk/test-ml-datahoarding"
ds_destination = "audio-test-15"

argilla_addr = "http://localhost:6900"
argilla_api = "argilla.apikey"

# Load dataset
ds = load_dataset(ds_source)


# Helpers
def convert_wav_file_to_base64_ogg(wav_file_path):
    data, sr = sphn.read(wav_file_path)

    sphn.write_opus("tmp.ogg", data, sr)

    with open("tmp.ogg", "rb") as ogg_file:
        ogg_data = ogg_file.read()
    ogg_base64 = base64.b64encode(ogg_data).decode("utf-8")

    return ogg_base64


# Create Argilla client
client = rg.Argilla(
    api_url=argilla_addr,
    api_key=argilla_api,
)

# Add settings
settings = rg.Settings(
    guidelines="Correct transcription field after listening to the audio.",
    fields=[
        rg.CustomField(
            name="audio_file",
            title="Audio",
            template="""<div style="content-visibility: auto">
<audio controlslist="noplaybackrate nodownload nofullscreen" preload="metadata" class="h-dvh max-h-[2.25rem] w-full min-w-[300px] max-w-xs" controls=""><source src="data:audio/ogg;base64,{{record.fields.audio_file.source_base64}}" type="audio/ogg"></audio>
</div>""",
            advanced_mode=False,
            required=True,
            description="Audio",
        ),
        rg.TextField(
            name="language",
            title="Language of the audio",
        ),
        rg.TextField(
            name="transcription",
            title="Original transcription",
        ),
        rg.TextField(
            name="file_name",
            title="File name",
        ),
    ],
    questions=[
        rg.LabelQuestion(
            name="correct_language", title="Language", labels=["uk", "ru", "mix"]
        ),
        rg.TextField(
            name="fixed_transcription",
            title="Transcription",
        ),
    ],
)

# Create Argilla dataset
dataset = rg.Dataset(
    name=ds_destination,
    settings=settings,
)

dataset.create()

# Generate records
for row in ds["train"]:
    audio = row["audio"]
    audio_file = audio["path"]

    # Write the audio file to a temporary location
    sf.write(audio_file, audio["array"], samplerate=audio["sampling_rate"])

    transcription = row["transcription"]
    lang = "uk"

    ogg_base64 = convert_wav_file_to_base64_ogg(audio_file)

    record = {
        "file_name": audio_file,
        "audio_file": {
            "source_base64": ogg_base64,
        },
        "language": lang,
        "correct_language": lang,
        "transcription": transcription,
        "fixed_transcription": transcription,
    }

    # Remove the temporary audio file
    if os.path.exists(audio_file):
        os.remove(audio_file)

    # Remove the temporary ogg file
    if os.path.exists("tmp.ogg"):
        os.remove("tmp.ogg")

    # Upload record to the Argilla dataset
    dataset.records.log([record])
