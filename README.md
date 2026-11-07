# Text Summarization

End-to-end project that fine-tunes **Pegasus** on **SAMSum** dialogue data and serves summaries in a Streamlit web page.

The pipeline downloads the dataset, validates the splits, tokenizes the text, trains the model, and scores it with ROUGE. After training, `streamlit_app.py` loads the saved weights and summarizes new text. The page has **Short**, **Medium**, and **Long** length settings.

## Project description

Abstractive summarization writes a shorter version of the input in new words. This project does that for chats and short English documents.

1. SAMSum conversations are downloaded and stored under `artifacts/data_ingestion/`.
2. Dialogues are tokenized with the Pegasus tokenizer.
3. `google/pegasus-cnn_dailymail` is fine-tuned for one epoch on those dialogues.
4. The fine-tuned model is saved and scored on the test split.
5. The Streamlit app loads that saved model and returns a summary.

The web page is the main user interface. FastAPI in `app.py` is optional.

## Model

| Item | Value |
| --- | --- |
| Base model | [`google/pegasus-cnn_dailymail`](https://huggingface.co/google/pegasus-cnn_dailymail) |
| Architecture | Pegasus (`PegasusForConditionalGeneration`), encoder–decoder seq2seq |
| Pretraining | CNN / DailyMail news summarization |
| Fine-tuning data | SAMSum (train / validation / test dialogues) |
| Saved model | `artifacts/model_trainer/pegasus-samsum-model` |
| Saved tokenizer | `artifacts/model_trainer/tokenizer` |
| Generation | beam search, `num_beams=8` |
| Metrics | ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum in `artifacts/model_evaluation/metrics.csv` |

Pegasus is a Transformer built for abstractive summarization. The CNN / DailyMail checkpoint already summarizes articles. Fine-tuning on SAMSum adapts it to chat-style text (`Name: message`).

The Streamlit app calls `model.generate`. Input is truncated at 1024 tokens. Length on the page maps to:

| Setting | Result | Token range | Length penalty |
| --- | --- | --- | --- |
| Short | A few sentences | 8–40 | 0.5 |
| Medium | Default recap | 20–80 | 0.8 |
| Long | More detail | 40–128 | 1.2 |

These knobs change generation length only. They do not retrain the model. SAMSum targets are short, so **Long** is still a paragraph.

## Repository layout

```text
Text-Summarization-Project/
├── main.py                          # runs the five training stages
├── streamlit_app.py                 # web UI for prediction
├── app.py                           # optional FastAPI /predict and /train
├── params.yaml                      # recorded training hyperparameters
├── setup.py                         # installs the textSummarizer package
├── requirements.txt                 # Python dependencies
├── template.py                      # original project scaffolding
├── config/
│   └── config.yaml                  # paths, dataset URL, model name
├── .streamlit/
│   └── config.toml                  # Streamlit theme and telemetry off
├── research/                        # notebooks for each stage
├── artifacts/                       # data, tokenizer, fine-tuned model
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   ├── model_trainer/
│   │   ├── pegasus-samsum-model/
│   │   └── tokenizer/
│   └── model_evaluation/
└── src/textSummarizer/
    ├── config/configuration.py      # reads config.yaml and params.yaml
    ├── constants/
    ├── entity/                      # config dataclasses
    ├── logging/
    ├── utils/
    ├── conponents/                  # ingestion, validation, transform, train, evaluate
    └── pipeline/                    # stage runners and prediction.py
```

`config/config.yaml` points training at `google/pegasus-cnn_dailymail` and writes the fine-tuned files under `artifacts/model_trainer/`.

## Setup

Run commands from the project root (the folder that contains `main.py`).

```bash
git clone https://github.com/manesh-meghwar/Text-Summarization-Project.git
cd Text-Summarization-Project

conda create -n summary python=3.10 -y
conda activate summary
pip install -r requirements.txt truststore streamlit accelerate
```

If Hugging Face is blocked during training:

```bash
HF_ENDPOINT=https://hf-mirror.com python main.py
```

## Train

```bash
python main.py
```

| Stage | Script | Output |
| --- | --- | --- |
| Data ingestion | `src/textSummarizer/conponents/data_ingestion.py` | `artifacts/data_ingestion/` |
| Data validation | `src/textSummarizer/conponents/data_validation.py` | `artifacts/data_validation/status.txt` |
| Data transformation | `src/textSummarizer/conponents/data_transformation.py` | `artifacts/data_transformation/samsum_dataset/` |
| Model training | `src/textSummarizer/conponents/model_trainer.py` | `pegasus-samsum-model/` and `tokenizer/` |
| Model evaluation | `src/textSummarizer/conponents/model_evaluation.py` | `artifacts/model_evaluation/metrics.csv` |

Training uses 1 epoch, batch size 1, and gradient accumulation 16. A GPU is recommended. Logs go to `logs/running_logs.log`.

## Web app

After the model folders exist:

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`.

1. Paste a dialogue or paragraph, or load the sample.
2. Choose **Short**, **Medium**, or **Long**.
3. Click **Summarize**.

The app reads only:

- `artifacts/model_trainer/pegasus-samsum-model`
- `artifacts/model_trainer/tokenizer`

Those two folders can be copied from a GPU machine onto a local PC. Place them in the same paths next to `streamlit_app.py`. The rest of `artifacts` is not required for prediction.

The first load is slow. Later clicks reuse the model in memory. Dialogues in SAMSum format work best; any English text can be pasted.

## Optional API

```bash
python app.py
```

`http://localhost:8080/docs`

- `POST /predict?text=...` returns a summary through `src/textSummarizer/pipeline/prediction.py`
- `GET /train` runs `python main.py`

## Author

Manesh Kumar  
Email: munnyvalasai5@gmail.com
