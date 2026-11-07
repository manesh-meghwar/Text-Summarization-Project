# Text Summarization

Fine-tune [Pegasus](https://huggingface.co/google/pegasus-cnn_dailymail) on the SAMSum dialogue dataset, score it with ROUGE, and summarize new text in a local web page.

The training pipeline downloads the dataset, checks the files, tokenizes the dialogues, fine-tunes the model, and writes a metrics file. After training, a Streamlit page loads the saved model. You can choose a **short**, **medium**, or **long** summary.

## What you need

- Python 3.10
- Git
- An internet connection. The first training run downloads the dataset and the Pegasus checkpoint.
- An NVIDIA GPU is recommended for training. Pegasus is large. Training on CPU can stop with an out-of-memory error.
- Conda, or another virtual environment tool

Run every command below from the project root, the folder that contains `main.py` and `config`.

## Setup

Clone the repository and enter it:

```bash
git clone https://github.com/manesh-meghwar/Text-Summarization-Project.git
cd Text-Summarization-Project
```

Create and activate an environment:

```bash
conda create -n summary python=3.10 -y
conda activate summary
```

Install the project libraries. Use `requirements-new.txt` on a CUDA GPU machine. Use `requirements.txt` on CPU-only Windows.

```bash
pip install -r requirements-new.txt truststore streamlit
```

CPU-only install:

```bash
pip install -r requirements.txt truststore streamlit accelerate
pip install "transformers<5" sentencepiece
```

`truststore` helps `main.py` download data on networks that fail SSL checks. `streamlit` runs the web page. `accelerate` is required by the Hugging Face trainer. Transformers 5 removed the old summarization pipeline, so local inference is pinned below version 5 if you need that API. The Streamlit app loads Pegasus with `generate` and works on both 4.x and 5.x.

If Hugging Face is blocked, prefix training with the China mirror:

```bash
HF_ENDPOINT=https://hf-mirror.com python main.py
```

## Train and evaluate

```bash
python main.py
```

`main.py` runs five stages in order:

| Stage | What it does | Output |
| --- | --- | --- |
| 1. Data ingestion | Downloads and unzips the SAMSum files | `artifacts/data_ingestion/` |
| 2. Data validation | Checks that `train`, `test`, and `validation` exist | `artifacts/data_validation/status.txt` |
| 3. Data transformation | Tokenizes dialogues and summaries | `artifacts/data_transformation/samsum_dataset/` |
| 4. Model training | Fine-tunes Pegasus and saves the model | `artifacts/model_trainer/pegasus-samsum-model/` and `artifacts/model_trainer/tokenizer/` |
| 5. Evaluation | Scores the full test split with ROUGE | `artifacts/model_evaluation/metrics.csv` |

Training uses 1 epoch, batch size 1, and gradient accumulation of 16. Those values are set in `src/textSummarizer/conponents/model_trainer.py` and match `params.yaml`. Logs are appended to `logs/running_logs.log`.

Stage 5 generates a summary for every test dialogue with beam search, then writes `rouge1`, `rouge2`, `rougeL`, and `rougeLsum`. On a CPU this stage is slow. It runs only after stage 4 has saved the model folders.

## Run the website

Start the page after the two model folders exist:

```bash
streamlit run streamlit_app.py
```

Open the URL printed in the terminal, usually `http://localhost:8501`.

The page loads:

- `artifacts/model_trainer/pegasus-samsum-model`
- `artifacts/model_trainer/tokenizer`

### Page controls

1. Paste a dialogue or paragraph, or click **Load sample dialogue**.
2. Choose **Short**, **Medium**, or **Long**.
3. Click **Summarize**.

| Setting | What you get | Length range |
| --- | --- | --- |
| Short | A few sentences | 8–40 tokens |
| Medium | Default recap | 20–80 tokens |
| Long | More detail | 40–128 tokens |

The model was trained on SAMSum chats (`Name: message`). Any English text can be pasted. Dialogues summarize best. Input longer than 1024 tokens is truncated. The first load is slow. Later clicks reuse the model in memory.

### Copy the model to another computer

Training can finish on a GPU server. Copy these two folders onto the local machine, next to `streamlit_app.py`, then run Streamlit in a matching environment. You do not need the rest of `artifacts` for prediction.

## Optional API

`app.py` exposes the same model through FastAPI on port 8080:

```bash
python app.py
```

Open `http://localhost:8080/docs`.

- `POST /predict?text=Your dialogue here` returns the summary.
- `GET /train` starts `python main.py` from the API process.

The root URL redirects to the API docs. Use the Streamlit page for everyday summarization.

## Project layout

```text
.
├── main.py                         # runs all five stages
├── app.py                          # FastAPI predict and train routes
├── streamlit_app.py                # browser page, with length controls
├── params.yaml                     # training numbers recorded for the config
├── config/config.yaml              # paths, dataset URL, and model name
├── requirements.txt                # CPU-friendly install
├── requirements-new.txt            # CUDA 12.1 PyTorch install
└── src/textSummarizer
    ├── conponents                  # ingestion, validation, transform, train, evaluate
    ├── pipeline                    # one runner per stage, plus prediction
    └── config/configuration.py     # reads config.yaml and params.yaml
```

## Configuration

`config/config.yaml` sets the dataset URL and the checkpoint name `google/pegasus-cnn_dailymail`. The fine-tuned model is written under `artifacts/model_trainer/`.

`params.yaml` lists the training arguments. The trainer reads its live arguments from `model_trainer.py`, which currently uses the same numbers: 1 epoch, warmup 500, batch size 1, weight decay 0.01, eval every 500 steps, and gradient accumulation 16.

## If something fails

- **`No such file or directory` for the model or tokenizer.** Stage 4 did not finish, or the two folders were not copied next to `streamlit_app.py`.
- **`The task summarization has been removed`.** That message comes from Transformers 5. This Streamlit app does not use that pipeline. Restart the page after pulling the latest `streamlit_app.py`.
- **`DefaultCPUAllocator: not enough memory`.** The machine does not have enough free RAM for Pegasus. Train on a GPU machine.
- **Streamlit hangs on an email prompt or `data.streamlit.io`.** Press Enter with a blank email. Usage stats are already turned off in `.streamlit/config.toml`.
- **Hugging Face download warnings.** On a blocked network, set `HF_ENDPOINT=https://hf-mirror.com` before `python main.py`.
- **Commands fail to find `config/config.yaml`.** Start a new terminal, `cd` into the project root, and run the command there.

## Author

Manesh Kumar  
Email: munnyvalasai5@gmail.com
