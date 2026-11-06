# Text Summarization

Fine-tune [Pegasus](https://huggingface.co/google/pegasus-cnn_dailymail) on the SAMSum dialogue dataset, score it with ROUGE, and summarize new text in a local web page.

The training pipeline downloads the dataset, checks the files, tokenizes the dialogues, fine-tunes the model, and writes a metrics file. After training, a Streamlit page loads the saved model and returns a short summary.

## What you need

- Python 3.10
- Git
- An internet connection. The first run downloads the dataset and the Pegasus checkpoint from Hugging Face.
- A NVIDIA GPU is recommended. Pegasus is large. Training on CPU can stop with an out-of-memory error.
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

Install the project and its libraries. The last line of `requirements.txt` installs this repository in editable mode, so the `textSummarizer` package can be imported.

```bash
pip install -r requirements.txt
pip install truststore streamlit accelerate
```

`truststore` is imported by `main.py` so dataset downloads can complete on networks that fail SSL certificate checks. `streamlit` runs the summary page. `accelerate` is required by the Hugging Face trainer.

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

## Summarize text in the browser

Start the page after training has created the two model folders:

```bash
streamlit run streamlit_app.py
```

Open the URL printed in the terminal, usually `http://localhost:8501`. Paste a dialogue or paragraph, then click **Summarize**.

The page loads:

- `artifacts/model_trainer/pegasus-samsum-model`
- `artifacts/model_trainer/tokenizer`

The first summary is slow because the model is loaded into memory. Later summaries in the same session reuse that load.

## Optional API

`app.py` exposes the same model through FastAPI on port 8080:

```bash
python app.py
```

Open `http://localhost:8080/docs`.

- `POST /predict?text=Your dialogue here` returns the summary.
- `GET /train` starts `python main.py` from the API process.

The root URL redirects to the API docs. The Streamlit page is the text box for everyday use.

## Project layout

```text
.
├── main.py                         # runs all five stages
├── app.py                          # FastAPI predict and train routes
├── streamlit_app.py                # browser page for summarization
├── params.yaml                     # training numbers recorded for the config
├── config/config.yaml              # paths, dataset URL, and model name
├── requirements.txt
└── src/textSummarizer
    ├── conponents                  # ingestion, validation, transform, train, evaluate
    ├── pipeline                    # one runner per stage, plus prediction
    └── config/configuration.py     # reads config.yaml and params.yaml
```

## Configuration

`config/config.yaml` sets the dataset URL and the checkpoint name `google/pegasus-cnn_dailymail`. The fine-tuned model is written under `artifacts/model_trainer/`.

`params.yaml` lists the training arguments. The trainer reads its live arguments from `model_trainer.py`, which currently uses the same numbers: 1 epoch, warmup 500, batch size 1, weight decay 0.01, eval every 500 steps, and gradient accumulation 16.

## If something fails

- **`No such file or directory` for the model or tokenizer.** Stage 4 did not finish. Run `python main.py` and wait until both folders exist under `artifacts/model_trainer/`.
- **`DefaultCPUAllocator: not enough memory`.** The machine does not have enough free RAM for Pegasus. Use a GPU machine, or close other programs and run again. A failed run does not leave a usable model.
- **`ImportError` for `truststore`, `streamlit`, or `accelerate`.** Install the three packages in the setup section.
- **Hugging Face download warnings.** The checkpoint still downloads without an account. A token raises the rate limit: set `HF_TOKEN` in the environment before `python main.py`.
- **Commands fail to find `config/config.yaml`.** Start a new terminal, `cd` into the project root, and run the command there.

## Author

Manesh Kumar  
Email: munnyvalasai5@gmail.com
