# Multi-Agent Framework for Adverse Drug Reaction (ADR) Detection in Clinical Notes

A collaborative multi-agent LLM framework that detects adverse drug reactions from clinical notes through a three-step pipeline: **Discovery → Screening → Validation**.

<p align="center">
  <img src="dashboard/Multi-Agent_Framework.png" width="800"/>
</p>

## Overview

Clinical notes contain critical ADR information in diverse formats — from structured shorthand (`MTF → GI trouble`) to full narrative sentences. This framework uses six specialized LLM agents organized in two phases to achieve high-precision ADR extraction:

| Phase | Agent | Role |
|-------|-------|------|
| **A — Discovery** | Context Agent | Converts shorthand notes into readable narrative (shorthand style only) |
| | Medications Agent | Extracts all drug mentions (generic, brand, abbreviations, drug classes) |
| | ADR Candidates Agent | Identifies all plausible drug–symptom pairs |
| **B — Screening** | Confounders Check Agent | Flags candidates with alternative clinical explanations |
| | Confounders Validation Agent | Verifies confounder claims against original note evidence spans |
| | ADR Final Agent | Synthesizes all evidence into final ADR determinations |

The framework supports two clinical note styles:
- **Shorthand**: Abbreviated/symbolic notes (e.g., EMR shorthand with arrows, slashes, indentation)
- **Narrative**: Full-sentence clinical notes (e.g., MIMIC discharge summaries)

## Project Structure

```
multi-agent-ADR-detection/
├── agents.py           # Six LLM agent definitions with style-aware prompts
├── pipeline.py         # Three-step pipeline runner (CLI)
├── utils.py            # Binary label conversion utility
├── requirements.txt    # Python dependencies
├── .env.example        # API key template (copy to .env)
├── dashboard/
│   ├── app.py          # Streamlit interactive dashboard
│   ├── .env.example    # API key template for dashboard
│   ├── requirements.txt
│   ├── note_sample.xlsx            # Sample input file
│   ├── Multi-Agent_Framework.png   # Framework diagram
│   └── note_example.jpg            # Input format example
├── docs/
│   └── screenshots/    # Web interface screenshots
└── README.md
```

## Setup

### Prerequisites

- Python 3.10 or higher
- An OpenRouter API key ([OpenRouter](https://openrouter.ai/keys))

### 1. Clone the repository

```bash
git clone https://github.com/sungmoonie/multi-agent-ADR-detection.git
cd multi-agent-ADR-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including `pandas`, `streamlit`, and others. All LLM calls are made through the [OpenRouter](https://openrouter.ai/) API, with **Gemini 3 Flash** as the default model. To use a different model, find the model identifier on [OpenRouter Models](https://openrouter.ai/models) and pass it via the `--model` option.

### 4. Set up your API key

You need an [OpenRouter API key](https://openrouter.ai/keys) to run the pipeline.

**Option A — Enter directly in the dashboard (easiest)**

Simply launch the dashboard (`streamlit run app.py`). The UI provides a built-in API key input where you can paste your key directly — no file setup required. Your key is held in session memory only and is never written to disk.

**Option B — Use a `.env` file**

For the CLI pipeline or to avoid re-entering your key each session, create a `.env` file:

```bash
cp .env.example .env
```

Edit `.env`:

```
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

For the dashboard, also set up its `.env`:

```bash
cp dashboard/.env.example dashboard/.env
```

The dashboard will automatically detect the key from `.env` and display a confirmation message. You can still override it with a custom key via the UI at any time.

## Usage

### Option A: Command-Line Pipeline

Run the pipeline directly on an Excel file:

```bash
# Shorthand style (default)
python pipeline.py data.xlsx --style shorthand --note-col note_preprocessed

# Narrative style
python pipeline.py data.xlsx --style narrative --note-col text

# With a different model (via OpenRouter)
python pipeline.py data.xlsx --style narrative --note-col text --model anthropic/claude-sonnet-4
```

**Quick test with sample data:**

```bash
python pipeline.py note_sample.xlsx --style shorthand --note-col note_preprocessed
```

**Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `filename` | *(required)* | Input Excel file (.xlsx) |
| `--style` | `shorthand` | `shorthand` or `narrative` |
| `--note-col` | auto | Column containing note text (`note_preprocessed` for shorthand, `text` for narrative) |
| `--model` | `google/gemini-3-flash-preview` | OpenRouter model name |

**Output:** `<input_name>_results.xlsx` — a single file containing all intermediate and final columns.

### Option B: Interactive Dashboard

Launch the Streamlit dashboard for a visual, interactive experience:

```bash
cd dashboard
streamlit run app.py
```

The dashboard provides:
- **File Upload**: Upload CSV/XLSX files with multiple notes
- **Single Note**: Paste a single clinical note for instant analysis
- **Note Style Selection**: Choose between shorthand and narrative
- **Overview**: ADR distribution charts and top drug–symptom pairs
- **Note Browser**: Drill into individual notes with full reasoning trace
- **Drug & ADR Analysis**: Explore ADR profiles per drug across all notes

<p align="center">
  <img src="docs/screenshots/interface_overview.png" width="800"/>
  <br><em>Dataset-level overview — final label distribution, top confirmed ADR drugs, and frequent drug–symptom pairs</em>
</p>

<details>
<summary><b>View all dashboard pages</b> (click to expand)</summary>
<br>

#### Data Upload
<p align="center">
  <img src="docs/screenshots/interface_upload.png" width="800"/>
</p>

Upload CSV/XLSX files containing multiple clinical notes or paste a single note as free text. Select note format, choose input columns, and execute the ADR detection pipeline.

#### Note Browser — Reasoning Process
<p align="center">
  <img src="docs/screenshots/interface_reasoning.png" width="800"/>
</p>

Inspect intermediate outputs for each note: ADR candidate extraction, confounder judgment, confounder validation with note-grounded evidence, and final adjudication. This view enables transparent review of how each ADR decision was derived.

#### Drug & ADR Analysis
<p align="center">
  <img src="docs/screenshots/interface_analysis.png" width="800"/>
</p>

Explore the frequency distribution of confirmed ADR symptoms and drug–symptom pairs across all uploaded notes with interactive controls.

#### Drug Detail View
<p align="center">
  <img src="docs/screenshots/interface_drug_detail.png" width="800"/>
</p>

Select a normalized drug name to review all associated ADR candidates and confirmed ADRs across notes, including note identifiers, symptoms, validation outcomes, and supporting source text.

</details>

## Input Format

Your input Excel file needs at minimum **one column containing clinical note text**. You specify which column to use via `--note-col` (CLI) or the column selector (dashboard).

<p align="center">
  <img src="dashboard/note_example.jpg" width="600"/>
  <br><em>Example input format</em>
</p>

| Column | Required | Description |
|--------|----------|-------------|
| Note text column | Yes | Raw clinical note text (name is configurable) |
| ID column | No | Note identifier (auto-generated if absent) |

## Output Columns

The pipeline adds the following columns to the output file:

| Column | Step | Description |
|--------|------|-------------|
| `context_note` | 1 | Narrative reconstruction (shorthand style only) |
| `medications` | 1 | Extracted medications (JSON) |
| `ADR_candidates` | 1 | Drug–symptom candidate pairs (JSON) |
| `adr_candidates_side_effect` | 1 | De-novo ADR detection result |
| `adr_candidates_side_effect_binary` | 1 | Binary label (0/1) |
| `confounders` | 2 | Confounder evaluation for each candidate (JSON) |
| `confounder_side_effect` | 2 | ADR detection with confounder info |
| `confounder_side_effect_binary` | 2 | Binary label (0/1) |
| `confounder_validation` | 3 | Evidence-based validation of confounders (JSON) |
| `validation_side_effect` | 3 | **Final ADR detection** (JSON) |
| `validation_side_effect_binary` | 3 | **Final binary label** (0/1) |

## License

This project is released for academic and research purposes.

## Citation

If you use this framework in your research, please cite our paper:

```bibtex
@article{
  title={...},
  author={...},
  journal={...},
  year={2025}
}
```
