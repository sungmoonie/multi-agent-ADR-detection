import streamlit as st
import pandas as pd
import plotly.express as px
import ast
import json
import os
import glob
import subprocess
import sys
import pathlib
from collections import Counter

# ── Auto-load .env file (GEMINI_API_KEY, etc.) ───────────────────────────────
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    load_dotenv(dotenv_path=_env_path, override=False)
except ImportError:
    pass

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Note-Insight: ADR Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Sidebar shell (White Background) ─────────────────────────────── */
section[data-testid="stSidebar"] > div:first-child {
    background-color: #FFFFFF !important;
    border-right: 1px solid #E2E8F0;
    padding-top: 0 !important;
}

/* ── Sidebar brand block ────────────────────────────────────────────── */
.sidebar-brand {
    padding: 35px 24px 25px 24px;
    border-bottom: 1px solid #F1F5F9;
    margin-bottom: 15px;
}
.sidebar-brand h1 {
    font-size: 22px !important;
    font-weight: 800 !important;
    color: #1E293B !important; /* Dark Slate */
    margin: 0 0 4px 0 !important;
    letter-spacing: -0.5px;
}
.sidebar-brand p {
    font-size: 12px !important;
    color: #64748B !important; /* Muted Blue Grey */
    margin: 0 !important;
    line-height: 1.5;
}

/* ── Nav section label ──────────────────────────────────────────────── */
.nav-section-label {
    font-size: 11px !important;
    font-weight: 700 !important;
    letter-spacing: 1.2px !important;
    color: #94A3B8 !important;
    padding: 10px 24px 8px 24px;
    text-transform: uppercase;
}

/* ── Radio → nav menu (Clean Hover Effect) ───────────────────────────── */
section[data-testid="stSidebar"] [data-testid="stRadio"] {
    padding: 0 12px;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > label:first-child {
    display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stRadio"] > div {
    gap: 4px !important;
}
/* Each nav item */
section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    display: flex !important;
    align-items: center !important;
    width: 100% !important;
    padding: 12px 16px !important;
    border-radius: 10px !important;
    background: transparent !important;
    border: none !important;
    color: #475569 !important; /* Default Text */
    font-size: 14px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
/* Hover item */
section[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
    background: #F8FAFC !important;
    color: #2563EB !important;
}
/* Active / selected item (Blue Accent) */
section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
    background: #EFF6FF !important; /* Light Blue */
    color: #2563EB !important;      /* Royal Blue */
    font-weight: 700 !important;
}

/* ── Sidebar footer stat box (Light Gray) ────────────────────────── */
.sidebar-footer {
    margin-top: 28px;
    border-top: 1px solid #E2E8F0;
    padding: 18px 12px 8px 12px;
    background: #F8FAFC;
    border-radius: 10px;
}
.stat-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
}
.stat-label {
    font-size: 12px;
    color: #64748B;
    font-weight: 500;
}
.stat-value {
    font-size: 14px;
    font-weight: 700;
    color: #1E293B;
}

/* Hide Streamlit default icons/circles */
section[data-testid="stSidebar"] [data-testid="stRadio"] [data-baseweb="radio"] > div:first-child {
    display: none !important;
}

            
.kpi-card {
    background-color: #FFFFFF;
    border: 1px solid #E2E8F0;
    padding: 15px;
    border-radius: 12px;
    text-align: left;
}
.kpi-label {
    font-size: 14px;
    color: #64748B;
    font-weight: 500;
    margin-bottom: 8px;
}
.kpi-value {
    font-size: 24px;
    font-weight: 700;
    color: #1E293B;
}
.kpi-unit {
    font-size: 14px;
    color: #94A3B8;
    font-weight: 400;
    margin-left: 4px;
}
</style>
""", unsafe_allow_html=True)

# ── Data loading ──────────────────────────────────────────────────────────────
@st.cache_data
def load_data(filepath, id_col="id_note"):
    # Convert relative path to absolute (supports output/ subfolder)
    if not os.path.isabs(filepath):
        filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_excel(filepath)

    # ── Initialize pipeline output columns if missing (raw input fallback) ──
    for _col in ["medications", "ADR_candidates", "confounders",
                 "validation_side_effect", "confounder_validation"]:
        if _col not in df.columns:
            df[_col] = "{}"
    for _col in ["validation_side_effect_binary",
                 "adr_candidates_side_effect_binary",
                 "confounder_side_effect_binary"]:
        if _col not in df.columns:
            df[_col] = 0

    def safe_parse(val):
        if pd.isna(val):
            return {}
        try:
            parsed = ast.literal_eval(str(val))
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            try:
                parsed = json.loads(str(val))
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

    df["_medications_parsed"]    = df["medications"].apply(safe_parse)
    df["_adr_parsed"]            = df["ADR_candidates"].apply(safe_parse)
    df["_confounder_parsed"]     = df["confounders"].apply(safe_parse)
    df["_validation_parsed"]     = df["validation_side_effect"].apply(safe_parse)
    df["_confounder_val_parsed"] = df["confounder_validation"].apply(safe_parse)

    df["n_meds"]       = df["_medications_parsed"].apply(lambda d: len(d.get("medications", [])))
    df["n_adrs"]       = df["_adr_parsed"].apply(lambda d: len(d.get("adr_candidates", [])))
    df["n_confounders"]= df["_confounder_parsed"].apply(lambda d: len(d.get("results", [])))
    df["n_validated"]  = df["_validation_parsed"].apply(lambda d: len(d.get("results", [])))

    # id_note: use specified column, or auto-generate (1, 2, 3, ...)
    if id_col and id_col != "__auto__" and id_col in df.columns:
        df["id_note"] = df[id_col].astype(str)
    else:
        df["id_note"] = [str(i + 1) for i in range(len(df))]

    # ── Unified drug name column ─────────────────────────────────────────────
    # Map medications[].text (abbreviation) → medication (standardized name),
    # then add any drugs from validation_side_effect not already mapped.
    def extract_all_drugs(row):
        seen_lower = set()
        abbr_map   = {}
        result     = []

        # 1) From medications: add standardized names, store abbreviation map
        for m in row["_medications_parsed"].get("medications", []):
            if not isinstance(m, dict):
                continue
            abbr = m.get("text", "").strip()
            name = m.get("medication", "").strip()
            if abbr:
                abbr_map[abbr.lower()] = name if name else abbr
            if name and name.lower() not in seen_lower:
                seen_lower.add(name.lower())
                result.append(name)

        # 2) From validation results: skip already-mapped abbreviations
        for item in row["_validation_parsed"].get("results", []):
            if not isinstance(item, dict):
                continue
            drug = item.get("drug", "").strip()
            if not drug:
                continue
            drug_lower = drug.lower()
            if drug_lower in abbr_map:
                continue
            if drug_lower not in seen_lower:
                seen_lower.add(drug_lower)
                result.append(drug)

        return ", ".join(result) if result else ""

    df["all_drugs_str"] = df.apply(extract_all_drugs, axis=1)
    df["n_all_drugs"]   = df["all_drugs_str"].apply(
        lambda s: len(s.split(", ")) if s else 0
    )
    return df


DEFAULT_FILE = None  # No default — user uploads via Data Upload page
if "data_file" not in st.session_state:
    st.session_state["data_file"] = DEFAULT_FILE
if "id_col" not in st.session_state:
    st.session_state["id_col"] = "id_note"
if "_data_version" not in st.session_state:
    st.session_state["_data_version"] = 0


def _bump_data_version():
    """Increment data version counter to invalidate all downstream caches."""
    st.session_state["_data_version"] = st.session_state.get("_data_version", 0) + 1
    load_data.clear()


def _empty_dataframe():
    """Return a minimal empty DataFrame with all expected pipeline columns."""
    df = pd.DataFrame()
    for _col in ["medications", "ADR_candidates", "confounders",
                 "validation_side_effect", "confounder_validation"]:
        df[_col] = pd.Series(dtype="object")
    for _col in ["validation_side_effect_binary",
                 "adr_candidates_side_effect_binary",
                 "confounder_side_effect_binary"]:
        df[_col] = pd.Series(dtype="int64")
    for _col in ["_medications_parsed", "_adr_parsed", "_confounder_parsed",
                 "_validation_parsed", "_confounder_val_parsed"]:
        df[_col] = pd.Series(dtype="object")
    for _col in ["n_meds", "n_adrs", "n_confounders", "n_validated", "n_all_drugs"]:
        df[_col] = pd.Series(dtype="int64")
    df["id_note"] = pd.Series(dtype="str")
    df["all_drugs_str"] = pd.Series(dtype="str")
    return df


_data_file = st.session_state["data_file"]
if _data_file is not None:
    _abs_path = _data_file if os.path.isabs(_data_file) else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _data_file)
    if os.path.exists(_abs_path):
        df = load_data(_data_file, st.session_state["id_col"])
    else:
        st.warning(f"File not found: `{_data_file}`. Please upload data via the Data Upload page.")
        df = _empty_dataframe()
else:
    df = _empty_dataframe()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <h1>Adverse Drug Reaction-Insight</h1>
        <p>Collaborative Multi-Agent Framework<br>for Medical Text ADR Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Nav label
    st.markdown('<p class="nav-section-label">MENU</p>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "Data Upload",
            "Overview",
            "Note Browser",
            "Drug & ADR Analysis",
        ],
        label_visibility="collapsed",
    )

    # Footer stats
    total     = len(df)
    n_adr     = int(df["validation_side_effect_binary"].sum()) if total > 0 else 0
    n_no_adr  = total - n_adr
    _pct_adr    = f"{n_adr/total*100:.1f}%" if total > 0 else "—"
    _pct_no_adr = f"{n_no_adr/total*100:.1f}%" if total > 0 else "—"
    st.markdown(f"""
    <div class="sidebar-footer">
        <div class="stat-row">
            <span class="stat-label">Total Notes</span>
            <span class="stat-value">{total:,}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">ADR Positive</span>
            <span class="stat-value" style="color:#DC2626;">{n_adr:,} ({_pct_adr})</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">ADR Negative</span>
            <span class="stat-value" style="color:#16A34A;">{n_no_adr:,} ({_pct_no_adr})</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ── Pipeline execution helper (shared by File & Text tabs) ───────────────────
def _run_pipeline(tmp_rel, log_path, search_dir, proc_env, style="shorthand"):
    """Run pipeline.py as a subprocess with live log streaming.
    Returns dict: {timed_out, exc, proc, log_lines}
    """
    import importlib.util as _ilu
    _PKG_MAP = {
        "google.genai": "google-genai",
        "dotenv":       "python-dotenv",
        "tqdm":         "tqdm",
        "openpyxl":     "openpyxl",
    }
    def _pkg_ok(name):
        try:
            return _ilu.find_spec(name) is not None
        except (ModuleNotFoundError, ValueError):
            return False

    _WARN_SUPPRESS = (
        "non-text parts in the response",
        "returning concatenated text result from text parts",
        "Check the full candidates.content.parts",
        "thought_signature",
    )

    _pipe_proc      = None
    _pipe_timed_out = False
    _pipe_exc       = None
    _all_log_lines  = []

    # Resolve project root (parent of dashboard/)
    _project_root = os.path.dirname(search_dir)
    _pipeline_script = os.path.join(_project_root, "pipeline.py")
    # File path relative to project root
    _input_from_root = os.path.join("dashboard", tmp_rel)

    with st.status("Running ADR pipeline...", expanded=True) as _status:
        _missing = [pip for imp, pip in _PKG_MAP.items() if not _pkg_ok(imp)]
        if _missing:
            st.write(f"**Pre-check:** Installing missing packages: `{', '.join(_missing)}`...")
            _pip = subprocess.run(
                [sys.executable, "-m", "pip", "install", "--quiet"] + _missing,
                capture_output=True, text=True,
            )
            st.write("All packages ready." if _pip.returncode == 0
                     else f"Package install issue: {_pip.stderr[:200]}")
        else:
            st.write("**Pre-check:** All packages already installed.")

        st.write("**Pipeline running** — live progress:")
        _log_area = st.empty()

        try:
            with open(log_path, "w", encoding="utf-8") as _log_f:
                _pipe_proc = subprocess.Popen(
                    [sys.executable, "-u", _pipeline_script, _input_from_root,
                     "--style", style,
                     "--note-col", "note_preprocessed",
                     "--provider", "gemini"],
                    cwd=_project_root,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=proc_env,
                )
                for raw_line in iter(_pipe_proc.stdout.readline, ""):
                    _log_f.write(raw_line)
                    _log_f.flush()
                    for part in raw_line.replace("\r", "\n").split("\n"):
                        part = part.strip()
                        if not part or any(w in part for w in _WARN_SUPPRESS):
                            continue
                        # Deduplicate tqdm: if the new line shares a prefix
                        # (e.g. "Step 1 Extraction:") with the last line, replace it.
                        if _all_log_lines and "%" in part and "it" in part:
                            _prefix = part.split(":")[0]
                            _prev_prefix = _all_log_lines[-1].split(":")[0]
                            if _prefix == _prev_prefix:
                                _all_log_lines[-1] = part
                                continue
                        _all_log_lines.append(part)
                    _log_area.code("\n".join(_all_log_lines[-30:]), language="")

                _pipe_proc.wait(timeout=7200)

            if _pipe_proc.returncode == 0:
                _status.update(label="✅ Pipeline completed!", state="complete")
            else:
                _status.update(label="❌ Pipeline failed.", state="error")

        except subprocess.TimeoutExpired:
            if _pipe_proc:
                _pipe_proc.kill()
            _pipe_timed_out = True
            _status.update(label="⏰ Timed out after 2 hours.", state="error")
        except Exception as _e:
            _pipe_exc = _e
            _status.update(label="❌ Unexpected error.", state="error")

    return {"timed_out": _pipe_timed_out, "exc": _pipe_exc,
            "proc": _pipe_proc, "log_lines": _all_log_lines}


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 0 – Data Upload
# ═════════════════════════════════════════════════════════════════════════════
if page == "Data Upload":
    st.title("Data Upload")
    st.caption("Upload a CSV/XLSX file to analyze multiple notes, or paste a single note directly as text.")

    st.warning(
    "**Privacy Notice**  \n"
    "This application assumes that all uploaded notes have already been **de-identified** "
    "(i.e., patient names, dates, identifiers, and other personal information have been removed "
    "prior to upload).  \n"
    "The pipeline does **not** perform anonymization. "
    "Do not upload notes containing personally identifiable information (PII) or "
    "protected health information (PHI).",
    icon="⚠️",
    )
    
    search_dir  = os.path.dirname(os.path.abspath(__file__))

    # ── Input format example image ────────────────────────────────────────────
    _ex_img_path = os.path.join(search_dir, "note_example.jpg")
    if os.path.exists(_ex_img_path):
        with st.expander("📋 Input Format Examples — Multiple Notes (CSV/XLSX) & Single Note (Text)", expanded=False):
            _c1, _c2, _c3 = st.columns([1, 3, 1])
            with _c2:
                st.image(_ex_img_path, use_container_width=True)

    _output_dir = os.path.join(search_dir, "output")
    os.makedirs(_output_dir, exist_ok=True)

    # ── Shared Gemini API Key detection (both tabs) ─────────────────────────
    _secrets_toml_paths = [
        pathlib.Path.home() / ".streamlit" / "secrets.toml",
        pathlib.Path(__file__).parent / ".streamlit" / "secrets.toml",
    ]
    _key_from_secrets = ""
    if any(p.exists() for p in _secrets_toml_paths):
        try:
            _key_from_secrets = st.secrets.get("GEMINI_API_KEY", "")
        except Exception:
            pass
    _key_from_env = os.environ.get("GEMINI_API_KEY", "")

    # ── Tab layout ─────────────────────────────────────────────────────────
    tab_file, tab_text = st.tabs(["📁 File Upload", "📝 Single Note"])

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 1 – File Upload
    # ══════════════════════════════════════════════════════════════════════════
    with tab_file:
        # ── ① Upload File ─────────────────────────────────────────────────
        st.subheader("① Upload File")
        st.info("📂 Upload **one** CSV or XLSX file. The file may contain one or multiple notes (rows).")
        uploaded_file = st.file_uploader(
            "Select a CSV or XLSX file",
            type=["csv", "xlsx"],
            accept_multiple_files=False,
            key="uploader_file",
        )
        if uploaded_file:
            save_path = os.path.join(search_dir, uploaded_file.name)
            with open(save_path, "wb") as out:
                out.write(uploaded_file.read())
            st.success(f"Saved: **{uploaded_file.name}** → `{save_path}`")

        st.divider()

        # ── ② Configure & Load ────────────────────────────────────────────
        st.subheader("② Configure & Load")

        _root_files = [os.path.basename(p) for p in
                       glob.glob(os.path.join(search_dir, "*.csv")) +
                       glob.glob(os.path.join(search_dir, "*.xlsx"))]
        _out_files  = [os.path.join("output", os.path.basename(p)) for p in
                       glob.glob(os.path.join(_output_dir, "*.csv")) +
                       glob.glob(os.path.join(_output_dir, "*.xlsx"))]
        available = sorted(_root_files) + sorted(_out_files)

        if not available:
            st.warning("No CSV/XLSX files found. Please upload a file first.")
        else:
            current     = st.session_state.get("data_file", DEFAULT_FILE)
            default_idx = available.index(current) if current in available else 0
            selected    = st.selectbox("Select file", available, index=default_idx)

            try:
                _path    = os.path.join(search_dir, selected)
                _preview = (pd.read_csv(_path, nrows=0) if selected.endswith(".csv")
                            else pd.read_excel(_path, nrows=0))
                file_cols = list(_preview.columns)
            except Exception:
                file_cols = []

            if file_cols:
                st.caption(f"Columns in file: `{'`,  `'.join(file_cols)}`")

            AUTO_LABEL     = "Auto-generate (1, 2, 3, ...)"
            id_options     = [AUTO_LABEL] + file_cols
            cur_id         = st.session_state.get("id_col", "id_note")
            id_default_idx = (file_cols.index(cur_id) + 1) if cur_id in file_cols else 0
            selected_id    = st.selectbox(
                "Note ID column", id_options, index=id_default_idx,
                help="Column to use as the note identifier in Note Browser. Auto-generates 1, 2, 3… if not selected.",
            )

            _PIPELINE_COLS      = ["medications", "ADR_candidates", "confounders",
                                   "validation_side_effect", "confounder_validation"]
            has_pipeline_output = all(c in file_cols for c in _PIPELINE_COLS)

            st.divider()

            if has_pipeline_output:
                st.success("✅ This file already contains ADR pipeline results.")
            else:
                st.info("ℹ️ This file does not yet contain ADR pipeline results.  \n"
                        "Configure the options below and click **Run Pipeline & Load**.")

            # Note style selector
            note_style = st.radio(
                "Clinical note style",
                ["shorthand", "narrative"],
                horizontal=True, index=0,
                disabled=has_pipeline_output,
                key="note_style_file",
                help=(
                    "**Shorthand**: Abbreviated/symbolic notes. "
                    "Uses Context Agent to convert shorthand to narrative.  \n"
                    "**Narrative**: Full-sentence clinical notes."
                ),
            )

            # Note text column
            def _best_note_col(cols, preview_path, file_ext):
                """Return index of the column most likely to contain note text."""
                try:
                    _pv = (pd.read_csv(preview_path, nrows=5) if file_ext == ".csv"
                           else pd.read_excel(preview_path, nrows=5))
                    if "note_preprocessed" in cols and _pv["note_preprocessed"].notna().any():
                        return cols.index("note_preprocessed")
                    best = max(cols, key=lambda c: _pv[c].astype(str).str.len().sum()
                               if c in _pv.columns else 0)
                    return cols.index(best)
                except Exception:
                    return 0

            _note_col_default = _best_note_col(
                file_cols, _path, os.path.splitext(selected)[1]
            ) if file_cols and not has_pipeline_output else 0

            note_text_col = st.selectbox(
                "Select note text column",
                file_cols if file_cols else [""],
                index=_note_col_default,
                help="Column containing the raw medical note text to analyze.",
                disabled=has_pipeline_output,
            )

            if not has_pipeline_output and note_text_col and file_cols:
                try:
                    _col_check = (pd.read_csv(_path, usecols=[note_text_col]) if selected.endswith(".csv")
                                  else pd.read_excel(_path, usecols=[note_text_col]))
                    _n_null  = _col_check[note_text_col].isna().sum()
                    _n_total = len(_col_check)
                    if _n_null == _n_total:
                        st.error(f"⚠️ **Column `{note_text_col}` is completely empty** "
                                 f"({_n_null}/{_n_total} rows are NaN).  \n"
                                 "Please select a different column that contains the note text.")
                    elif _n_null > 0:
                        st.warning(f"⚠️ Column `{note_text_col}` has **{_n_null} empty rows** "
                                   f"out of {_n_total}. These rows will be skipped by the pipeline.")
                    else:
                        st.caption(f"✅ `{note_text_col}`: {_n_total} rows, no empty values.")
                except Exception:
                    pass

            # API Key
            _api_source_f = st.radio(
                "Gemini API Key",
                ["🔑 Use key from .env (auto)", "✏️ Enter custom key"],
                horizontal=True, index=0,
                disabled=has_pipeline_output,
                key="api_src_file",
                help=(
                    "**Use .env (auto)**: Reads `GEMINI_API_KEY` automatically from the `.env` file.  \n"
                    "**Enter custom key**: Held in session memory only — never written to disk or logs."
                ),
            )
            if _api_source_f == "🔑 Use key from .env (auto)":
                _effective_key = _key_from_secrets or _key_from_env
                if _effective_key:
                    st.success("✅ Gemini API Key detected automatically "
                               "(source: `st.secrets` / `.env` / environment variable).")
                else:
                    _proj_dir = os.path.dirname(os.path.abspath(__file__))
                    st.warning(f"⚠️ `GEMINI_API_KEY` not found.  \n"
                               f"Create a **`.env`** file in `{_proj_dir}` and add:  \n"
                               "`GEMINI_API_KEY=your_key_here`")
            else:
                _effective_key = st.text_input(
                    "Custom Gemini API Key", type="password",
                    placeholder="AIza... paste your key here",
                    disabled=has_pipeline_output, key="api_key_file",
                )

            st.divider()

            # Action button: branch by file state
            if has_pipeline_output:
                if st.button("✅ Load This File"):
                    st.session_state["data_file"] = selected
                    st.session_state["id_col"]    = (
                        "__auto__" if selected_id == AUTO_LABEL else selected_id
                    )
                    _bump_data_version()
                    st.rerun()
            else:
                run_ready = bool(file_cols and _effective_key and note_text_col)
                if st.button("🚀 Run Pipeline & Load", disabled=not run_ready):
                    _input_path = os.path.join(search_dir, selected)
                    _tmp_df = (pd.read_csv(_input_path) if selected.endswith(".csv")
                               else pd.read_excel(_input_path))
                    if note_text_col != "note_preprocessed":
                        if "note_preprocessed" in _tmp_df.columns:
                            _tmp_df = _tmp_df.drop(columns=["note_preprocessed"])
                        _tmp_df = _tmp_df.rename(columns={note_text_col: "note_preprocessed"})

                    _n_valid = int(_tmp_df["note_preprocessed"].notna().sum())
                    st.info(f"📋 **Column mapping confirmed.** \n"
                            f"Rows with valid note text: **{_n_valid}** / {len(_tmp_df)}")

                    _base_name = os.path.splitext(os.path.basename(selected))[0]
                    if _base_name.startswith("_pipeline_"):
                        _base_name = _base_name[len("_pipeline_"):]
                    _tmp_name = f"_pipeline_{_base_name}.xlsx"
                    _log_name = f"_pipeline_{_base_name}_log.txt"
                    _tmp_rel  = os.path.join("output", _tmp_name)
                    _tmp_path = os.path.join(_output_dir, _tmp_name)
                    _log_path = os.path.join(_output_dir, _log_name)
                    _tmp_df.to_excel(_tmp_path, index=False)

                    _proc_env                   = os.environ.copy()
                    _proc_env["GEMINI_API_KEY"] = _effective_key

                    _r = _run_pipeline(_tmp_rel, _log_path, search_dir, _proc_env,
                                       style=note_style)

                    # Result file: pipeline writes {base}_results{ext}
                    _base_out, _ext_out = os.path.splitext(_tmp_rel)
                    _result_rel = f"{_base_out}_results{_ext_out}"
                    _result_name = os.path.basename(_result_rel)

                    if _r["timed_out"]:
                        st.error("Pipeline timed out after 2 hours.")
                    elif _r["exc"] is not None:
                        st.error(f"Unexpected error: {_r['exc']}")
                    elif _r["proc"] is not None:
                        if _r["proc"].returncode == 0:
                            st.session_state["data_file"] = _result_rel
                            st.session_state["id_col"]    = (
                                "__auto__" if selected_id == AUTO_LABEL else selected_id
                            )
                            _bump_data_version()
                            st.success(f"Results saved → **output/{_result_name}**  \n"
                                       f"Log saved → **output/{_log_name}**")
                            st.rerun()
                        else:
                            st.error("The pipeline encountered an error.")
                            st.caption(f"📄 Full log saved to: `output/{_log_name}`")
                            if _r["log_lines"]:
                                with st.expander("🔴 Error log (last 60 lines)"):
                                    st.code("\n".join(_r["log_lines"][-60:]), language="")

                if not run_ready and file_cols:
                    st.caption("⬆️ Enter your Gemini API Key above to enable the pipeline.")

            # Display: selected file vs currently loaded file
            cur_id_display = st.session_state.get("id_col", "id_note")
            id_label       = "Auto-generated" if cur_id_display == "__auto__" else cur_id_display
            _active_file   = st.session_state.get("data_file", DEFAULT_FILE)
            col_sel, col_act = st.columns(2)
            with col_sel:
                _sel_icon = "✅" if selected == _active_file else "📂"
                st.info(f"{_sel_icon} **Selected:** `{selected}`")
            with col_act:
                st.success(f"✅ **Loaded in Dashboard:** `{_active_file}`  \n"
                           f"ID Column: `{id_label}`")

    # ══════════════════════════════════════════════════════════════════════════
    # Tab 2 – Single Note
    # ══════════════════════════════════════════════════════════════════════════
    with tab_text:
        st.subheader("Single Note Analysis")

        _note_text = st.text_area(
            "Medical Note",
            height=300,
            placeholder="Paste your medical note here…",
            help="The text will be stored with id=1 assigned automatically.",
            key="note_text_input",
        )

        st.divider()

        # API Key
        _api_source_t = st.radio(
            "Gemini API Key",
            ["🔑 Use key from .env (auto)", "✏️ Enter custom key"],
            horizontal=True, index=0,
            key="api_src_text",
        )
        if _api_source_t == "🔑 Use key from .env (auto)":
            _effective_key_t = _key_from_secrets or _key_from_env
            if _effective_key_t:
                st.success("✅ Gemini API Key detected automatically.")
            else:
                st.warning("⚠️ `GEMINI_API_KEY` not found in `.env` or environment.")
        else:
            _effective_key_t = st.text_input(
                "Custom Gemini API Key", type="password",
                placeholder="AIza... paste your key here",
                key="api_key_text",
            )

        st.divider()

        # Note style selector
        _note_style_t = st.radio(
            "Clinical note style",
            ["shorthand", "narrative"],
            horizontal=True, index=0,
            key="note_style_text",
            help=(
                "**Shorthand**: Abbreviated/symbolic notes. "
                "**Narrative**: Full-sentence clinical notes."
            ),
        )

        st.divider()

        _text_ready = bool(_note_text.strip() and _effective_key_t)
        if st.button("Run Pipeline & Load", disabled=not _text_ready, key="btn_run_text"):
            import datetime
            _ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            _tmp_name = f"_pipeline_text_{_ts}.xlsx"
            _log_name = f"_pipeline_text_{_ts}_log.txt"
            _tmp_rel  = os.path.join("output", _tmp_name)
            _tmp_path = os.path.join(_output_dir, _tmp_name)
            _log_path = os.path.join(_output_dir, _log_name)

            _df_text = pd.DataFrame({"id": [1], "note_preprocessed": [_note_text.strip()]})
            _df_text.to_excel(_tmp_path, index=False)
            st.info(f"**Note saved** to `output/{_tmp_name}`  \n"
                    "id=1 and `note_preprocessed` column assigned automatically.")

            _proc_env                   = os.environ.copy()
            _proc_env["GEMINI_API_KEY"] = _effective_key_t

            _r = _run_pipeline(_tmp_rel, _log_path, search_dir, _proc_env,
                               style=_note_style_t)

            # Result file: pipeline writes {base}_results{ext}
            _base_out_t, _ext_out_t = os.path.splitext(_tmp_rel)
            _result_rel_t = f"{_base_out_t}_results{_ext_out_t}"
            _result_name_t = os.path.basename(_result_rel_t)

            if _r["timed_out"]:
                st.error("Pipeline timed out after 2 hours.")
            elif _r["exc"] is not None:
                st.error(f"Unexpected error: {_r['exc']}")
            elif _r["proc"] is not None:
                if _r["proc"].returncode == 0:
                    st.session_state["data_file"] = _result_rel_t
                    st.session_state["id_col"]    = "id"
                    _bump_data_version()
                    st.success(f"Results saved → **output/{_result_name_t}**  \n"
                               f"Log saved → **output/{_log_name}**")
                    st.rerun()
                else:
                    st.error("The pipeline encountered an error.")
                    st.caption(f"📄 Full log saved to: `output/{_log_name}`")
                    if _r["log_lines"]:
                        with st.expander("🔴 Error log (last 60 lines)"):
                            st.code("\n".join(_r["log_lines"][-60:]), language="")

        if not _text_ready and _note_text.strip():
            st.caption("⬆️ Enter your Gemini API Key above to enable the pipeline.")

    st.divider()

    # ── ③ Current Data Preview ────────────────────────────────────────────────
    st.subheader("③ Current Data Preview")
    st.caption(f"Total: **{len(df):,}** notes  |  Columns: **{len(df.columns)}**")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Overview
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Overview":
    st.title("Overview")

    # ── Multi-Agent Framework Explainer ───────────────────────────────────────
    with st.expander("🤖 About this System — Multi-Agent ADR Detection Framework",
                     expanded=False):

        # ── Top: description + model card ─────────────────────────────────
        _desc_col, _card_col = st.columns([2, 1], gap="large")
        with _desc_col:
            st.markdown("""
This dashboard goes beyond displaying results — it makes the **reasoning process transparent**.
For each clinical note, the system answers two key questions:

- ***Why*** is this symptom suspected as an adverse drug reaction?
- ***Where*** in the original text is the evidence?

This **text-grounded** approach minimises hallucination and supports a
**Human-in-the-Loop** workflow, where the physician retains final clinical judgment.
            """)
        with _card_col:
            st.markdown("""
<div style="background:#EFF6FF; border-left:4px solid #2563EB;
            padding:16px 20px; border-radius:6px;
            font-size:14px; line-height:2.2;">
  <div><span style="color:#64748B;">Model</span>&nbsp;&nbsp;·&nbsp;&nbsp;<b>Gemini 3 Flash</b></div>
  <div><span style="color:#64748B;">Validation</span>&nbsp;&nbsp;·&nbsp;&nbsp;<b>864 clinical notes</b></div>
  <div><span style="color:#64748B;">F1-Score</span>&nbsp;&nbsp;·&nbsp;&nbsp;<b style="color:#2563EB;">94.6 %</b></div>
  <div><span style="color:#64748B;">Architecture</span>&nbsp;&nbsp;·&nbsp;&nbsp;<b>6 specialised agents</b></div>
  <div><span style="color:#64748B;">Pipeline</span>&nbsp;&nbsp;·&nbsp;&nbsp;<b>2-phase (A → B)</b></div>
</div>
            """, unsafe_allow_html=True)

        st.divider()

        # ── Framework diagram ──────────────────────────────────────────────
        _img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "Multi-Agent_Framework.png")
        if os.path.exists(_img_path):
            st.image(_img_path, use_container_width=True)
        else:
            st.info("📌 Framework diagram not found (`Multi-Agent_Framework.png` "
                    "— place it in the project root folder).")

        st.divider()

        # ── Phase A | Phase B ──────────────────────────────────────────────
        _pa, _pb = st.columns(2, gap="large")
        with _pa:
            st.markdown("#### Phase A — Coverage-Oriented Discovery")
            st.caption("Goal: maximise recall — leave no candidate behind.")
            st.markdown("""
**① Context Understanding**
Reconstructs shorthand clinical notes (abbreviations, symbols `→`, indentation)
into a readable narrative while preserving the clinical timeline —
analogous to an intern's verbal handover.

**② Medication Extraction**
Extracts every drug mentioned in the note (brand names, generic names,
abbreviations) to prevent false-negative omissions.

**③ ADR Candidate Generation**
Enumerates all plausible drug–symptom associations comprehensively,
feeding the full candidate pool into Phase B.
            """)
        with _pb:
            st.markdown("#### Phase B — Text-Grounded Screening")
            st.caption("Goal: maximise precision — verify every claim against source text.")
            st.markdown("""
**④ Confounder Check**
Investigates whether each symptom is better explained by an underlying
condition or co-treatment rather than the suspected drug.

**⑤ Confounder Validation**
Verifies each confounder claim via **Evidence Spans** — direct quotes
from the original note — blocking unsupported or hallucinated assertions.

**⑥ ADR Final Detection**
Synthesises all evidence and validation outputs to produce a final
ADR determination with full, traceable reasoning.
            """)

    st.divider()

    total         = len(df)
    n_adr         = int(df["validation_side_effect_binary"].sum())
    n_meds_rows   = int((df["n_meds"] > 0).sum())
    n_adr_cand    = int((df["n_adrs"] > 0).sum())
    n_conf_cancel = int(
        ((df["adr_candidates_side_effect_binary"] == 1) & (df["confounder_side_effect_binary"] == 0)).sum()
    )

    # KPI cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Notes",            f"{total:,}")
    c2.metric("Notes w/ ADR Positive",     f"{n_adr:,}")
    c3.metric("Notes w/ Medications",   f"{n_meds_rows:,}")
    c4.metric("Notes w/ ADR Candidates",f"{n_adr_cand:,}")


    st.divider()

    # ── Final Label Distribution  +  Top 10 Confirmed ADR Drugs ─────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Final Label Distribution")
        vc = df["validation_side_effect_binary"].value_counts().reset_index()
        vc.columns = ["label", "count"]
        vc["label_name"] = vc["label"].map({0: "No ADR", 1: "ADR"})
        fig = px.pie(
            vc, names="label_name", values="count",
            color="label_name",
            color_discrete_map={"No ADR": "#93C5FD", "ADR": "#1D4ED8"},
            hole=0.45,
        )
        fig.update_traces(textinfo="percent+value")
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Top 10 Confirmed ADR Drugs")

        @st.cache_data
        def _ov_top_drugs(_df, data_file="", ver=0):
            cnts = Counter()
            for _, row in _df.iterrows():
                abbr_map = {}
                for m in row["_medications_parsed"].get("medications", []):
                    if not isinstance(m, dict):
                        continue
                    abbr = m.get("text", "").strip()
                    name = m.get("medication", "").strip()
                    if abbr:
                        abbr_map[abbr.lower()] = name if name else abbr
                for item in row["_validation_parsed"].get("results", []):
                    if not isinstance(item, dict):
                        continue
                    drug = item.get("drug", "").strip()
                    norm = abbr_map.get(drug.lower(), drug) if drug else drug
                    if norm: cnts[norm] += 1
            return cnts

        _cur_file = st.session_state.get("data_file", "")
        _ver = st.session_state.get("_data_version", 0)
        top_drugs_ov = _ov_top_drugs(df[df["validation_side_effect_binary"] == 1],
                                      data_file=_cur_file, ver=_ver)
        top10_d = pd.DataFrame(top_drugs_ov.most_common(10), columns=["Drug", "Count"])
        fig_td = px.bar(top10_d, x="Count", y="Drug", orientation="h",
                        color="Count", color_continuous_scale="Blues")
        fig_td.update_layout(yaxis=dict(autorange="reversed"),
                             margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig_td, use_container_width=True)

    # ── Top 10 Confirmed Drug → Symptom Pairs ─────────────────────────────────
    st.subheader("Top 10 Confirmed Drug → Symptom Pairs")

    @st.cache_data
    def _ov_top_pairs(_df, data_file="", ver=0):
        cnts = Counter()
        canonical = {}
        for _, row in _df.iterrows():
            abbr_map = {}
            for m in row["_medications_parsed"].get("medications", []):
                if not isinstance(m, dict):
                    continue
                abbr = m.get("text", "").strip()
                name = m.get("medication", "").strip()
                if abbr:
                    abbr_map[abbr.lower()] = name if name else abbr
            for item in row["_validation_parsed"].get("results", []):
                if not isinstance(item, dict):
                    continue
                drug = item.get("drug", "").strip()
                sym  = item.get("symptom", "")
                if isinstance(sym, list): sym = ", ".join(sym)
                sym = sym.strip()
                norm_drug = abbr_map.get(drug.lower(), drug) if drug else drug
                if sym:
                    key = sym.lower()
                    if key not in canonical:
                        canonical[key] = sym
                    sym = canonical[key]
                if norm_drug and sym:
                    cnts[f"{norm_drug} → {sym}"] += 1
        return cnts

    top_pairs_ov = _ov_top_pairs(df[df["validation_side_effect_binary"] == 1],
                                  data_file=st.session_state.get("data_file", ""),
                                  ver=st.session_state.get("_data_version", 0))
    top10_p = pd.DataFrame(top_pairs_ov.most_common(10), columns=["Pair", "Count"])
    fig_tp = px.bar(top10_p, x="Count", y="Pair", orientation="h",
                    color="Count", color_continuous_scale="Blues")
    fig_tp.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=12)),
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_tp, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Note Browser
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Note Browser":
    st.title("Note Browser")

    with st.expander("Filters", expanded=True):
        f1, f2, f3 = st.columns(3)
        with f1:
            model_filter = st.multiselect(
                "Final ADR Label",
                options=[0, 1],
                default=[0, 1],
                format_func=lambda x: "No ADR" if x == 0 else "ADR Positive",
            )
        with f2:
            conf_adr_filter = st.selectbox(
                "Confirmed ADRs",
                ["All", "Has Confirmed ADRs", "No Confirmed ADRs"],
            )
        with f3:
            search_id = st.text_input("Search by ID", placeholder="Enter ID...")

    filtered = df[df["validation_side_effect_binary"].isin(model_filter)].copy()
    if conf_adr_filter == "Has Confirmed ADRs":
        filtered = filtered[filtered["n_validated"] > 0]
    elif conf_adr_filter == "No Confirmed ADRs":
        filtered = filtered[filtered["n_validated"] == 0]
    if search_id.strip():
        filtered = filtered[filtered["id_note"].str.contains(search_id.strip())]

    st.caption(f"Showing **{len(filtered):,}** notes")

    # ── Table ────────────────────────────────────────────────────────────────
    tbl = filtered[["id_note", "validation_side_effect_binary",
                     "n_all_drugs", "n_adrs", "n_validated"]].rename(columns={
        "id_note":                       "ID",
        "validation_side_effect_binary": "Final ADR",
        "n_all_drugs":                   "# Medications",
        "n_adrs":                        "# ADR Candidates",
        "n_validated":                   "# Confirmed ADRs",
    }).reset_index(drop=True)

    st.dataframe(
        tbl,
        use_container_width=True,
        height=260,
        hide_index=True,
        column_config={
            "ID": st.column_config.TextColumn("ID", width="small"),
            "Final ADR": st.column_config.NumberColumn("Final ADR", format="%d", width="small"),
            "# Medications": st.column_config.NumberColumn("# Medications", format="%d", width="small"),
            "# ADR Candidates": st.column_config.NumberColumn("# ADR Candidates", format="%d", width="small"),
            "# Confirmed ADRs": st.column_config.NumberColumn("# Confirmed ADRs", format="%d", width="small"),
        },
    )

    # ── Note selection: selectbox (scroll ↕ or type to filter) ──────────
    note_ids = filtered["id_note"].tolist()

    selected_note = st.selectbox(
        "Select ID to view details",
        options=note_ids,
        index=None,
        placeholder="— Select or type a note ID —",
        key="browser_id_select",
    )

    if selected_note:
        row = df[df["id_note"] == selected_note].iloc[0]

        st.divider()
        final_lbl = "ADR Positive" if row["validation_side_effect_binary"] == 1 else "No ADR"
        h1, h2, h3 = st.columns(3)
        h1.metric("ID",             row["id_note"])
        h2.metric("Model Final Verdict", final_lbl)
        h3.metric("Confirmed ADRs",      int(row["n_validated"]))

        tab_meds, tab_reason, tab_final = st.tabs(
            ["Medications", "Reasoning Process", "Final Result"]
        )

        # ── Tab 1: Medications ────────────────────────────────────────────────
        with tab_meds:
            meds = row["_medications_parsed"].get("medications", [])
            if meds:
                st.dataframe(
                    pd.DataFrame(meds)[["text", "medication"]].rename(
                        columns={"text": "Raw Text in Note", "medication": "Normalized Drug Name"}
                    ),
                    use_container_width=True,
                )
            else:
                st.info("No medications extracted from this note.")

        # ── Tab 2: Reasoning process ──────────────────────────────────────────
        with tab_reason:
            adrs  = row["_adr_parsed"].get("adr_candidates", [])
            confs = row["_confounder_parsed"].get("results", [])
            cv_qs = row["_confounder_val_parsed"].get("validation_questions", [])

            def find_confounder(drug, confs):
                for r in confs:
                    if not isinstance(r, dict):
                        continue
                    if r.get("drug", "").strip().upper() == drug.strip().upper():
                        return r
                return None

            def find_cv_question(drug, cv_qs):
                for vq in cv_qs:
                    if not isinstance(vq, dict):
                        continue
                    cand = vq.get("adr_candidate", "")
                    if drug.strip().upper() in cand.upper():
                        return vq
                return None

            if not adrs:
                st.info("No ADR candidates were extracted from this note.")
            else:
                st.caption(
                    f"**{len(adrs)} ADR candidate(s)** found — "
                    "each card shows extraction evidence → confounder judgment → validation review."
                )
                for adr in adrs:
                    if not isinstance(adr, dict):
                        continue
                    drug    = adr.get("drug", "-")
                    symptom = adr.get("symptom", "-")
                    text    = adr.get("text", "")

                    conf   = find_confounder(drug, confs)
                    cv_q   = find_cv_question(drug, cv_qs)

                    conf_status = conf.get("status", "?") if conf else None
                    cv_verdict  = cv_q.get("validation_verdict", "?") if cv_q else None

                    if cv_verdict == "Correct" and conf_status == "Yes ADR":
                        badge = "🔴 Confirmed ADR"
                    elif conf_status == "No ADR":
                        badge = "🟡 Removed by Confounder"
                    elif cv_verdict == "Incorrect":
                        badge = "🟠 Reversed by Validation"
                    else:
                        badge = "⚪ Unclear"

                    with st.expander(f"**{drug}  →  {symptom}**   |   {badge}", expanded=True):

                        # Step 1
                        st.markdown("##### ① ADR Candidate Extraction")
                        if text:
                            st.markdown(f"> {text}")
                        else:
                            st.markdown("*(No source text snippet)*")

                        st.divider()

                        # Step 2
                        st.markdown("##### ② Confounder Agent Judgment")
                        if conf:
                            icon = "✅" if conf_status == "Yes ADR" else "❌"
                            st.markdown(f"**Verdict:** {icon} `{conf_status}`")
                            reasoning = conf.get("reasoning", "")
                            if reasoning:
                                st.markdown(f"**Reasoning:**  \n{reasoning}")
                        else:
                            st.markdown("*(No confounder result available)*")

                        st.divider()

                        # Step 3
                        st.markdown("##### ③ Confounder Validation")
                        if cv_q:
                            icon2 = "✅" if cv_verdict == "Correct" else "❌"
                            st.markdown(f"**Validation Verdict:** {icon2} `{cv_verdict}`")
                            summary = cv_q.get("confounder_judgment_summary", "")
                            if summary:
                                st.markdown(f"**Summary:**  \n{summary}")
                            ctx_ev  = cv_q.get("context_note_evidences", "")
                            if text:
                                st.markdown("**Note Evidence:**")
                                st.code(text, language="")
                            if ctx_ev and ctx_ev != "N/A":
                                st.markdown(f"**Contextual Evidence:**  \n{ctx_ev}")
                            reasons = cv_q.get("reasoning", [])
                            if reasons:
                                st.markdown("**Detailed Reasoning:**")
                                for i, r in enumerate(reasons, 1):
                                    st.markdown(f"**{i}.** {r}")
                        else:
                            st.markdown("*(No validation result available)*")

        # ── Tab 3: Final result ───────────────────────────────────────────────
        with tab_final:
            val_data  = row["_validation_parsed"]
            val_items = val_data.get("results", [])

            if val_items:
                st.success(f"**{len(val_items)} ADR(s) confirmed** in this note")
                for item in val_items:
                    if not isinstance(item, dict):
                        continue
                    drug    = item.get("drug", "-")
                    symptom = item.get("symptom", "-")
                    if isinstance(symptom, list):
                        symptom = ", ".join(symptom)
                    explanation = item.get("explanation", "-")
                    raw_text    = item.get("text", "")

                    with st.expander(f"{drug}  →  {symptom}"):
                        if raw_text:
                            st.markdown("**Source Text:**")
                            st.code(raw_text, language="")
                        st.markdown(f"**Confirmation Rationale:**  \n{explanation}")
            else:
                result_str = val_data.get("result", "No Side Effect")
                st.info(f"Final result: {result_str}")
    else:
        st.info("Select a Note ID from the dropdown above to view detailed note analysis.")


# ═════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Drug & ADR Analysis
# ═════════════════════════════════════════════════════════════════════════════
elif page == "Drug & ADR Analysis":
    st.title("Drug & ADR Analysis")

    @st.cache_data
    def build_drug_detail_index(_df, data_file="", ver=0):
        """Per-drug lookup: ADR candidates & confirmed ADRs across all notes."""
        candidates = {}   # norm_drug → [{"Note ID", "Symptom", "Confounder", "Validation", "Source Text"}]
        confirmed  = {}   # norm_drug → [{"Note ID", "Symptom", "Source Text", "Explanation"}]

        for _, row in _df.iterrows():
            note_id  = row["id_note"]
            abbr_map = {}
            for m in row["_medications_parsed"].get("medications", []):
                if not isinstance(m, dict): continue
                abbr = m.get("text", "").strip()
                name = m.get("medication", "").strip()
                if abbr:
                    abbr_map[abbr.lower()] = name if name else abbr

            # ── ADR candidates ────────────────────────────────────────────
            confs_list = row["_confounder_parsed"].get("results", [])
            cv_list    = row["_confounder_val_parsed"].get("validation_questions", [])

            for a in row["_adr_parsed"].get("adr_candidates", []):
                if not isinstance(a, dict): continue
                drug    = a.get("drug", "").strip()
                symptom = a.get("symptom", "").strip()
                text    = a.get("text", "")
                norm    = abbr_map.get(drug.lower(), drug) if drug else drug
                if not norm: continue

                conf_status, cv_verdict = None, None
                for c in confs_list:
                    if isinstance(c, dict) and c.get("drug","").strip().upper() == drug.upper():
                        conf_status = c.get("status")
                        break
                for vq in cv_list:
                    if isinstance(vq, dict) and drug.upper() in vq.get("adr_candidate","").upper():
                        cv_verdict = vq.get("validation_verdict")
                        break

                candidates.setdefault(norm, []).append({
                    "Note ID":     note_id,
                    "Symptom":     symptom,
                    "Confounder":  conf_status or "—",
                    "Validation":  cv_verdict  or "—",
                    "Source Text": text,
                })

            # ── Confirmed ADRs ─────────────────────────────────────────────
            for item in row["_validation_parsed"].get("results", []):
                if not isinstance(item, dict): continue
                drug    = item.get("drug", "").strip()
                symptom = item.get("symptom", "")
                if isinstance(symptom, list): symptom = ", ".join(symptom)
                symptom = symptom.strip()
                text    = item.get("text", "")
                expl    = item.get("explanation", "")
                norm    = abbr_map.get(drug.lower(), drug) if drug else drug
                if not norm: continue

                confirmed.setdefault(norm, []).append({
                    "Note ID":     note_id,
                    "Symptom":     symptom,
                    "Source Text": text,
                    "Explanation": expl,
                })

        return candidates, confirmed

    @st.cache_data
    def get_med_counts(_df, data_file="", ver=0):
        all_meds = []
        for parsed in _df["_medications_parsed"]:
            for m in parsed.get("medications", []):
                if not isinstance(m, dict):
                    continue
                name = m.get("medication", "").strip()
                if name:
                    all_meds.append(name)
        return Counter(all_meds)

    @st.cache_data
    def get_adr_counts(_df, data_file="", ver=0):
        pairs      = []
        drug_cnts  = Counter()
        sym_cnts   = Counter()
        for _, row in _df.iterrows():
            # Build abbreviation → standardized name map per note
            abbr_map = {}
            for m in row["_medications_parsed"].get("medications", []):
                if not isinstance(m, dict):
                    continue
                abbr = m.get("text", "").strip()
                name = m.get("medication", "").strip()
                if abbr:
                    abbr_map[abbr.lower()] = name if name else abbr
            for a in row["_adr_parsed"].get("adr_candidates", []):
                if not isinstance(a, dict):
                    continue
                drug    = a.get("drug", "").strip()
                symptom = a.get("symptom", "").strip()
                # Use standardized name if abbreviation exists, else keep as-is
                normalized = abbr_map.get(drug.lower(), drug) if drug else drug
                if normalized: drug_cnts[normalized] += 1
                if symptom:    sym_cnts[symptom] += 1
                if normalized and symptom:
                    pairs.append(f"{normalized} → {symptom}")
        return drug_cnts, sym_cnts, Counter(pairs)

    @st.cache_data
    def get_validated_adr_counts(_df, data_file="", ver=0):
        drug_cnts     = Counter()
        sym_cnts      = Counter()
        pairs         = []
        sym_canonical = {}  # symptom.lower() → first-seen canonical form

        for _, row in _df.iterrows():
            # 1) Abbreviation → standardized drug name map
            abbr_map = {}
            for m in row["_medications_parsed"].get("medications", []):
                if not isinstance(m, dict):
                    continue
                abbr = m.get("text", "").strip()
                name = m.get("medication", "").strip()
                if abbr:
                    abbr_map[abbr.lower()] = name if name else abbr

            # 2) Count drugs and symptoms from validation results
            for item in row["_validation_parsed"].get("results", []):
                if not isinstance(item, dict):
                    continue
                drug    = item.get("drug", "").strip()
                symptom = item.get("symptom", "")
                if isinstance(symptom, list): symptom = ", ".join(symptom)
                symptom = symptom.strip()

                normalized_drug = abbr_map.get(drug.lower(), drug) if drug else drug

                # Normalize symptom casing: keep the first-seen form as canonical
                sym_key = symptom.lower()
                if sym_key and sym_key not in sym_canonical:
                    sym_canonical[sym_key] = symptom
                normalized_sym = sym_canonical.get(sym_key, symptom)

                if normalized_drug: drug_cnts[normalized_drug] += 1
                if normalized_sym:  sym_cnts[normalized_sym] += 1
                if normalized_drug and normalized_sym:
                    pairs.append(f"{normalized_drug} → {normalized_sym}")

        return drug_cnts, sym_cnts, Counter(pairs)

    _cur_file = st.session_state.get("data_file", "")
    _ver = st.session_state.get("_data_version", 0)
    med_counts                           = get_med_counts(df, data_file=_cur_file, ver=_ver)
    drug_counts, sym_counts, pair_cnts   = get_adr_counts(df, data_file=_cur_file, ver=_ver)
    val_drug_cnts, val_sym_cnts, val_pair_cnts = get_validated_adr_counts(
        df[df["validation_side_effect_binary"] == 1], data_file=_cur_file, ver=_ver
    )

    # ── Top N control: slider + number input (bidirectional sync) ────────────
    if "top_n" not in st.session_state:
        st.session_state["top_n"] = 20
    if "_tn_slider" not in st.session_state:
        st.session_state["_tn_slider"] = st.session_state["top_n"]
    if "_tn_input" not in st.session_state:
        st.session_state["_tn_input"] = st.session_state["top_n"]

    def _sync_slider():
        v = st.session_state["_tn_slider"]
        st.session_state["top_n"]    = v
        st.session_state["_tn_input"] = v

    def _sync_input():
        v = max(1, int(st.session_state["_tn_input"]))
        st.session_state["top_n"]     = v
        st.session_state["_tn_slider"] = min(v, 100)

    col_s, col_n = st.columns([4, 1])
    with col_s:
        st.slider("Top N to display", 1, 100,
                  key="_tn_slider", on_change=_sync_slider)
    with col_n:
        st.number_input("Enter N", min_value=1, step=1,
                        key="_tn_input", on_change=_sync_input)

    top_n = st.session_state["top_n"]

    # Section 1
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Medications in All Notes")
        top_meds = pd.DataFrame(med_counts.most_common(top_n), columns=["Drug", "Count"])
        fig_meds = px.bar(top_meds, x="Count", y="Drug", orientation="h",
                          color="Count", color_continuous_scale="Blues")
        fig_meds.update_layout(yaxis=dict(autorange="reversed"),
                               margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig_meds, use_container_width=True)

    with col2:
        st.subheader("Medications in Confirmed ADRs")
        top_conf_drugs = pd.DataFrame(val_drug_cnts.most_common(top_n), columns=["Drug", "Count"])
        fig_conf_drugs = px.bar(top_conf_drugs, x="Count", y="Drug", orientation="h",
                                color="Count", color_continuous_scale="Oranges")
        fig_conf_drugs.update_layout(yaxis=dict(autorange="reversed"),
                                     margin=dict(t=10, b=10, l=10, r=10), coloraxis_showscale=False)
        st.plotly_chart(fig_conf_drugs, use_container_width=True)

    # Section 2
    st.divider()

    # ── Section 2 Top N control ──────────────────────────────────────────────
    if "top_n_s2" not in st.session_state:
        st.session_state["top_n_s2"] = 20
    if "_tn_s2_slider" not in st.session_state:
        st.session_state["_tn_s2_slider"] = st.session_state["top_n_s2"]
    if "_tn_s2_input" not in st.session_state:
        st.session_state["_tn_s2_input"] = st.session_state["top_n_s2"]

    def _sync_s2_slider():
        v = st.session_state["_tn_s2_slider"]
        st.session_state["top_n_s2"]     = v
        st.session_state["_tn_s2_input"] = v

    def _sync_s2_input():
        v = max(1, int(st.session_state["_tn_s2_input"]))
        st.session_state["top_n_s2"]      = v
        st.session_state["_tn_s2_slider"] = min(v, 100)

    col_s2, col_n2 = st.columns([4, 1])
    with col_s2:
        st.slider("Top N to display (Section 2)", 1, 100,
                  key="_tn_s2_slider", on_change=_sync_s2_slider)
    with col_n2:
        st.number_input("Enter N ", min_value=1, step=1,
                        key="_tn_s2_input", on_change=_sync_s2_input)

    top_n_s2 = st.session_state["top_n_s2"]

    st.subheader("ADR Symptoms")
    top_syms = pd.DataFrame(val_sym_cnts.most_common(top_n_s2), columns=["Symptom", "Count"])
    fig_sym = px.bar(top_syms, x="Count", y="Symptom", orientation="h",
                     color="Count", color_continuous_scale="Reds")
    fig_sym.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(t=10, b=10, l=10, r=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_sym, use_container_width=True)

    top_pairs = pd.DataFrame(val_pair_cnts.most_common(top_n_s2), columns=["Pair", "Count"])
    fig_pair = px.bar(top_pairs, x="Count", y="Pair", orientation="h",
                      color="Count", color_continuous_scale="Purples",
                      title="Drug → Symptom Pairs (Confirmed ADRs)")
    fig_pair.update_layout(
        yaxis=dict(autorange="reversed", tickfont=dict(size=13)),
        margin=dict(t=40, b=10, l=10, r=10),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig_pair, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # Section 3 – Drug Detail View
    # ══════════════════════════════════════════════════════════════════════════
    st.divider()
    st.subheader("🔍 Drug Detail View")
    st.caption(
        "Select a normalized drug name to explore all ADR candidates "
        "and confirmed ADRs associated with it across every note."
    )

    _cand_idx, _conf_idx = build_drug_detail_index(df, data_file=_cur_file, ver=_ver)

    # Drug list sorted by candidate count (most suspicious drugs first)
    _all_drugs = sorted(
        set(list(_cand_idx.keys()) + list(_conf_idx.keys())),
        key=lambda d: len(_cand_idx.get(d, [])),
        reverse=True,
    )

    _sel_drug = st.selectbox(
        "Select a drug",
        options=_all_drugs,
        index=None,
        placeholder="— Select or type a drug name —",
        key="drug_detail_select",
    )

    if _sel_drug:
        _cands = _cand_idx.get(_sel_drug, [])
        _confs = _conf_idx.get(_sel_drug, [])
        _conf_rate = (f"{len(_confs)/len(_cands)*100:.1f}%" if _cands else "—")

        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("ADR Candidates",    len(_cands))
        _m2.metric("Confirmed ADRs",    len(_confs))
        _m3.metric("Confirmation Rate", _conf_rate)

        _tab_c, _tab_v = st.tabs(["📋 ADR Candidates", "✅ Confirmed ADRs"])

        # ── Tab 1: ADR Candidates ─────────────────────────────────────────
        with _tab_c:
            if _cands:
                _n_notes_c = len(set(r["Note ID"] for r in _cands))
                _n_syms_c  = len(set(r["Symptom"] for r in _cands))
                st.caption(
                    f"**{len(_cands)}** candidate(s) across **{_n_notes_c}** note(s)"
                    f" — **{_n_syms_c}** unique symptom(s)"
                )

                # Symptom frequency bar chart
                _sym_ctr_c = Counter(r["Symptom"] for r in _cands if r["Symptom"])
                if _sym_ctr_c:
                    _sym_df_c = pd.DataFrame(
                        _sym_ctr_c.most_common(20), columns=["Symptom", "Count"]
                    )
                    _fig_c = px.bar(
                        _sym_df_c, x="Count", y="Symptom", orientation="h",
                        color="Count", color_continuous_scale="Blues",
                    )
                    _fig_c.update_layout(
                        yaxis=dict(autorange="reversed"),
                        margin=dict(t=10, b=10, l=10, r=10),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(_fig_c, use_container_width=True)

                st.dataframe(
                    pd.DataFrame(_cands),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Note ID":     st.column_config.TextColumn("Note ID",     width="small"),
                        "Symptom":     st.column_config.TextColumn("Symptom",     width="medium"),
                        "Confounder":  st.column_config.TextColumn("Confounder",  width="small"),
                        "Validation":  st.column_config.TextColumn("Validation",  width="small"),
                        "Source Text": st.column_config.TextColumn("Source Text", width="large"),
                    },
                )
            else:
                st.info("No ADR candidates found for this drug.")

        # ── Tab 2: Confirmed ADRs ─────────────────────────────────────────
        with _tab_v:
            if _confs:
                _n_notes_v = len(set(r["Note ID"] for r in _confs))
                _n_syms_v  = len(set(r["Symptom"] for r in _confs))
                st.caption(
                    f"**{len(_confs)}** confirmed ADR(s) across **{_n_notes_v}** note(s)"
                    f" — **{_n_syms_v}** unique symptom(s)"
                )

                # Symptom frequency bar chart
                _sym_ctr_v = Counter(r["Symptom"] for r in _confs if r["Symptom"])
                if _sym_ctr_v:
                    _sym_df_v = pd.DataFrame(
                        _sym_ctr_v.most_common(20), columns=["Symptom", "Count"]
                    )
                    _fig_v = px.bar(
                        _sym_df_v, x="Count", y="Symptom", orientation="h",
                        color="Count", color_continuous_scale="Oranges",
                    )
                    _fig_v.update_layout(
                        yaxis=dict(autorange="reversed"),
                        margin=dict(t=10, b=10, l=10, r=10),
                        coloraxis_showscale=False,
                    )
                    st.plotly_chart(_fig_v, use_container_width=True)

                st.dataframe(
                    pd.DataFrame(_confs),
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Note ID":     st.column_config.TextColumn("Note ID",     width="small"),
                        "Symptom":     st.column_config.TextColumn("Symptom",     width="medium"),
                        "Source Text": st.column_config.TextColumn("Source Text", width="large"),
                        "Explanation": st.column_config.TextColumn("Explanation", width="large"),
                    },
                )
            else:
                st.info("No confirmed ADRs found for this drug.")

    else:
        st.info("⬆️ Select a drug above to explore its ADR profile.")

