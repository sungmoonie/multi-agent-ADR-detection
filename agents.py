"""
Multi-Agent Prompts for ADR (Adverse Drug Reaction) Detection.

Supports two clinical note styles:
  - "shorthand": Abbreviated/symbolic clinical notes (e.g., Korean EMR shorthand).
                  Uses Context_Agent to convert shorthand → narrative before downstream agents.
  - "narrative": Full-sentence narrative notes (e.g., MIMIC discharge summaries).
                  Skips Context_Agent; agents work directly on the raw note.

Uses OpenRouter API (OpenAI-compatible) for LLM calls.
"""

import os
import json
import re
import warnings

from openai import OpenAI
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# ============================================================================
# Base Agent
# ============================================================================
class BaseAgent:
    """Shared initialization, LLM calling, and JSON cleaning logic."""

    def __init__(self, model="google/gemini-3-flash-preview"):
        self.model = model
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # ----- helpers ----------------------------------------------------------
    @staticmethod
    def _clean_json_string(text):
        """Strip markdown ```json ... ``` wrappers from LLM output."""
        text = text.strip()
        match = re.search(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
        return match.group(1) if match else text

    # ----- core LLM call ----------------------------------------------------
    def _call_llm(self, instruction, messages, temperature=0.0,
                  json_mode=False, **_ignored):
        try:
            kwargs = dict(
                model=self.model,
                messages=[
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": messages},
                ],
                temperature=temperature,
                extra_body={
                    "provider": {
                        "zdr": True,
                    },
                },
            )
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content

        except Exception as e:
            print(f"[LLM Error] {self.model}: {e}")
            raise

    def _call_and_parse_json(self, instruction, messages, temperature=0.0,
                             json_mode=True, fallback=None, max_retries=2,
                             **_ignored):
        """Call LLM, clean response, parse as JSON.
        Retries on transient errors up to *max_retries* times.
        """
        import time as _time

        last_err = None
        for attempt in range(1 + max_retries):
            try:
                raw = self._call_llm(instruction, messages, temperature,
                                     json_mode)
                cleaned = self._clean_json_string(raw)
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    if fallback is not None:
                        print(f"[Warning] JSON parse failed (attempt {attempt+1}). "
                              f"Using fallback. Raw snippet: {raw[:200]}")
                        return fallback
                    print(f"JSON Parsing Failed. Raw: {raw}")
                    return fallback or {}
            except Exception as e:
                last_err = e
                if attempt < max_retries:
                    wait = 2 ** attempt
                    print(f"[Retry {attempt+1}/{max_retries}] LLM call failed: {e}. "
                          f"Retrying in {wait}s...")
                    _time.sleep(wait)
                else:
                    print(f"[Error] LLM call failed after {max_retries} retries: {e}")
                    if fallback is not None:
                        return fallback
                    return fallback or {}


# ============================================================================
# Context Agent  (shorthand style only)
# ============================================================================
class Context_Agent(BaseAgent):
    """Transforms shorthand clinical notes into structured verbal reports."""

    def prompt(self, note):
        instruction = """
        You are an Agent tasked with transforming clinical notes into a structured verbal report without omitting any information.
        Your role is to reconstruct the content of the clinical note in a natural, accurate manner—just as an intern would verbally report a patient's condition to a senior resident during morning rounds.
        """

        task = f"""
        ## 1. Core Principles
        * **Completeness (No Omission):** You must include all information from the note, including medications, lab values, symptoms, and plans.
        * **Factuality (No Hallucination):** Do not add personal guesses, interpretations, or inferences. If information is ambiguous, report it exactly as written.
        * **Maintain Sequence:** Follow the exact order of the clinical note to preserve the patient's timeline.
        * **Tone:** Use a professional, polite, and clear verbal reporting tone (e.g., "The patient reports...", "It is noted that...", "Records show...").

        ## 2. Special Instructions: Reporting Drug-Symptom Associations (ADR)
        Clinical notes often use shorthand to link drugs and symptoms. You must distinguish between the following three types to ensure accurate reporting of potential Adverse Drug Reactions (ADR).

        ### Type A) Explicit Causal Syntax
        * **Condition:** The drug and symptom are connected by arrows (`->`, `→`), slashes (`/`), `due to`, `d/t`, `2/2`, or `secondary to`.
        * **Action:** Explicitly state the causal relationship based on the record.
        * **Example:** "Metformin -> Diarrhea"
            * (Output) "According to the note, diarrhea occurred after taking Metformin."

        ### Type B) Shorthand Adjacency (Parallel Syntax)
        * **Condition:** A drug name and a symptom are written side-by-side without prepositions or verbs.
        * **Action:** State that the symptom occurred after taking the drug, citing the record.
        * **Example:** "MTF GI trouble", "Duvie Edema@@"
            * (Output) "According to the note, GI trouble occurred after taking MTF."
            * (Output) "According to the note, Edema@@ occurred after taking Duvie." (Preserve special markers like '@@')

        ### Type C) Grouping & Indentation
        * **Condition:** Symptoms appear immediately after a drug header or are listed/indented under a drug name.
        * **Action:** Treat all indented/listed symptoms as belonging to that specific drug context until a new header or section appears.
        * **Action:** Use phrases like "In relation to [Drug], [Symptoms] are recorded" or "Under the [Drug] entry, [Symptoms] are noted."

        ### Type D) Sequential Switching (Multi-line Syntax)
        * **Condition:** A line **starting** with an arrow (`->`, `→`) or a hyphen (`-`) indicates a transition from the previous entry.
        * **Action:** Connect the previous line's item to the current line's item as a switch or sequence. If a reason (e.g., `d/t`, `due to`) is provided, explicitly state that the change occurred because of that reason.

        ## 3. Handling Subjective Expressions
        * **Target:** Vague patient complaints like "doesn't fit me," "can't handle it," "uncomfortable," "not good."
        * **Guideline:** Unless a specific symptom is named, **DO NOT** translate these into medical terms like "Side effect" or "Intolerance."
        * **Action:** Quote the patient's expression exactly to preserve the nuance.
            * (Correct) "The patient expressed that the medication 'doesn't fit' them."
            * (Incorrect) "The patient has a drug intolerance."

        ## Output Format
        * Start the report immediately without an introduction (e.g., "Here is the report") or conclusion.
        * Use only **plain text**. Do not use Markdown tables, code blocks, or JSON.
        * Ensure the language flows naturally as spoken speech.

        ---
        **[Input Clinical Note]**
        {note}
"""
        return self._call_llm(instruction, task, temperature=0.5)


# ============================================================================
# Medications Agent
# ============================================================================
class Medications_Agent(BaseAgent):
    """Extracts all medication mentions from a clinical note."""

    def prompt(self, note):
        instruction = """
        You are a clinical pharmacist Agent specializing in identifying any and all medication-related mentions in clinical notes.
        You must identify not only individual drugs (generic, brand, abbreviation) but also mentions of drug classes, therapeutic categories, and general medication types.
        """

        task = f"""
        Your task is to extract all medication-related terms from the following clinical note.
        This includes:
        - Generic drug names (e.g., metformin)
        - Brand names (e.g., Glucophage)
        - Abbreviations (e.g., MFM, MTF, ASA, Duvie, etc.)
        - Drug classes (e.g., steroids, oral hypoglycemic agents, contrast agents, antibiotics, etc.)
        - Descriptive phrases (e.g., oral diabetes medication, etc.)


        If no medication or drug-related expression is present, return:
        {{
        "medications": []
        }}

        For each detected mention, return:
        - "text": the exact word or phrase as written in the note
        - "medication": the corresponding standardized drug name

        Return the final output as a JSON array. No explanations or commentary—just valid JSON.

        Output Format Example:
        {{
        "medications": [
            {{
            "text": "ASA",
            "medication": "Aspirin"
            }},
            {{
            "text": "MFM",
            "medication": "Metformin"
            }},
            {{
            "text": "MTF",
            "medication": "Metformin"
            }}
        ]
        }}

        ## Clinical Note:
        {note}
        """

        return self._call_and_parse_json(
            instruction, task,
            temperature=0.0, json_mode=True,
            fallback={"medications": []},
        )


# ============================================================================
# ADR Candidates Agent
# ============================================================================
class ADRcandidates_Agent(BaseAgent):
    """Extracts drug-symptom pairs as ADR candidates from clinical notes.

    Args:
        note_style: "shorthand" or "narrative" — controls prompt wording.
    """

    def __init__(self, note_style="shorthand", **kwargs):
        super().__init__(**kwargs)
        self.note_style = note_style

    def prompt(self, drugs, note, context=None):
        drugs_str = json.dumps(drugs, ensure_ascii=False)

        instruction = """
        You are a highly skilled Clinical Pharmacology Specialist Agent.
        Your goal is to extract structured JSON data regarding potential Adverse Drug Reactions (ADR) from clinical notes.
        You strictly output standard JSON format only. No markdown, no conversational text.
        """

        if self.note_style == "shorthand":
            task = f"""
        Given the clinical note content (summarized for readability), and a list of medications mentioned in the note,
        extract all candidate drug-symptom pairs where a drug and an adverse symptom are mentioned in the same context.
        Include both current and past events.

        Extract all **possible** drug–symptom (ADR) pairs that are either explicitly or implicitly connected.
        Include the following types of co-mention:
        - Past history or known adverse events
        - Discontinuation, dose reduction, or switching of a drug following a symptom
        - Indirect or causative expressions implying an adverse effect, such as "induced," "aggravated," "exacerbated," etc.
        - Compressed shorthand such as "drug symptom", drug → symptom, or symbolic notations like slashes or arrows
        - A drug mention followed immediately by a symptom in the next sentence or line (e.g., "lantus 18U" followed by "nausea", "HA")
        - Grouped symptoms following a medication line, even if not directly connected

        Do not infer beyond what is written in the note, but do allow medically reasonable linkage from structural or compressed formats.
        For each candidate, the "text" field must be a direct quote from the clinical note.
        Do not paraphrase or reword.

        ### Inputs:
        ## Extracted Medications:
        {drugs_str}

        ## Clinical Note Summary (Context):
        {context}

        ## Clinical Note:
        {note}
        ---

        ### Output Format:
        Return only a valid JSON object. Do not include markdown formatting (like ```json).

        If no drugs were mentioned, return:
        {{
        "adr_candidates": []
        }}

        For each ADR candidate, return:
        {{
            "adr_candidates": [
                {{
                    "text": "Direct excerpt from the note proving the link",
                    "drug": "Name of the drug involved",
                    "symptom": "The mentioned adverse symptom"
                }}
            ]
        }}
        """
        else:  # narrative
            task = f"""
        Given the clinical note, and a list of medications mentioned in the note,
        extract all candidate drug–symptom pairs where a medication and an adverse symptom (ADR / side effect) are mentioned in the same context.
        Include both current and past events.

        Extract **all possible** candidate drug–symptom or drug-AE keyword(ADR) pairs that are explicitly or implicitly connected within the same context.
        Include the following co-mention patterns:
        - Past history or known adverse events/Allergies
        - Discontinuation, dose reduction, or switching of a drug, with the symptom explicitly stated as the reason
        - Causal/temporal expressions (including "induced," "aggravated," "exacerbated," "due to," "secondary to," "attributed to," "after starting," "since starting," "while taking", etc.)

        Do not infer anything beyond what is written in the note.
        For each candidate, the "text" field must be a direct quote from the clinical note.
        Do not paraphrase or reword.

        ### Inputs:
        ## Extracted Medications:
        {drugs_str}

        ## Clinical Note:
        {note}
        ---

        ### Output Format:
        Return only a valid JSON object. Do not include markdown formatting (like ```json).

        If no drugs were mentioned, return:
        {{
        "adr_candidates": []
        }}

        For each ADR candidate, return:
        {{
            "adr_candidates": [
                {{
                    "text": "Direct excerpt from the note proving the link",
                    "drug": "Name of the drug involved",
                    "symptom": "The mentioned adverse symptom"
                }}
            ]
        }}
        """

        return self._call_and_parse_json(
            instruction, task,
            temperature=0.0, json_mode=True,
            fallback={"adr_candidates": []},
        )


# ============================================================================
# Confounders Check Agent
# ============================================================================
class ConfoundersCheck_Agent(BaseAgent):
    """Screens ADR candidates for confounders or invalidities.

    Args:
        note_style: "shorthand" or "narrative" — controls whether context is used.
    """

    def __init__(self, note_style="shorthand", **kwargs):
        super().__init__(**kwargs)
        self.note_style = note_style

    def prompt(self, adr_candidates, note, context=None):
        if not isinstance(adr_candidates, str):
            adr_candidates_str = json.dumps(adr_candidates, ensure_ascii=False)
        else:
            adr_candidates_str = adr_candidates

        instruction = """
        You are a Clinical Review Agent acting as a 'Skeptic' focusing on Differential Diagnosis.
        Your goal is to screen potential Drug-Adverse Drug Reaction (ADR) candidates and output the results in structured JSON format.
        """
        if self.note_style == "narrative":
            instruction += """
        You strictly output standard JSON format only. No markdown, no conversational text.
        """

        task = f"""
        You will be given a clinical note, a context note, and a list of drug–adverse symptom (ADR) candidate pairs.

        Task: Evaluate ADR candidates for possible confounders or invalidities.
        For each ADR candidate, do one or more of the following:
        - If the symptom could have other clinical explanations, list those as possible confounders.
        - If the ADR candidate is ambiguous, not a real symptom, or clinically invalid, flag it and explain why.
        - If the ADR candidate's text does not actually appear in the clinical note or the drug–symptom pair is contextually unrelated, explicitly flag this as an error.
        - If a medication was discontinued or reduced in dose without a specific symptom being mentioned, flag that it cannot be assumed to be an ADR (e.g., could be due to improvement or other reasons).

        Examples of ADR candidates that require clinical consideration:
        - Vague phrases like "안 맞음", "잘 안 맞음" (not tolerating, discomfort, or non-adherence) without a specific symptom
        - Psychological states like "fear of injection" that are not post-treatment physiological symptoms
        - Symptoms that are expected effects of treatment (e.g., fatigue during chemotherapy)
        - ADR text or symptom not actually found in or supported by the clinical note
        - Medication stopped or reduced without accompanying symptom

        Guidelines:
        - Do not infer beyond what is written in the note.
        - Explicitly note if the ADR candidate appears to be hallucinated, contextually mismatched, or unsupported by the clinical note.

        If no confounders are identified and the ADR candidate appears valid, explicitly write: "No relevant confounders found."

        ---
        ### Inputs"""

        if self.note_style == "shorthand":
            task += f"""
        1. **Context Summary:** {context}
        2. **Raw Clinical Note:** {note}
        3. **ADR Candidates:** {adr_candidates_str}"""
        else:
            task += f"""
        1. **Raw Clinical Note:** {note}
        2. **ADR Candidates:** {adr_candidates_str}"""

        task += f"""

        ---
        ### Output Format (JSON Only)
        Return a JSON object with a list of "results".

        Example:
        {{
            "results": [
                {{
                    "drug": [ADR Candidate Drug],
                    "symptom": [ADR Candidate Symptom],
                    "status": [Yes ADR / No ADR],
                    "reasoning": [Explain the specific reason for rejection by citing the Summary (Context) or specific phrases from the Raw Note. If Yes ADR, write "No confounders found."]
                }},
                ,,,
            ]
        }}
        """

        return self._call_and_parse_json(
            instruction, task,
            temperature=0.0, json_mode=True, gemini_json_mime=True,
            fallback={"checked_results": []},
        )


# ============================================================================
# Confounders Validation Agent
# ============================================================================
class ConfoundersValidation_Agent(BaseAgent):
    """Validates confounder agent's interpretations against the original note.

    Args:
        note_style: "shorthand" or "narrative" — controls prompt, temperature, and JSON mode.
    """

    def __init__(self, note_style="shorthand", **kwargs):
        super().__init__(**kwargs)
        self.note_style = note_style

    def ver1_prompt(self, note, confounders, context=None):
        if self.note_style == "shorthand":
            return self._shorthand_prompt(note, context, confounders)
        else:
            return self._narrative_prompt(note, confounders)

    def _shorthand_prompt(self, note, context, confounders):
        instruction = """
        You are a Validation Agent tasked with verifying whether the Confounder Agent's interpretation of an ADR (Adverse Drug Reaction) candidate is accurate, reasonable, and faithful to the clinical records (Original Note & Context Note).
        """

        task = f"""
        ### Data Principles (Ground Truth)
        1. **Original Note:** The raw record containing abbreviations and symbols. This is the primary evidence and the **Ground Truth** for all judgments.
        2. **Context Note:** A version of the raw note converted into natural, grammatical sentences. It serves as supplementary reference. If there is a conflict or discrepancy in information density between the two, the **Original Note takes precedence.**

        ### Core Objective
        Determine whether the Confounder Agent's results reflect the context accurately, or if they exhibit hallucinations, under-interpretation, or over-interpretation.

        ### Symptom / AE Keyword Definition (Scope)
        For this task, "Symptom" includes:
        1) **Clinical symptoms** documented in the note (objective findings or subjective patient-reported symptoms), and
        2) **AE keywords** that explicitly indicate an adverse reaction even without naming a specific symptom,
        such as "side effect", "S/E", "부작용", "allergy", "rash", "intolerance", "ADR", "AE".

        ### Key Clinical Reasoning & Special Considerations (Required)
        1. **Recognition of Shorthand Notation:** Unless clearly contradicted, the following patterns are valid linkage candidate of an ADR:
        - "Drug Adverse Symptom" (e.g., "GMP itching")
        - "Drug → Adverse Symptom" or symbolic links (e.g., →, /, :)
        - Adverse symptoms appearing immediately after a drug mention (next line/sentence) or symptoms listed directly under a drug name.
        2. **Evidence of Causality:**  Medication changes (discontinuation/switch/dose reduction) do NOT constitute an ADR candidate by themselves. They are supportive evidence ONLY IF an explicit symptom/AE keyword is documented as the reason.
        3. **Decision Thresholds:**
        	- Likely ADR: An ADR is supported when an explicit symptom/AE keyword (including "side effect", "S/E", "부작용") or a symptom phrase is documented and linked to a drug by proximity or explicit connectors. Subjective intolerance phrases such as "uncomfortable," "hard to tolerate," and "not tolerating" qualify as symptoms when explicitly written in the note.
	        - Strengthening Evidence: A prescription change (discontinuation/switch/dose reduction) strengthens the ADR hypothesis only when the symptom/AE is explicitly documented as the reason in the text. A prescription change alone is not an ADR.
	        - Handling intolerance-only mentions: If only an intolerance phrase is documented (with no objective symptom/AE keyword) and there is no clinician action (discontinuation/switch/dose reduction), do not "confirm" an ADR; label it as low-likelihood/insufficient evidence.
            - Allergy / history ADR rule: Any record that documents drug "allergy," "hypersensitivity," "intolerance," "induced," or explicit past history MUST be treated as a patient-level ADR history and MUST be extracted as an ADR entry labeled "Historical ADR," even if it is not a new/current event.

        ### Evidence Span Extraction Rules
        - `original_note_evidences` MUST be a single continuous verbatim span from the Original Note.
        - The span MUST contain BOTH the drug mention and the AE/symptom (or AE keyword such as "S/E", "부작용") within:
        (a) the same line, OR
        (b) adjacent lines within the same indentation/block.
        - Do NOT stitch together distant parts of the note. Do NOT skip intervening unrelated lines.
        - When multiple drugs are listed, attribute the symptom to the closest preceding drug within the same indentation/block unless an explicit connector assigns it differently.

        ### Input Data
        1. **Original Note (RAW):** {note}
        2. **Context Note (Converted):** {context}
        3. **Confounder Agent Output:** {confounders}

        ### Output Format (JSON Only)

        ```json
        {{
        "validation_questions": [
            {{
            "adr_candidate": "[Drug – Symptom]",
            "confounder_judgment_summary": "[1-line summary of confounder's reasoning]",
            "validation_verdict": "Correct / Incorrect",
            "original_note_evidences": "[Direct verbatim quote or span from the Original Note]",
            "context_note_evidences": "[Relevant summary from the Context Note]",
            "reasoning": [
            "Compare the evidence in the Original Note and the Context Note, and determine whether the confounder judgment is supported by the documented facts.",
            "Explain specifically how the confounder agent’s reasoning aligns or conflicts with the note evidence. If the confounder reasoning is valid but does not fully negate the ADR candidate, explicitly state that both a valid confounder and an explicit drug–event cue coexist.",
            "Identify any missed explicit cues in the note, including explicit symptoms/AE keywords and explicit drug–symptom connectors. Make clear whether these cues preserve ADR plausibility even when a confounder is present. Prescription changes may be cited ONLY as supporting context after a symptom/AE is explicitly identified; do NOT create ADR candidates from prescription changes alone, and do NOT infer unstated reasons behind those changes."
        ]
            }}
        ]
        }}
        ```
        """

        return self._call_and_parse_json(
            instruction, task,
            temperature=0.5, json_mode=False,
            fallback={"checked_results": []},
        )

    def _narrative_prompt(self, note, confounders):
        instruction = """
        You are a Validation Agent tasked with verifying whether the Confounder Agent's interpretation of an ADR (Adverse Drug Reaction) candidate is accurate, reasonable, and faithful to the clinical records (Original Note).
        """

        task = f"""
        ### Data Principles (Ground Truth)
        1. **Original Note:** The raw clinical record written in narrative form. This is the primary evidence and the **Ground Truth** for all judgments.

        ### Core Objective
        Determine whether the Confounder Agent's results accurately reflect the note context, or whether they exhibit hallucination, under-interpretation, or over-interpretation.

        ### Symptom / AE Keyword Definition (Scope)
        For this task, "Symptom" includes:
        1) **Clinical symptoms** documented in the note (objective findings or subjective patient-reported symptoms), and
        2) **AE keywords** that explicitly indicate an adverse reaction even without naming a specific symptom,
        such as "side effect", "S/E", "부작용", "allergy", "allergies", "rash", "intolerance", "ADR", "AE".

        ### Key Clinical Reasoning & Special Considerations (Required)
        1. **Narrative context recognition:** Unless clearly contradicted, the following patterns are valid linkage candidates for an ADR:
        - The drug and the symptom/AE keyword are mentioned within the same sentence.
        - A symptom/AE keyword appears in the same narrative context following a drug mention.
        - The drug–symptom relation is indicated by compressed expressions similar to "Drug Adverse Symptom" or by explicit connectors (e.g., due to, secondary to, attributed to, induced).

        2. **Evidence of causality:** Medication changes (discontinuation/switch/dose reduction) do NOT constitute an ADR candidate by themselves. They may be used as ADR evidence ONLY IF an explicit symptom/AE keyword is documented as the reason in the text.

        3. **Decision thresholds:**
            - Likely ADR: An ADR is supported when an objective symptom/AE keyword (including "side effect", "S/E", "부작용", "Allergies", etc.) or a symptom phrase is explicitly documented and linked to a drug by proximity or explicit connectors. Subjective intolerance phrases such as "uncomfortable," "hard to tolerate," and "not tolerating" qualify as symptoms when explicitly written in the note.
            - Strengthening evidence: A prescription change (discontinuation/switch/dose reduction) strengthens the ADR hypothesis only when the symptom/AE is explicitly documented as the reason in the text. A prescription change alone is not an ADR.
            - Handling intolerance-only mentions: If only an intolerance phrase is documented (with no objective symptom/AE keyword) and there is no clinician action (discontinuation/switch/dose reduction), do not "confirm" an ADR; label it as low-likelihood/insufficient evidence.
            - Allergy / historical ADR rule: ** Any mention of "Allergy" or "known adverse events/side effects" MUST be validated as a **valid ADR**. If the Confounder Agent dismisses these as "not a new event," it is an **Under-interpretation**. These are patient-level ADRs and must be included.

        ### Evidence Span Extraction Rules
        - `original_note_evidences` MUST be a single continuous verbatim span from the Original Note.
        - The span MUST contain BOTH the drug mention and the AE/symptom (objective clinical symptom) OR an AE keyword such as "S/E" or "부작용" OR an allergy/ADR-history cue (e.g., allergy, hypersensitivity, intolerance, 알레르기, 과민, 부작용) within:
        (a) the same sentence, OR
        (b) the same narrative context window following the drug mention.
        - Do NOT stitch together distant parts of the note. Do NOT skip intervening unrelated sentences.
        - When multiple drugs are mentioned consecutively, attribute the symptom to the closest preceding drug unless an explicit connector assigns it differently.

        ### Input Data
        1. **Original Note (RAW, Narrative):** {note}
        2. **Confounder Agent Output:** {confounders}

        ### Output Format (JSON Only)

        ```json
        {{
        "validation_questions": [
            {{
            "adr_candidate": "[Drug – Symptom]",
            "confounder_judgment_summary": "[One-line summary of the confounder's reasoning]",
            "validation_verdict": "Correct / Incorrect",
            "original_note_evidences": "[Direct verbatim quote or continuous span from the Original Note]",
            "reasoning": [
            "Compare the evidence in the Original Note and the Context Note, and determine whether the confounder judgment is supported by the documented facts.",
            "Explain specifically how the confounder agent’s reasoning aligns or conflicts with the note evidence. If the confounder reasoning is valid but does not fully negate the ADR candidate, explicitly state that both a valid confounder and an explicit drug–event cue coexist.",
            "Identify any missed explicit cues in the note, including explicit symptoms/AE keywords and explicit drug–symptom connectors. Make clear whether these cues preserve ADR plausibility even when a confounder is present. Prescription changes may be cited ONLY as supporting context after a symptom/AE is explicitly identified; do NOT create ADR candidates from prescription changes alone, and do NOT infer unstated reasons behind those changes."
        ]
            }}
        ]
        }}
        ```
        """

        return self._call_and_parse_json(
            instruction, task,
            temperature=0.5, json_mode=True,
            fallback={"checked_results": []},
        )


# ============================================================================
# ADR Agent (Final Integration)
# ============================================================================
class ADR_Agent(BaseAgent):
    """Final agent that synthesizes all previous outputs to determine ADRs.

    Args:
        note_style: "shorthand" or "narrative" — controls prompt and temperature.
    """

    def __init__(self, note_style="shorthand", **kwargs):
        super().__init__(**kwargs)
        self.note_style = note_style

    def prompt(self, adr_candidates, confounders, note, context=None):
        instruction = """
        You are the Final Clinical Data Integration Agent.
        Your mission is to synthesize the analysis results (Reasoning) from multiple validation stages along with the raw clinical note to extract the most credible and definitive list of Adverse Drug Reactions (ADRs).
        """

        # Mode Configuration
        if not confounders or str(confounders).strip() == "":
            mode_instruction = """
            **[MODE: Independent Extraction]** No previous validation data is available. Analyze the raw clinical note and clinical context independently to extract ADRs.
            """
            validation_data = "N/A"
        else:
            mode_instruction = """
            **[MODE: Validation-Based Reconstruction]**
            1) If `original_note_evidences` are provided (non-empty): treat them as the PRIMARY evidence (Ground Truth).
            - Use `Reasoning` to map/attribute the symptom to the correct drug; correct drug-symptom ADR relation (e.g., switch-reason vs post-drug adverse effect).

            2) If `original_note_evidences` are missing or empty:
            - Use `Reasoning` as guidance (a hint).
            """
            validation_data = confounders

        # Input Data section differs by style
        if self.note_style == "shorthand":
            input_section = f"""
        **[1] ADR Candidates (For Reference):** {adr_candidates}
        **[2] Validation Summary (Primary Evidence):** {validation_data}
        **[3] Clinical Context:** {context}
        **[4] Raw Clinical Note:** {note}"""
        else:
            input_section = f"""
        **[1] ADR Candidates (For Reference):** {adr_candidates}
        **[2] Validation Summary (Primary Evidence):** {validation_data}
        **[3] Raw Clinical Note:** {note}"""

        task = f"""
        ## Objective
        Extract **valid ADR data** confirmed by synthesizing the raw clinical note and the provided validation reasoning.

        ## Core Instructions
        {mode_instruction}

        1. **Analyze & Extract:** Examine the sentence structure, prescription intent, and medication changes in the raw note. Prioritize the causal logic mentioned in the 'Validation Summary' to extract actual ADRs.
        2. **Refine Candidates:** If an entry in the `ADR Candidates` was judged as inappropriate (Disagree) in the Reasoning, exclude it. Conversely, if the Reasoning supports a valid drug–symptom relationship NOT present in the initial candidates, include it as a new entry.
        3. **Provide Explanation:** In the `explanation` field, concisely describe why this specific ADR was selected (e.g., "Validated by excluding alternative drugs in reasoning," "Consistent with discontinuation evidence in the note," etc.).
        4. **Output Format:**
            - If one or more valid ADRs exist: Return them in the `"results"` array within a JSON object.
            - If no valid ADRs exist: Return the `{{"result": "No Side Effect"}}` object.

        ---

        ## Input Data
        {input_section}

        ---

        ### Output Format (JSON Only)
        You MUST wrap the response in a JSON object. Ensure the output is valid JSON.

        **[Scenario A] If one or more valid ADRs exist**
        ```json
        {{
            "results": [
                {{
                    "text": "<Specific evidence span from the raw note>",
                    "drug": "<Confirmed Drug Name>",
                    "symptom": ["<Confirmed Symptom Name>"],
                    "explanation": "<Reason for final selection/reconstruction based on reasoning>"
                }},
                ...
            ]
        }}
        ```

        **[Scenario B] If NO valid ADRs exist**
        ```json
        {{
            "result": "No Side Effect"
        }}
        ```
        """

        return self._call_and_parse_json(
            instruction, task,
            temperature=0.5, json_mode=True,
            fallback={"adr_candidates": []},
        )
