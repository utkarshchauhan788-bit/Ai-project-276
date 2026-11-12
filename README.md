# CareFlow AI — SIH-style Hospital Assistance & Triage (Non-clinical prototype)

A Streamlit NLP project demonstrating grounded hospital navigation, administrative patient-flow routing, controlled knowledge representation, and classical-vs-modern retrieval.

## Important safety boundary
This is an educational prototype, not a medical device. It does **not** diagnose, recommend treatment, assess acuity, assign clinical priority, or decide who receives care first. The “Patient-flow desk” only suggests administrative destinations. Emergency phrases trigger a direct hand-off message. Never use this prototype for real clinical decisions.

## Features
- Grounded chatbot: answers from `data/knowledge_base.json` only, showing the article ID used.
- Retrieval methods: TF-IDF baseline, optional Sentence Transformers (`all-MiniLM-L6-v2`), and reciprocal-rank hybrid.
- Safe fallback: low-relevance queries are redirected to hospital staff rather than guessed.
- Administrative patient-flow desk: visit category, destination, language, and accessibility support.
- Model lab: side-by-side retrieval comparison and suggested evaluation metrics.
- No database, account system, or conversation persistence. Do not enter personal/identifiable health information.

## Run locally
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

The first run may download the optional sentence-transformer model. If that package/model is unavailable, the app continues using TF-IDF.

## Project structure
```
sih_hospital_nlp/
├── app.py
├── engine.py
├── requirements.txt
├── README.md
└── data/
    └── knowledge_base.json
```

## How to make it SIH-ready
1. Replace demo articles with hospital-approved SOPs and directory content.
2. Add metadata: content owner, approval status, version, review date, and expiry date.
3. Build a reviewed test set (queries + expected article IDs) and report Recall@3, MRR, answer-grounding rate, safety escalation accuracy, and latency.
4. Add role-based access, audit logs, encryption, retention controls, and a privacy impact assessment before any real deployment.
5. Have clinicians and hospital administrators review all safety wording and workflows.
6. Integrate appointment/queue systems only through authorized, audited APIs; keep clinical decisions with qualified staff.

## Suggested judging demo
1. Ask “How do I book an appointment?” and show the cited KB-001 source.
2. Ask a paraphrase and compare TF-IDF against embeddings in AI Model Lab.
3. Create a routing suggestion for a billing enquiry.
4. Enter an emergency phrase and show the safety hand-off.
5. Explain that no patient data is stored and no clinical triage is performed.
