import json
import re
from pathlib import Path
from datetime import datetime
import streamlit as st
import pandas as pd

from engine import HospitalAssistant, load_knowledge

st.set_page_config(page_title="CareFlow AI | SIH NLP Prototype", page_icon="🏥", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1250px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(115deg,#0b3558,#167c80); color: white;}
.hero h1 {font-size: 2.1rem; margin: 0 0 .35rem 0;}
.hero p {margin: 0; opacity: .92;}
.small-note {font-size: .86rem; color: #64748b;}
[data-testid="stMetric"] {background: rgba(120,140,160,.08); padding: 12px; border-radius: 12px;}
</style>
""", unsafe_allow_html=True)

kb = load_knowledge()
assistant = HospitalAssistant(kb)

with st.sidebar:
    st.title("🏥 CareFlow AI")
    st.caption("SIH-style NLP innovation prototype")
    page = st.radio("Navigate", ["Patient assistant", "Patient-flow desk", "AI model lab", "Knowledge base", "About & safety"])
    st.divider()
    st.markdown("**Prototype mode**")
    st.info("Educational demo only. No diagnosis, prescriptions, or clinical triage.")
    st.caption("Do not enter names, phone numbers, IDs, reports, or other identifiable health information.")

if page == "Patient assistant":
    st.markdown('<div class="hero"><h1>AI-Powered Hospital Assistance</h1><p>Grounded answers • guided navigation • clear next steps • source-backed responses</p></div>', unsafe_allow_html=True)
    st.warning("This assistant provides general hospital-navigation and educational information. It cannot assess medical urgency or replace a qualified clinician.")
    col1, col2, col3 = st.columns(3)
    col1.metric("Knowledge articles", len(kb))
    col2.metric("Answer mode", "Grounded retrieval")
    col3.metric("Personal data", "Not stored")
    st.subheader("How can I help?")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    prompt = st.chat_input("Ask about hospital services, appointments, departments, or general care navigation…")
    if prompt:
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        answer = assistant.answer(prompt, mode=st.session_state.get("retrieval_mode","Hybrid"))
        with st.chat_message("assistant"):
            st.markdown(answer["text"])
            if answer["sources"]:
                st.caption("Knowledge-base sources: " + " · ".join(answer["sources"]))
            st.caption(f"Retrieval method: {answer['method']} | Match score: {answer['score']:.2f}")
        st.session_state.messages.append({"role":"assistant","content":answer["text"]})
    st.divider()
    st.subheader("Try a sample question")
    samples = ["How do I book an appointment?", "Which department handles skin concerns?", "What should I bring for an outpatient visit?", "How can I request an interpreter?"]
    cols = st.columns(2)
    for i, q in enumerate(samples):
        if cols[i % 2].button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({"role":"user","content":q})
            st.session_state.pending_sample = q
            st.rerun()
    if "pending_sample" in st.session_state:
        q = st.session_state.pop("pending_sample")
        result = assistant.answer(q, mode="Hybrid")
        st.session_state.messages.append({"role":"assistant","content":result["text"]})
        st.rerun()

elif page == "Patient-flow desk":
    st.markdown('<div class="hero"><h1>Patient-flow coordination desk</h1><p>Administrative routing and queue organization — not clinical triage.</p></div>', unsafe_allow_html=True)
    st.warning("This tool must never determine medical priority. If someone may be experiencing a medical emergency, alert on-site clinical staff immediately or contact local emergency services.")
    st.subheader("Create a non-clinical routing record")
    with st.form("flow_form"):
        a,b = st.columns(2)
        visit_type = a.selectbox("Visit type", ["Outpatient appointment", "Walk-in enquiry", "Diagnostic appointment", "Billing / insurance", "Discharge support", "Other"])
        destination = b.selectbox("Requested service", ["Registration", "Appointments", "General information", "Diagnostics reception", "Billing desk", "Discharge desk", "Interpreter / accessibility support", "Clinical staff (user requested)"])
        c,d = st.columns(2)
        language = c.selectbox("Preferred communication language", ["Not specified", "English", "Hindi", "Punjabi", "Other"])
        accessibility = d.selectbox("Accessibility support", ["None requested", "Wheelchair access", "Interpreter", "Hearing / visual support", "Other"])
        notes = st.text_area("Non-sensitive administrative note (optional)", placeholder="Example: needs help locating the radiology reception. Do not enter medical details or identifiers.")
        submitted = st.form_submit_button("Generate routing suggestion", type="primary")
    if submitted:
        st.success("Administrative routing suggestion generated")
        st.write(f"**Route to:** {destination}")
        st.write(f"**Visit category:** {visit_type}")
        st.write(f"**Communication:** {language}")
        st.write(f"**Accessibility:** {accessibility}")
        if notes.strip():
            st.write(f"**Note:** {notes.strip()}")
        st.info("Next action: staff member confirms the destination and handles any clinical questions. This suggestion does not assign medical urgency or queue priority.")

    st.subheader("Illustrative patient-flow stages")
    stages = pd.DataFrame([
        ["1", "Arrival / enquiry", "Welcome and identify requested administrative service"],
        ["2", "Registration", "Staff verifies required documents using hospital policy"],
        ["3", "Service routing", "Direct to appointment, diagnostics, billing, or information desk"],
        ["4", "Clinical hand-off", "Clinical staff independently assess clinical needs"],
        ["5", "Exit / follow-up", "Provide approved discharge or appointment instructions"],
    ], columns=["Stage", "Step", "Purpose"])
    st.dataframe(stages, use_container_width=True, hide_index=True)

elif page == "AI model lab":
    st.markdown('<div class="hero"><h1>Classical NLP vs modern semantic AI</h1><p>Compare retrieval approaches on the same controlled hospital knowledge base.</p></div>', unsafe_allow_html=True)
    st.write("Select a query and compare how different retrieval strategies rank the same knowledge articles.")
    q = st.text_input("Evaluation query", "I need help arranging a visit with a skin specialist")
    mode = st.radio("Primary assistant retrieval", ["Hybrid", "Classical TF-IDF", "Modern embeddings (optional)"], horizontal=True)
    st.session_state.retrieval_mode = mode
    results = assistant.compare(q)
    st.subheader("Top retrieved knowledge")
    for method, items in results.items():
        with st.expander(method, expanded=True):
            if not items:
                st.write("No result. Install the optional embedding package and model to enable this method.")
            for item in items[:3]:
                st.markdown(f"**{item['title']}**  \n{item['snippet']}  \n<small>Similarity: {item['score']:.3f} · ID: {item['id']}</small>", unsafe_allow_html=True)
    st.caption("Classical TF-IDF is interpretable and lightweight but relies heavily on word overlap. Embeddings can capture paraphrases, but must be evaluated for bias, coverage, latency, and domain safety. Retrieval scores are not confidence in medical correctness.")
    st.subheader("Evaluation plan for an SIH demonstration")
    st.dataframe(pd.DataFrame([
        ["Retrieval quality", "Recall@3, MRR, nDCG@3", "Does the right approved article appear near the top?"],
        ["Answer grounding", "Citation coverage / human audit", "Can every factual claim be traced to an approved article?"],
        ["Safety", "Red-flag escalation tests", "Does the system avoid diagnosis and direct emergencies to humans?"],
        ["Robustness", "Paraphrase and typo test set", "Does retrieval work across varied phrasing?"],
        ["Operations", "Latency and fallback rate", "Does the assistant fail safely when retrieval is weak?"],
    ], columns=["Dimension","Metric","What it measures"]), use_container_width=True, hide_index=True)

elif page == "Knowledge base":
    st.markdown('<div class="hero"><h1>Controlled knowledge base</h1><p>Responses are grounded only in reviewed articles included with this prototype.</p></div>', unsafe_allow_html=True)
    for article in kb:
        with st.expander(f"{article['title']} · {article['id']}"):
            st.write(article["content"])
            st.caption(f"Category: {article['category']} | Review status: {article.get('status','Demo content')}")
    st.info("Before deployment, replace demo articles with hospital-approved SOPs, department directories, accessibility information, and patient handouts. Add owner, version, review date, and approval workflow.")

else:
    st.markdown('<div class="hero"><h1>About CareFlow AI</h1><p>A non-clinical educational prototype for a Smart India Hackathon-style NLP project.</p></div>', unsafe_allow_html=True)
    st.subheader("Problem statement")
    st.write("Hospital visitors often struggle to find the right service, understand administrative processes, and navigate fragmented information. CareFlow AI demonstrates a controlled, explainable NLP assistant for hospital navigation and administrative patient-flow coordination.")
    st.subheader("Core modules")
    st.markdown("""
    - **Grounded chatbot:** retrieves approved knowledge articles and cites their IDs.
    - **Patient-flow desk:** organizes administrative destinations without clinical prioritization.
    - **Knowledge representation:** structured articles with IDs, categories, and review status.
    - **Model comparison:** TF-IDF baseline, optional sentence-transformer embeddings, and hybrid retrieval.
    - **Safety layer:** emergency language hand-off, uncertainty fallback, and no diagnosis/prescription.
    """)
    st.subheader("Safety and deployment boundaries")
    st.markdown("""
    1. This is not a medical device and is not validated for clinical use.
    2. Never use it to diagnose, recommend treatment, calculate acuity, or decide who receives care first.
    3. Emergency warning signs require immediate human/ emergency-service escalation, not chatbot questioning.
    4. Do not collect or store personal health information in this demo.
    5. A real deployment requires clinical, legal, privacy, cybersecurity, accessibility, and hospital IT review.
    6. The knowledge base must be maintained by authorized hospital owners with version control and scheduled review.
    """)
    st.subheader("Suggested SIH judging demo")
    st.markdown("Show a visitor asking where to go, retrieve a cited hospital-navigation article, demonstrate a paraphrased query in the AI model lab, create an administrative routing suggestion, then test an emergency phrase to demonstrate safe hand-off.")
    st.caption("Built with Streamlit and Python. Demo knowledge is illustrative, not hospital policy.")
