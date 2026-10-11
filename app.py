"""Streamlit front end for the PDF RAG Lab."""

import streamlit as st

from rag import (
    build_index,
    extract_pdf_pages,
    generate_answer,
    load_generator,
    retrieve,
    split_into_chunks,
)

st.set_page_config(page_title="PDF RAG Lab", page_icon="📚", layout="wide")
st.title("📚 PDF RAG Lab — Ask Questions from Your PDF")
st.write(
    "A beginner-friendly Retrieval-Augmented Generation (RAG) lab. Upload a text-based PDF, "
    "build its searchable index, and ask questions grounded in the document."
)
st.info(
    "Free model: google/flan-t5-small runs inside this app and does not need a paid API key. "
    "There is no guaranteed unlimited hosted compute or unlimited token service."
)

with st.expander("What happens in a RAG system?"):
    st.markdown(
        "1. **Extract:** PyMuPDF reads text from each PDF page.\n"
        "2. **Chunk:** The text is split into overlapping passages.\n"
        "3. **Index:** TF-IDF creates a searchable representation of the passages.\n"
        "4. **Retrieve:** Cosine similarity finds passages relevant to the question.\n"
        "5. **Generate:** FLAN-T5-small uses those passages to draft an answer.\n\n"
        "**Important:** This does not fine-tune/train the model's weights. The PDF is indexed "
        "for retrieval. Scanned image-only PDFs need OCR before this app can read them."
    )

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_pdf is not None:
    if st.button("Read PDF and build RAG index", type="primary"):
        try:
            with st.spinner("Extracting text and building the searchable index..."):
                pages = extract_pdf_pages(uploaded_pdf.getvalue())
                if not pages:
                    st.error(
                        "No selectable text was found. This is probably a scanned PDF. "
                        "Use a text-based PDF or run OCR on it first."
                    )
                    st.stop()
                chunks = split_into_chunks(pages)
                vectorizer, matrix = build_index(chunks)
                st.session_state["rag_data"] = {
                    "filename": uploaded_pdf.name,
                    "pages": pages,
                    "chunks": chunks,
                    "vectorizer": vectorizer,
                    "matrix": matrix,
                }
            st.success(
                f"Index ready for **{uploaded_pdf.name}** — "
                f"{len(pages)} text pages and {len(chunks)} chunks."
            )
        except Exception as error:
            st.error(f"Could not process this PDF: {error}")

rag_data = st.session_state.get("rag_data")
if rag_data:
    st.divider()
    st.subheader(f"Ask a question about: {rag_data['filename']}")
    question = st.text_input(
        "Your question", placeholder="Example: What are the main findings of this paper?"
    )
    top_k = st.slider("Number of passages to retrieve", min_value=1, max_value=5, value=3)

    if st.button("Get answer", type="primary", disabled=not question.strip()):
        passages = retrieve(
            question.strip(),
            rag_data["chunks"],
            rag_data["vectorizer"],
            rag_data["matrix"],
            top_k=top_k,
        )
        if not passages:
            st.warning(
                "No matching passage was found. Try using keywords that appear in the PDF."
            )
        else:
            try:
                with st.spinner(
                    "Loading the free model on first use and generating the answer..."
                ):
                    generator = load_generator()
                    answer = generate_answer(question.strip(), passages, generator)
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Retrieved source passages")
                for number, passage in enumerate(passages, start=1):
                    with st.expander(
                        f"Passage {number} — page {passage['page']} "
                        f"(similarity {passage['score']:.3f})"
                    ):
                        st.write(passage["text"])
            except Exception as error:
                st.error(
                    "The language model could not start or generate an answer. Free hosting may "
                    "have insufficient RAM or the model download may have failed. "
                    f"Details: {error}"
                )

st.caption(
    "Educational prototype • Python + Streamlit + PyMuPDF + scikit-learn + FLAN-T5-small. "
    "First model load requires an internet connection. Do not upload confidential documents "
    "to a public hosted app."
)
