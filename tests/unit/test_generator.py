"""Grounding, citations and prompt assembly. No network, no model."""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.rag.generator import NO_ANSWER, Citation, GeneratedAnswer, Generator


class FakeLLM:
    """Records the messages it was sent and replies with a fixed string."""

    def __init__(self, reply="La aprobación la firma el propietario del servicio [1]."):
        self.reply = reply
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return AIMessage(content=self.reply)

    @property
    def last_prompt(self) -> str:
        """The human message of the most recent call."""
        return self.calls[-1][1].content

    @property
    def last_system(self) -> str:
        return self.calls[-1][0].content


@pytest.fixture
def documents():
    return [
        Document(
            page_content="Toda release requiere la firma del propietario del servicio.",
            metadata={
                "source": "data/raw/handbook.pdf",
                "page": 3,
                "headings": ["4 Despliegue", "4.2 Aprobación"],
            },
        ),
        Document(
            page_content="El gestor de release obtiene autorización escrita.",
            metadata={"source": "data/raw/process.pdf", "page": 7, "headings": []},
        ),
    ]


def test_no_documents_refuses_without_calling_the_model(documents):
    """Calling the model with no context invites it to answer from training data."""
    llm = FakeLLM()
    generator = Generator(llm=llm)

    answer = generator.generate("¿Quién aprueba?", [])

    assert answer.text == NO_ANSWER
    assert answer.is_refusal
    assert llm.calls == [], "the LLM must not be called without context"
    assert answer.citations == []


def test_citations_are_numbered_in_retrieval_order(documents):
    generator = Generator(llm=FakeLLM())

    answer = generator.generate("¿Quién aprueba?", documents)

    assert [c.marker for c in answer.citations] == [1, 2]
    assert answer.citations[0].source == "data/raw/handbook.pdf"
    assert answer.citations[0].page == 3
    assert answer.citations[1].source == "data/raw/process.pdf"


def test_context_block_carries_provenance_per_fragment(documents):
    """Provenance inside the block is what lets the model attribute claims."""
    llm = FakeLLM()
    Generator(llm=llm).generate("¿Quién aprueba?", documents)

    prompt = llm.last_prompt
    assert "[1] Fuente: data/raw/handbook.pdf" in prompt
    assert "Página: 3" in prompt
    assert "Sección: 4 Despliegue > 4.2 Aprobación" in prompt
    assert "[2] Fuente: data/raw/process.pdf" in prompt


def test_document_text_reaches_the_prompt(documents):
    llm = FakeLLM()
    Generator(llm=llm).generate("¿Quién aprueba?", documents)

    for doc in documents:
        assert doc.page_content in llm.last_prompt


def test_question_is_passed_verbatim(documents):
    llm = FakeLLM()
    Generator(llm=llm).generate("¿Quién firma la aprobación?", documents)

    assert "PREGUNTA: ¿Quién firma la aprobación?" in llm.last_prompt


def test_system_prompt_carries_the_grounding_contract(documents):
    """Weakening these rules is what turns retrieval into fabrication."""
    llm = FakeLLM()
    Generator(llm=llm).generate("¿Quién aprueba?", documents)

    system = llm.last_system
    assert "exclusivamente" in system
    assert NO_ANSWER in system, "the model must be told the exact refusal string"


def test_fragment_order_is_preserved(documents):
    """Reranked order must survive into the prompt: strongest evidence first."""
    llm = FakeLLM()
    Generator(llm=llm).generate("q", documents)

    prompt = llm.last_prompt
    assert prompt.index("handbook.pdf") < prompt.index("process.pdf")


def test_refusal_from_the_model_is_detected(documents):
    """A refusal must be machine-detectable, not parsed out of prose."""
    generator = Generator(llm=FakeLLM(reply=NO_ANSWER))

    answer = generator.generate("¿Cuál es la capital de Francia?", documents)

    assert answer.is_refusal


def test_a_normal_answer_is_not_flagged_as_refusal(documents):
    answer = Generator(llm=FakeLLM()).generate("¿Quién aprueba?", documents)

    assert not answer.is_refusal
    assert "propietario del servicio" in answer.text


def test_context_documents_are_returned_for_auditing(documents):
    answer = Generator(llm=FakeLLM()).generate("q", documents)

    assert answer.context_documents == documents


def test_custom_system_prompt_is_used(documents):
    llm = FakeLLM()
    Generator(llm=llm, system_prompt="REGLA PROPIA").generate("q", documents)

    assert llm.last_system == "REGLA PROPIA"


def test_whitespace_is_stripped_from_the_answer(documents):
    answer = Generator(llm=FakeLLM(reply="  respuesta  \n")).generate("q", documents)

    assert answer.text == "respuesta"


# ── Citation formatting ───────────────────────────────────────


def test_citation_label_includes_page_and_section():
    citation = Citation(
        marker=1,
        source="handbook.pdf",
        page=3,
        headings=["4 Despliegue", "4.2 Aprobación"],
    )

    assert citation.label() == "[1] handbook.pdf, p. 3, 4.2 Aprobación"


def test_citation_label_degrades_when_metadata_is_missing():
    assert Citation(marker=2, source="scan.pdf").label() == "[2] scan.pdf"


def test_missing_page_metadata_does_not_break_generation():
    """PyMuPDF fallback documents carry less metadata than Docling ones."""
    docs = [Document(page_content="texto", metadata={"source": "scan.pdf"})]
    llm = FakeLLM()

    answer = Generator(llm=llm).generate("q", docs)

    assert answer.citations[0].page is None
    assert "Página" not in llm.last_prompt


def test_formatted_sources_lists_every_citation():
    answer = GeneratedAnswer(
        text="respuesta",
        citations=[
            Citation(marker=1, source="a.pdf", page=1),
            Citation(marker=2, source="b.pdf"),
        ],
    )

    assert answer.formatted_sources() == "[1] a.pdf, p. 1\n[2] b.pdf"
