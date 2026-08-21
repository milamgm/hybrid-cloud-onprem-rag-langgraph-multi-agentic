"""Streamlit UI and readiness entrypoint for Onyx."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv


def _load_environment(path: str | None = None) -> None:
    configured_file = path or os.getenv("ONYX_ENV_FILE")
    load_dotenv()
    if configured_file:
        env_path = Path(configured_file)
        if env_path.exists():
            load_dotenv(env_path, override=True)


def readiness_command(env_file: str | None = None) -> int:
    _load_environment(env_file)
    from src.app.readiness import check_readiness

    report = check_readiness()
    print(f"Onyx readiness ({report.mode})")
    for check in report.checks:
        marker = "OK" if check.ok else "FAIL"
        print(f"[{marker}] {check.name}: {check.detail}")
    return 0 if report.ready else 1


def render_streamlit() -> None:
    _load_environment()
    import streamlit as st

    st.set_page_config(page_title="Onyx RAG", page_icon="◆", layout="wide")
    st.title("Onyx · Governed RAG")

    @st.cache_resource
    def application():
        from src.app.composition import build_rag_application

        return build_rag_application()

    try:
        app = application()
    except Exception as error:
        st.error(f"Onyx no puede iniciar: {error}")
        st.code("uv run python main.py check", language="bash")
        st.stop()

    with st.sidebar:
        st.subheader("Deployment")
        st.write(f"Mode: `{app.mode}`")
        st.write(f"Security: `{app.security_profile}`")
        if app.security_profile == "development":
            st.warning("Controles locales de desarrollo; no usar con datos reales.")
        try:
            st.metric("Indexed chunks", app.indexed_chunks)
        except Exception as error:
            st.caption(f"Index status unavailable: {error}")
        classification = st.selectbox("Classification", ["public", "internal"])

    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if question := st.chat_input("Pregunta sobre los documentos indexados"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"), st.spinner("Consultando el corpus…"):
            try:
                result = app.agent.invoke(
                    question,
                    request_id=str(uuid4()),
                    tenant_id="local-development",
                    subject_id="local-user",
                    thread_id=st.session_state.thread_id,
                    roles=("rag.user",),
                    data_classification=classification,
                )
                answer = result["response_text"]
                st.markdown(answer)
                if result["citations"]:
                    with st.expander("Fuentes"):
                        for citation in result["citations"]:
                            page = (
                                f", página {citation['page']}"
                                if citation.get("page") is not None
                                else ""
                            )
                            st.write(
                                f"[{citation['marker']}] {citation['source']}{page}"
                            )
            except Exception as error:
                answer = f"Error de ejecución: {error}"
                st.error(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


def _running_under_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx(suppress_warning=True) is not None
    except ImportError:
        return False


def main() -> int:
    if _running_under_streamlit():
        render_streamlit()
        return 0
    parser = argparse.ArgumentParser(description="Onyx local application")
    parser.add_argument("command", choices=["check"], help="Operation to run")
    parser.add_argument("--env-file", help="Environment profile to load")
    args = parser.parse_args()
    return readiness_command(args.env_file)


if __name__ == "__main__":
    sys.exit(main())
