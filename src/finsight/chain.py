"""LCEL RAG chain: (condense) -> retriever -> prompt -> LLM -> PydanticOutputParser -> source resolution.

Source references are attached deterministically from retrieved-chunk metadata;
the LLM only states WHICH numbered excerpts it used, so it cannot invent a page number.

Phase 3 addition - history-aware retrieval: with memory, a follow-up such as
"How much did that grow on a currency-neutral basis?" was sent to the retriever as-is, and the
word "that" carries no meaning for a vector search. When there is chat history, the LLM first
rewrites the follow-up into a standalone question (CONDENSE_PROMPT) and that rewritten question
is used for retrieval. The first turn of a conversation skips this step (no extra LLM call).
"""
from operator import itemgetter
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, RunnableBranch
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


class FinancialAnswer(BaseModel):
    answer: str = Field(description="Answer using ONLY the excerpts; quote figures exactly as written")
    answerable: bool = Field(description="False if the excerpts do not contain the answer")
    used_excerpts: list[int] = Field(description="Numbers of the excerpts [n] that support the answer")


parser = PydanticOutputParser(pydantic_object=FinancialAnswer)

SYSTEM = (
    "You are a financial report analyst. Answer the question using ONLY the numbered "
    "excerpts from the company filing below. Do not use outside knowledge. If the "
    "excerpts do not contain the answer, set answerable=false and say so.\n\n"
    "Excerpts:\n{context}\n\n{format_instructions}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    MessagesPlaceholder("history", optional=True),
    ("human", "{question}"),
]).partial(format_instructions=parser.get_format_instructions())

CONDENSE_SYSTEM = (
    "Rewrite the user's latest question as a single standalone question about the company "
    "filing, using the conversation so far to resolve references such as 'it', 'that' or "
    "'the year before'. Do not answer it. Return only the rewritten question."
)

condense_prompt = ChatPromptTemplate.from_messages([
    ("system", CONDENSE_SYSTEM),
    MessagesPlaceholder("history"),
    ("human", "{question}"),
])


def format_docs(docs) -> str:
    return "\n\n".join(
        f"[{i}] ({d.metadata['source']}, p.{d.metadata['page']})\n{d.page_content}"
        for i, d in enumerate(docs, 1))


def resolve_sources(x) -> dict:
    ans: FinancialAnswer = x["parsed"]
    docs = x["docs"]
    sources = [{"excerpt": n, "source": docs[n - 1].metadata["source"],
                "page": docs[n - 1].metadata["page"],
                "start_index": docs[n - 1].metadata.get("start_index"),
                "passage": docs[n - 1].page_content[:300]}
               for n in ans.used_excerpts if 1 <= n <= len(docs)]
    return {"answer": ans.answer, "answerable": ans.answerable, "sources": sources,
            "retrieval_query": x.get("retrieval_query", x.get("question")),
            "retrieved_pages": [d.metadata["page"] for d in docs]}


def build_chain(retriever, llm, condense: bool = True):
    """condense=False reproduces the Phase 2 behaviour (raw follow-up sent to the retriever)."""
    if condense:
        retrieval_query = RunnableBranch(
            (lambda x: bool(x.get("history")), condense_prompt | llm | StrOutputParser()),
            itemgetter("question"))
    else:
        retrieval_query = itemgetter("question")
    return (
        RunnablePassthrough.assign(retrieval_query=retrieval_query)
        | RunnablePassthrough.assign(docs=itemgetter("retrieval_query") | retriever)
        | RunnablePassthrough.assign(context=lambda x: format_docs(x["docs"]))
        | RunnablePassthrough.assign(parsed=prompt | llm | parser)
        | RunnableLambda(resolve_sources)
    )


_store: dict[str, InMemoryChatMessageHistory] = {}


def with_memory(chain):
    def get_history(session_id):
        return _store.setdefault(session_id, InMemoryChatMessageHistory())
    return RunnableWithMessageHistory(
        chain, get_history, input_messages_key="question",
        history_messages_key="history", output_messages_key="answer")
