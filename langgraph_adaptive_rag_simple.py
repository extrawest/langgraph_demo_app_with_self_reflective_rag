"""
Adaptive RAG Implementation

This module implements an Adaptive RAG (Retrieval Augmented Generation) system
that routes queries between web search and vector store retrieval based on content analysis.
"""

import os

os.environ["USER_AGENT"] = "ActiveRAG/1.0"

from typing import List, Dict, Any, Literal, Optional
from typing_extensions import TypedDict

from langchain import hub
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic.v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import END, StateGraph, START


class RouteQuery(BaseModel):
    """Routes a user query to the most relevant datasource."""
    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Data source to route the query to (vectorstore or web_search)"
    )


class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses question."""
    binary_score: Literal["yes", "no"] = Field(
        ...,
        description="Answer addresses the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """Represents the state of the workflow graph."""
    question: str
    generation: Optional[str]
    documents: Optional[List[Document]]


class AdaptiveRAG:
    """Main class for the Adaptive RAG implementation."""

    def __init__(self):
        """Initialize components for the Adaptive RAG system."""
        self.retriever = None
        self.vectorstore = None

        self.query_router = None
        self.doc_grader = None
        self.hallucination_grader = None
        self.answer_grader = None
        self.query_rewriter = None
        self.web_search_tool = None

        self.rag_chain = None
        self.rag_llm = None
        self.rag_prompt = None

        self.route_prompt = None
        self.grade_prompt = None
        self.hallucination_prompt = None
        self.answer_prompt = None
        self.rewrite_prompt = None
        
        # Workflow
        self.app = None

    async def initialize(self) -> None:
        """Initialize all components and build the graph."""
        self.vectorstore = await self._initialize_vector_store()
        self.retriever = self.vectorstore.as_retriever()

        llm_router = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        self.query_router = llm_router.with_structured_output(RouteQuery, method="function_calling")
        system_router = """You are an expert at routing a user question to a vectorstore or web search.
        The vectorstore contains documents related to agents, prompt engineering, and adversarial attacks.
        Use the vectorstore for questions on these topics. Otherwise, use web-search."""
        self.route_prompt = ChatPromptTemplate.from_messages([
            ("system", system_router),
            ("human", "{question}"),
        ])

        llm_doc_grader = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        self.doc_grader = llm_doc_grader.with_structured_output(GradeDocuments, method="function_calling")
        system_doc_grader = """You are a grader assessing relevance of a retrieved document to a user question.
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant.
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals.
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        self.grade_prompt = ChatPromptTemplate.from_messages([
            ("system", system_doc_grader),
            ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
        ])

        llm_hallucination = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

        self.hallucination_grader = llm_hallucination.with_structured_output(GradeHallucinations, method="function_calling")
        system_hallucination = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
        self.hallucination_prompt = ChatPromptTemplate.from_messages([
            ("system", system_hallucination),
            ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
        ])

        llm_answer = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)

        self.answer_grader = llm_answer.with_structured_output(GradeAnswer, method="function_calling")
        system_answer = """You are a grader assessing whether an answer addresses / resolves a question.
        Give a binary score 'yes' or 'no'. Yes' means that the answer resolves the question."""
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", system_answer),
            ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
        ])

        llm_rewriter = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
        system_rewriter = """You are a question re-writer that converts an input question to a better version that is optimized
        for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", system_rewriter),
            ("human", "Here is the initial question: \n\n {question} \n Formulate an improved question."),
        ])
        self.query_rewriter = self.rewrite_prompt | llm_rewriter | StrOutputParser()

        self.web_search_tool = TavilySearchResults(k=3)

        self.rag_prompt = hub.pull("rlm/rag-prompt")
        self.rag_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.rag_chain = self.rag_prompt | self.rag_llm | StrOutputParser()

        self.app = self._build_graph()

    async def _initialize_vector_store(self) -> Chroma:
        """Initialize and populate the vector store with documents."""
        embeddings = OpenAIEmbeddings()

        urls = [
            "https://lilianweng.github.io/posts/2023-06-23-agent/",
            "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
            "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
        ]

        docs = []
        for url in urls:
            loader = WebBaseLoader(url)
            docs.extend(loader.load())

        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=500, chunk_overlap=0
        )
        doc_splits = text_splitter.split_documents(docs)

        vectorstore = await Chroma.afrom_documents(
            documents=doc_splits,
            collection_name="rag-chroma",
            embedding=embeddings,
        )

        return vectorstore

    async def retrieve(self, state: GraphState) -> GraphState:
        """Retrieve documents from the vector store."""
        print("---RETRIEVE---")
        question = state["question"]

        documents = await self.retriever.ainvoke(question)
        return {"documents": documents, "question": question}

    async def generate(self, state: GraphState) -> GraphState:
        """Generate answer based on documents and question."""
        print("---GENERATE---")
        question = state["question"]
        documents = state["documents"]

        generation = await self.rag_chain.ainvoke({"context": documents, "question": question})
        return {"documents": documents, "question": question, "generation": generation}

    async def grade_documents(self, state: GraphState) -> GraphState:
        """Grade documents for relevance to the question."""
        print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
        question = state["question"]
        documents = state["documents"]

        filtered_docs = []
        for doc in documents:
            score = await (self.grade_prompt | self.doc_grader).ainvoke(
                {"question": question, "document": doc.page_content}
            )
            if score.binary_score == "yes":
                print("---GRADE: DOCUMENT RELEVANT---")
                filtered_docs.append(doc)
            else:
                print("---GRADE: DOCUMENT NOT RELEVANT---")

        return {"documents": filtered_docs, "question": question}

    async def transform_query(self, state: GraphState) -> GraphState:
        """Transform the query to a better question for retrieval."""
        print("---TRANSFORM QUERY---")
        question = state["question"]
        documents = state["documents"]

        better_question = await self.query_rewriter.ainvoke({"question": question})
        return {"documents": documents, "question": better_question}

    async def web_search(self, state: GraphState) -> GraphState:
        """Perform web search for the question."""
        print("---WEB SEARCH---")
        question = state["question"]

        docs = await self.web_search_tool.ainvoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)

        return {"documents": [web_results], "question": question}

    async def route_question(self, state: GraphState) -> str:
        """Route question to web search or RAG."""
        print("---ROUTE QUESTION---")
        question = state["question"]
        source = await (self.route_prompt | self.query_router).ainvoke({"question": question})

        if source.datasource == "web_search":
            print("---ROUTE QUESTION TO WEB SEARCH---")
            return "web_search"
        else:
            print("---ROUTE QUESTION TO RAG---")
            return "vectorstore"

    async def decide_to_generate(self, state: GraphState) -> str:
        """Decide whether to generate an answer or transform query."""
        print("---ASSESS GRADED DOCUMENTS---")
        filtered_documents = state["documents"]

        if not filtered_documents:
            print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---")
            return "transform_query"
        else:
            print("---DECISION: GENERATE---")
            return "generate"

    async def grade_generation(self, state: GraphState) -> str:
        """Grade the generation against documents and question."""
        print("---CHECK HALLUCINATIONS---")
        question = state["question"]
        documents = state["documents"]
        generation = state["generation"]

        docs_content = "\n\n".join(doc.page_content for doc in documents)
        hallucination_score = await (self.hallucination_prompt | self.hallucination_grader).ainvoke(
            {"documents": docs_content, "generation": generation}
        )

        if hallucination_score.binary_score == "yes":
            print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")

            print("---GRADE GENERATION vs QUESTION---")
            answer_score = await (self.answer_prompt | self.answer_grader).ainvoke(
                {"question": question, "generation": generation}
            )

            if answer_score.binary_score == "yes":
                print("---DECISION: GENERATION ADDRESSES QUESTION---")
                return "useful"
            else:
                print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
                return "not_useful"
        else:
            print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
            return "not_supported"

    def _build_graph(self) -> Any:
        """Build and compile the workflow graph."""
        workflow = StateGraph(GraphState)

        workflow.add_node("web_search", self.web_search)
        workflow.add_node("vectorstore", self.retrieve)
        workflow.add_node("grade_documents", self.grade_documents)
        workflow.add_node("generate", self.generate)
        workflow.add_node("transform_query", self.transform_query)

        workflow.add_conditional_edges(
            START,
            self.route_question,
            {
                "web_search": "web_search",
                "vectorstore": "vectorstore",
            },
        )
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("vectorstore", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            self.decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "vectorstore")
        workflow.add_conditional_edges(
            "generate",
            self.grade_generation,
            {
                "not_supported": "generate",
                "useful": END,
                "not_useful": "transform_query",
            },
        )

        return workflow.compile()

    async def process_query(self, question: str) -> Dict[str, Any]:
        """Process a question through the workflow."""
        print(f"\n\n===== PROCESSING: '{question}' =====\n")

        inputs = {"question": question}
        final_state = None

        async for output in self.app.astream(inputs):
            for key, value in output.items():
                print(f"Node: '{key}'")
            print("---")
            final_state = value

        if final_state and "generation" in final_state:
            print("\n===== ANSWER =====")
            print(final_state["generation"])

        return final_state

async def main() -> None:
    """Main application entry point."""
    rag_system = AdaptiveRAG()
    await rag_system.initialize()

    questions = [
        "What player are the Bears expected to draft first in the 2024 NFL draft?",
        "What are the types of agent memory?",
    ]

    for question in questions:
        await rag_system.process_query(question)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
