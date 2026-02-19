"""
Universal Self-RAG System
Works with any PDF documents using Google Gemini
"""

from typing import List, TypedDict, Literal
from pydantic import BaseModel, Field
import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API key
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file")


class State(TypedDict):
    """Graph state for Self-RAG workflow"""
    question: str
    retrieval_query: str
    rewrite_tries: int
    need_retrieval: bool
    docs: List[Document]
    relevant_docs: List[Document]
    context: str
    answer: str
    issup: Literal["fully_supported", "partially_supported", "no_support"]
    evidence: List[str]
    retries: int
    isuse: Literal["useful", "not_useful"]
    use_reason: str


class SelfRAG:
    """Universal Self-RAG system for any documents"""
    
    def __init__(self, documents_folder: str = "documents", chunk_size: int = 800, chunk_overlap: int = 200):
        """
        Initialize Self-RAG system
        
        Args:
            documents_folder: Folder containing PDF documents
            chunk_size: Size of text chunks for retrieval
            chunk_overlap: Overlap between chunks
        """
        self.documents_folder = documents_folder
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            convert_system_messages_to_human=True
        )
        
        # Load and process documents
        self.retriever = self._load_documents()
        
        # Build workflow
        self.app = self._build_workflow()
        
        print("✅ Self-RAG system initialized successfully!")
    
    def _load_documents(self):
        """Load PDF documents and create vector store"""
        print(f"📂 Loading documents from {self.documents_folder}...")
        
        # Find all PDF files
        pdf_files = list(Path(self.documents_folder).glob("*.pdf"))
        if not pdf_files:
            raise ValueError(f"No PDF files found in {self.documents_folder}")
        
        print(f"📄 Found {len(pdf_files)} PDF files")
        
        # Load all PDFs
        docs = []
        for pdf_file in pdf_files:
            print(f"   Loading {pdf_file.name}...")
            docs.extend(PyPDFLoader(str(pdf_file)).load())
        
        print(f"✅ Loaded {len(docs)} pages")
        
        # Chunk documents
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        ).split_documents(docs)
        
        print(f"✅ Created {len(chunks)} chunks")
        
        # Create vector store
        print("🔄 Creating vector store with Gemini embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        
        print("✅ Vector store created")
        return retriever
    
    def _build_workflow(self):
        """Build the Self-RAG workflow graph"""
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("decide_retrieval", self._decide_retrieval)
        workflow.add_node("generate_direct", self._generate_direct)
        workflow.add_node("retrieve", self._retrieve)
        workflow.add_node("is_relevant", self._is_relevant)
        workflow.add_node("generate_from_context", self._generate_from_context)
        workflow.add_node("no_answer_found", self._no_answer_found)
        workflow.add_node("is_sup", self._is_sup)
        workflow.add_node("accept_answer", self._accept_answer)
        workflow.add_node("revise_answer", self._revise_answer)
        workflow.add_node("is_use", self._is_use)
        workflow.add_node("rewrite_question", self._rewrite_question)
        
        # Add edges
        workflow.add_edge(START, "decide_retrieval")
        workflow.add_conditional_edges("decide_retrieval", self._route_after_decide)
        workflow.add_edge("generate_direct", END)
        workflow.add_edge("retrieve", "is_relevant")
        workflow.add_conditional_edges("is_relevant", self._route_after_relevance)
        workflow.add_edge("generate_from_context", "is_sup")
        workflow.add_conditional_edges("is_sup", self._route_after_issup)
        workflow.add_edge("accept_answer", "is_use")
        workflow.add_edge("revise_answer", "is_sup")
        workflow.add_conditional_edges("is_use", self._route_after_isuse)
        workflow.add_edge("rewrite_question", "retrieve")
        workflow.add_edge("no_answer_found", END)
        
        return workflow.compile()
    
    # Node 1: Decide Retrieval
    def _decide_retrieval(self, state: State):
        class RetrieveDecision(BaseModel):
            should_retrieve: bool = Field(..., description="True if documents needed")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Decide if retrieval from documents is needed.\n"
                      "Return JSON with key: should_retrieve (boolean).\n"
                      "True if question needs specific facts from documents.\n"
                      "False for general knowledge questions."),
            ("human", "Question: {question}")
        ])
        
        llm = self.llm.with_structured_output(RetrieveDecision)
        decision = llm.invoke(prompt.format_messages(question=state["question"]))
        return {"need_retrieval": decision.should_retrieve}
    
    def _route_after_decide(self, state: State):
        return "retrieve" if state["need_retrieval"] else "generate_direct"
    
    # Node 2: Direct Answer
    def _generate_direct(self, state: State):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer using only your general knowledge.\n"
                      "If it requires specific document info, say: 'I need to check the documents.'"),
            ("human", "{question}")
        ])
        out = self.llm.invoke(prompt.format_messages(question=state["question"]))
        return {"answer": out.content}
    
    # Node 3: Retrieve
    def _retrieve(self, state: State):
        q = state.get("retrieval_query") or state["question"]
        return {"docs": self.retriever.invoke(q)}
    
    # Node 4: Filter Relevance
    def _is_relevant(self, state: State):
        class RelevanceDecision(BaseModel):
            is_relevant: bool = Field(..., description="True if document is relevant")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Judge if document is relevant to the question.\n"
                      "Return JSON with key: is_relevant (boolean).\n"
                      "Relevant means it discusses the same topic or contains related information."),
            ("human", "Question:\n{question}\n\nDocument:\n{document}")
        ])
        
        llm = self.llm.with_structured_output(RelevanceDecision)
        relevant_docs = []
        
        for doc in state.get("docs", []):
            decision = llm.invoke(prompt.format_messages(
                question=state["question"],
                document=doc.page_content
            ))
            if decision.is_relevant:
                relevant_docs.append(doc)
        
        return {"relevant_docs": relevant_docs}
    
    def _route_after_relevance(self, state: State):
        if state.get("relevant_docs") and len(state["relevant_docs"]) > 0:
            return "generate_from_context"
        return "no_answer_found"
    
    # Node 5: Generate from Context
    def _generate_from_context(self, state: State):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant.\n"
                      "Answer the question based on the provided context.\n"
                      "Be clear, accurate, and thorough.\n"
                      "Don't mention 'context' in your answer."),
            ("human", "Question:\n{question}\n\nContext:\n{context}")
        ])
        
        context = "\n\n---\n\n".join([d.page_content for d in state.get("relevant_docs", [])]).strip()
        if not context:
            return {"answer": "No relevant content found.", "context": ""}
        
        out = self.llm.invoke(prompt.format_messages(question=state["question"], context=context))
        return {"answer": out.content, "context": context}
    
    def _no_answer_found(self, state: State):
        return {"answer": "No relevant content found in the documents.", "context": ""}
    
    # Node 6: Verify Support (IsSUP)
    def _is_sup(self, state: State):
        class IsSUPDecision(BaseModel):
            issup: Literal["fully_supported", "partially_supported", "no_support"]
            evidence: List[str] = Field(default_factory=list)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Verify if the ANSWER is supported by the CONTEXT.\n"
                      "Return JSON with keys: issup, evidence.\n"
                      "- fully_supported: All claims backed by context\n"
                      "- partially_supported: Core facts supported but includes interpretation\n"
                      "- no_support: Claims not supported\n"
                      "Evidence: up to 3 quotes from context"),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}\n\nContext:\n{context}")
        ])
        
        llm = self.llm.with_structured_output(IsSUPDecision)
        decision = llm.invoke(prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        ))
        return {"issup": decision.issup, "evidence": decision.evidence}
    
    def _route_after_issup(self, state: State):
        if state.get("issup") == "fully_supported":
            return "accept_answer"
        if state.get("retries", 0) >= 3:
            return "accept_answer"
        return "revise_answer"
    
    def _accept_answer(self, state: State):
        return {}
    
    # Node 7: Revise Answer
    def _revise_answer(self, state: State):
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Revise the answer to be more grounded in the context.\n"
                      "Use direct quotes and examples from the context.\n"
                      "Be accurate and factual."),
            ("human", "Question:\n{question}\n\nCurrent Answer:\n{answer}\n\nContext:\n{context}")
        ])
        
        out = self.llm.invoke(prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", ""),
            context=state.get("context", "")
        ))
        return {"answer": out.content, "retries": state.get("retries", 0) + 1}
    
    # Node 8: Check Usefulness (IsUSE)
    def _is_use(self, state: State):
        class IsUSEDecision(BaseModel):
            isuse: Literal["useful", "not_useful"]
            reason: str = Field(..., description="Short reason")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Judge if the ANSWER is useful for the QUESTION.\n"
                      "Return JSON with keys: isuse, reason.\n"
                      "- useful: Directly answers the question\n"
                      "- not_useful: Generic, off-topic, or missing key info"),
            ("human", "Question:\n{question}\n\nAnswer:\n{answer}")
        ])
        
        llm = self.llm.with_structured_output(IsUSEDecision)
        decision = llm.invoke(prompt.format_messages(
            question=state["question"],
            answer=state.get("answer", "")
        ))
        return {"isuse": decision.isuse, "use_reason": decision.reason}
    
    def _route_after_isuse(self, state: State):
        if state.get("isuse") == "useful":
            return "END"
        if state.get("rewrite_tries", 0) >= 2:
            return "no_answer_found"
        return "rewrite_question"
    
    # Node 9: Rewrite Query
    def _rewrite_question(self, state: State):
        class RewriteDecision(BaseModel):
            retrieval_query: str = Field(..., description="Rewritten query")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Rewrite the question for better document retrieval.\n"
                      "Keep it short (6-16 words).\n"
                      "Include key terms and remove filler words.\n"
                      "Return JSON with key: retrieval_query"),
            ("human", "Question:\n{question}\n\nPrevious query:\n{retrieval_query}")
        ])
        
        llm = self.llm.with_structured_output(RewriteDecision)
        decision = llm.invoke(prompt.format_messages(
            question=state["question"],
            retrieval_query=state.get("retrieval_query", "")
        ))
        
        return {
            "retrieval_query": decision.retrieval_query,
            "rewrite_tries": state.get("rewrite_tries", 0) + 1,
            "docs": [],
            "relevant_docs": [],
            "context": ""
        }
    
    def ask(self, question: str, verbose: bool = False):
        """
        Ask a question to the Self-RAG system
        
        Args:
            question: Your question
            verbose: Show detailed metadata
            
        Returns:
            dict: Result with answer and metadata
        """
        result = self.app.invoke({"question": question})
        
        print("\n" + "="*80)
        print("Q:", question)
        print("\n" + "-"*80)
        print("A:", result["answer"])
        
        if verbose:
            print("\n" + "-"*80)
            print("METADATA:")
            print(f"  Retrieval: {result.get('need_retrieval')}")
            print(f"  Support: {result.get('issup')}")
            print(f"  Useful: {result.get('isuse')}")
            if result.get('evidence'):
                print(f"  Evidence: {result.get('evidence')}")
        print("="*80 + "\n")
        
        return result


if __name__ == "__main__":
    # Example usage
    rag = SelfRAG(documents_folder="documents")
    
    # Ask questions
    rag.ask("What is the main topic of the documents?")
    rag.ask("Summarize the key points.", verbose=True)
