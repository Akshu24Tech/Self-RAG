"""
Example usage of Universal Self-RAG System
"""

from self_rag import SelfRAG

def main():
    print("🚀 Universal Self-RAG System")
    print("="*80)
    
    # Initialize Self-RAG with your documents
    print("\n📂 Initializing Self-RAG system...")
    rag = SelfRAG(documents_folder="documents")
    
    print("\n" + "="*80)
    print("💡 You can now ask questions about your documents!")
    print("="*80)
    
    # Example questions
    example_questions = [
        "What is the main topic of the documents?",
        "Summarize the key points.",
        "What are the important details mentioned?",
    ]
    
    print("\n📝 Example questions:")
    for i, q in enumerate(example_questions, 1):
        print(f"   {i}. {q}")
    
    print("\n" + "="*80)
    print("🎯 Asking first example question...")
    print("="*80)
    
    # Ask the first question with verbose output
    rag.ask(example_questions[0], verbose=True)
    
    print("\n" + "="*80)
    print("✅ Done! You can now use the system with your own questions.")
    print("="*80)
    
    # Interactive mode (optional)
    print("\n💬 Want to ask more questions? (y/n): ", end="")
    try:
        choice = input().strip().lower()
        if choice == 'y':
            while True:
                print("\n❓ Your question (or 'quit' to exit): ", end="")
                question = input().strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                if question:
                    rag.ask(question, verbose=True)
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    
    print("\n✨ Thank you for using Universal Self-RAG!")


if __name__ == "__main__":
    main()
