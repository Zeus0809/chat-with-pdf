#!/usr/bin/env python3
"""
Test script to reproduce the exact FunctionAgent workflow error
"""

import sys
import os
import asyncio
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, '.')

load_dotenv(verbose=True)

from llama_index.core import VectorStoreIndex, Settings, SimpleDirectoryReader
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.tools import QueryEngineTool
from llamaindex_utils.integrations import LlamaCppEmbedding, DockerLLM

async def test_full_workflow():
    """Test the complete workflow that's causing the error"""
    
    print("1. Setting up models...")
    # Initialize embedding model
    Settings.embed_model = LlamaCppEmbedding(model_path=os.getenv('EMBED_MODEL_PATH'), verbose=False)
    
    # Initialize chat model
    chat_model = DockerLLM(model="ai/gemma3n")
    Settings.llm = chat_model
    
    print("2. Creating test document...")
    # Create a simple test document
    test_content = "This is a test document about cats. Cats are amazing animals that like to sleep and play."
    with open("test_doc.txt", "w") as f:
        f.write(test_content)
    
    print("3. Creating index and query engine...")
    documents = SimpleDirectoryReader(input_files=["test_doc.txt"]).load_data()
    index = VectorStoreIndex.from_documents(documents)
    
    # Test both ways - with and without explicit LLM
    print("   Testing query engine with explicit LLM...")
    query_engine = index.as_query_engine(llm=chat_model, streaming=True)
    
    print("4. Creating RAG tool...")
    rag_tool = QueryEngineTool.from_defaults(
        query_engine,
        name="rag_query",
        description="Use this tool to answer questions about the document."
    )
    
    print("5. Creating FunctionAgent...")
    function_agent = FunctionAgent(
        tools=[rag_tool],
        llm=chat_model,
        system_prompt="You are a helpful assistant that can answer questions about documents."
    )
    
    print("6. Testing agent workflow...")
    try:
        print("   Calling agent.run()...")
        handler = await function_agent.run(user_msg="What are cats like?")
        print(f"   SUCCESS: Got handler: {type(handler)}")
        
        print("   Testing handler.stream_events()...")
        count = 0
        for event in handler.stream_events():
            print(f"   Event {count}: {event.delta if hasattr(event, 'delta') else str(event)[:50]}")
            count += 1
            if count >= 3:  # Just test first few events
                break
        
        print("SUCCESS: Full workflow completed!")
        return True
        
    except Exception as e:
        print(f"ERROR in workflow: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up
        if os.path.exists("test_doc.txt"):
            os.remove("test_doc.txt")

if __name__ == "__main__":
    success = asyncio.run(test_full_workflow())
    print(f"\nTest result: {'PASSED' if success else 'FAILED'}")
