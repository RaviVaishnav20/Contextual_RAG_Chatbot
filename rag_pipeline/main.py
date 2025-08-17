import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    
import os
import json
import time
from rag_pipeline.data_extraction import convert_to_markdown
from rag_pipeline.chunking import chunk_documents
from rag_pipeline.contextual_retrieval import get_context_for_chunk
from rag_pipeline.embedding_storage import process_and_store_embeddings
from config.config_manager import ConfigManager
from rag_pipeline.file_hash_manager import FileHashManager
from tqdm import tqdm
import asyncio
import signal



def print_banner():
    print("="*60)
    print("🤖 CONTEXTUAL RAG CHATBOT PIPELINE")
    print("="*60)

class GracefulKiller:
    kill_now = False
    def __init__(self):
        signal.signal(signal.SIGINT, self._graceful_exit)
        signal.signal(signal.SIGTERM, self._graceful_exit)

    def _graceful_exit(self, signum, frame):
        print("\n🛑 Graceful shutdown requested...")
        self.kill_now = True

def save_progress(chunks_with_context, metadata_context_chunk_file, current_index, total_chunks):
    """Save progress after each chunk to allow resuming"""
    # Save the contextual chunks
    with open(metadata_context_chunk_file, 'w') as f:
        json.dump(chunks_with_context, f, indent=4)
    
    # Save current progress
    progress_file = metadata_context_chunk_file.replace('.json', '_progress.json')
    with open(progress_file, 'w') as f:
        json.dump({
            'last_processed_index': current_index,
            'total_chunks': total_chunks,
            'timestamp': time.time()
        }, f)

def load_progress(metadata_context_chunk_file):
    """Load progress to resume from where left off"""
    progress_file = metadata_context_chunk_file.replace('.json', '_progress.json')
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
            return progress_data.get('last_processed_index', 0), progress_data.get('total_chunks', 0)
    return 0, 0

def cleanup_progress(metadata_context_chunk_file):
    """Clean up progress file when processing is complete"""
    progress_file = metadata_context_chunk_file.replace('.json', '_progress.json')
    if os.path.exists(progress_file):
        os.remove(progress_file)

async def main():
    print_banner()
    killer = GracefulKiller()
    
    config = ConfigManager()
    file_hash_manager = FileHashManager(config)
    
    # Get configuration
    directories = config.get_directories()
    RESOURCES_FOLDER = directories['resources']
    MARKDOWN_ARTIFACTS_FOLDER = directories['markdown_artifacts']
    METADATA_CHUNK_FILE = directories['metadata_chunk_file']
    METADATA_CONTEXT_CHUNK_FILE = directories['metadata_context_chunk_file']
    
    print(f"📁 Resources folder: {RESOURCES_FOLDER}")
    
    # Ensure markdown artifacts directory exists
    os.makedirs(MARKDOWN_ARTIFACTS_FOLDER, exist_ok=True)
    print(f"✅ Markdown artifacts directory ready: {MARKDOWN_ARTIFACTS_FOLDER}")
    
    # Load existing metadata
    existing_chunks = []
    existing_context_chunks = []
    
    if os.path.exists(METADATA_CHUNK_FILE):
        with open(METADATA_CHUNK_FILE, 'r') as f:
            existing_chunks = json.load(f)
        print(f"📚 Loaded {len(existing_chunks)} existing chunks")
    
    if os.path.exists(METADATA_CONTEXT_CHUNK_FILE):
        with open(METADATA_CONTEXT_CHUNK_FILE, 'r') as f:
            existing_context_chunks = json.load(f)
        print(f"🎯 Loaded {len(existing_context_chunks)} existing context chunks")
    
    # Get changed or new files
    print("\n🔍 Checking for new or changed files...")
    changed_files = file_hash_manager.get_changed_or_new_files(RESOURCES_FOLDER)
    
    if not changed_files:
        print("✅ No new or changed files found.")
        # Still process contextual chunks if needed
        if len(existing_chunks) > len(existing_context_chunks):
            print(f"🎯 Found {len(existing_chunks) - len(existing_context_chunks)} chunks needing context processing...")
        else:
            print("Pipeline complete!")
            return
    else:
        print(f"📋 Found {len(changed_files)} files to process: {list(changed_files)}")
    
    all_chunks_with_metadata = existing_chunks.copy()
    
    # Step 1: Convert documents to markdown and chunk (only if new files)
    if changed_files:
        print("\n" + "="*40)
        print("📄 STEP 1: DOCUMENT PROCESSING")
        print("="*40)
        
        for filename in tqdm(changed_files, desc="Processing files"):
            if killer.kill_now:
                print("🛑 Stopping document processing...")
                break
                
            file_path = os.path.join(RESOURCES_FOLDER, filename)
            print(f"\n🔄 Processing: {filename}")
            try:
                print("  📝 Converting to markdown...")
                whole_document_content = convert_to_markdown(file_path, MARKDOWN_ARTIFACTS_FOLDER)
                
                print("  ✂️  Creating chunks...")
                chunks = chunk_documents(whole_document_content, filename)
                print(f"  ✅ Created {len(chunks)} chunks")
                
                for i, chunk_content in enumerate(chunks):
                    metadata = {
                        "source": filename,
                        "chunk_id": len(all_chunks_with_metadata) + i,
                        "original_file_path": file_path,
                    }
                    all_chunks_with_metadata.append({
                        "chunk_content": chunk_content,
                        "metadata": metadata,
                        "whole_document_content": whole_document_content
                    })
            except Exception as e:
                print(f"  ❌ Error processing {filename}: {e}")
                continue
        
        if not killer.kill_now:
            # Save updated chunks
            with open(METADATA_CHUNK_FILE, 'w') as f:
                json.dump(all_chunks_with_metadata, f, indent=4)
            print(f"\n💾 Saved {len(all_chunks_with_metadata)} total chunks")
    
    if killer.kill_now:
        print("🛑 Exiting due to interrupt...")
        return
    
    # Step 2: Generate contextual chunks for new items only with resumable processing
    new_chunks_start_index = len(existing_context_chunks)
    new_chunks_to_process = all_chunks_with_metadata[new_chunks_start_index:]
    
    if new_chunks_to_process:
        print("\n" + "="*40)
        print("🧠 STEP 2: CONTEXTUAL PROCESSING")
        print("="*40)
        
        # Check for existing progress
        last_processed_index, total_from_progress = load_progress(METADATA_CONTEXT_CHUNK_FILE)
        
        if last_processed_index > 0:
            print(f"📍 Resuming from chunk {last_processed_index + 1} (previously processed {last_processed_index} chunks)")
            # Skip already processed chunks
            new_chunks_to_process = new_chunks_to_process[last_processed_index:]
            current_start_index = last_processed_index
        else:
            current_start_index = 0
            print(f"🎯 Starting fresh - processing {len(new_chunks_to_process)} new chunks...")
        
        chunks_with_context = existing_context_chunks.copy()
        total_chunks_to_process = len(all_chunks_with_metadata[new_chunks_start_index:])
        
        for i, item in enumerate(new_chunks_to_process):
            current_global_index = new_chunks_start_index + current_start_index + i
            current_processing_index = current_start_index + i
            
            if killer.kill_now:
                print(f"\n🛑 Stopping at chunk {current_processing_index}...")
                save_progress(chunks_with_context, METADATA_CONTEXT_CHUNK_FILE, current_processing_index, total_chunks_to_process)
                print(f"💾 Progress saved. Run again to resume from chunk {current_processing_index + 1}")
                return
            
            try:
                print(f"\n🔄 Processing chunk {current_processing_index + 1}/{total_chunks_to_process} (Global: {current_global_index + 1}/{len(all_chunks_with_metadata)})")
                print(f"   📄 Source: {item['metadata']['source']}")
                
                context_and_chunk_list = get_context_for_chunk(
                    whole_document=item["whole_document_content"],
                    chunk_content=item["chunk_content"],
                    config=config
                )
                
                contextual_chunk_content = context_and_chunk_list[0] 
                original_chunk_content = context_and_chunk_list[1] 

                updated_metadata = item["metadata"].copy()
                updated_metadata["contextual_chunk_content"] = contextual_chunk_content
                
                chunks_with_context.append({
                    "chunk_content": original_chunk_content,
                    "metadata": updated_metadata,
                })
                
                # Save progress after EVERY chunk
                save_progress(chunks_with_context, METADATA_CONTEXT_CHUNK_FILE, current_processing_index + 1, total_chunks_to_process)
                print(f"   ✅ Chunk {current_processing_index + 1} completed and saved")
                
            except Exception as e:
                print(f"   ❌ Error processing chunk {current_processing_index + 1}: {e}")
                # Add empty context to maintain consistency
                updated_metadata = item["metadata"].copy()
                updated_metadata["contextual_chunk_content"] = f"Error generating context: {str(e)}"
                chunks_with_context.append({
                    "chunk_content": item["chunk_content"],
                    "metadata": updated_metadata,
                })
                # Still save progress even on error
                save_progress(chunks_with_context, METADATA_CONTEXT_CHUNK_FILE, current_processing_index + 1, total_chunks_to_process)
                print(f"   ⚠️  Error handled, progress saved")
                continue
        
        if not killer.kill_now:
            print(f"\n🎉 All contextual processing completed!")
            print(f"💾 Final save: {len(chunks_with_context)} contextual chunks")
            
            # Final save
            with open(METADATA_CONTEXT_CHUNK_FILE, 'w') as f:
                json.dump(chunks_with_context, f, indent=4)
            
            # Clean up progress file
            cleanup_progress(METADATA_CONTEXT_CHUNK_FILE)
            print("🧹 Progress tracking cleaned up")
            
            # Step 3: Create embeddings for new chunks only
            print("\n" + "="*40)
            print("🔮 STEP 3: EMBEDDING CREATION")
            print("="*40)
            
            new_chunks_for_embedding = [f"{item["chunk_content"]}\n\n{item['metadata']['contextual_chunk_content']}" for item in chunks_with_context[len(existing_context_chunks):]]
            new_metadata_for_embedding = [item["metadata"] for item in chunks_with_context[len(existing_context_chunks):]]
            
            if new_chunks_for_embedding:
                print(f"🚀 Creating embeddings for {len(new_chunks_for_embedding)} new chunks...")
                db_config = config.get_database_config()
                index, vector_store = await process_and_store_embeddings(
                    chunks=new_chunks_for_embedding,
                    metadata_list=new_metadata_for_embedding,
                    db_config=db_config,
                    config=config
                )
                print("✅ Embedding creation completed!")
            else:
                print("ℹ️  No new chunks to embed")
    else:
        print("✅ All chunks already have context. Pipeline complete!")

async def re_embed_existing_data():
    print_banner()
    print("\n" + "="*40)
    print("🔄 RE-EMBEDDING EXISTING DATA")
    print("="*40)

    config = ConfigManager()
    directories = config.get_directories()
    METADATA_CONTEXT_CHUNK_FILE = directories['metadata_context_chunk_file']

    if not os.path.exists(METADATA_CONTEXT_CHUNK_FILE):
        print(f"❌ Error: {METADATA_CONTEXT_CHUNK_FILE} not found. Please run the full pipeline first.")
        return

    print(f"📚 Loading existing context chunks from {METADATA_CONTEXT_CHUNK_FILE}...")
    with open(METADATA_CONTEXT_CHUNK_FILE, 'r') as f:
        chunks_with_context = json.load(f)
    print(f"✅ Loaded {len(chunks_with_context)} contextual chunks.")

    chunks_for_embedding = [f"{item["chunk_content"]}\n\n{item['metadata']['contextual_chunk_content']}" for item in chunks_with_context]
    metadata_for_embedding = [item["metadata"] for item in chunks_with_context]

    if chunks_for_embedding:
        print(f"🚀 Creating embeddings for {len(chunks_for_embedding)} existing chunks...")
        db_config = config.get_database_config()
        index, vector_store = await process_and_store_embeddings(
            chunks=chunks_for_embedding,
            metadata_list=metadata_for_embedding,
            db_config=db_config,
            config=config
        )
        print("✅ Re-embedding completed successfully!")
    else:
        print("ℹ️  No chunks found in metadata_context_chunk.json to re-embed.")

if __name__ == "__main__":
   import argparse
   parser = argparse.ArgumentParser(description="Run the RAG chatbot pipeline or re-embed existing data.")
   parser.add_argument('--re-embed', action='store_true', help='Only re-embed existing contextual chunks.')
   args = parser.parse_args()
   try:
       if args.re_embed:
           asyncio.run(re_embed_existing_data())
       else:
           asyncio.run(main())
   except KeyboardInterrupt:
       print("\n🛑 Pipeline interrupted by user")
       sys.exit(0)
   except Exception as e:
       print(f"\n❌ Pipeline failed with error: {e}")
       sys.exit(1)