
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable,List
import json
import os
import signal
from contextual_rag.infrastructure.llm import generate_content
from contextual_rag.infrastructure.config_manager import ConfigManager
from contextual_rag.settings import settings

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

@dataclass
class Chunk:
    document_name: str
    chunk_id: str
    text: str
    metadata: dict

def chunk_with_llm(path: Path, word_limit: int = 512, overlap: int = 50) -> List[str]:
    cm = ConfigManager()
    chunk_cfg = cm.get_chunking_config() or {}
    WORD_LIMIT = chunk_cfg.get('parameters', {}).get('word_limit', 512)
    
    primary_provider = chunk_cfg.get('model', {}).get('primary_provider', 'ollama')
    primary_model = chunk_cfg.get('model', {}).get('primary_model_name', 'llama3:8b')
    fallback_provider = chunk_cfg.get('model', {}).get('fallback_provider', 'gemini')
    fallback_model = chunk_cfg.get('model', {}).get('fallback_model_name', 'gemini-2.5-flash')
    
    text = path.read_text(encoding="utf-8")
    words = text.split()
    i = 0
    final_chunks = []

    while i < len(words):
        chunk_words = words[i:i + WORD_LIMIT]
        chunk_text = " ".join(chunk_words).strip()

        prompt = f"""
                    You are a markdown document segmenter.

                    Here is a portion of a markdown document:

                    ---
                    {chunk_text}
                    ---

                    If this chunk clearly contains **more than one distinct topic or section**, reply ONLY with the **second part**, starting from the first sentence or heading of the new topic.

                    If it's only one topic, reply with NOTHING.

                    Keep markdown formatting intact.
                    """
        reply = ""
        try:
            reply = generate_content(
                provider=primary_provider,
                model_name=primary_model,
                prompt=prompt
            ).strip()
            if reply:
                split_point = chunk_text.find(reply)

                if split_point != -1:
                    first_part = chunk_text[:split_point].strip()
                    second_part = reply.strip()

                    final_chunks.append(first_part)

                    leftover_words = second_part.split()
                    words = leftover_words + words[i + WORD_LIMIT:]
                    
                    i = 0
                    continue
                else:
                    final_chunks.append(chunk_text)
                    # print(f"Chunk: {len(final_chunks)}")
            else:
                final_chunks.append(chunk_text)   
                # print(f"Chunk: {len(final_chunks)}") 
        except Exception as e:
            try:
                reply = generate_content(
                    provider=fallback_provider,
                    model_name=fallback_model,
                    prompt=prompt
                ).strip()
                
                if reply:
                    split_point = chunk_text.find(reply)
                    if split_point != -1:
                        first_part = chunk_text[:split_point].strip()
                        second_part = reply.strip()
                        final_chunks.append(first_part)
                        leftover_words = second_part.split()
                        words = leftover_words + words[i + WORD_LIMIT:]
                        i = 0
                        continue
                    else:
                        final_chunks.append(chunk_text)
                        # print(f"Chunk: {len(final_chunks)}")
                else:
                    final_chunks.append(chunk_text)
                    # print(f"Chunk: {len(final_chunks)}")
            except Exception as fallback_e:
                final_chunks.append(chunk_text)
                # print(f"Chunk: {len(final_chunks)}")

        i += WORD_LIMIT
    # print("final_chunks")
    # print(final_chunks)
    return final_chunks


def build_chunks(path: Path) -> List[Chunk]:
    
    all_chunks: List[Chunk] = []
    pieces = chunk_with_llm(path)
    document_name = path.name
    for i, piece in enumerate(pieces):
            all_chunks.append(
                Chunk(
                    document_name=document_name,
                    chunk_id=f"chunk-{i}",
                    text=piece,
                    metadata={"source": "markdown"},

                )
            )
    return all_chunks

def _summarize_and_deduplicate_context(contexts: list[str], config: ConfigManager, timeout_seconds: int = 60) -> str:
    """Summarizes and de-duplicates a list of contexts using an LLM call with timeout."""
    if not contexts:
        return ""
    if len(contexts) == 1:
        return contexts[0]

    cleaned_contexts = [c for c in contexts if isinstance(c, str)]
    if not cleaned_contexts:
        return "Error: No valid contexts to summarize."

    combined_context = "\n---\n".join(cleaned_contexts)
    
    # Truncate if too long
    if len(combined_context) > 50000:
        combined_context = combined_context[:50000] + "..."

    summarization_prompt = f"""
        The following are several pieces of context related to a document chunk.
        Please combine them into a single, succinct, and non-redundant context that best describes the chunk's place within the overall document.
        Do not add new information beyond what is provided in the contexts.

        Contexts:
        {combined_context}

        Combined Succinct Context:
        """
    cm = ConfigManager()
    chunk_cfg = cm.get_contextual_config() or {}
    primary_provider = chunk_cfg.get('primary', {}).get('provider', 'ollama')
    primary_model = chunk_cfg.get('primary', {}).get('model_name', 'llama3:8b')
    secondary_provider = chunk_cfg.get('secondary', {}).get('provider', 'gemini')
    secondary_model = chunk_cfg.get('secondary', {}).get('model_name', 'gemini-2.5-flash')
    tertiary_provider = chunk_cfg.get('tertiary', {}).get('provider', 'bedrock')
    tertiary_model = chunk_cfg.get('tertiary', {}).get('model_name', 'claude')

    timeout_seconds = chunk_cfg.get('parameters', {}).get('timeout_seconds', 60)
  
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        summarized_context = generate_content(
            provider=primary_provider,
            model_name=primary_model,
            prompt=summarization_prompt
        )
        signal.alarm(0)  # Cancel timeout
        if summarized_context is None:
            return "Error: Primary model summarization returned None. " + combined_context[:1000]
        return summarized_context
    except (Exception, TimeoutError) as e:
        signal.alarm(0)  # Cancel timeout
        try:
            signal.alarm(timeout_seconds)
            summarized_context = generate_content(
                provider=secondary_provider,
                model_name=secondary_model,
                prompt=summarization_prompt
            )
            signal.alarm(0)
            if summarized_context is None:
                return "Error: Secondary model summarization returned None. " + combined_context[:1000]
            return summarized_context
        except (Exception, TimeoutError) as e:
            signal.alarm(0)
            try:
                signal.alarm(timeout_seconds)
                summarized_context = generate_content(
                    provider=tertiary_provider,
                    model_name=tertiary_model,
                    prompt=summarization_prompt
                )
                signal.alarm(0)
                if summarized_context is None:
                    return "Error: Tertiary model summarization returned None. " + combined_context[:1000]
                return summarized_context
            except (Exception, TimeoutError) as groq_e:
                signal.alarm(0)
                return "Combined Context (failed to summarize/deduplicate): " + combined_context[:1000]


def get_context_for_chunk(whole_document: str, chunk_content: str, timeout_seconds: int = 60) -> list[str]:
    cm = ConfigManager()
    chunk_cfg = cm.get_contextual_config() or {}
    
    primary_provider = chunk_cfg.get('primary', {}).get('provider', 'ollama')
    primary_model = chunk_cfg.get('primary', {}).get('model_name', 'llama3:8b')
    secondary_provider = chunk_cfg.get('secondary', {}).get('provider', 'gemini')
    secondary_model = chunk_cfg.get('secondary', {}).get('model_name', 'gemini-2.5-flash')
    tertiary_provider = chunk_cfg.get('tertiary', {}).get('provider', 'bedrock')
    tertiary_model = chunk_cfg.get('tertiary', {}).get('model_name', 'claude')

    max_document_length = chunk_cfg.get('parameters', {}).get('max_document_length', 90000)
    timeout_seconds = chunk_cfg.get('parameters', {}).get('timeout_seconds', 60)
    
    prompt_template = """
        <document>
        {{WHOLE_DOCUMENT}}
        </document>
        Here is the chunk we want to situate within the whole document
        <chunk>
        {{CHUNK_CONTENT}}
        </chunk>
        Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. 
        Answer only with the succinct context and nothing else. """

    generated_contexts_for_segments = []
    final_context = ""

    if len(whole_document) > max_document_length:
        print(f"    📄 Large document detected ({len(whole_document)} chars), splitting...")
        document_segments = [whole_document[i:i + max_document_length] for i in range(0, len(whole_document), max_document_length)]
        print(f"    ✂️  Split into {len(document_segments)} segments")

        for i, segment in enumerate(document_segments):
            print(f"    🔄 Processing segment {i+1}/{len(document_segments)}")
            segment_formatted_prompt = prompt_template.replace("{{WHOLE_DOCUMENT}}", segment).replace("{{CHUNK_CONTENT}}", chunk_content)

            context_for_segment = None
            
            # Set up timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(timeout_seconds)
            
            try:
                context_for_segment = generate_content(
                    provider=primary_provider,
                    model_name=primary_model,
                    prompt=segment_formatted_prompt
                )
                signal.alarm(0)  # Cancel timeout
            except (Exception, TimeoutError) as e:
                signal.alarm(0)
                print(f"    ⚠️  Primary model failed for segment {i+1}, trying secondary...")
                try:
                    signal.alarm(timeout_seconds)
                    context_for_segment = generate_content(
                        provider=secondary_provider,
                        model_name=secondary_model,
                        prompt=segment_formatted_prompt
                    )
                    signal.alarm(0)
                except (Exception, TimeoutError) as e:
                    signal.alarm(0)
                    print(f"    ⚠️  Secondary model failed for segment {i+1}, trying tertiary...")
                    try:
                        signal.alarm(timeout_seconds)
                        context_for_segment = generate_content(
                            provider=tertiary_provider,
                            model_name=tertiary_model,
                            prompt=segment_formatted_prompt
                        )
                        signal.alarm(0)
                    except (Exception, TimeoutError) as groq_e:
                        signal.alarm(0)
                        print(f"    ❌ All models failed for segment {i+1}")
                        context_for_segment = ""

            if context_for_segment is None:
                context_for_segment = ""
                continue
            generated_contexts_for_segments.append(context_for_segment)

        if generated_contexts_for_segments:
            print(f"    🔗 Combining {len(generated_contexts_for_segments)} segment contexts...")
            final_context = _summarize_and_deduplicate_context(generated_contexts_for_segments, config, timeout_seconds)
        else:
            final_context = ""

    else:
        formatted_prompt = prompt_template.replace("{{WHOLE_DOCUMENT}}", whole_document).replace("{{CHUNK_CONTENT}}", chunk_content)

        final_context = None
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            final_context = generate_content(
                provider=primary_provider,
                model_name=primary_model,
                prompt=formatted_prompt
            )
            signal.alarm(0)  # Cancel timeout
        except (Exception, TimeoutError) as e:
            signal.alarm(0)
            try:
                signal.alarm(timeout_seconds)
                final_context = generate_content(
                    provider=secondary_provider,
                    model_name=secondary_model,
                    prompt=formatted_prompt
                )
                signal.alarm(0)
            except (Exception, TimeoutError) as e:
                signal.alarm(0)
                try:
                    signal.alarm(timeout_seconds)
                    final_context = generate_content(
                        provider=tertiary_provider,
                        model_name=tertiary_model,
                        prompt=formatted_prompt
                    )
                    signal.alarm(0)
                except (Exception, TimeoutError) as groq_e:
                    signal.alarm(0)
                    final_context = ""

        if final_context is None:
            final_context = ""

    return final_context

def build_chunks_context(path: Path, sementic_chunk:List[Chunk]) -> List[Chunk]:
    
    all_chunks: List[Chunk] = []

    whole_document = path.read_text(encoding="utf-8")
    document_name = path.name
    for i, c in enumerate(sementic_chunk):
        chunk_content = c.text
        context = get_context_for_chunk(whole_document, chunk_content)
        new_chunk = context + chunk_content
        all_chunks.append(
            Chunk(
                document_name=document_name,
                chunk_id=f"chunk-{i}",
                text=new_chunk,
                metadata={"source": "markdown"},
            )
        )
    return all_chunks