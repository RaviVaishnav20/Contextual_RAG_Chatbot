import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # go up 1 more level (project root)

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
import os

import asyncio
import json
from typing import List, Optional
import signal
import time
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from llm.llm import generate_content
from config.config_manager import ConfigManager

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

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
    
    contextual_config = config.get_contextual_config()
    primary = contextual_config.get('primary', {})
    secondary = contextual_config.get('secondary', {})
    tertiary = contextual_config.get('tertiary', {})
    
    # Set up timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        summarized_context = generate_content(
            provider=primary.get('provider', 'ollama'),
            model_name=primary.get('model_name', 'llama3:8b'),
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
                provider=secondary.get('provider', 'gemini'),
                model_name=secondary.get('model_name', 'gemini-2.5-flash'),
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
                    provider=tertiary.get('provider', 'groq'),
                    model_name=tertiary.get('model_name', 'llama-3.3-70b-versatile'),
                    prompt=summarization_prompt
                )
                signal.alarm(0)
                if summarized_context is None:
                    return "Error: Tertiary model summarization returned None. " + combined_context[:1000]
                return summarized_context
            except (Exception, TimeoutError) as groq_e:
                signal.alarm(0)
                return "Combined Context (failed to summarize/deduplicate): " + combined_context[:1000]


def get_context_for_chunk(whole_document: str, chunk_content: str, config: ConfigManager, timeout_seconds: int = 120) -> list[str]:
    contextual_config = config.get_contextual_config()
    max_document_length = contextual_config.get('parameters', {}).get('max_document_length', 100000)
    
    primary = contextual_config.get('primary', {})
    secondary = contextual_config.get('secondary', {})
    tertiary = contextual_config.get('tertiary', {})
    
    prompt_template = """
<document>
{{WHOLE_DOCUMENT}}
</document>
Here is the chunk we want to situate within the whole document
<chunk>
{{CHUNK_CONTENT}}
</chunk>
Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else. """

    generated_contexts_for_segments = []
    final_context = "No context generated due to an error."

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
                    provider=primary.get('provider', 'ollama'),
                    model_name=primary.get('model_name', 'llama3:8b'),
                    prompt=segment_formatted_prompt
                )
                signal.alarm(0)  # Cancel timeout
            except (Exception, TimeoutError) as e:
                signal.alarm(0)
                print(f"    ⚠️  Primary model failed for segment {i+1}, trying secondary...")
                try:
                    signal.alarm(timeout_seconds)
                    context_for_segment = generate_content(
                        provider=secondary.get('provider', 'gemini'),
                        model_name=secondary.get('model_name', 'gemini-2.5-flash'),
                        prompt=segment_formatted_prompt
                    )
                    signal.alarm(0)
                except (Exception, TimeoutError) as e:
                    signal.alarm(0)
                    print(f"    ⚠️  Secondary model failed for segment {i+1}, trying tertiary...")
                    try:
                        signal.alarm(timeout_seconds)
                        context_for_segment = generate_content(
                            provider=tertiary.get('provider', 'groq'),
                            model_name=tertiary.get('model_name', 'llama-3.3-70b-versatile'),
                            prompt=segment_formatted_prompt
                        )
                        signal.alarm(0)
                    except (Exception, TimeoutError) as groq_e:
                        signal.alarm(0)
                        print(f"    ❌ All models failed for segment {i+1}")
                        context_for_segment = f"Error for segment {i+1}: Could not generate context."

            if context_for_segment is None:
                context_for_segment = f"Error: LLM call returned None for segment {i+1}."
            generated_contexts_for_segments.append(context_for_segment)

        if generated_contexts_for_segments:
            print(f"    🔗 Combining {len(generated_contexts_for_segments)} segment contexts...")
            final_context = _summarize_and_deduplicate_context(generated_contexts_for_segments, config, timeout_seconds)
        else:
            final_context = "No context could be generated for any segment."

    else:
        formatted_prompt = prompt_template.replace("{{WHOLE_DOCUMENT}}", whole_document).replace("{{CHUNK_CONTENT}}", chunk_content)

        final_context = None
        
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        try:
            final_context = generate_content(
                provider=primary.get('provider', 'ollama'),
                model_name=primary.get('model_name', 'llama3:8b'),
                prompt=formatted_prompt
            )
            signal.alarm(0)  # Cancel timeout
        except (Exception, TimeoutError) as e:
            signal.alarm(0)
            try:
                signal.alarm(timeout_seconds)
                final_context = generate_content(
                    provider=secondary.get('provider', 'gemini'),
                    model_name=secondary.get('model_name', 'gemini-2.5-flash'),
                    prompt=formatted_prompt
                )
                signal.alarm(0)
            except (Exception, TimeoutError) as e:
                signal.alarm(0)
                try:
                    signal.alarm(timeout_seconds)
                    final_context = generate_content(
                        provider=tertiary.get('provider', 'groq'),
                        model_name=tertiary.get('model_name', 'llama-3.3-70b-versatile'),
                        prompt=formatted_prompt
                    )
                    signal.alarm(0)
                except (Exception, TimeoutError) as groq_e:
                    signal.alarm(0)
                    final_context = "Error: Could not generate context using available models."

        if final_context is None:
            final_context = "Error: LLM call returned None for single document."

    return [final_context, chunk_content]