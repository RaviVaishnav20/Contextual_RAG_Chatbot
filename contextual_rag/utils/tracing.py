from contextual_rag.infrastructure.config_manager import ConfigManager
from contextlib import contextmanager
from openinference.instrumentation.llama_index import LlamaIndexInstrumentor
from phoenix.otel import register
import opentelemetry.trace as trace
# Initialize tracing components (only when needed)
tracer_provider = None
tracer = None

def initialize_tracing():
    """Initialize tracing components lazily"""
    global tracer_provider, tracer
    if tracer_provider is None:
        try:
            cm = ConfigManager()
            phoenix_cfg = cm.get_phoenix_config() or {}
            project_name = phoenix_cfg.get('tracing', {}).get('project_name', 'contextual_rag_chatbot')
            endpoint = phoenix_cfg.get('tracing', {}).get('trace_endpoint', 'http://localhost:6006/v1/traces')
           
            tracer_provider = register(
                project_name=project_name,
                endpoint=endpoint,
                auto_instrument=True
            )
            LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
            tracer = trace.get_tracer(__name__)
            return True
        except Exception as e:
            print(f"Failed to initialize tracing: {e}")
            return False
    return True

# def initialize_tracing():
#     """Initialize tracing components lazily"""
#     global tracer_provider, tracer
#     if tracer_provider is None:
#         try:
#             cm = ConfigManager()
#             phoenix_cfg = cm.get_phoenix_config() or {}
#             project_name = phoenix_cfg.get('tracing', {}).get('project_name', 'contextual_rag_chatbot')
#             endpoint = phoenix_cfg.get('tracing', {}).get('trace_endpoint', 'http://localhost:6006/v1/traces')
            
#             # Avoid re-registering if already set
#             if trace.get_tracer_provider().__class__.__name__ != "ProxyTracerProvider":
#                 return True  

#             tracer_provider = register(
#                 project_name=project_name,
#                 endpoint=endpoint,
#                 auto_instrument=True,
#                 set_global_tracer_provider=True
#             )
#             LlamaIndexInstrumentor().instrument(tracer_provider=tracer_provider)
#             tracer = trace.get_tracer(__name__)
#             return True
#         except Exception as e:
#             print(f"Failed to initialize tracing: {e}")
#             return False
#     return True

@contextmanager
def conditional_span(span_name: str, enable_tracing: bool = False, **attributes):
    """Context manager for conditional tracing"""
    if enable_tracing and initialize_tracing():
        with tracer.start_as_current_span(span_name) as span:
            # Set attributes if provided
            for key, value in attributes.items():
                try:
                    span.set_attribute(key, str(value))
                except:
                    pass
            
            try:
                yield span
            except Exception as e:
                span.set_attribute("error", True)
                span.set_attribute("error_message", str(e))
                raise
    else:
        # No-op span for consistency
        class NoOpSpan:
            def set_attribute(self, key, value): pass
            def get_span_context(self): 
                class NoOpContext:
                    trace_id = 0
                return NoOpContext()
        
        yield NoOpSpan()