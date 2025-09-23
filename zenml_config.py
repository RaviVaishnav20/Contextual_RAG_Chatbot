"""ZenML configuration and materializer registration."""

from contextual_rag.infrastructure.materializers import ChunkMaterializer, ChunkListMaterializer

def register_materializers():
    """Register custom materializers with ZenML."""
    print("✅ Custom materializers are available for use in steps")
    print("Materializers will be used automatically when specified in @step decorators")

if __name__ == "__main__":
    register_materializers()
