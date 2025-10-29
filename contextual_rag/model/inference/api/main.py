from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from contextual_rag.model.inference.api.routers import health, rag, openai_api

app = FastAPI(
    title="Contextual RAG ChatBot API",
    description="Enhanced RAG API with modular routers",
    version="2.2.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(health.router, prefix="/health", tags=["health"])
app.include_router(rag.router, tags=["rag"])
app.include_router(openai_api.router, prefix="/v1", tags=["openai"])
# app.include_router(evaluation.router, prefix="/evaluate", tags=["evaluation"])

@app.get("/")
async def root():
    return {"message": "Contextual RAG ChatBot API is running", "version": "2.2.0"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "contextual_rag.model.inference.api.main:app",  # module:variable
#         host="0.0.0.0",
#         port=8000,
#         reload=True
#     )