from typing import Optional
import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, APIRouter

from api.core.main_router import router as main_router
from api.graph.graphrag_manager import GraphRAGManager
from api.graph.views import router as graph_router
from api.utils.logger import setup_logger
from api.utils.constants import ENGINES

load_dotenv(".env")

root_router = APIRouter()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize GraphRAGManager once at startup

    root_path = os.environ["ROOT_PATH"]
    folder = os.environ["FOLDER"]

    graphrag_manager = GraphRAGManager(
        root_path=root_path if root_path else "../data",
        folder=folder if folder else "20240903-194043"
    )

    # Load data or setup LLM models during app startup
    local_search_engine = graphrag_manager.create_local_search_engine()
    question_gen_engine = graphrag_manager.create_question_generation_engine()
    global_search_engine = graphrag_manager.create_global_search_engine()

    ENGINES["local_search_engine"] = local_search_engine
    ENGINES["question_gen_engine"] = question_gen_engine
    ENGINES["global_search_engine"] = global_search_engine

    yield  # Application is running

    # Clean up resources, if needed, during shutdown
    ENGINES.clear()


app = FastAPI(
    lifespan=lifespan,
    description="API for GraphRAG",
    title="GraphRAGAPI",
    version="0.1.0",
    contact={
        "name": "GraphRAG",
        "url": "https://microsoft.github.io/graphrag/",
    },
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(main_router)
app.include_router(graph_router)
app.include_router(root_router)

setup_logger()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="debug")
