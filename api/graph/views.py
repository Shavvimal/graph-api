from fastapi import HTTPException, Query, Depends, APIRouter
from fastapi.responses import JSONResponse
from api.utils.utils import convert_response_to_string, process_context_data, serialize_search_result
from api.core.auth import verify_token
from api.utils.constants import ENGINES

router = APIRouter(dependencies=[Depends(verify_token)])

@router.get("/search/global")
async def global_search(
    query: str = Query(..., description="Search query for global context")
):
    try:
        result = await ENGINES["global_search_engine"].asearch(query)
        response_dict = {
            "response": convert_response_to_string(result.response),
            "context_data": process_context_data(result.context_data),
            "context_text": result.context_text,
            "completion_time": result.completion_time,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
            "reduce_context_data": process_context_data(result.reduce_context_data),
            "reduce_context_text": result.reduce_context_text,
            "map_responses": [serialize_search_result(result) for result in result.map_responses],
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/local")
async def local_search(
    query: str = Query(..., description="Search query for local context"),
):
    try:
        result = await ENGINES["local_search_engine"].asearch(query)
        response_dict = {
            "response": convert_response_to_string(result.response),
            "context_data": process_context_data(result.context_data),
            "context_text": result.context_text,
            "completion_time": result.completion_time,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,            
        }
        return JSONResponse(content=response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search/question-generation")
async def generate_questions(
    question_history: list[str] = Query(..., description="List of previous user questions"),
):
    try:
        result = await ENGINES["question_gen_engine"].agenerate(
            question_history=question_history, context_data=None, question_count=5
        )
        response_dict = {
            "response": result.response,
            "context_data": process_context_data(result.context_data),
            "completion_time": result.completion_time,
            "llm_calls": result.llm_calls,
            "prompt_tokens": result.prompt_tokens,
        }
        return JSONResponse(response_dict)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))