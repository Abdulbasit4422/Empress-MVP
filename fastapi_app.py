from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import os
import sys

# Add the current directory to the Python path to import rag_pipeline
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_pipeline import (
    chatbot_qa,
    doctor_symptoms_matching,
    affirmation_recommendation,
    product_recommendation
)

# Initialize FastAPI app
app = FastAPI(
    title="Empress RAG API",
    description="A RAG-based API for peri+menopausal healthcare Q&A, doctor matching, affirmations, and product recommendations",
    version="1.0.0"
)

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Pydantic models for request bodies
class QARequest(BaseModel):
    query: str

class DoctorMatchingRequest(BaseModel):
    symptoms: str

class AffirmationRequest(BaseModel):
    categories: List[str]

class ProductRecommendationRequest(BaseModel):
    user_input: str

# Response models
class QAResponse(BaseModel):
    response: str
    retrieved_documents_count: int

class DoctorMatchingResponse(BaseModel):
    response: str
    retrieved_documents_count: int

class AffirmationResponse(BaseModel):
    response: str
    affirmations: List[str]
    retrieved_documents_count: int

class ProductRecommendationResponse(BaseModel):
    response: str
    products: List[str]
    retrieved_documents_count: int

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to the Empress RAG API",
        "endpoints": {
            "/qa": "Q&A Chatbot - answers questions based on the knowledge base",
            "/doctor-matching": "Doctor Symptoms Matching - matches symptoms to doctors",
            "/affirmations": "Affirmation Recommendation - suggests affirmations based on categories",
            "/product-recommendations": "Product Recommendation - recommends products based on user input"
        }
    }

@app.post("/qa", response_model=QAResponse)
async def qa_endpoint(request: QARequest):
    """
    Q&A Chatbot endpoint - answers user questions based on the PDF knowledge base.
    """
    try:
        result = chatbot_qa(request.query)
        return QAResponse(
            response=result["response"],
            retrieved_documents_count=len(result["retrieved_documents"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing Q&A request: {str(e)}")

@app.post("/doctor-matching", response_model=DoctorMatchingResponse)
async def doctor_matching_endpoint(request: DoctorMatchingRequest):
    """
    Doctor Symptoms Matching endpoint - maps patient symptoms to doctors based on the knowledge base.
    """
    try:
        result = doctor_symptoms_matching(request.symptoms)
        return DoctorMatchingResponse(
            response=result["response"],
            retrieved_documents_count=len(result["retrieved_documents"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing doctor matching request: {str(e)}")

@app.post("/affirmations", response_model=AffirmationResponse)
async def affirmations_endpoint(request: AffirmationRequest):
    """
    Affirmation Recommendation endpoint - suggests 3 affirmations at random from the chosen categories.
    """
    try:
        result = affirmation_recommendation(request.categories)
        return AffirmationResponse(
            response=result["response"],
            affirmations=result.get("affirmations", []),
            retrieved_documents_count=len(result["retrieved_documents"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing affirmation request: {str(e)}")

@app.post("/product-recommendations", response_model=ProductRecommendationResponse)
async def product_recommendations_endpoint(request: ProductRecommendationRequest):
    """
    Product Recommendation endpoint - recommends products relevant to user input by querying the knowledge base.
    """
    try:
        result = product_recommendation(request.user_input)
        return ProductRecommendationResponse(
            response=result["response"],
            products=result.get("products", []),
            retrieved_documents_count=len(result["retrieved_documents"])
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing product recommendation request: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "message": "Empress RAG API is running"}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # use Render's port if available
    uvicorn.run(app, host="0.0.0.0", port=port)

