from typing import List, Optional
from pydantic import BaseModel, Field

class CollectorResponse(BaseModel):
    status: str = Field(description="Overall status of the collection")
    total_articles: int = Field(description="Total number of articles collected")
    sources_summary: List[str] = Field(description="Summary of articles per source (e.g., 'reddit: 5')")

class FilterResponse(BaseModel):
    scored_count: int = Field(description="Number of articles scored")
    urgent_ids: List[str] = Field(description="IDs of articles marked as urgent")
    summary: str = Field(description="Brief summary of the filtering results")

class SummarizerResponse(BaseModel):
    summarized_ids: List[str] = Field(description="IDs of articles successfully summarized")
    summary: str = Field(description="Brief summary of the summarization results")

class PublisherResponse(BaseModel):
    pdf_path: Optional[str] = Field(description="Path to the generated PDF newspaper")
    article_count: int = Field(description="Number of articles included in the newspaper")
    summary: str = Field(description="Status of the publishing task")

class MemoryResponse(BaseModel):
    updated: bool = Field(description="Whether the user profile or facts were updated")
    summary: str = Field(description="Description of what was updated")

class SupportResponse(BaseModel):
    answer: str = Field(description="The final answer to the user's question")
    sources_used: List[str] = Field(description="List of article IDs or web URLs used to formulate the answer")
