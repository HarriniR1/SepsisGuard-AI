import json
import os
from typing import Literal

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# REQUEST SCHEMAS
# ============================================================


class Demographics(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: str


class Vitals(BaseModel):
    heart_rate: float | None = None
    oxygen_saturation: float | None = None
    temperature_c: float | None = None
    systolic_blood_pressure: float | None = None
    mean_arterial_pressure: float | None = None
    diastolic_blood_pressure: float | None = None
    respiratory_rate: float | None = None


class Labs(BaseModel):
    bun: float | None = None
    creatinine: float | None = None
    glucose: float | None = None
    lactate: float | None = None
    wbc: float | None = None
    platelets: float | None = None
    hematocrit: float | None = None
    hemoglobin: float | None = None
    potassium: float | None = None


class ModelContributor(BaseModel):
    feature: str
    display_name: str | None = None
    value: float | None = None
    contribution: float | None = None
    direction: Literal["increased risk", "decreased risk", "neutral"]


class ModelOutput(BaseModel):
    sepsis_risk_score: float = Field(..., ge=0.0, le=1.0)
    operating_threshold: float = Field(0.35, ge=0.0, le=1.0)
    risk_category: str
    top_contributors: list[ModelContributor] = []


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class SepsisExplanationRequest(BaseModel):
    patient_id: str | None = None
    demographics: Demographics
    vitals: Vitals
    labs: Labs
    model_output: ModelOutput
    question: str = "Why was this patient flagged?"
    clinician_context: str | None = ""
    chat_history: list[ChatMessage] = []
    model_name: str = "llama-3.3-70b-versatile"
    temperature: float = Field(0.35, ge=0.0, le=1.0)
    max_tokens: int = Field(500, ge=64, le=1200)


class SepsisExplanationResponse(BaseModel):
    patient_id: str | None
    sepsis_risk_score: float
    operating_threshold: float
    risk_category: str
    explanation: str
    llm_payload: dict


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="SepsisGuard AI Explanation Backend",
    version="3.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def compact_dict(model_obj: BaseModel) -> dict:
    return {
        key: value
        for key, value in model_obj.model_dump().items()
        if value not in [None, "", [], {}]
    }


def build_llm_payload(req: SepsisExplanationRequest) -> dict:
    """
    Build the structured JSON sent to the LLM.

    Patient ID is intentionally excluded from the external LLM payload.
    """

    return {
        "demographics": compact_dict(req.demographics),
        "available_vitals": compact_dict(req.vitals),
        "available_labs": compact_dict(req.labs),
        "model_output": {
            "sepsis_risk_score": req.model_output.sepsis_risk_score,
            "operating_threshold": req.model_output.operating_threshold,
            "risk_category": req.model_output.risk_category,
            "top_contributors": [
                contributor.model_dump()
                for contributor in req.model_output.top_contributors
            ],
        },
        "clinician_question": req.question,
        "clinician_context": req.clinician_context or "None provided",
    }


def build_messages(req: SepsisExplanationRequest) -> list[dict]:
    patient_context = build_llm_payload(req)

    system_prompt = """
You are SepsisGuard AI, a clinical decision-support explanation assistant.

You are discussing one ICU patient with a clinician. The patient context supplied
below is the sole source of truth for this conversation.

YOUR TASK:
- Answer the clinician's current question directly.
- Use the patient's actual recorded values whenever relevant.
- Connect related findings into a coherent clinical explanation.
- Use prior conversation turns to understand follow-up questions such as
  "that value," "those findings," or "explain it more simply."
- Vary the response structure naturally. Do not repeat the same template for
  every question.
- Keep the answer focused on what was asked.
- Use clinician-friendly language and explain technical concepts plainly.
- Refer to the result as "sepsis propensity" or "risk signal."
- Use the term "AI model" only when necessary.

SAFETY AND GROUNDING:
- Do not diagnose, confirm, or rule out sepsis.
- Do not recommend treatment, medication, testing, procedures, or disposition.
- Do not invent values, symptoms, diagnoses, infection sources, history, or events.
- Clearly distinguish recorded facts from interpretation.
- If requested information is unavailable, say it was not provided.
- Contributors reflect predictive association, not clinical causality.
- Do not reveal hidden reasoning or chain-of-thought.
- Provide only the concise final answer.

For broad summary questions, organize the answer clearly.
For narrow follow-up questions, answer naturally without restating the full case.
""".strip()

    context_message = "Patient context for this conversation:\n\n" + json.dumps(
        patient_context, indent=2
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": context_message},
    ]

    for message in req.chat_history[-10:]:
        messages.append(
            {
                "role": message.role,
                "content": message.content,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": req.question,
        }
    )

    return messages


def call_groq(
    messages: list[dict],
    model_name: str,
    temperature: float,
    max_tokens: int,
) -> str:

    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY is not set in environment variables.",
        )

    response = requests.post(
        GROQ_API_URL,
        headers={
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": messages,
        },
        timeout=45,
    )

    if response.status_code != 200:
        raise HTTPException(
            status_code=response.status_code,
            detail={
                "message": "Groq API call failed",
                "groq_response": response.text,
            },
        )

    data = response.json()

    explanation = data.get("choices", [{}])[0].get("message", {}).get("content")

    if not explanation:
        raise HTTPException(
            status_code=502,
            detail="Groq response did not include explanation content.",
        )

    return explanation.strip()


# ============================================================
# ENDPOINTS
# ============================================================


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "SepsisGuard AI Explanation Backend",
        "version": "3.0.0",
    }


@app.post(
    "/explain-sepsis-risk",
    response_model=SepsisExplanationResponse,
)
def explain_sepsis_risk(req: SepsisExplanationRequest):

    llm_payload = build_llm_payload(req)

    messages = build_messages(req)

    explanation = call_groq(
        messages=messages,
        model_name=req.model_name,
        temperature=req.temperature,
        max_tokens=req.max_tokens,
    )

    return SepsisExplanationResponse(
        patient_id=req.patient_id,
        sepsis_risk_score=req.model_output.sepsis_risk_score,
        operating_threshold=req.model_output.operating_threshold,
        risk_category=req.model_output.risk_category,
        explanation=explanation,
        llm_payload=llm_payload,
    )


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "sepsis_quick_chat_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
