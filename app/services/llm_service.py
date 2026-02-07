
import os
import time
import json
import logging
from openai import AzureOpenAI
from app.core.config import AOAI_KEY, AOAI_ENDPOINT, AOAI_MODEL, AOAI_API_VER

logger = logging.getLogger(__name__)

# Initialize client if keys function
openai_client = None
if all([AOAI_KEY, AOAI_ENDPOINT, AOAI_MODEL]):
    openai_client = AzureOpenAI(api_key=AOAI_KEY, azure_endpoint=AOAI_ENDPOINT, api_version=AOAI_API_VER)

def llm_call_qa_fields(transcript_text: str) -> dict:
    if not openai_client:
        logger.error("Azure OpenAI client not initialized.")
        return {}

    schema = {
      "type":"object",
      "properties":{
        "conversation_feel":{"type":"string","enum":["Positive","Neutral","Negative"]},
        "agent_improvement_areas":{"type":"array","items":{"type":"string"}},
        "Agent_performance_summary":{"type":"string"},
        "ScriptAdherenceScore":{"type":"integer","minimum":1,"maximum":5},
        "PolitenessProfessionalismScore":{"type":"integer","minimum":1,"maximum":5},
        "ResolutionEffectivenessScore":{"type":"integer","minimum":1,"maximum":5},
        "CsatPrediction":{"type":"number","minimum":1,"maximum":5},
        "CallDisposition":{"type":"string"},
        "FollowUpRequired":{"type":"boolean"},
        "CrossSellUpsellAttempts":{"type":"boolean"},
        "CrossSellUpsellDetails":{"type":"string"}
      },
      "required":["conversation_feel","agent_improvement_areas","Agent_performance_summary",
                  "ScriptAdherenceScore","PolitenessProfessionalismScore",
                  "ResolutionEffectivenessScore","CsatPrediction","CallDisposition",
                  "FollowUpRequired","CrossSellUpsellAttempts","CrossSellUpsellDetails"]
    }
    SYSTEM = (
        "You are a QA analyst for call centers. Score 1-5 (5 best).\n"
        "Definitions:\n"
        "- Script adherence: greeting, verification, resolution, closing.\n"
        "- Resolution effectiveness: whether issue addressed.\n"
        "- Politeness/professionalism: tone, respect, no off-topic remarks.\n"
        "Use only the provided transcript. If unknown, infer conservatively."
    )
    diarization_like = f"[00:00:00–00:00:00] Speaker 1: {transcript_text}"
    prompt = (
        f"Overall sentiment: Neutral. Reason: n/a.\n\n"
        f"Transcript:\n{diarization_like}\n\n"
        "Return strict JSON that matches the schema."
    )
    req = {
        "model": AOAI_MODEL,
        "messages": [
            {"role":"system","content":SYSTEM},
            {"role":"user","content":prompt}
        ],
        "response_format":{"type":"json_schema","json_schema":{"name":"callqa","schema":schema}},
    }
    if not AOAI_MODEL.lower().startswith("o4"):
        try:
            req["temperature"] = float(os.getenv("AOAI_TEMPERATURE","0.2"))
        except ValueError:
            pass
    t0 = time.time()
    try:
        resp = openai_client.chat.completions.create(**req)
        logger.info("LLM completion done in %d ms (model=%s)", int((time.time()-t0)*1000), AOAI_MODEL)
        msg = resp.choices[0].message
        if hasattr(msg, "parsed") and getattr(msg, "parsed"):
            return msg.parsed  # type: ignore[attr-defined]
        return json.loads(msg.content or "{}")
    except Exception as e:
        logger.error("Failed to call LLM or parse JSON: %s", e)
        return {}
