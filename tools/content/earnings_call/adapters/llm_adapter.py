"""LLM Driven Adapter for Earnings Call Summarization."""
from functools import lru_cache
import httpx

from application.earnings_call.errors import EarningsCallProviderUnavailableError
from core.llm_factory import get_llm, detect_provider
from core.logger import get_logger
from core.model_registry import get_model_name
from core.security import anonymize_pii

log = get_logger(__name__)

EARNINGS_CALL_SYSTEM_PROMPT = """คุณคือนักวิเคราะห์ปัจจัยพื้นฐานชั้นนำ (Senior Equity Research Analyst)
ช่วยสรุป Earnings Call Transcript ของหุ้นที่ระบุอย่างกระชับ เจาะลึก และเน้นประเด็นสำคัญที่มีผลต่อมูลค่าบริษัท

กรุณาสรุปโดยจัดโครงสร้างเป็น 5 หัวข้อหลักดังนี้ (ในรูปแบบ Markdown):

### 1. 📊 Key Financial Highlights & Operational Metrics
- ผลประกอบการเปรียบเทียบกับความคาดหวังของตลาด (Beat / Miss / In-line ในแง่รายได้และกำไร)
- รายได้, อัตรากำไร (Gross/Operating/Net Margins), EPS และตัวเลขชี้วัดเฉพาะธุรกิจ (เช่น Demand trends, Segment breakdown)

### 2. 🔮 Forward Guidance & Strategic Outlook
- เป้าหมายหรือประมาณการไตรมาสถัดไปและทั้งปี (Guidance Raise / Maintain / Lower)
- แผนการลงทุน (CapEx), การขยายกำลังการผลิต หรือทิศทางผลิตภัณฑ์ใหม่

### 3. 🎙️ Analyst Q&A Key Takeaways
- ประเด็นสำคัญที่นักวิเคราะห์ซักถามมากที่สุด และคำชี้แจงของผู้บริหาร
- จุดที่น่าสังเกตในการตอบคำถาม

### 4. 🚩 Risks, Headwinds & Bottlenecks
- ความเสี่ยงและอุปสรรคทั้งระยะสั้นและระยะยาว (เช่น สภาวะเศรษฐกิจมหภาค, Supply Chain, ต้นทุน, การแข่งขัน)

### 5. 🎯 Executive Tone & Strategic Sentiment
- น้ำเสียงและทัศนคติของผู้บริหาร (Bullish / Cautious / Defensive) พร้อมเหตุผลประกอบสั้นๆ
"""


@lru_cache(maxsize=2)
def _get_earnings_call_llm(model_name: str):
    """Cache LLM client + retry wrapper per model_name."""
    provider = detect_provider(model_name)
    return get_llm(provider=provider, model_name=model_name).with_retry(
        retry_if_exception_type=(
            httpx.TimeoutException,
            httpx.ConnectError,
            httpx.RemoteProtocolError,
            TimeoutError,
            ConnectionError,
        ),
        wait_exponential_jitter=True,
        stop_after_attempt=3,
    )


class LlmEarningsCallSummarizerAdapter:
    """Implements EarningsCallLlmPort using configured LLM from model registry."""

    def __init__(self, slot_key: str = "earnings_call_summarizer") -> None:
        self._slot_key = slot_key

    def summarize(self, ticker: str, period: str, transcript: str) -> str:
        model_name = get_model_name(self._slot_key)
        anonymized_transcript, _ = anonymize_pii(transcript)

        log.info(
            "LLM Call | purpose=earnings_call_summarize | ticker=%s | period=%s | model=%s",
            ticker,
            period,
            model_name,
        )

        try:
            llm = _get_earnings_call_llm(model_name)
            response = llm.invoke(
                [
                    {"role": "system", "content": EARNINGS_CALL_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": f"Company Ticker: {ticker}\nEarnings Period: {period}\n\nFull Transcript:\n{anonymized_transcript}",
                    },
                ]
            )

            content = response.content
            if isinstance(content, list):
                content = (
                    content[0].get("text", "")
                    if len(content) > 0 and isinstance(content[0], dict)
                    else str(content)
                )
            return str(content).strip()
        except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError, TimeoutError, ConnectionError) as exc:
            log.warning("LLM provider unavailable during earnings call summarization: %s", exc)
            raise EarningsCallProviderUnavailableError(f"LLM provider is currently unavailable or timed out: {exc}") from exc
        except Exception as exc:
            # Check for rate limit or provider error strings in LangChain wrappers
            err_msg = str(exc).lower()
            if "rate limit" in err_msg or "resource exhausted" in err_msg or "503" in err_msg or "timeout" in err_msg:
                log.warning("LLM provider error mapped to unavailable: %s", exc)
                raise EarningsCallProviderUnavailableError(f"LLM provider error: {exc}") from exc
            raise
