"""Bootstrap factory for Earnings Call Application Service."""
from application.earnings_call.ports import (
    EarningsCallLlmPort,
    EarningsCallNoteWriterPort,
    EarningsCallKanbanPort,
    EarningsCallWorkflowPort,
)
from application.earnings_call.service import EarningsCallApplicationService


def build_earnings_call_service(
    llm_port: EarningsCallLlmPort,
    writer_port: EarningsCallNoteWriterPort,
    workflow_port: EarningsCallWorkflowPort,
    kanban_port: EarningsCallKanbanPort,
    prompt_version: str = "v1",
    max_attempts: int = 5,
) -> EarningsCallApplicationService:
    """Constructs EarningsCallApplicationService with injected outbound ports and configuration."""
    return EarningsCallApplicationService(
        llm_port=llm_port,
        writer_port=writer_port,
        workflow_port=workflow_port,
        kanban_port=kanban_port,
        prompt_version=prompt_version,
        max_attempts=max_attempts,
    )
