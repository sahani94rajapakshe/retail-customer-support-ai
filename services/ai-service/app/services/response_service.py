"""Response generation — fine-tuned T5 or template fallback."""

from __future__ import annotations

import torch

from app.services.intent_service import classify_intent
from app.services.model_registry import registry
from app.services.rag_service import build_grounded_answer, search_knowledge


def generate_response(
    text: str,
    intent: str | None = None,
    context: str | None = None,
) -> tuple[str, str, str]:
    if intent is None:
        intent, _, _ = classify_intent(text)

    if registry.response_model is not None:
        prefix = "generate response: "
        input_text = prefix + text
        if context:
            input_text = prefix + f"[{intent}] {context} {text}"

        tokenizer = registry.response_tokenizer
        model = registry.response_model
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=128, num_beams=4, early_stopping=True)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip(), intent, registry.response_mode

    sources = search_knowledge(text, top_k=1)
    if context:
        sources = [{"title": "Context", "content": context, "source_type": "Agent"}] + sources

    if intent == "OrderTracking":
        template = (
            f"Hi, thank you for your message about order tracking. "
            f"{build_grounded_answer(text, sources)} "
            f"If you need further help, please reply with your order number."
        )
    elif intent in ("Refund", "ReturnRequest"):
        template = (
            f"We're sorry to hear you need help with a {intent.replace('Request', '').lower()}. "
            f"{build_grounded_answer(text, sources)}"
        )
    else:
        template = build_grounded_answer(text, sources)

    return template, intent, registry.response_mode
