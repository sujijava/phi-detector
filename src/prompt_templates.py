"""
Prompt templates for the Canadian Privacy Compliance Assistant.
"""

# System prompt - base instructions for all interactions
SYSTEM_PROMPT = """You are a Canadian Privacy Compliance Assistant specializing in personal information protection.

Your expertise includes:
- Canadian privacy legislation (PIPEDA, provincial privacy acts)
- Protected Health Information (PHI) identification and handling
- Personal Information Protection and Electronic Documents Act (PIPEDA) compliance
- Provincial privacy laws (BC PIPA, Alberta PIPA, Quebec Law 25, etc.)
- Privacy best practices for Canadian organizations

Guidelines:
- Provide accurate, professional advice on Canadian privacy matters
- Reference specific legislation and regulations when applicable
- Explain privacy concepts clearly and concisely
- Focus on practical compliance steps
- Always prioritize protecting individual privacy rights
- Cite sources when providing regulatory guidance

Maintain a professional, helpful tone while ensuring privacy protection remains the top priority."""


# Detection prompt - format PHI detection results with risk assessment
DETECTION_PROMPT = """You are analyzing a support ticket for personal information and privacy risks.

**Ticket Content:**
{ticket_text}

**Detected Personal Information:**
{detections}

**Overall Risk Level:** {risk_level}

**Your Task:**
Provide a professional analysis of the detected personal information including:

1. **Summary of Findings**
   - List each type of personal information detected
   - Explain the privacy implications

2. **Risk Assessment**
   - Evaluate the sensitivity of the information
   - Identify potential compliance concerns
   - Highlight any high-risk elements

3. **Recommended Actions**
   - Suggest immediate steps to protect the information
   - Recommend handling procedures
   - Advise on compliance requirements

4. **Risk Indicators:**
   Use these indicators for risk levels:
   - 🟢 LOW: Common business information with minimal privacy risk
   - 🟡 MEDIUM: Personal information requiring standard protection measures
   - 🟠 HIGH: Sensitive personal information requiring enhanced security
   - 🔴 CRITICAL: Highly sensitive information (health, financial) requiring maximum protection

Format your response clearly with appropriate risk indicators. Be specific and actionable."""


# Policy question prompt - answer using RAG context
POLICY_QUESTION_PROMPT = """You are a Canadian Privacy Compliance Assistant answering a question about privacy regulations and policies.

**Question:**
{question}

**Relevant Policy Context:**
{retrieved_context}

**Your Task:**
Provide a comprehensive answer to the question based on the retrieved policy context.

**Guidelines:**
1. **Answer Structure**
   - Start with a direct answer to the question
   - Provide relevant details and context
   - Include practical implications or examples if helpful

2. **Citations**
   - Reference the specific policy sections or regulations
   - Indicate which documents or acts your answer is based on
   - Use clear citations like [Source: <document name>]

3. **Canadian Privacy Focus**
   - Prioritize Canadian privacy legislation (PIPEDA, provincial acts)
   - Explain requirements in the Canadian regulatory context
   - Distinguish between federal and provincial requirements when relevant

4. **Clarity and Accuracy**
   - Use clear, professional language
   - Avoid ambiguity in regulatory interpretations
   - If the context doesn't fully answer the question, acknowledge limitations
   - Recommend consulting legal counsel for specific compliance situations

5. **Practical Guidance**
   - Explain how the policy applies in practice
   - Provide actionable compliance steps when relevant
   - Highlight key obligations or requirements

Format your answer professionally with clear structure and citations."""


# Risk level formatting helper
def format_risk_indicator(risk_level: str) -> str:
    """
    Get the appropriate emoji indicator for a risk level.

    Args:
        risk_level: Risk level (LOW, MEDIUM, HIGH, CRITICAL)

    Returns:
        Formatted risk indicator with emoji
    """
    risk_indicators = {
        "LOW": "🟢 LOW RISK",
        "MEDIUM": "🟡 MEDIUM RISK",
        "HIGH": "🟠 HIGH RISK",
        "CRITICAL": "🔴 CRITICAL RISK"
    }

    return risk_indicators.get(risk_level.upper(), "⚪ UNKNOWN RISK")


# Detection formatting helper
def format_detections(detections: list) -> str:
    """
    Format detection results into a readable string.

    Args:
        detections: List of detection dictionaries with 'type', 'value', 'start', 'end'

    Returns:
        Formatted string of detections
    """
    if not detections:
        return "No personal information detected."

    formatted = []

    # Group detections by type
    detections_by_type = {}
    for detection in detections:
        det_type = detection.get('type', 'UNKNOWN')
        if det_type not in detections_by_type:
            detections_by_type[det_type] = []
        detections_by_type[det_type].append(detection)

    # Format each type
    for det_type, items in detections_by_type.items():
        count = len(items)
        examples = [item.get('value', 'N/A') for item in items[:3]]  # Show max 3 examples

        formatted.append(f"- **{det_type}** ({count} occurrence{'s' if count > 1 else ''})")
        for example in examples:
            formatted.append(f"  - {example}")

        if count > 3:
            formatted.append(f"  - ... and {count - 3} more")

    return "\n".join(formatted)


# Context formatting helper
def format_rag_context(context_chunks: list) -> str:
    """
    Format RAG context chunks into a readable string.

    Args:
        context_chunks: List of tuples (text, metadata, score)

    Returns:
        Formatted context string
    """
    if not context_chunks:
        return "No relevant policy context found."

    formatted = []

    for i, (text, metadata, score) in enumerate(context_chunks, 1):
        source = metadata.get('source', 'Unknown source')
        # Extract filename from path
        source_name = source.split('/')[-1] if '/' in source else source

        formatted.append(f"[Context {i}] Source: {source_name}")
        formatted.append(text.strip())
        formatted.append("")  # Empty line between contexts

    return "\n".join(formatted)


# Complete prompt builders
def build_detection_prompt(ticket_text: str, detections: list, risk_level: str) -> str:
    """
    Build complete detection analysis prompt.

    Args:
        ticket_text: The support ticket text
        detections: List of detected personal information
        risk_level: Overall risk level (LOW, MEDIUM, HIGH, CRITICAL)

    Returns:
        Complete formatted prompt
    """
    formatted_detections = format_detections(detections)
    risk_indicator = format_risk_indicator(risk_level)

    return DETECTION_PROMPT.format(
        ticket_text=ticket_text,
        detections=formatted_detections,
        risk_level=risk_indicator
    )


def build_policy_question_prompt(question: str, context_chunks: list) -> str:
    """
    Build complete policy question prompt with RAG context.

    Args:
        question: User's question about privacy policy
        context_chunks: Retrieved context from RAG system (list of tuples)

    Returns:
        Complete formatted prompt
    """
    formatted_context = format_rag_context(context_chunks)

    return POLICY_QUESTION_PROMPT.format(
        question=question,
        retrieved_context=formatted_context
    )


def build_complete_prompt(system_prompt: str, user_prompt: str) -> str:
    """
    Combine system and user prompts for LLM.

    Args:
        system_prompt: System-level instructions
        user_prompt: User's specific request/question

    Returns:
        Complete prompt for LLM
    """
    return f"{system_prompt}\n\n---\n\n{user_prompt}"


if __name__ == "__main__":
    # Example usage
    print("=== SYSTEM PROMPT ===")
    print(SYSTEM_PROMPT)
    print("\n" + "="*80 + "\n")

    # Example detection
    sample_detections = [
        {"type": "PERSON_NAME", "value": "John Doe", "start": 0, "end": 8},
        {"type": "PHONE_NUMBER", "value": "604-555-0123", "start": 20, "end": 32},
        {"type": "EMAIL", "value": "john@example.com", "start": 40, "end": 56}
    ]

    sample_ticket = "Customer John Doe called from 604-555-0123 regarding his account. Email: john@example.com"

    print("=== DETECTION ANALYSIS PROMPT ===")
    detection_prompt = build_detection_prompt(
        ticket_text=sample_ticket,
        detections=sample_detections,
        risk_level="MEDIUM"
    )
    print(detection_prompt)
    print("\n" + "="*80 + "\n")

    # Example policy question
    sample_context = [
        (
            "Personal information must be collected with consent and used only for disclosed purposes.",
            {"source": "data/policies/pipeda.txt", "chunk_id": "5"},
            0.85
        ),
        (
            "Organizations must protect personal information with appropriate security safeguards.",
            {"source": "data/policies/pipeda.txt", "chunk_id": "12"},
            0.78
        )
    ]

    print("=== POLICY QUESTION PROMPT ===")
    policy_prompt = build_policy_question_prompt(
        question="What are the consent requirements for collecting personal information?",
        context_chunks=sample_context
    )
    print(policy_prompt)
