"""
ChatBot for Canadian Privacy Compliance - combines PHI detection, RAG, and LLM.
"""

import re
import logging
from typing import Dict, Optional, List

from .phi_detector import PHIDetector
from .rag_system import RAGSystem
from .ollama_client import OllamaClient
from .prompt_templates import (
    SYSTEM_PROMPT,
    build_detection_prompt,
    build_policy_question_prompt,
    build_complete_prompt
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatBot:
    """
    Canadian Privacy Compliance ChatBot.

    Combines PHI detection, RAG-based policy retrieval, and LLM generation
    to provide privacy compliance assistance.
    """

    def __init__(
        self,
        model: str = "gemma2:2b",
        ollama_base_url: str = "http://localhost:11434",
        rag_collection: str = "privacy_policies",
        rag_persist_dir: str = "./chroma_db"
    ):
        """
        Initialize the ChatBot with all required components.

        Args:
            model: Ollama model to use
            ollama_base_url: Base URL for Ollama API
            rag_collection: ChromaDB collection name
            rag_persist_dir: Directory for ChromaDB persistence
        """
        try:
            logger.info("Initializing ChatBot...")

            # Initialize PHI Detector
            logger.info("Loading PHI Detector...")
            self.phi_detector = PHIDetector()

            # Initialize RAG System
            logger.info("Loading RAG System...")
            self.rag = RAGSystem(
                collection_name=rag_collection,
                persist_directory=rag_persist_dir
            )

            # Initialize Ollama Client
            logger.info("Loading Ollama Client...")
            self.llm = OllamaClient(
                model=model,
                base_url=ollama_base_url
            )

            # Verify Ollama is running
            if not self.llm.check_health():
                logger.warning(
                    "Ollama health check failed. The chatbot will work for detection "
                    "but LLM-enhanced responses will not be available."
                )

            logger.info("ChatBot initialized successfully!")

        except Exception as e:
            logger.error(f"Failed to initialize ChatBot: {e}")
            raise

    def _is_ticket(self, message: str) -> bool:
        """
        Determine if a message looks like a support ticket.

        A message is considered a ticket if it:
        - Contains the word "ticket" (case-insensitive)
        - Is multi-line (has newlines)
        - Contains PHI patterns (names, emails, phones, etc.)
        - Is longer than typical questions (>100 chars)

        Args:
            message: Input message text

        Returns:
            True if message appears to be a ticket, False otherwise
        """
        try:
            if not message or not message.strip():
                return False

            message_lower = message.lower()

            # Check for explicit ticket keywords
            ticket_keywords = ['ticket', 'incident', 'case', 'issue #', 'ticket #']
            has_ticket_keyword = any(keyword in message_lower for keyword in ticket_keywords)

            # Check if multi-line
            has_multiple_lines = '\n' in message.strip()

            # Check for PHI patterns (quick check without full detection)
            # Email pattern
            has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message))

            # Phone pattern
            has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message))

            # SSN/SIN pattern (includes Canadian SIN format)
            has_ssn = bool(re.search(r'\b\d{3}-\d{2,3}-\d{3,4}\b', message))

            # Account number pattern
            has_account = bool(re.search(r'\b(?:account|acct|id|patient)\s*#?\s*\d{5,}\b', message_lower))

            # Count PHI indicators
            phi_count = sum([has_email, has_phone, has_ssn, has_account])
            has_phi_patterns = phi_count > 0

            # Check length
            is_long = len(message) > 100

            # Context clues that suggest a ticket
            ticket_context_words = ['user', 'customer', 'patient', 'client', 'reports', 'issue', 'problem', 'request']
            has_context = any(word in message_lower for word in ticket_context_words)

            # Decision logic:
            # If it has "ticket" keyword, it's likely a ticket
            if has_ticket_keyword:
                return True

            # If it's multi-line AND has PHI patterns, likely a ticket
            if has_multiple_lines and has_phi_patterns:
                return True

            # If it's long AND has PHI patterns, likely a ticket
            if is_long and has_phi_patterns:
                return True

            # If it has multiple PHI indicators (2+), it's likely a ticket
            if phi_count >= 2:
                return True

            # If it has PHI AND ticket context words, likely a ticket
            if has_phi_patterns and has_context:
                return True

            # Otherwise, treat as a question
            return False

        except Exception as e:
            logger.error(f"Error in _is_ticket: {e}")
            # Default to treating as question if check fails
            return False

    def _assess_risk_level(self, detections: List[Dict]) -> str:
        """
        Assess overall risk level based on detected PHI types.

        Args:
            detections: List of PHI detections

        Returns:
            Risk level: LOW, MEDIUM, HIGH, or CRITICAL
        """
        if not detections:
            return "LOW"

        # High-risk PHI types
        critical_types = {
            'SSN', 'MEDICAL_RECORD_NUMBER', 'HEALTH_PLAN_NUMBER',
            'DIAGNOSIS_CODE', 'PRESCRIPTION'
        }

        high_risk_types = {
            'CREDIT_CARD', 'BANK_ACCOUNT', 'DATE_OF_BIRTH',
            'BIOMETRIC', 'FULL_FACE_PHOTO'
        }

        medium_risk_types = {
            'PERSON_NAME', 'PHONE_NUMBER', 'EMAIL',
            'IP_ADDRESS', 'VEHICLE_ID'
        }

        detected_types = {d['type'] for d in detections}

        # Check for critical PHI
        if detected_types & critical_types:
            return "CRITICAL"

        # Check for high-risk PHI
        if detected_types & high_risk_types:
            return "HIGH"

        # Check for medium-risk PHI
        if detected_types & medium_risk_types:
            # If many medium-risk items, escalate to HIGH
            if len(detections) > 5:
                return "HIGH"
            return "MEDIUM"

        # Default to LOW
        return "LOW"

    def _build_detection_prompt(
        self,
        ticket: str,
        detections: List[Dict],
        risk: str
    ) -> str:
        """
        Build a detection analysis prompt using the DETECTION_PROMPT template.

        Args:
            ticket: Support ticket text
            detections: List of PHI detections
            risk: Risk level (LOW, MEDIUM, HIGH, CRITICAL)

        Returns:
            Formatted prompt for LLM
        """
        try:
            # Use the imported prompt builder
            user_prompt = build_detection_prompt(
                ticket_text=ticket,
                detections=detections,
                risk_level=risk
            )

            # Combine with system prompt
            full_prompt = build_complete_prompt(SYSTEM_PROMPT, user_prompt)

            return full_prompt

        except Exception as e:
            logger.error(f"Error building detection prompt: {e}")
            raise

    def _build_policy_prompt(self, question: str, rag_results: List) -> str:
        """
        Build a policy question prompt using the POLICY_QUESTION_PROMPT template.

        Args:
            question: User's question about privacy policies
            rag_results: Retrieved context chunks from RAG system (list of tuples)

        Returns:
            Formatted prompt for LLM
        """
        try:
            # Use the imported prompt builder
            user_prompt = build_policy_question_prompt(
                question=question,
                context_chunks=rag_results
            )

            # Combine with system prompt
            full_prompt = build_complete_prompt(SYSTEM_PROMPT, user_prompt)

            return full_prompt

        except Exception as e:
            logger.error(f"Error building policy prompt: {e}")
            raise

    def analyze_ticket(self, text: str) -> str:
        """
        Analyze a support ticket for PHI and privacy risks.

        Args:
            text: Support ticket text

        Returns:
            LLM-generated analysis of the ticket with risk assessment
        """
        try:
            logger.info("Analyzing ticket for PHI...")

            # Detect PHI using analyze() method
            result = self.phi_detector.analyze(text)
            detections = result.get('detections', [])
            logger.info(f"Found {len(detections)} PHI detections")

            # Assess risk level
            risk_level = self._assess_risk_level(detections)
            logger.info(f"Risk level: {risk_level}")

            # Build detection prompt using helper method
            full_prompt = self._build_detection_prompt(
                ticket=text,
                detections=detections,
                risk=risk_level
            )

            # Generate response with LLM
            logger.info("Generating LLM response...")
            response = self.llm.generate(
                prompt=full_prompt,
                max_tokens=1500,
                temperature=0.3  # Lower temperature for more consistent analysis
            )

            if response:
                return response
            else:
                # Fallback if LLM fails
                logger.warning("LLM generation failed, returning basic analysis")
                return self._format_basic_detection(detections, risk_level)

        except Exception as e:
            logger.error(f"Error analyzing ticket: {e}")
            return f"Error analyzing ticket: {str(e)}"

    def answer_question(self, question: str, top_k: int = 3) -> str:
        """
        Answer a privacy policy question using RAG and LLM.

        Args:
            question: User's question about privacy policies
            top_k: Number of context chunks to retrieve

        Returns:
            LLM-generated answer with citations
        """
        try:
            logger.info(f"Answering question: {question}")

            # Query RAG system for relevant context
            logger.info("Retrieving relevant policy context...")
            context_chunks = self.rag.query(question, top_k=top_k)

            if not context_chunks:
                return (
                    "I don't have enough policy information to answer that question. "
                    "Please ensure policy documents have been loaded into the system."
                )

            logger.info(f"Retrieved {len(context_chunks)} relevant chunks")

            # Build policy question prompt using helper method
            full_prompt = self._build_policy_prompt(
                question=question,
                rag_results=context_chunks
            )

            # Generate response with LLM
            logger.info("Generating LLM response...")
            response = self.llm.generate(
                prompt=full_prompt,
                max_tokens=1500,
                temperature=0.4  # Slightly higher for more natural language
            )

            if response:
                return response
            else:
                # Fallback if LLM fails
                logger.warning("LLM generation failed, returning context directly")
                return self._format_basic_answer(question, context_chunks)

        except ValueError as e:
            # Handle empty collection
            logger.error(f"RAG error: {e}")
            return (
                "The policy knowledge base is not yet loaded. "
                "Please load policy documents before asking questions."
            )

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"Error processing question: {str(e)}"

    def chat(self, message: str) -> str:
        """
        Main chat interface - routes messages to appropriate handler.

        Args:
            message: User's input message

        Returns:
            ChatBot response (either ticket analysis or question answer)
        """
        try:
            if not message or not message.strip():
                return "Please provide a message or question."

            logger.info(f"Processing message (length: {len(message)} chars)")

            # Determine message type and route
            if self._is_ticket(message):
                logger.info("Message identified as ticket")
                return self.analyze_ticket(message)
            else:
                logger.info("Message identified as question")
                return self.answer_question(message)

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return f"An error occurred: {str(e)}"

    def _format_basic_detection(self, detections: List[Dict], risk_level: str) -> str:
        """
        Fallback formatter for detections when LLM is unavailable.

        Args:
            detections: List of PHI detections
            risk_level: Overall risk level

        Returns:
            Formatted detection report
        """
        if not detections:
            return "No personal information detected. ✅"

        risk_indicators = {
            "LOW": "🟢",
            "MEDIUM": "🟡",
            "HIGH": "🟠",
            "CRITICAL": "🔴"
        }

        indicator = risk_indicators.get(risk_level, "⚪")

        output = [
            f"\n{indicator} **Risk Level: {risk_level}**\n",
            "**Detected Personal Information:**\n"
        ]

        # Group by type
        by_type = {}
        for det in detections:
            det_type = det['type']
            if det_type not in by_type:
                by_type[det_type] = []
            by_type[det_type].append(det['value'])

        for det_type, values in by_type.items():
            output.append(f"- {det_type}: {len(values)} occurrence(s)")

        output.append("\n**Recommendation:** Review and handle this information according to privacy policies.")

        return "\n".join(output)

    def _format_basic_answer(self, question: str, context_chunks: List) -> str:
        """
        Fallback formatter for answers when LLM is unavailable.

        Args:
            question: Original question
            context_chunks: Retrieved context

        Returns:
            Formatted answer with context
        """
        output = [
            f"**Question:** {question}\n",
            "**Relevant Policy Information:**\n"
        ]

        for i, (text, metadata, score) in enumerate(context_chunks, 1):
            source = metadata.get('source', 'Unknown')
            source_name = source.split('/')[-1] if '/' in source else source

            output.append(f"\n**[{i}] From {source_name}:**")
            output.append(text.strip())

        return "\n".join(output)


if __name__ == "__main__":
    # Example usage
    try:
        # Initialize chatbot
        print("Initializing ChatBot...")
        chatbot = ChatBot()

        # Example ticket
        print("\n" + "="*80)
        print("EXAMPLE 1: Ticket Analysis")
        print("="*80)

        ticket = """Ticket #12345
Customer John Doe called regarding account access.
Phone: 604-555-0123
Email: john.doe@example.com
Date of Birth: 01/15/1985

Issue: Unable to access medical records online."""

        response = chatbot.chat(ticket)
        print(response)

        # Example question
        print("\n" + "="*80)
        print("EXAMPLE 2: Policy Question")
        print("="*80)

        question = "What are the consent requirements for collecting personal information?"
        response = chatbot.chat(question)
        print(response)

    except Exception as e:
        logger.error(f"Error in main: {e}")
