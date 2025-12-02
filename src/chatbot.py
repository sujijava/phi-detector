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

    def _classify_message(self, message: str) -> str:
        """
        Classify message into one of three types:
        - 'pii_ticket': Support ticket with actual PII to detect
        - 'dev_ticket': Development/PM ticket about requirements and compliance
        - 'question': General policy question

        Args:
            message: Input message text

        Returns:
            Message type: 'pii_ticket', 'dev_ticket', or 'question'
        """
        try:
            if not message or not message.strip():
                return 'question'

            message_lower = message.lower()

            # Check for actual PII patterns (real user data)
            has_email = bool(re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', message))
            has_phone = bool(re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', message))
            has_ssn = bool(re.search(r'\b\d{3}-\d{2,3}-\d{3,4}\b', message))
            has_account = bool(re.search(r'\b(?:account|acct|patient)\s*#?\s*:?\s*[A-Z0-9]{5,}\b', message, re.IGNORECASE))
            has_dob = bool(re.search(r'\b(?:dob|date of birth|born)[\s:]+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', message_lower))

            phi_count = sum([has_email, has_phone, has_ssn, has_account, has_dob])

            # If has 2+ actual PII items, it's a support ticket with PII
            if phi_count >= 2:
                logger.info("Classified as: PII Support Ticket")
                return 'pii_ticket'

            # Check for development/PM ticket indicators
            dev_indicators = [
                'pm ', 'product manager', 'project manager',
                'developing', 'building', 'creating', 'implementing',
                'feature', 'functionality', 'system', 'application',
                'storing', 'store', 'database', 'save',
                'encryption', 'encrypt', 'security measure',
                'technology', 'technical', 'architecture',
                'requirement', 'compliance', 'regulation',
                'chat message', 'messaging system', 'communication',
                'between user', 'between patient', 'between client'
            ]

            has_dev_indicators = sum(1 for ind in dev_indicators if ind in message_lower)

            # Check for question words (common in dev tickets)
            question_words = ['should', 'need', 'require', 'must', 'recommend', 'what', 'how']
            has_questions = any(qw in message_lower for qw in question_words)

            # If it has dev indicators AND questions, it's a dev ticket
            if has_dev_indicators >= 2 and has_questions:
                logger.info("Classified as: Development/Requirements Ticket")
                return 'dev_ticket'

            # Default to general question
            logger.info("Classified as: General Policy Question")
            return 'question'

        except Exception as e:
            logger.error(f"Error in _classify_message: {e}")
            return 'question'

    def answer_dev_ticket(self, ticket_text: str, top_k: int = 3, return_prompt: bool = False):
        """
        Answer a development/PM ticket about technical requirements and compliance.

        Args:
            ticket_text: Development ticket text
            top_k: Number of context chunks to retrieve
            return_prompt: If True, return tuple with prompt and metadata

        Returns:
            LLM-generated answer with policy guidance and technical recommendations
        """
        try:
            logger.info("Processing development ticket...")

            # Query RAG system for relevant compliance context
            logger.info("Retrieving relevant policy context...")
            context_chunks = self.rag.query(ticket_text, top_k=top_k)

            if not context_chunks:
                msg = (
                    "I don't have enough policy information to provide guidance. "
                    "Please ensure policy documents have been loaded into the system."
                )
                if return_prompt:
                    return msg, "", []
                return msg

            logger.info(f"Retrieved {len(context_chunks)} relevant chunks")

            # Build enhanced prompt for development requirements
            context_text = "\n\n".join([
                f"[Source: {metadata.get('source', 'Unknown').split('/')[-1]}]\n{text}"
                for text, metadata, score in context_chunks
            ])

            full_prompt = f"""You are a privacy compliance advisor helping development teams.

DEVELOPMENT TICKET:
{ticket_text}

RELEVANT PRIVACY REGULATIONS AND POLICIES:
{context_text}

Please provide:
1. **Compliance Requirements**: What privacy regulations apply?
2. **Required Security Measures**: What technical safeguards are required (encryption, access controls, etc.)?
3. **Technology Recommendations**: Specific technologies or approaches recommended for compliance
4. **Risk Assessment**: What privacy risks need to be addressed?
5. **Best Practices**: Additional recommendations for implementation

Be specific and actionable in your recommendations."""

            # Generate response with LLM
            logger.info("Generating LLM response...")
            response = self.llm.generate(
                prompt=full_prompt,
                max_tokens=600,  # More tokens for comprehensive guidance
                temperature=0.4
            )

            if response:
                if return_prompt:
                    return response, full_prompt, context_chunks
                return response
            else:
                # Fallback
                logger.warning("LLM generation failed, returning context directly")
                fallback = self._format_basic_answer(ticket_text, context_chunks)
                if return_prompt:
                    return fallback, full_prompt, context_chunks
                return fallback

        except ValueError as e:
            logger.error(f"RAG error: {e}")
            error_msg = (
                "The policy knowledge base is not yet loaded. "
                "Please load policy documents before processing development tickets."
            )
            if return_prompt:
                return error_msg, "", []
            return error_msg

        except Exception as e:
            logger.error(f"Error processing development ticket: {e}")
            error_msg = f"Error processing development ticket: {str(e)}"
            if return_prompt:
                return error_msg, "", []
            return error_msg

    def _is_ticket(self, message: str) -> bool:
        """
        Determine if a message looks like a support ticket (backward compatibility).
        Now uses the new classification system internally.

        Args:
            message: Input message text

        Returns:
            True if message is any type of ticket, False if it's a question
        """
        msg_type = self._classify_message(message)
        return msg_type in ['pii_ticket', 'dev_ticket']

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

    def analyze_ticket(self, text: str, return_prompt: bool = False):
        """
        Analyze a support ticket for PHI and privacy risks using RAG-grounded responses.

        Args:
            text: Support ticket text
            return_prompt: If True, return tuple of (response, prompt, detections, risk_level, context_chunks)

        Returns:
            LLM-generated analysis of the ticket with risk assessment based on policy documents
            Or tuple if return_prompt is True
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

            # Build RAG query based on detected PII types and risk level
            detected_types = set(d['type'] for d in detections)
            rag_query = f"How should we handle and protect {', '.join(detected_types)} information? What are the security requirements and privacy obligations for {risk_level} risk personal information?"

            # Retrieve relevant policy context using RAG
            logger.info("Retrieving relevant policy guidance from knowledge base...")
            try:
                context_chunks = self.rag.query(rag_query, top_k=3)
            except Exception as rag_error:
                logger.warning(f"RAG query failed: {rag_error}, continuing without policy context")
                context_chunks = []

            # Build enhanced prompt with policy context
            if context_chunks:
                context_text = "\n\n".join([
                    f"[Policy Source: {metadata.get('source', 'Unknown').split('/')[-1]}]\n{text}"
                    for text, metadata, score in context_chunks
                ])

                full_prompt = f"""You are a privacy compliance advisor. Analyze this support ticket for PII and provide guidance based on Canadian privacy regulations.

SUPPORT TICKET:
{text}

DETECTED PII:
{self._format_detections_for_prompt(detections)}

RISK LEVEL: {risk_level}

RELEVANT PRIVACY REGULATIONS:
{context_text}

Based on the detected PII and the privacy regulations above, provide:
1. **Summary of Detected PII**: List what was found
2. **Risk Assessment**: Explain the privacy risks
3. **Handling Requirements**: What regulations apply and what's required
4. **Recommended Actions**: Specific steps to protect this information
5. **Compliance Notes**: Any additional compliance considerations

Be specific and cite the relevant regulations."""
            else:
                # Fallback if RAG fails
                full_prompt = self._build_detection_prompt(
                    ticket=text,
                    detections=detections,
                    risk=risk_level
                )

            # Generate response with LLM
            logger.info("Generating LLM response...")
            response = self.llm.generate(
                prompt=full_prompt,
                max_tokens=600,  # Increased for more comprehensive guidance
                temperature=0.3  # Lower temperature for more consistent analysis
            )

            if response:
                if return_prompt:
                    return response, full_prompt, detections, risk_level, context_chunks
                return response
            else:
                # Fallback if LLM fails
                logger.warning("LLM generation failed, returning basic analysis")
                fallback = self._format_basic_detection(detections, risk_level)
                if return_prompt:
                    return fallback, full_prompt, detections, risk_level, context_chunks
                return fallback

        except Exception as e:
            logger.error(f"Error analyzing ticket: {e}")
            error_msg = f"Error analyzing ticket: {str(e)}"
            if return_prompt:
                return error_msg, "", [], "LOW", []
            return error_msg

    def _format_detections_for_prompt(self, detections: List[Dict]) -> str:
        """Format detections for inclusion in prompt."""
        if not detections:
            return "No PII detected"

        lines = []
        for det in detections:
            lines.append(f"- {det['type']}: {det['value']}")
        return "\n".join(lines)

    def answer_question(self, question: str, top_k: int = 2, return_prompt: bool = False):
        """
        Answer a privacy policy question using RAG and LLM.

        Args:
            question: User's question about privacy policies
            top_k: Number of context chunks to retrieve
            return_prompt: If True, return tuple of (response, prompt, context_chunks)

        Returns:
            LLM-generated answer with citations
            Or tuple if return_prompt is True
        """
        try:
            logger.info(f"Answering question: {question}")

            # Query RAG system for relevant context
            logger.info("Retrieving relevant policy context...")
            context_chunks = self.rag.query(question, top_k=top_k)

            if not context_chunks:
                msg = (
                    "I don't have enough policy information to answer that question. "
                    "Please ensure policy documents have been loaded into the system."
                )
                if return_prompt:
                    return msg, "", []
                return msg

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
                max_tokens=400,  # Reduced for faster response
                temperature=0.4  # Slightly higher for more natural language
            )

            if response:
                if return_prompt:
                    return response, full_prompt, context_chunks
                return response
            else:
                # Fallback if LLM fails
                logger.warning("LLM generation failed, returning context directly")
                fallback = self._format_basic_answer(question, context_chunks)
                if return_prompt:
                    return fallback, full_prompt, context_chunks
                return fallback

        except ValueError as e:
            # Handle empty collection
            logger.error(f"RAG error: {e}")
            error_msg = (
                "The policy knowledge base is not yet loaded. "
                "Please load policy documents before asking questions."
            )
            if return_prompt:
                return error_msg, "", []
            return error_msg

        except Exception as e:
            logger.error(f"Error answering question: {e}")
            error_msg = f"Error processing question: {str(e)}"
            if return_prompt:
                return error_msg, "", []
            return error_msg

    def chat(self, message: str, return_prompt: bool = False):
        """
        Main chat interface - routes messages to appropriate handler.

        Args:
            message: User's input message
            return_prompt: If True, return tuple with prompt and metadata

        Returns:
            ChatBot response (PII ticket, dev ticket, or question answer)
            Or tuple if return_prompt is True: (response, prompt, metadata, message_type)
        """
        try:
            if not message or not message.strip():
                msg = "Please provide a message or question."
                if return_prompt:
                    return msg, "", {}, 'question'
                return msg

            logger.info(f"Processing message (length: {len(message)} chars)")

            # Classify message type
            msg_type = self._classify_message(message)
            logger.info(f"Message classified as: {msg_type}")

            if msg_type == 'pii_ticket':
                # Support ticket with actual PII
                logger.info("Processing as PII support ticket")
                if return_prompt:
                    response, prompt, detections, risk_level, context_chunks = self.analyze_ticket(message, return_prompt=True)
                    metadata = {
                        'detections': detections,
                        'risk_level': risk_level,
                        'context_chunks': context_chunks
                    }
                    return response, prompt, metadata, 'pii_ticket'
                else:
                    return self.analyze_ticket(message)

            elif msg_type == 'dev_ticket':
                # Development/PM ticket about requirements
                logger.info("Processing as development/requirements ticket")
                if return_prompt:
                    response, prompt, context_chunks = self.answer_dev_ticket(message, return_prompt=True)
                    metadata = {'context_chunks': context_chunks, 'ticket_type': 'development'}
                    return response, prompt, metadata, 'dev_ticket'
                else:
                    return self.answer_dev_ticket(message)

            else:
                # General policy question
                logger.info("Processing as policy question")
                if return_prompt:
                    response, prompt, context_chunks = self.answer_question(message, return_prompt=True)
                    metadata = {'context_chunks': context_chunks}
                    return response, prompt, metadata, 'question'
                else:
                    return self.answer_question(message)

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            error_msg = f"An error occurred: {str(e)}"
            if return_prompt:
                return error_msg, "", {}, 'question'
            return error_msg

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
