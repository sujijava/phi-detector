#!/usr/bin/env python3
"""
Streamlit app for PII Detection Chatbot
"""

import streamlit as st
from src.chatbot import ChatBot

# Page config
st.set_page_config(
    page_title="PII Detection Chatbot",
    layout="wide"
)

# Initialize chatbot and load documents
@st.cache_resource
def get_chatbot(_version=4):  # Increment version to force cache refresh
    bot = ChatBot()
    # Auto-load documents if collection is empty
    stats = bot.rag.get_collection_stats()
    if stats.get('total_chunks', 0) == 0:
        try:
            chunks = bot.rag.load_documents("./data/knowledge_base")
            bot.rag.add_documents(chunks)
        except Exception as e:
            pass  # Will be handled by UI checks later
    return bot

def reload_policy_documents(bot):
    """Reload policy documents into the RAG system."""
    try:
        # Clear existing collection first
        bot.rag.clear_collection()
        # Load documents from the knowledge base directory
        chunks = bot.rag.load_documents("./data/knowledge_base")
        bot.rag.add_documents(chunks)
        return True, f"Successfully loaded {len(chunks)} chunks from policy documents"
    except Exception as e:
        return False, f"Error loading documents: {str(e)}"

# Initialize session state for chat history
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

if 'current_input' not in st.session_state:
    st.session_state.current_input = ""

# Title and description
st.title("PII Detection Chatbot")
st.markdown("""
This chatbot analyzes text to detect and flag Protected Health Information (PHI) and
Personally Identifiable Information (PII). It can help you identify sensitive data that
may need to be redacted or handled securely.
""")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    ### PII Detection Chatbot

    This tool helps identify:
    - Email addresses
    - Social Insurance Numbers (SIN)
    - Phone numbers
    - Names and other PII

    **How to use:**
    1. Documents auto-load on startup
    2. Enter text in the main area
    3. Click "Analyze" to detect PII
    4. Review the results
    """)

    st.divider()

    # Document loading section
    st.subheader("Knowledge Base")

    # Get chatbot instance
    bot = get_chatbot()

    # Check current document count
    stats = bot.rag.get_collection_stats()
    doc_count = stats.get('total_chunks', 0)

    if doc_count > 0:
        st.success(f"✅ Documents loaded: {doc_count} chunks")
        st.session_state.documents_loaded = True
    else:
        st.warning("⚠️ No documents loaded")

    if st.button("Reload Policy Documents", use_container_width=True):
        with st.spinner("Reloading policy documents..."):
            success, message = reload_policy_documents(bot)
            if success:
                st.success(message)
                st.session_state.documents_loaded = True
                st.rerun()
            else:
                st.error(message)

    st.divider()

    st.subheader("Examples")

    st.markdown("**PII Support Tickets:**")
    if st.button("📋 Support Ticket", use_container_width=True):
        st.session_state.current_input = "User test@test.com (SIN: 123-456-789) reports issue with login. Contact them at 555-1234."

    if st.button("🏥 Medical Record", use_container_width=True):
        st.session_state.current_input = """Patient: Sarah Johnson
DOB: 03/15/1978
Email: sjohnson@email.com
Phone: 416-555-9876
Medical Record #: MR-448821

Chief Complaint: Patient reports persistent headaches and requests refill of prescription."""

    st.markdown("**Development Tickets:**")

    if st.button("🛠️ Chat Encryption", use_container_width=True):
        st.session_state.current_input = "As a PM, I'm creating a ticket for storing chat messages between users and healthcare providers. Does this data require encryption? What technology measures are recommended for compliance?"

    if st.button("🔐 Patient Portal", use_container_width=True):
        st.session_state.current_input = "We're building a patient portal where users can view their medical records. What security measures and access controls do we need to implement to comply with privacy regulations?"

    st.markdown("**Questions:**")

    if st.button("❓ What is PII?", use_container_width=True):
        st.session_state.current_input = "What types of personal information are considered PII under Canadian privacy law?"

    if st.button("🔒 Consent Requirements", use_container_width=True):
        st.session_state.current_input = "What are the consent requirements for collecting personal information?"

    if st.button("📊 Data Retention", use_container_width=True):
        st.session_state.current_input = "How long should we retain personal health information?"

    if st.button("🔐 Security Obligations", use_container_width=True):
        st.session_state.current_input = "What are the security safeguard requirements for protecting personal information?"

    if st.button("📧 Breach Notification", use_container_width=True):
        st.session_state.current_input = "What should we do if there is a privacy breach?"

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")

    # If current_input has content and input_area doesn't match, update input_area
    if st.session_state.current_input and st.session_state.get('input_area', '') != st.session_state.current_input:
        st.session_state.input_area = st.session_state.current_input
        st.session_state.current_input = ""  # Clear after setting

    # Text area managed by session state via key
    user_input = st.text_area(
        "Enter text to analyze:",
        height=200,
        key="input_area"
    )

    analyze_button = st.button("Analyze", type="primary", use_container_width=True)

with col2:
    st.subheader("Results")

    if analyze_button and user_input:
        # First, detect what type of input it is
        with st.spinner("🔎 Detecting input type..."):
            msg_type = bot._classify_message(user_input)

        # Display detection result based on message type
        if msg_type == 'pii_ticket':
            st.success("✅ **Detected as:** PII Support Ticket (will analyze for personal information)")
            spinner_msg = "🔍 Detecting PII..."
        elif msg_type == 'dev_ticket':
            st.info("🛠️ **Detected as:** Development Ticket (will provide compliance guidance)")
            spinner_msg = "💡 Analyzing requirements..."
        else:
            st.info("ℹ️ **Detected as:** Policy Question (will search knowledge base)")
            spinner_msg = "💬 Searching for answer..."

        # Process the input
        with st.spinner(spinner_msg):
            # Get response with prompt
            response, prompt, metadata, message_type = bot.chat(user_input, return_prompt=True)

            # Store in chat history
            st.session_state.messages.append({
                'input': user_input,
                'output': response,
                'message_type': message_type,
                'prompt': prompt,
                'metadata': metadata
            })

            # Display response
            st.markdown("### Analysis Result")
            st.info(response)

            # Display prompt used in an expander
            if prompt:
                with st.expander("🔧 View Prompt Used", expanded=False):
                    st.markdown("**Prompt sent to LLM:**")
                    st.code(prompt, language="text")
                    st.caption(f"Prompt length: {len(prompt)} characters")

            # Display metadata based on message type
            if message_type == 'pii_ticket':
                # For PII tickets, show detections AND policy references
                detections = metadata.get('detections', [])
                risk_level = metadata.get('risk_level', 'UNKNOWN')
                context_chunks = metadata.get('context_chunks', [])

                if detections:
                    with st.expander(f"📋 PII Detection Details ({len(detections)} items found)", expanded=False):
                        st.markdown(f"**Risk Level:** `{risk_level}`")
                        st.markdown("**Detected PII:**")
                        for det in detections:
                            st.markdown(f"- **{det['type']}**: `{det['value']}`")

                # Show policy references used for guidance
                if context_chunks:
                    with st.expander(f"📚 Policy References ({len(context_chunks)} sources)", expanded=False):
                        st.markdown("**Compliance guidance based on:**")
                        for i, (text, chunk_metadata, score) in enumerate(context_chunks, 1):
                            source = chunk_metadata.get('source', 'Unknown')
                            source_name = source.split('/')[-1] if '/' in source else source
                            st.markdown(f"**[{i}] {source_name}** (relevance: {score:.4f})")
                            with st.expander(f"View excerpt {i}", expanded=False):
                                st.text(text[:400] + "..." if len(text) > 400 else text)

            elif message_type == 'dev_ticket':
                # For dev tickets, show context chunks used for guidance
                context_chunks = metadata.get('context_chunks', [])
                if context_chunks:
                    with st.expander(f"📚 Policy References ({len(context_chunks)} sources)", expanded=False):
                        st.markdown("**Compliance documents referenced:**")
                        for i, (text, chunk_metadata, score) in enumerate(context_chunks, 1):
                            source = chunk_metadata.get('source', 'Unknown')
                            source_name = source.split('/')[-1] if '/' in source else source
                            st.markdown(f"**[{i}] {source_name}** (relevance: {score:.4f})")
                            with st.expander(f"View excerpt {i}", expanded=False):
                                st.text(text[:400] + "..." if len(text) > 400 else text)

            else:
                # For general questions, show context chunks
                context_chunks = metadata.get('context_chunks', [])
                if context_chunks:
                    with st.expander(f"📚 Retrieved Context ({len(context_chunks)} chunks)", expanded=False):
                        for i, (text, chunk_metadata, score) in enumerate(context_chunks, 1):
                            source = chunk_metadata.get('source', 'Unknown')
                            source_name = source.split('/')[-1] if '/' in source else source
                            st.markdown(f"**Chunk {i}** (from `{source_name}`, score: {score:.4f})")
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                            st.divider()

    elif analyze_button and not user_input:
        st.warning("Please enter some text to analyze.")

# Chat History
if st.session_state.messages:
    st.divider()
    st.subheader("Chat History")

    for idx, msg in enumerate(reversed(st.session_state.messages[-5:])):  # Show last 5
        # Get message type (new) or fallback to old format
        message_type = msg.get('message_type', None)

        if message_type == 'pii_ticket':
            detection_badge = "🎫 PII Ticket"
            type_label = "PII Support Ticket"
        elif message_type == 'dev_ticket':
            detection_badge = "🛠️ Dev Ticket"
            type_label = "Development Ticket"
        elif message_type == 'question':
            detection_badge = "❓ Question"
            type_label = "Policy Question"
        else:
            # Fallback for old messages
            is_ticket = msg.get('is_ticket', None)
            detection_badge = "🎫 Ticket" if is_ticket else "❓ Question" if is_ticket is not None else "📝 Entry"
            type_label = "Support Ticket" if is_ticket else "Policy Question" if is_ticket is not None else "Unknown"

        with st.expander(f"{detection_badge} | Query {len(st.session_state.messages) - idx}: {msg['input'][:50]}..."):
            st.markdown("**Input:**")
            st.text(msg['input'])
            st.markdown(f"**Type:** {type_label}")
            st.markdown("**Output:**")
            st.info(msg['output'])

            # Show prompt if available
            prompt = msg.get('prompt', '')
            if prompt:
                with st.expander("🔧 View Prompt", expanded=False):
                    st.code(prompt, language="text")
                    st.caption(f"Prompt length: {len(prompt)} characters")

            # Show metadata based on message type
            metadata = msg.get('metadata', {})

            if message_type == 'pii_ticket':
                # Show PII detections
                if metadata.get('detections'):
                    detections = metadata['detections']
                    risk_level = metadata.get('risk_level', 'UNKNOWN')
                    with st.expander(f"📋 PII Details ({len(detections)} items)", expanded=False):
                        st.markdown(f"**Risk Level:** `{risk_level}`")
                        for det in detections:
                            st.markdown(f"- **{det['type']}**: `{det['value']}`")

                # Show policy references
                if metadata.get('context_chunks'):
                    context_chunks = metadata['context_chunks']
                    with st.expander(f"📚 Policy Refs ({len(context_chunks)})", expanded=False):
                        for i, (text, chunk_metadata, score) in enumerate(context_chunks, 1):
                            source = chunk_metadata.get('source', 'Unknown')
                            source_name = source.split('/')[-1] if '/' in source else source
                            st.markdown(f"**[{i}] {source_name}**")
                            st.text(text[:150] + "..." if len(text) > 150 else text)

            elif (message_type == 'dev_ticket' or message_type == 'question') and metadata.get('context_chunks'):
                context_chunks = metadata['context_chunks']
                with st.expander(f"📚 References ({len(context_chunks)} sources)", expanded=False):
                    for i, (text, chunk_metadata, score) in enumerate(context_chunks, 1):
                        source = chunk_metadata.get('source', 'Unknown')
                        source_name = source.split('/')[-1] if '/' in source else source
                        st.markdown(f"**[{i}] {source_name}** (score: {score:.4f})")
                        st.text(text[:200] + "..." if len(text) > 200 else text)
