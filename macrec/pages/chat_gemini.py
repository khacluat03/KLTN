import os
import streamlit as st
from macrec.systems import ChatSystem
from macrec.utils import read_json
import uuid
import json
from datetime import datetime

def chat_gemini_page():
    """Dedicated chat page using Gemini for conversational recommendation."""
    
    # Initialize API
    api_cfg = read_json('config/api-config.json')
    provider = (api_cfg.get('provider') or api_cfg.get('API_TYPE') or 'openai').lower()
    if provider == 'gemini':
        from macrec.utils.init import init_gemini_api
        init_gemini_api(api_cfg)
    
    # Set page config
    st.set_page_config(
        page_title="MACRec Chat - Gemini",
        page_icon="💬",
        layout="wide",
    )
    
    # Title and description
    st.title("🤖 MACRec Conversational Recommendation")
    st.markdown("""
    ### Powered by Google Gemini 2.0 Flash Experimental
    
    Chat with our AI recommendation system to get personalized movie recommendations!
    """)
    
    # Initialize chat system
    if 'chat_system' not in st.session_state:
        try:
            # Load the chat_gemini config
            config_path = 'config/systems/chat/chat_gemini.json'
            st.session_state.chat_system = ChatSystem(config_path=config_path, task='chat')
            st.session_state.chat_system.reset()
            st.success("✅ Chat system initialized successfully!")
        except Exception as e:
            st.error(f"❌ Failed to initialize chat system: {e}")
            st.stop()

    # Initialize session id for feedback tracking
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # Explainability toggle
    if 'explain_toggle' not in st.session_state:
        st.session_state.explain_toggle = True
    
    # Chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (message, role) in enumerate(st.session_state.chat_history):
                if role == 'user':
                    with st.chat_message("user"):
                        st.write(message)
                else:
                    with st.chat_message("assistant"):
                        st.write(message)
        
        # Chat input row
        prompt = st.chat_input("Ask about movies you'd like to watch...")
        if prompt is not None and len(prompt.strip()) > 0:
            # Add user message to history
            st.session_state.chat_history.append((prompt, 'user'))
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # Optionally append explainability request
                        user_prompt = prompt
                        if st.session_state.explain_toggle:
                            user_prompt = f"{prompt}\n\nAlso explain briefly why these recommendations fit my preferences."

                        # Get response from chat system
                        response = st.session_state.chat_system(
                            user_input=user_prompt,
                            reset=False
                        )
                        
                        # Add response to history
                        st.session_state.chat_history.append((response, 'assistant'))
                        st.write(response)

                        # Render feedback controls for the latest assistant message
                        _render_feedback_controls(
                            user_prompt=prompt,
                            model_response=response,
                            session_id=st.session_state.session_id,
                            explain_enabled=st.session_state.explain_toggle,
                        )
                        
                    except Exception as e:
                        error_msg = f"Sorry, I encountered an error: {str(e)}"
                        st.session_state.chat_history.append((error_msg, 'assistant'))
                        st.error(error_msg)
    
    with col2:
        st.markdown("### 🎯 Quick Actions")
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chat_system.reset()
            st.rerun()
        
        if st.button("🎬 Sample Questions", use_container_width=True):
            sample_questions = [
                "I want to watch a romantic comedy",
                "Recommend some action movies",
                "What are the best sci-fi films?",
                "I like movies with strong female leads",
                "Show me some classic movies from the 90s"
            ]
            
            selected_question = st.selectbox("Choose a sample question:", sample_questions)
            if selected_question:
                st.session_state.sample_input = selected_question
                st.rerun()
        
        st.markdown("### 📊 System Info")
        st.info(f"""
        **Model:** gemini-2.0-flash-exp  
        **Temperature:** 0.3 (optimized for speed)  
        **Max Tokens:** 512 (optimized for speed)  
        **Provider:** Google Gemini
        """)
        
        # Explainability toggle
        st.markdown("### ✨ Explainability")
        st.session_state.explain_toggle = st.toggle(
            "Ask the system to explain recommendations",
            value=st.session_state.explain_toggle,
            help="When enabled, the model will add a brief justification for its recommendations."
        )

        st.markdown("### 💡 Tips")
        st.markdown("""
        - Be specific about your preferences
        - Ask for explanations of recommendations
        - Try different genres or time periods
        - Ask follow-up questions
        """)
    
    # Handle sample input
    if 'sample_input' in st.session_state:
        sample_prompt = st.session_state.sample_input
        del st.session_state.sample_input
        
        # Add sample message to history
        st.session_state.chat_history.append((sample_prompt, 'user'))
        
        # Generate response
        with st.spinner("Thinking..."):
            try:
                user_prompt = sample_prompt
                if st.session_state.explain_toggle:
                    user_prompt = f"{sample_prompt}\n\nAlso explain briefly why these recommendations fit my preferences."
                response = st.session_state.chat_system(
                    user_input=user_prompt,
                    reset=False
                )
                st.session_state.chat_history.append((response, 'assistant'))
                # Render feedback after sample response
                _render_feedback_controls(
                    user_prompt=sample_prompt,
                    model_response=response,
                    session_id=st.session_state.session_id,
                    explain_enabled=st.session_state.explain_toggle,
                )
                st.rerun()
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}"
                st.session_state.chat_history.append((error_msg, 'assistant'))
                st.rerun()


def _render_feedback_controls(user_prompt: str, model_response: str, session_id: str, explain_enabled: bool) -> None:
    """Render thumbs up/down and optional reason/comment, then log feedback."""
    import streamlit as st
    from pathlib import Path

    with st.expander("Feedback on this answer", expanded=False):
        col_like, col_dislike = st.columns(2)
        with col_like:
            like = st.button("👍 Helpful", key=f"like_{uuid.uuid4()}" )
        with col_dislike:
            dislike = st.button("👎 Not helpful", key=f"dislike_{uuid.uuid4()}" )

        selected = None
        if like:
            selected = "up"
        elif dislike:
            selected = "down"

        reason = st.selectbox(
            "Primary reason",
            ["", "Relevant to my taste", "Clear explanation", "Novel suggestions", "Too generic", "Irrelevant", "Poor explanation"],
            index=0,
            key=f"reason_{uuid.uuid4()}"
        )
        comment = st.text_area("Additional comments (optional)", key=f"comment_{uuid.uuid4()}")
        submit = st.button("Submit Feedback", type="primary", key=f"submit_{uuid.uuid4()}")

        if submit:
            feedback = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "session_id": session_id,
                "signal": selected or "",
                "reason": reason,
                "comment": comment,
                "explain_enabled": explain_enabled,
                "user_prompt": user_prompt,
                "model_response": model_response,
            }
            _log_feedback(feedback)
            st.success("Thanks for your feedback!")


def _log_feedback(record: dict) -> None:
    """Append feedback to CSV and JSONL under logs/."""
    from pathlib import Path
    import csv

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    # JSONL
    jsonl_path = logs_dir / "web_feedback.jsonl"
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # CSV (create header if not exists)
    csv_path = logs_dir / "web_feedback.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "timestamp",
                "session_id",
                "signal",
                "reason",
                "comment",
                "explain_enabled",
                "user_prompt",
                "model_response",
            ],
        )
        if write_header:
            writer.writeheader()
        writer.writerow(record)

if __name__ == "__main__":
    chat_gemini_page()
