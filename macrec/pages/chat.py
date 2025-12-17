import streamlit as st
from loguru import logger

from macrec.systems import ChatSystem, CollaborationSystem
from macrec.utils import add_chat_message

def chat_page(system: ChatSystem | CollaborationSystem) -> None:
    # Render Sidebar History
    with st.sidebar:
        st.markdown("---")
        col1, col2 = st.columns([0.8, 0.2])
        with col1:
            st.markdown("### 🕒 Chat History")
        with col2:
            # Lock History Checkbox (Compact)
            lock_history = st.checkbox("🔒", value=False, help="Lock/Unlock Chat History")
        
        # New Chat Button
        if not lock_history:
            if st.button("➕ New Chat", use_container_width=True, type="primary"):
                st.session_state.chat_history = []
                if hasattr(system, 'reset'):
                    system.reset(clear=True)
                st.rerun()
            
        # Display history items with delete option
        if not st.session_state.chat_history:
            st.caption("No history yet.")
        else:
            st.markdown("### Today")
            
            # We need to handle deletion safely. Collect indices to delete.
            indices_to_delete = []
            
            for i, chat in enumerate(st.session_state.chat_history):
                if chat['role'] == 'user':
                    col1, col2 = st.columns([0.8, 0.2])
                    msg = chat['message']
                    display_msg = msg[:25] + "..." if len(msg) > 25 else msg
                    
                    with col1:
                        st.text(f"{display_msg}")
                    with col2:
                        # Only show delete button if history is unlocked
                        if not lock_history:
                            if st.button("🗑️", key=f"del_{i}", help="Delete this message"):
                                indices_to_delete.append(i)
            
            # Process deletions
            if indices_to_delete:
                # Delete the user message AND the subsequent system message(s)
                # We delete from end to start to avoid index shifting issues if multiple (though button click is usually 1)
                for idx in sorted(indices_to_delete, reverse=True):
                    # Remove the user message at idx
                    if idx < len(st.session_state.chat_history):
                        st.session_state.chat_history.pop(idx)
                        
                    # Remove subsequent system messages (if any) that belonged to this turn
                    # We check if the *new* item at idx is a system message
                    while idx < len(st.session_state.chat_history) and st.session_state.chat_history[idx]['role'] != 'user':
                        st.session_state.chat_history.pop(idx)
                
                st.rerun()

    # Render Main Chat Interface
    for chat in st.session_state.chat_history:
        if isinstance(chat['message'], str):
            # Clean message to ensure newlines are rendered correctly
            clean_message = chat['message'].replace('\\n', '\n').replace('\n', '  \n')
            st.chat_message(chat['role']).markdown(clean_message)
        elif isinstance(chat['message'], list):
            with st.chat_message(chat['role']):
                with st.expander("System Process", expanded=False):
                    for message in chat['message']:
                        # Filter out intermediate Analyst logs
                        if "Analyst" in message and ("Look up" in message or "Finish with results" in message):
                            continue
                        st.markdown(f'{message}')
    logger.debug('Initialization complete!')
    if prompt := st.chat_input():
        add_chat_message('user', prompt)
        with st.chat_message('assistant'):
            with st.expander("System Process", expanded=False):
                st.markdown('#### System is running...')
                response = system(prompt)
                for log in system.web_log:
                    # Filter out intermediate Analyst logs
                    if "Analyst" in log and ("Look up" in log or "Finish with results" in log):
                        continue
                    st.markdown(log)
            
            st.session_state.chat_history.append({
                'role': 'assistant',
                'message': system.web_log
            })
        add_chat_message('assistant', response)
        st.rerun()
