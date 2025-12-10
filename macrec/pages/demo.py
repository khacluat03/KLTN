import os
import streamlit as st

from macrec.pages.task import task_config
from macrec.pages.user import user_page
from macrec.systems import *
from macrec.utils import task2name, system2dir, read_json

all_tasks = ['rp', 'sr', 'gen', 'chat']

# System descriptions for better UX
SYSTEM_DESCRIPTIONS = {
    'ReActSystem': '🤖 Simple LLM (GPT/Gemini only)',
    'ReflectionSystem': '🔄 LLM + Self-Reflection',
    'AnalyseSystem': '📊 LLM + Data Analysis',
    'CollaborationSystem': '👥 Multi-Agent (Manager + Analyst + Searcher)',
    'ItemKNNSystem': '🔢 Pure Collaborative Filtering (No LLM)',
    'HybridSystem': '⭐ RECOMMENDED: Hybrid (MF/SASRec + LLM) for RP/SR',
}

# Recommended configs for each system
RECOMMENDED_CONFIGS = {
    'HybridSystem': 'cf_llm.json',
    'CollaborationSystem': 'reflect_analyse_search_interpret.json',  # Default for CollaborationSystem
    'ItemKNNSystem': 'itemknn.json',
}

def get_system_label(system_class):
    """Get formatted label for system selection"""
    return system_class.__name__

def demo():
    api_cfg = read_json('config/api-config.json')
    provider = (api_cfg.get('provider') or api_cfg.get('API_TYPE') or 'openai').lower()
    if provider == 'gemini':
        from macrec.utils.init import init_gemini_api
        init_gemini_api(api_cfg)
    else:
        from macrec.utils.init import init_openai_api
        init_openai_api(api_cfg)
    st.set_page_config(
        page_title="test",
        page_icon="🧠",
        layout="wide",
    )
    st.markdown("""
        <style>
               .block-container {
                    padding-top: 3rem;
                }
               [data-testid="stSidebar"] {
                    padding-top: 0rem;
                }
               [data-testid="stSidebar"] .css-1d391kg {
                    padding-top: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)
    st.sidebar.title('Customer Product Recommendation System')
    mode = st.sidebar.radio('Mode', ['Tasks', 'Users'])
    
    # Create a formatted help string with all descriptions
    system_help = "### System Descriptions:\n\n" + "\n".join([f"- **{k}**: {v}" for k, v in SYSTEM_DESCRIPTIONS.items()])
    
    system_type = st.sidebar.selectbox(
        'System', 
        SYSTEMS, 
        format_func=get_system_label,
        help=system_help
    )
    
    if mode == 'Users':
        dataset = st.sidebar.selectbox('Choose a dataset', ['ml-100k', 'Beauty'])
        user_page(dataset=dataset)
        return
    
    # For Tasks mode, get task selection first
    # Set 'chat' as default task (conversational recommendation)
    default_task_index = all_tasks.index('chat') if 'chat' in all_tasks else 0
    task = st.sidebar.selectbox(
        'Task',
        all_tasks,
        index=default_task_index,
        format_func=task2name,
        help="rp: Rating Prediction | sr: Sequential Recommendation | gen: Explanation Generation | chat: Chat"
    )
    
    # Choose the config with smart suggestions based on system and task
    config_dir = os.path.join('config', 'systems', system2dir(system_type.__name__))
    config_files = os.listdir(config_dir)
    
    # Determine recommended config
    recommended_config = RECOMMENDED_CONFIGS.get(system_type.__name__)
    
    # Override for CollaborationSystem + chat task
    if system_type.__name__ == 'CollaborationSystem' and task == 'chat':
        if 'chat.json' in config_files:
            recommended_config = 'chat.json'
    
    default_index = 0
    if recommended_config and recommended_config in config_files:
        default_index = config_files.index(recommended_config)
        st.sidebar.info(f"💡 Suggestion: `{recommended_config}` (Auto-selected)")
    
    config_file = st.sidebar.selectbox(
        'Config File', 
        config_files,
        index=default_index,
        help="Configuration file for the selected system"
    )
    
    config = read_json(os.path.join(config_dir, config_file))
    assert 'supported_tasks' in config, f'The config file {config_file} should contain the field "supported_tasks".'
    supported_tasks = config['supported_tasks']
    supported_tasks = [task for task in supported_tasks if task in system_type.supported_tasks()]
    
    if task not in supported_tasks:
        st.error(f'❌ Task `{task2name(task)}` is not supported by `{system_type.__name__}` with config `{config_file}`.')
        st.info(f"✅ Supported tasks: {', '.join([task2name(t) for t in supported_tasks])}")
        return
    
    task_config(task=task, system_type=system_type, config_path=os.path.join(config_dir, config_file))
