import os
import streamlit as st

from macrec.pages.task import task_config
from macrec.pages.user import user_page
from macrec.systems import *
from macrec.utils import task2name, system2dir, read_json

all_tasks = ['rp', 'sr', 'gen', 'chat']

# System descriptions for better UX
SYSTEM_DESCRIPTIONS = {
    'ReActSystem': '🤖 LLM đơn giản (Chỉ dùng GPT/Gemini)',
    'ReflectionSystem': '🔄 LLM + Tự phản biện',
    'AnalyseSystem': '📊 LLM + Phân tích dữ liệu',
    'CollaborationSystem': '👥 Multi-Agent (Manager + Analyst + Searcher)',
    'ItemKNNSystem': '🔢 Collaborative Filtering thuần túy (Không dùng LLM)',
    'HybridSystem': '⭐ KHUYÊN DÙNG: Hybrid (MF/SASRec + LLM) cho RP/SR',
}

# Recommended configs for each system
RECOMMENDED_CONFIGS = {
    'HybridSystem': 'cf_llm.json',
    'CollaborationSystem': 'reflect_analyse_search_interpret.json',  # Default for CollaborationSystem
    'ItemKNNSystem': 'itemknn.json',
}

def get_system_label(system_class):
    """Get formatted label for system selection"""
    name = system_class.__name__
    desc = SYSTEM_DESCRIPTIONS.get(name, '')
    return f"{name} - {desc}" if desc else name

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
    st.sidebar.title('MACRec Demo')
    mode = st.sidebar.radio('Mode', ['Tasks', 'Users'])
    
    # Choose a system with descriptions
    st.sidebar.markdown("### 🎯 Chọn Hệ thống")
    system_type = st.sidebar.radio(
        'System', 
        SYSTEMS, 
        format_func=get_system_label,
        help="HybridSystem được khuyên dùng cho Rating Prediction (rp) và Sequential Recommendation (sr)"
    )
    
    if mode == 'Users':
        dataset = st.sidebar.selectbox('Choose a dataset', ['ml-100k', 'Beauty'])
        user_page(dataset=dataset)
        return
    
    # For Tasks mode, get task selection first
    st.sidebar.markdown("### 📋 Chọn Task")
    # Set 'chat' as default task (conversational recommendation)
    default_task_index = all_tasks.index('chat') if 'chat' in all_tasks else 0
    task = st.sidebar.radio(
        'Task',
        all_tasks,
        index=default_task_index,
        format_func=task2name,
        help="rp: Dự đoán điểm | sr: Gợi ý tiếp theo | gen: Tạo giải thích | chat: Trò chuyện"
    )
    
    # Choose the config with smart suggestions based on system and task
    st.sidebar.markdown("### ⚙️ Chọn Cấu hình")
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
        st.sidebar.info(f"💡 Gợi ý: `{recommended_config}` (Đã chọn tự động)")
    
    config_file = st.sidebar.selectbox(
        'Config File', 
        config_files,
        index=default_index,
        help="File cấu hình cho hệ thống đã chọn"
    )
    
    config = read_json(os.path.join(config_dir, config_file))
    assert 'supported_tasks' in config, f'The config file {config_file} should contain the field "supported_tasks".'
    supported_tasks = config['supported_tasks']
    supported_tasks = [task for task in supported_tasks if task in system_type.supported_tasks()]
    
    if task not in supported_tasks:
        st.error(f'❌ Task `{task2name(task)}` không được hỗ trợ bởi `{system_type.__name__}` với config `{config_file}`.')
        st.info(f"✅ Các task được hỗ trợ: {', '.join([task2name(t) for t in supported_tasks])}")
        return
    
    task_config(task=task, system_type=system_type, config_path=os.path.join(config_dir, config_file))
