import os
import streamlit as st

from macrec.pages.task import task_config
from macrec.pages.user import user_page
from macrec.systems import *
from macrec.utils import task2name, system2dir, read_json

all_tasks = ['rp', 'sr', 'gen', 'chat']

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
    # choose a system
    system_type = st.sidebar.radio('Choose a system', SYSTEMS, format_func=lambda x: x.__name__)
    # choose the config
    config_dir = os.path.join('config', 'systems', system2dir(system_type.__name__))
    config_files = os.listdir(config_dir)
    config_file = st.sidebar.selectbox('Choose a config file', config_files)
    config = read_json(os.path.join(config_dir, config_file))
    assert 'supported_tasks' in config, f'The config file {config_file} should contain the field "supported_tasks".'
    supported_tasks = config['supported_tasks']
    supported_tasks = [task for task in supported_tasks if task in system_type.supported_tasks()]
    if mode == 'Users':
        dataset = st.sidebar.selectbox('Choose a dataset', ['ml-100k', 'Beauty'])
        user_page(dataset=dataset)
        return
    # elif mode == 'Chat':
    #     from macrec.pages.chat_gemini import chat_gemini_page
    #     chat_gemini_page()
    #     return
    else:
        # choose a task
        task = st.sidebar.radio('Choose a task', all_tasks, format_func=task2name)
        if task not in supported_tasks:
            st.error(f'The task {task2name(task)} is not supported by the system `{system_type.__name__}` with the config file `{config_file}`.')
            return
                # --- Bảng hướng dẫn chọn System – Task ---
        with st.sidebar.expander("📌 System – Task Mapping", expanded=True):

            st.markdown("""
            ### **ReActSystem** $\iff$ ⭐Rating Prediction  - ⭐Sequential Recommendation - ⭐Explanation Generation  
            _Hệ thống reasoning–acting phù hợp dự đoán & sinh giải thích._

            ---

            ### **ReflectionSystem** $\iff$ ⭐Rating Prediction - ⭐Sequential Recommendation - ⭐Explanation Generation
            _Giống ReAct nhưng có bước Reflection để cải thiện kết quả._

            ---

            ### **ChatSystem** $\iff$ ⭐Conversational Recommendation  
            ❗ _Không hỗ trợ Rating Prediction, Sequential hay Explanation._

            ---

            ### **AnalyseSystem** $\iff$ ⭐ Rating Prediction (tùy config) -⭐Sequential Recommendation (tùy config)  
            _Dùng để phân tích và đánh giá mô hình._

            ---

            ### **CollaborationSystem** $\iff$ ⭐Sequential Recommendation - ⭐Explanation Generation - ⭐Conversational Recommendation (tùy config)
            _Hệ multi-agent cộng tác cho task phức tạp._
            """)

        task_config(task=task, system_type=system_type, config_path=os.path.join(config_dir, config_file))
        