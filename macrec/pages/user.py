import os
import streamlit as st
import pandas as pd


def user_page(dataset: str) -> None:
    st.markdown('## `Users` browser')
    data_dir = os.path.join('data', dataset)
    user_csv_path = os.path.join(data_dir, 'user.csv')

    if not os.path.isfile(user_csv_path):
        st.error(f'User file not found: `{user_csv_path}`')
        return

    try:
        df = pd.read_csv(user_csv_path)
    except Exception as e:
        st.error(f'Failed to read `{user_csv_path}`: {e}')
        return

    expected_cols = ['user_id', 'age', 'gender', 'occupation', 'user_profile']
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        st.warning(f'Missing columns in `user.csv`: `{missing}`')
        present_cols = [c for c in expected_cols if c in df.columns]
    else:
        present_cols = expected_cols

    # Sidebar filters
    st.sidebar.markdown('### User Filters')
    user_ids = df['user_id'].unique().tolist() if 'user_id' in df.columns else []
    selected_user_id = st.sidebar.selectbox('Select user_id', ['(All)'] + [str(u) for u in user_ids]) if user_ids else '(All)'

    gender_values = sorted(df['gender'].dropna().unique().tolist()) if 'gender' in df.columns else []
    selected_gender = st.sidebar.multiselect('Gender', gender_values, default=gender_values)

    occupation_values = sorted(df['occupation'].dropna().unique().tolist()) if 'occupation' in df.columns else []
    selected_occupation = st.sidebar.multiselect('Occupation', occupation_values, default=occupation_values)

    # Apply filters
    filtered = df
    if 'user_id' in filtered.columns and selected_user_id != '(All)':
        try:
            selected_value = int(selected_user_id)
        except ValueError:
            selected_value = selected_user_id
        filtered = filtered[filtered['user_id'] == selected_value]

    if 'gender' in filtered.columns and selected_gender:
        filtered = filtered[filtered['gender'].isin(selected_gender)]

    if 'occupation' in filtered.columns and selected_occupation:
        filtered = filtered[filtered['occupation'].isin(selected_occupation)]

    st.markdown('### Users')
    st.dataframe(filtered[present_cols] if present_cols else filtered)


