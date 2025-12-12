"""
Minimal test app
"""

import streamlit as st
import os
import pickle

st.title("📚 RAG System Test")

# Check database
db_path = "vector_db.pkl"
st.write(f"Looking for: {db_path}")
st.write(f"Current directory: {os.getcwd()}")

if os.path.exists(db_path):
    st.success("✅ Database file found!")

    try:
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
        st.success(f"✅ Database loaded with {len(db.get('chunks', []))} chunks")

        # Show sample
        if db.get('chunks'):
            st.write("Sample content:")
            st.write(db['chunks'][0][:200] + "...")
    except Exception as e:
        st.error(f"❌ Error: {e}")
else:
    st.error("❌ Database file not found!")
    st.write("Files in directory:")
    files = os.listdir('.')
    for f in files:
        st.write(f"- {f}")