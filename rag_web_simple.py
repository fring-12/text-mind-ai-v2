"""
Simple RAG Web App - Debug Version
"""

import streamlit as st
import os
import pickle
import numpy as np
import requests
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="RAG System",
    page_icon="📚",
    layout="wide"
)

st.title("📚 English Textbook RAG System")

# Debug info
st.write("### Debug Information")
st.write(f"Current working directory: {os.getcwd()}")
st.write(f"Python path: {os.path.dirname(os.path.abspath(__file__))}")

# Check for database
db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vector_db.pkl")
st.write(f"Database path: {db_path}")
st.write(f"Database exists: {os.path.exists(db_path)}")

if os.path.exists(db_path):
    st.write(f"Database size: {os.path.getsize(db_path)} bytes")

    # Try to load
    try:
        with open(db_path, 'rb') as f:
            db = pickle.load(f)
        st.success(f"✅ Database loaded! Contains {len(db.get('chunks', []))} chunks")

        # Show sample
        if db.get('chunks'):
            st.write("Sample chunk:")
            st.write(db['chunks'][0][:200] + "...")
    except Exception as e:
        st.error(f"❌ Error loading database: {e}")
        import traceback
        st.code(traceback.format_exc())
else:
    st.error("❌ Database file not found!")

st.write("---")
st.write("If the database loads successfully above, the main app should work.")