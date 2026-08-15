# streamlit_app.py - Streamlit frontend for TruthGuard

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import requests
import json
from typing import Optional

# Page configuration
st.set_page_config(
    page_title="TruthGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark glassmorphism theme
st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
    }
    
    /* Header styling */
    .header-container {
        background: rgba(26, 26, 46, 0.8);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    .header-title {
        font-size: 2.5em;
        font-weight: bold;
        color: #ffffff;
        text-shadow: 0 0 20px rgba(100, 150, 255, 0.5);
    }
    
    .header-subtitle {
        font-size: 1.1em;
        color: #a0a0c0;
        margin-top: 5px;
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(30, 30, 50, 0.9);
        border-radius: 15px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Risk level banner */
    .risk-banner {
        padding: 15px 25px;
        border-radius: 10px;
        font-size: 1.3em;
        font-weight: bold;
        text-align: center;
        margin: 15px 0;
        border: 2px solid;
    }
    
    .risk-low {
        background: rgba(34, 197, 94, 0.2);
        border-color: #22c55e;
        color: #22c55e;
    }
    
    .risk-moderate {
        background: rgba(245, 158, 11, 0.2);
        border-color: #f59e0b;
        color: #f59e0b;
    }
    
    .risk-high {
        background: rgba(220, 38, 38, 0.2);
        border-color: #dc2626;
        color: #dc2626;
    }
    
    /* Text areas and inputs */
    .stTextArea > div > div > textarea {
        background: rgba(20, 20, 35, 0.9);
        border: 1px solid rgba(100, 100, 150, 0.3);
        color: #ffffff;
        border-radius: 10px;
    }
    
    .stTextInput > div > div > input {
        background: rgba(20, 20, 35, 0.9);
        border: 1px solid rgba(100, 100, 150, 0.3);
        color: #ffffff;
        border-radius: 8px;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 25px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(59, 130, 246, 0.4);
    }
    
    /* Progress bars */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        border-radius: 10px;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(30, 30, 50, 0.9);
        border-radius: 10px;
        border: 1px solid rgba(100, 100, 150, 0.3);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(15, 15, 26, 0.95);
    }
    
    /* Status indicator */
    .status-online {
        color: #22c55e;
        font-weight: bold;
    }
    
    .status-offline {
        color: #dc2626;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# API configuration
DEFAULT_API_URL = "http://localhost:8000"


def check_backend_health(api_url: str) -> tuple[bool, str]:
    """Check if the backend is online."""
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            return True, "Backend Online"
        return False, "Backend Error"
    except Exception:
        return False, "Backend Offline"


def verify_response(api_url: str, prompt: str, response: str, regenerate: bool) -> Optional[dict]:
    """Send verification request to the API."""
    try:
        payload = {
            "prompt": prompt,
            "response": response,
            "regenerate": regenerate
        }
        resp = requests.post(f"{api_url}/verify", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"API error: {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def generate_and_verify(api_url: str, prompt: str, regenerate: bool) -> Optional[dict]:
    """Send generate-and-verify request to the API."""
    try:
        payload = {
            "prompt": prompt,
            "regenerate": regenerate
        }
        resp = requests.post(f"{api_url}/generate-and-verify", json=payload, timeout=120)
        if resp.status_code == 200:
            return resp.json()
        return {"error": f"API error: {resp.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def render_gauge(value: float, label: str, color: str, min_val: float = 0.0, max_val: float = 1.0):
    """Render a circular gauge metric."""
    percentage = int((value / max_val) * 100)
    return f"""
    <div style="text-align: center; padding: 15px;">
        <div style="font-size: 2.5em; font-weight: bold; color: {color};">
            {percentage:.1f}%
        </div>
        <div style="color: #a0a0c0; font-size: 0.9em;">{label}</div>
    </div>
    """


# Header
st.markdown("""
<div class="header-container">
    <div class="header-title">🛡️ TruthGuard AI</div>
    <div class="header-subtitle">Heterogeneous Hallucination Detection & Grounding Framework</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API URL
    api_url = st.text_input(
        "API Endpoint",
        value=DEFAULT_API_URL,
        help="Backend API URL"
    )
    
    # Backend status
    is_online, status_text = check_backend_health(api_url)
    status_class = "status-online" if is_online else "status-offline"
    status_indicator = "●" if is_online else "○"
    st.markdown(f'<p class="{status_class}">{status_indicator} {status_text}</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Model info (placeholder - would come from API in production)
    st.subheader("📊 Model Configuration")
    st.info("**Base Model:** Qwen-2.5-72B\n\n**Judge Model:** Claude-3.5-Sonnet")
    
    st.divider()
    
    # Threshold slider
    hallucination_threshold = st.slider(
        "Hallucination Threshold",
        min_value=0.3,
        max_value=0.9,
        value=0.6,
        step=0.05,
        help="Threshold above which regeneration is triggered"
    )

# Main content area
tab1, tab2 = st.tabs(["🔍 Verify Existing Response", "✨ Generate & Verify"])

with tab1:
    st.subheader("Verify an existing LLM response")
    
    col1, col2 = st.columns(2)
    with col1:
        user_prompt = st.text_area(
            "User Prompt",
            height=150,
            placeholder="Enter the original user prompt...",
            key="verify_prompt"
        )
    
    with col2:
        llm_response = st.text_area(
            "LLM-Generated Response",
            height=150,
            placeholder="Enter the LLM response to verify...",
            key="verify_response"
        )
    
    regenerate_check = st.checkbox(
        "Enable auto-regeneration if hallucination detected",
        value=False,
        key="verify_regenerate"
    )
    
    verify_btn = st.button("🔍 Verify Response", type="primary", use_container_width=True)
    
    if verify_btn:
        if not user_prompt or not llm_response:
            st.error("Please enter both prompt and response")
        elif not is_online:
            st.error("Backend is offline. Please check the API endpoint.")
        else:
            with st.spinner("Analyzing response..."):
                result = verify_response(api_url, user_prompt, llm_response, regenerate_check)
                
                if result and "result" in result:
                    verification = result["result"]
                    
                    # Display results
                    st.success("✅ Verification completed!")
                    
                    # Top metrics row
                    col1, col2, col3 = st.columns(3)
                    
                    truth_score = verification.get("truth_score", 0)
                    confidence_score = verification.get("confidence_score", 0)
                    hallucination_prob = verification.get("hallucination_probability", 0)
                    
                    with col1:
                        st.markdown(render_gauge(
                            truth_score, 
                            "Truth Score", 
                            "#22c55e" if truth_score > 0.7 else "#f59e0b" if truth_score > 0.4 else "#dc2626"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(render_gauge(
                            confidence_score, 
                            "Confidence Score", 
                            "#3b82f6"
                        ), unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(render_gauge(
                            hallucination_prob, 
                            "Hallucination Probability", 
                            "#dc2626" if hallucination_prob > 0.6 else "#f59e0b" if hallucination_prob > 0.3 else "#22c55e"
                        ), unsafe_allow_html=True)
                    
                    # Risk level banner
                    risk_level = verification.get("risk_level", "MODERATE")
                    risk_class = f"risk-{risk_level.lower()}"
                    st.markdown(f"""
                    <div class="risk-banner {risk_class}">
                        ⚠️ RISK LEVEL: {risk_level}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Module scores
                    st.subheader("📈 Module-Wise Analysis")
                    module_scores = verification.get("module_scores", {})
                    
                    if module_scores:
                        col1, col2 = st.columns(2)
                        with col1:
                            if module_scores.get("blackbox") is not None:
                                st.progress(module_scores["blackbox"], text=f"Black-Box Consistency: {module_scores['blackbox']:.2%}")
                            if module_scores.get("judge") is not None:
                                st.progress(module_scores["judge"], text=f"LLM Judge Score: {module_scores['judge']:.2%}")
                        with col2:
                            if module_scores.get("whitebox") is not None:
                                st.progress(module_scores["whitebox"], text=f"White-Box Token Confidence: {module_scores['whitebox']:.2%}")
                            if module_scores.get("grounding") is not None:
                                st.progress(module_scores["grounding"], text=f"External Grounding: {module_scores['grounding']:.2%}")
                    
                    # Explanation
                    st.subheader("📝 Audit Explanation")
                    st.info(verification.get("explanation", "No explanation available"))
                    
                    # Grounding evidence
                    grounding = verification.get("grounding_evidence")
                    if grounding:
                        with st.expander("🔗 External Grounding Evidence", expanded=False):
                            if grounding.get("query"):
                                st.markdown(f"**Search Query:** `{grounding['query']}`")
                            
                            snippets = grounding.get("snippets", [])
                            sources = grounding.get("sources", [])
                            
                            if snippets:
                                st.markdown("**Evidence Snippets:**")
                                for i, snippet in enumerate(snippets, 1):
                                    st.markdown(f"{i}. {snippet}")
                                
                                if sources:
                                    st.markdown("**Sources:**")
                                    for i, source in enumerate(sources, 1):
                                        st.markdown(f"[{i}] [{source}]({source})")
                            
                            st.markdown(f"- **Supported:** {'✅ Yes' if grounding.get('supported') else '❌ No'}")
                            st.markdown(f"- **Contradicted:** {'❌ Yes' if grounding.get('contradicted') else '✅ No'}")
                            st.markdown(f"- **Insufficient:** {'⚠️ Yes' if grounding.get('insufficient') else '✅ No'}")
                    
                    # Regenerated response
                    if verification.get("regenerated_response"):
                        st.subheader("🔄 Regenerated Safer Response")
                        st.warning(verification["regenerated_response"])
                    
                    # Errors
                    if verification.get("errors"):
                        with st.expander("⚠️ Errors/Warnings"):
                            for error in verification["errors"]:
                                st.warning(error)
                
                elif result and "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.error("❌ Failed to get verification result")


with tab2:
    st.subheader("Generate a response and verify it")
    
    gen_prompt = st.text_area(
        "User Prompt",
        height=150,
        placeholder="Enter your prompt...",
        key="gen_prompt"
    )
    
    gen_regenerate = st.checkbox(
        "Enable auto-regeneration if hallucination detected",
        value=False,
        key="gen_regenerate"
    )
    
    gen_btn = st.button("✨ Generate & Verify", type="primary", use_container_width=True)
    
    if gen_btn:
        if not gen_prompt:
            st.error("Please enter a prompt")
        elif not is_online:
            st.error("Backend is offline. Please check the API endpoint.")
        else:
            with st.spinner("Generating and analyzing response..."):
                result = generate_and_verify(api_url, gen_prompt, gen_regenerate)
                
                if result and "verification" in result:
                    # Show generated response first
                    st.subheader("📝 Generated Response")
                    st.markdown(result.get("generated_response", "No response generated"))
                    
                    st.divider()
                    
                    # Show verification results
                    verification = result["verification"]
                    
                    if verification:
                        st.success("✅ Verification completed!")
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        
                        truth_score = verification.get("truth_score", 0)
                        confidence_score = verification.get("confidence_score", 0)
                        hallucination_prob = verification.get("hallucination_probability", 0)
                        
                        with col1:
                            st.markdown(render_gauge(
                                truth_score, 
                                "Truth Score", 
                                "#22c55e" if truth_score > 0.7 else "#f59e0b" if truth_score > 0.4 else "#dc2626"
                            ), unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(render_gauge(
                                confidence_score, 
                                "Confidence Score", 
                                "#3b82f6"
                            ), unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(render_gauge(
                                hallucination_prob, 
                                "Hallucination Probability", 
                                "#dc2626" if hallucination_prob > 0.6 else "#f59e0b" if hallucination_prob > 0.3 else "#22c55e"
                            ), unsafe_allow_html=True)
                        
                        # Risk banner
                        risk_level = verification.get("risk_level", "MODERATE")
                        risk_class = f"risk-{risk_level.lower()}"
                        st.markdown(f"""
                        <div class="risk-banner {risk_class}">
                            ⚠️ RISK LEVEL: {risk_level}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Module scores
                        st.subheader("📈 Module-Wise Analysis")
                        module_scores = verification.get("module_scores", {})
                        
                        if module_scores:
                            col1, col2 = st.columns(2)
                            with col1:
                                if module_scores.get("blackbox") is not None:
                                    st.progress(module_scores["blackbox"], text=f"Black-Box Consistency: {module_scores['blackbox']:.2%}")
                                if module_scores.get("judge") is not None:
                                    st.progress(module_scores["judge"], text=f"LLM Judge Score: {module_scores['judge']:.2%}")
                            with col2:
                                if module_scores.get("whitebox") is not None:
                                    st.progress(module_scores["whitebox"], text=f"White-Box Token Confidence: {module_scores['whitebox']:.2%}")
                                if module_scores.get("grounding") is not None:
                                    st.progress(module_scores["grounding"], text=f"External Grounding: {module_scores['grounding']:.2%}")
                        
                        # Explanation
                        st.subheader("📝 Audit Explanation")
                        st.info(verification.get("explanation", "No explanation available"))
                        
                        # Grounding evidence
                        grounding = verification.get("grounding_evidence")
                        if grounding:
                            with st.expander("🔗 External Grounding Evidence", expanded=False):
                                if grounding.get("query"):
                                    st.markdown(f"**Search Query:** `{grounding['query']}`")
                                
                                snippets = grounding.get("snippets", [])
                                sources = grounding.get("sources", [])
                                
                                if snippets:
                                    st.markdown("**Evidence Snippets:**")
                                    for i, snippet in enumerate(snippets, 1):
                                        st.markdown(f"{i}. {snippet}")
                                    
                                    if sources:
                                        st.markdown("**Sources:**")
                                        for i, source in enumerate(sources, 1):
                                            st.markdown(f"[{i}] [{source}]({source})")
                                
                                st.markdown(f"- **Supported:** {'✅ Yes' if grounding.get('supported') else '❌ No'}")
                                st.markdown(f"- **Contradicted:** {'❌ Yes' if grounding.get('contradicted') else '✅ No'}")
                                st.markdown(f"- **Insufficient:** {'⚠️ Yes' if grounding.get('insufficient') else '✅ No'}")
                        
                        # Regenerated response
                        if verification.get("regenerated_response"):
                            st.subheader("🔄 Regenerated Safer Response")
                            st.warning(verification["regenerated_response"])
                
                elif result and "error" in result:
                    st.error(f"❌ Error: {result['error']}")
                else:
                    st.error("❌ Failed to get result")


# Footer
st.divider()
st.markdown("""
<div style="text-align: center; color: #606080; font-size: 0.8em;">
    TruthGuard AI v1.0.0 | Heterogeneous Hallucination Detection Framework
</div>
""", unsafe_allow_html=True)
