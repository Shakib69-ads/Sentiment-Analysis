from pathlib import Path
import html
import re

import joblib
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


st.set_page_config(
    page_title="ReviewLens | Movie Sentiment",
    page_icon="🎬",
    layout="wide"
)

STOP_WORDS = set(ENGLISH_STOP_WORDS) - {
    "no", "not", "nor", "never"
}


def preprocess_review(text):
    text = html.unescape(text)
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("’", "'")
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return " ".join(
        word for word in text.split()
        if word not in STOP_WORDS
    )


@st.cache_resource
def load_model():
    model_path = Path(__file__).parent / "sentiment_model.joblib"
    return joblib.load(model_path)


st.markdown("""
<style>
.stApp {background: #0b1120; color: #e8edf5;}
[data-testid="stHeader"] {background: transparent;}
.block-container {max-width: 1200px; padding-top: 2.4rem; padding-bottom: 2rem;}
.brand {font-size: 23px; font-weight: 800; letter-spacing: -.8px;}
.brand span {color: #75e2c1;}
.eyebrow {color: #75e2c1; font-size: 12px; letter-spacing: 2px; font-weight: 700; margin: 36px 0 14px;}
.hero {font-size: clamp(36px, 5vw, 60px); line-height: 1.08; font-weight: 800; letter-spacing: -2px; margin: 0 0 18px; color: #f5f7fc;}
.hero span {color: #75e2c1;}
.sub {color: #a5b2c7; max-width: 640px; font-size: 17px; line-height: 1.7; margin-bottom: 30px;}
.pill {display: inline-block; border: 1px solid #304354; color: #b2c5d6; padding: 6px 12px; border-radius: 30px; font-size: 12px; margin: 0 8px 12px 0;}
[data-testid="stVerticalBlockBorderWrapper"] > div {border-color: #283449 !important; border-radius: 18px !important; background: #121c2d;}
h3 {font-size: 22px !important; color: #f0f4fa !important;}
[data-testid="stTextArea"] textarea {background: #0d1625 !important; color: #edf3fb !important; font-size: 16px; line-height: 1.7; border-radius: 12px;}
[data-testid="stTextArea"] textarea::placeholder {color: #8493a9;}
[data-testid="stWidgetLabel"] p, [data-testid="stCaptionContainer"] p {color: #a5b2c7 !important;}
.stButton > button {border-radius: 10px; border-color: #34445b; background: #18253a; color: #e6eef9;}
.stButton > button[kind="primary"] {background: #75e2c1; color: #09241e; border: none; font-weight: 750; min-height: 48px;}
.result-symbol {font-size: 42px; margin: 12px 0;}
.result-title {font-size: 32px; font-weight: 750; letter-spacing: -1px; margin-bottom: 10px;}
.muted {color: #a5b2c7; line-height: 1.7; font-size: 14px;}
.score-row {display:flex; justify-content:space-between; font-size:14px; color:#d8e1ee; margin: 18px 0 8px;}
.track {height: 8px; background: #263348; border-radius: 9px; overflow:hidden;}
.track div {height:100%; border-radius:9px;}
.footer {border-top: 1px solid #263348; padding-top: 20px; margin-top: 28px; color:#8c9bb1; font-size:12px;}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="brand">◉ Review<span>Lens</span></div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">MOVIE REVIEW SENTIMENT ANALYZER</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Every review has<br>a <span>point of view.</span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Loved the story or left disappointed? Explore the sentiment behind your words in a single click.</div>', unsafe_allow_html=True)
st.markdown('<span class="pill">English movie reviews</span><span class="pill">Positive / Negative</span>', unsafe_allow_html=True)


def set_example(text):
    st.session_state.review = text
    st.session_state.pop("analysis", None)


left, right = st.columns([1.35, 1], gap="large")
with left:
    with st.container(border=True):
        st.subheader("Your review")
        st.caption("Tell us what you thought of the movie.")
        review = st.text_area(
            "Movie review", key="review", label_visibility="collapsed",
            placeholder="The cinematography was beautiful, and the ending stayed with me long after the credits…",
            height=220
        )
        st.caption(f"{len(review.split())} words · Try an example below")
        positive, negative, clear = st.columns([1, 1, .65])
        positive.button("Loved it", on_click=set_example, args=("A wonderful movie with brilliant acting and a beautiful story. I loved every minute!",), use_container_width=True)
        negative.button("Not a fan", on_click=set_example, args=("A terrible movie with boring dialogue and a predictable plot. I regret watching it.",), use_container_width=True)
        clear.button("Clear", on_click=set_example, args=("",), use_container_width=True)
        analyze = st.button("Analyze sentiment  →", type="primary", use_container_width=True)
        if analyze:
            cleaned = preprocess_review(review)
            if not cleaned:
                st.session_state.pop("analysis", None)
                st.warning("Enter a review with meaningful English words first.")
            else:
                try:
                    model = load_model()
                    prediction = model.predict([cleaned])[0]
                    scores = dict(zip(model.classes_, model.predict_proba([cleaned])[0]))
                    st.session_state.analysis = {"review": review, "prediction": prediction, "scores": scores}
                except FileNotFoundError:
                    st.error("Model file missing. Run the Save the Final Model cell in the notebook first.")

with right:
    with st.container(border=True):
        st.subheader("The verdict")
        result = st.session_state.get("analysis")
        if result and result["review"] == review:
            is_positive = result["prediction"] == "positive"
            color = "#75e2c1" if is_positive else "#ffad99"
            symbol = "↗" if is_positive else "↘"
            label = "Positive" if is_positive else "Negative"
            st.markdown(f'<div class="result-symbol" style="color:{color}">{symbol}</div><div class="result-title" style="color:{color}">{label} sentiment</div>', unsafe_allow_html=True)
            st.markdown('<div class="muted">Here is how the model scored your review.</div>', unsafe_allow_html=True)
            for sentiment, bar_color in [("positive", "#75e2c1"), ("negative", "#ffad99")]:
                score = float(result["scores"][sentiment]) * 100
                st.markdown(f'<div class="score-row"><span>{sentiment.title()}</span><strong>{score:.1f}%</strong></div><div class="track"><div style="width:{score:.2f}%;background:{bar_color}"></div></div>', unsafe_allow_html=True)
            st.caption("Model probabilities are estimates, not a guarantee of correctness.")
        else:
            st.markdown('<div class="result-symbol" style="color:#75e2c1">◎</div><div class="result-title">Words in.<br>Insight out.</div><div class="muted">Your result will appear here. Write a review or choose an example, then select <b>Analyze sentiment</b>.</div>', unsafe_allow_html=True)
        st.divider()
        st.caption("Trained on IMDb movie reviews. Sarcasm and mixed opinions can be harder to interpret.")

st.markdown('<div class="footer">ReviewLens · A sentiment analysis project &nbsp; / &nbsp; Built for English movie reviews</div>', unsafe_allow_html=True)
