import streamlit as st

def inject_ui_styles() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');

html, body, [class*="st-"] {
    font-family: "Manrope", "Segoe UI", sans-serif;
}

.block-container {
    padding-top: 1.1rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

.sme-hero {
    border: 1px solid rgba(56, 189, 248, 0.18);
    background: linear-gradient(130deg, rgba(15, 23, 42, 0.95), rgba(22, 78, 99, 0.35));
    border-radius: 14px;
    padding: 1.2rem 1.25rem;
    margin-bottom: 0.9rem;
}

.sme-hero h1 {
    margin: 0 0 0.3rem 0;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.3px;
}

.sme-hero p {
    margin: 0;
    opacity: 0.88;
    font-size: 0.95rem;
}

.sme-kpi {
    border: 1px solid rgba(100, 116, 139, 0.28);
    background: rgba(15, 23, 42, 0.42);
    border-radius: 12px;
    padding: 0.72rem 0.75rem;
    min-height: 106px;
}

.sme-kpi .kpi-title {
    font-size: 0.82rem;
    opacity: 0.86;
    font-weight: 700;
    letter-spacing: 0.2px;
}

.sme-kpi .kpi-value {
    margin-top: 0.28rem;
    font-size: 1.45rem;
    font-weight: 800;
}

.sme-kpi .kpi-caption {
    margin-top: 0.2rem;
    font-size: 0.76rem;
    opacity: 0.78;
}

.sme-kpi.bull {
    border-color: rgba(16, 185, 129, 0.35);
}

.sme-kpi.bear {
    border-color: rgba(244, 63, 94, 0.35);
}

.sme-kpi.warn {
    border-color: rgba(245, 158, 11, 0.35);
}

.sme-section {
    margin-top: 0.25rem;
}
</style>
        """,
        unsafe_allow_html=True,
    )

def render_kpi_card(title: str, value: str, caption: str, tone: str = "neutral") -> None:
    tone_class = {
        "bull": "bull",
        "bear": "bear",
        "warn": "warn",
    }.get(tone, "")
    st.markdown(
        f"""
<div class="sme-kpi {tone_class}">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value">{value}</div>
  <div class="kpi-caption">{caption}</div>
</div>
        """,
        unsafe_allow_html=True,
    )
