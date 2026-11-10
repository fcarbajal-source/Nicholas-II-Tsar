import streamlit as st
from google import genai
from google.genai import types

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Tsar Nicholas II - Historical Voice",
    page_icon="👑",
    layout="centered"
)

st.title("👑 Nicholas II: Autocrat of All the Russias")
st.caption("A historical dialogue designed for U.S. 10th-grade history students.")

# ---------------------------------------------------------
# API Key Verification
# ---------------------------------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.error("Missing `GEMINI_API_KEY`. Please configure it in your Streamlit secrets.")
    st.stop()

# Initialize Gemini Client
client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

# ---------------------------------------------------------
# System Persona Prompt
# ---------------------------------------------------------
SYSTEM_INSTRUCTION = """
You are Tsar Nicholas II (Nikolai Aleksandrovich Romanov), Emperor and Autocrat of All the Russias from 1894 to 1917.

AUDIENCE & TONE:
- Target Audience: 10th-grade high school history students in the United States.
- Voice: Dignified, deeply religious, paternalistic, sincere, and burdened by duty, yet expressed in clear English that a 10th grader easily comprehends. Avoid overly archaic language, but preserve authentic imperial gravity and personal warmth.

MANDATED HISTORICAL SOURCES & ARCHIVES:
Base your answers strictly on primary and secondary records, particularly:
- Personal letters and diaries: Letters to Empress Alexandra ('Alix'), the 'Willy-Nicky' telegrams to Kaiser Wilhelm II, and your private diary entries.
- Official documents: The Coronation speech (1895), October Manifesto (1905), Fundamental Laws (1906), Duma opening addresses, military orders at Mogilev Stavka, and the Abdication Manifesto (March 15, 1917).
- Digital Archival Holdings: State Archive of the Russian Federation (GARF, Moscow), Boris Yeltsin Presidential Library, Yale Beinecke Romanov Collection, Library of Congress (Prints, Photographs, and Chronicling America), Harvard Meeting of Frontiers, Tsarnicholas.org Repository, and RGAKFD.

STRICT OPERATIONAL CONSTRAINTS:
1. WORD COUNT: Responses MUST be between 15 and 120 words maximum.
2. DENSITY: Highly factual and precise. Mention concrete dates, figures (e.g., Stolypin, Witte, Grand Duke Nicholas), and locations (e.g., Tsarskoye Selo, Winter Palace, Mogilev).
3. MANDATORY CITATION: Every response must conclude with an exact bracketed archival or document reference indicating where the primary/secondary evidence originates.
Example citation format: [Source: State Archive of the Russian Federation (GARF), Fond 601 / Letters to Alexandra, 1915].
"""

# ---------------------------------------------------------
# Session State Management (Memory limited to last 6 questions)
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---------------------------------------------------------
# Chat Input & Model Execution
# ---------------------------------------------------------
user_input = st.chat_input("Ask Tsar Nicholas II about his reign, family, or decisions...")

if user_input:
    # Append user question
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Enforce memory constraint: Only retain the last 6 questions (and corresponding answers)
    # 6 questions + 6 answers = max 12 historical messages plus current input
    history_window = st.session_state.messages[-12:]

    # Construct conversation history payload for the model
    contents = []
    for turn in history_window:
        role = "user" if turn["role"] == "user" else "model"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=turn["content"])]
            )
        )

    # Model configuration (Gemini 3.5 Flash-Lite, Temp 0.3, Max Tokens 280)
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        temperature=0.3,
        max_output_tokens=280
    )

    with st.chat_message("assistant"):
        with st.spinner("The Emperor considers your inquiry..."):
            try:
                response = client.models.generate_content(
                    model="gemini-3.5-flash-lite",
                    contents=contents,
                    config=config
                )
                bot_reply = response.text.strip()
                st.markdown(bot_reply)

                # Save assistant response to session state
                st.session_state.messages.append({"role": "assistant", "content": bot_reply})

            except Exception as e:
                st.error(f"Error retrieving response: {e}")
