import json
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import google.generativeai as genai
import anthropic
from llamaapi import LlamaAPI
from openai import OpenAI
import speech_recognition as sr
from gtts import gTTS  
import tempfile  
from langsmith import wrappers
import openai
client = wrappers.wrap_openai(openai.Client())


st.title("Speak with AI Models")

API_KEY = st.sidebar.text_input("Enter Your API Key Here", type="password")

# Function to Generate Response and Provide Text and Speech Output
def display_response(response_text):
    if response_text:
        # Display the text response
        st.info(response_text)
        
        # Convert text to speech and play the audio
        tts = gTTS(text=response_text, lang="en")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_file:
            temp_file_path = tmp_file.name
        
        tts.save(temp_file_path)
        st.audio(temp_file_path, format="audio/mp3")
        
    else:
        st.error("No response received.")

def pipeline(input_text: str):  # LangChain pipeline function
    # Assuming `client` is defined earlier and configured correctly
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": input_text}]
    )
    content = response.choices[0].message.content
    display_response(content)


def openai(input_text):  # OpenAI
    model = ChatOpenAI(temperature=0.7, api_key=API_KEY)
    response = model.invoke(input_text)
    display_response(response.content)


def gemini(input_text):  # Gemini
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel("gemini-1.5-flash")
    response = model.generate_content(input_text)
    display_response(response.text)


def claude(input_text):  # Claude
    client = anthropic.Anthropic(api_key=API_KEY)
    response = client.messages.create(
        model="claude-3-5-sonnet-20241022",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": input_text}
        ]
    )
    display_response(response.get("content", "No response content available."))


def grok(input_text):  # Grok
    client = OpenAI(
        api_key=API_KEY,
        base_url="https://api.x.ai/v1",
    )
    completion = client.chat.completions.create(
        model="grok-beta",
        messages=[
            {"role": "system", "content": "You are Grok, a chatbot inspired by the Hitchhiker's Guide to the Galaxy."},
            {"role": "user", "content": input_text},
        ],
    )
    display_response(completion.choices[0].message.content)


def llama(input_text):  # Llama
    llama_client = LlamaAPI(API_KEY)
    api_request_json = {
        "model": "llama3.1-70b",
        "messages": [
            {"role": "user", "content": input_text}
        ]
    }
    response = llama_client.run(api_request_json)
    display_response(response)


def gemma(input_text):  # Gemma
    llama_client = LlamaAPI(API_KEY)
    api_request_json = {
        "model": "gemma-2b",
        "messages": [
            {"role": "user", "content": input_text}
        ]
    }
    response = llama_client.run(api_request_json)
    display_response(response)



st.header("Choose Input Method")
input_method = st.radio("How would you like to interact with the model?", ["Text", "Mic { Coming Soon...}"])


option = st.selectbox(
    "Which model do you want?",
    ("OpenAI", "Gemini", "Claude", "Grok", "Llama", "Gemma", "Langchain"),
)

st.write("You selected:", option)


model_functions = {
    "OpenAI": openai,
    "Gemini": gemini,
    "Claude": claude,
    "Grok": grok,
    "Llama": llama,
    "Gemma": gemma,
    "Langchain": pipeline
}


input_text = None

if input_method == "Text":
    input_text = st.text_input("Enter your prompt 👇")
elif input_method == "Audio":
    audio_value = st.file_uploader("Record or upload a voice message (WAV format)", type=["wav"])
    if audio_value is not None:
        try:
            # Convert audio to text using SpeechRecognition
            recognizer = sr.Recognizer()
            with sr.AudioFile(audio_value) as source:
                audio_data = recognizer.record(source)
                input_text = recognizer.recognize_google(audio_data)
                st.success(f"Recognized text: {input_text}")
        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")


if input_text and option in model_functions:
    model_function = model_functions[option]
    model_function(input_text)
elif not input_text:
    st.warning("Please provide some input to proceed.")
