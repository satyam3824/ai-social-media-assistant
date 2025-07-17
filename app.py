# app.py
import streamlit as st
import asyncio
import os
import time
import json 
import base64 
import requests 
from dotenv import load_dotenv 
import pandas as pd 
import numpy as np 
import subprocess 
import sys # Import sys for subprocess.check_call

load_dotenv() 

# Function to install a package
def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        st.success(f"Successfully installed {package}")
    except Exception as e:
        st.error(f"Failed to install {package}: {e}")

# Check and install docx and pypdf2 if not already installed
# These imports are placed here to ensure installation attempts happen before their use
try:
    import docx
except ImportError:
    st.warning("`python-docx` not found. Attempting to install...")
    install_package("python-docx")
    try:
        import docx # Try importing again after installation
    except ImportError:
        st.error("Failed to import `python-docx` even after installation attempt. Please install manually if issues persist.")

try:
    from PyPDF2 import PdfReader
except ImportError:
    st.warning("`PyPDF2` not found. Attempting to install...")
    install_package("PyPDF2")
    try:
        from PyPDF2 import PdfReader # Try importing again after installation
    except ImportError:
        st.error("Failed to import `PyPDF2` even after installation attempt. Please install manually if issues persist.")


# For LLM interaction
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# For YouTube transcript fetching
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# --- Configuration (from .env) ---
# Load API keys from environment variables
# For local testing, ensure these are set in your .env file
# For Canvas, GOOGLE_API_KEY will be automatically provided.
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# IMPORTANT: Get your Unsplash Access Key from https://unsplash.com/developers
UNSPLASH_ACCESS_KEY = os.getenv("UNSPLASH_ACCESS_KEY", "")

# Configure Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# --- Pydantic Models for Structured Output ---
class SocialMediaContent(BaseModel):
    post_text: str = Field(description="The main text for the social media post.")
    hashtags: list[str] = Field(description="A list of relevant hashtags for the post.")
    call_to_action: str = Field(description="A clear call to action for the post.")
    tone_suggestions: str = Field(description="Suggestions for different tones for the post.")

class Slide(BaseModel):
    title: str = Field(description="The concise title of the presentation slide.")
    bullet_points: list[str] = Field(description="A list of 3-5 key bullet points for the slide content.")

class Presentation(BaseModel):
    slides: list[Slide] = Field(description="A list of presentation slides, each with a title and bullet points.")


# --- LLM Interface Functions ---
async def generate_text_with_gemini(prompt_template, input_variables, output_parser=None, model_name="gemini-2.5-flash"):
    """
    Generic function to generate text using Gemini via Langchain.
    Can use an optional output_parser for structured output.
    """
    if not GOOGLE_API_KEY:
        st.error("Google API Key not set. Please ensure it's configured in your environment or Streamlit secrets, and linked to a project with access to the Gemini API.")
        return None

    llm = ChatGoogleGenerativeAI(model=model_name)
    
    if output_parser:
        chain = prompt_template | llm | output_parser
    else:
        chain = prompt_template | llm

    try:
        response = await chain.ainvoke(input_variables)
        return response
    except Exception as e:
        st.error(f"Error generating content with LLM: {e}")
        st.exception(e) # Display full exception traceback for debugging
        return "I'm sorry, an error occurred during content generation." # More specific error message

# --- Image Generation Function (using Unsplash) ---
def generate_image_with_unsplash(prompt_text):
    """
    Fetches an image from Unsplash based on the provided text prompt.
    Returns the URL of the image or a placeholder URL on error.
    """
    print(f"DEBUG: generate_image_with_unsplash called with prompt: {prompt_text}")
    if not UNSPLASH_ACCESS_KEY:
        st.error("Unsplash Access Key not set. Cannot generate image. Please ensure it's configured in your .env file.")
        print("DEBUG: UNSPLASH_ACCESS_KEY is not set.")
        return "https://placehold.co/400x250/FF0000/FFFFFF?text=Unsplash+Key+Missing"

    try:
        # Unsplash API endpoint for searching photos
        # Using a more specific search query for better results
        search_query = f"{prompt_text} social media post" 
        unsplash_api_url = f"https://api.unsplash.com/search/photos?query={requests.utils.quote(search_query)}&per_page=1&client_id={UNSPLASH_ACCESS_KEY}"
        print(f"DEBUG: Calling Unsplash API at: {unsplash_api_url}")

        response = requests.get(unsplash_api_url)
        print(f"DEBUG: Unsplash API response status code: {response.status_code}")
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        
        data = response.json()
        print(f"DEBUG: Unsplash API response JSON: {data}")

        if data.get("results") and len(data["results"]) > 0:
            image_url = data["results"][0]["urls"]["regular"] # Get the regular size image URL
            print(f"DEBUG: Image found: {image_url}")
            return image_url
        else:
            st.warning("Unsplash API did not return a valid image for your prompt. Try a different one.")
            print("DEBUG: Unsplash API did not return valid results.")
            return "https://placehold.co/400x250/FFA500/000000?text=No+Image+Found"
    except requests.exceptions.RequestException as e:
        st.error(f"Error calling Unsplash API: {e}. Check your Unsplash Access Key and network connection.")
        st.exception(e) # Display full exception traceback for debugging
        print(f"DEBUG: requests.exceptions.RequestException: {e}")
        return "https://placehold.co/400x250/FF0000/FFFFFF?text=Image+Error"
    except Exception as e:
        st.error(f"An unexpected error occurred during image generation: {e}")
        st.exception(e) # Display full exception traceback for debugging
        print(f"DEBUG: Unexpected Exception during image generation: {e}")
        return "https://placehold.co/400x250/FF0000/FFFFFF?text=Image+Error"

# --- Content Generation Functions ---
async def generate_social_media_post(topic, platform, tone, audience, length="short", rag_context=None):
    """Generates a social media post with hashtags and CTA."""
    parser = JsonOutputParser(pydantic_object=SocialMediaContent)
    
    system_prompt = "You are an expert social media content creator. Generate a compelling post with relevant hashtags and a clear call to action, formatted as JSON."
    if rag_context:
        system_prompt += f"\n\nUse the following context to inform your response: {rag_context}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", """
            Create a {length} social media post for {platform} on the topic: "{topic}".
            The post should have a {tone} tone and be aimed at an {audience} audience.
            Include 3-5 relevant hashtags and a clear call to action.

            {format_instructions}
            """)
        ]
    )
    response = await generate_text_with_gemini(
        prompt,
        {"topic": topic, "platform": platform, "tone": tone, "audience": audience, "length": length, "format_instructions": parser.get_format_instructions()},
        output_parser=parser
    )
    return response

async def generate_presentation_slides(topic, num_slides, language, tone, audience, scene, rag_context=None):
    """
    Generates presentation slide content using Google Gemini via Langchain.
    Returns a list of dictionaries, each representing a slide.
    """
    parser = JsonOutputParser(pydantic_object=Presentation)

    system_prompt = "You are an expert presentation content generator. Create a presentation outline based on the user's input, formatted as a JSON array of slides."
    if rag_context:
        system_prompt += f"\n\nUse the following context to inform your response: {rag_context}"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", """
            Generate a {num_slides}-slide presentation outline on the topic: "{topic}".
            Each slide should have a concise title and 3-5 bullet points.
            The presentation should be in {language}, with a {tone} tone, for an {audience} audience, and styled for a {scene} scene.

            {format_instructions}
            """)
        ]
    )

    response = await generate_text_with_gemini(
        prompt,
        {
            "topic": topic,
            "num_slides": num_slides,
            "language": language,
            "tone": tone,
            "audience": audience,
            "scene": scene,
            "format_instructions": parser.get_format_instructions()
        },
        output_parser=parser
    )
    return response.get('slides', []) if response else []


# --- Engagement Bot Functions ---
async def get_ai_response(user_message):
    """
    Generates an AI response for the chatbot.
    Conversation history is managed in Streamlit's session state.
    """
    # Format chat history into a string for direct injection into the prompt
    formatted_chat_history = ""
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            formatted_chat_history += f"Human: {msg['content']}\n"
        elif msg["role"] == "ai":
            formatted_chat_history += f"AI: {msg['content']}\n"

    # Define the prompt template without MessagesPlaceholder
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful social media engagement assistant. Respond concisely and professionally."),
        ("human", f"Conversation history:\n{formatted_chat_history}\nUser: {{input}}") # Inject history directly
    ])

    # Create the chain - no need for RunnablePassthrough.assign for chat_history anymore
    chain = prompt_template | ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    try:
        response = await chain.ainvoke({"input": user_message})
        return response.content
    except Exception as e:
        st.error(f"Error getting AI response: {e}")
        st.exception(e) # Display full exception traceback for debugging
        return "I'm sorry, I couldn't process that right now."

# --- Sentiment Analysis Function ---
async def analyze_sentiment(text):
    """Analyzes the sentiment of the given text."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert sentiment analysis AI. Analyze the sentiment of the following text and categorize it as 'Positive', 'Negative', 'Neutral', or 'Mixed'. Provide only the category."),
            ("human", "Analyze the sentiment of this text: '{text}'")
        ]
    )
    response = await generate_text_with_gemini(
        prompt,
        {"text": text},
        output_parser=None # We want raw text output for simplicity here
    )
    return response

# --- Utility Functions ---
def read_txt(file):
    """Reads content from a .txt file."""
    return file.read().decode("utf-8")

def read_docx(file):
    """Reads content from a .docx file."""
    # Ensure docx is imported, if not, it means installation failed or is pending
    try:
        import docx
        doc = docx.Document(file)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return "\n".join(full_text)
    except ImportError:
        st.error("`python-docx` library not found. Please ensure it's installed.")
        return ""


def read_pdf(file):
    """Reads content from a .pdf file."""
    # Ensure PyPDF2 is imported, if not, it means installation failed or is pending
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file)
        full_text = []
        for page in reader.pages:
            full_text.append(page.extract_text())
        return "\n".join(full_text)
    except ImportError:
        st.error("`PyPDF2` library not found. Please ensure it's installed.")
        return ""

def get_youtube_transcript(youtube_url):
    """
    Fetches the transcript from a YouTube video URL.
    Returns the concatenated transcript text or None if not available/error.
    """
    try:
        video_id = youtube_url.split("v=")[1].split("&")[0]
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        transcript = None
        try:
            transcript = transcript_list.find_transcript(['en'])
        except NoTranscriptFound:
            if transcript_list:
                for t in transcript_list:
                    transcript = t
                    break
        
        if transcript:
            full_transcript_data = transcript.fetch()
            # FIX: Access the 'text' attribute directly from the FetchedTranscriptSnippet object
            return " ".join([entry.text for entry in full_transcript_data])
        else:
            st.warning("No transcript found for the provided YouTube video in English or any other language.")
            return None
    except NoTranscriptFound:
        st.warning("No transcript found for this YouTube video. It might not have captions or they are disabled.")
        return None
    except TranscriptsDisabled:
        st.warning("Transcripts are disabled for this YouTube video.")
        return None
    except Exception as e:
        st.error(f"Error fetching YouTube transcript: {e}. Please check the URL and video availability.")
        st.exception(e) # Display full exception traceback for debugging
        return None

# --- RAG System (Simulated/Placeholder) ---
async def process_document_for_rag(uploaded_file):
    """
    Simulates processing a document for RAG. In a real application, this would
    involve chunking, embedding, and storing in a vector database.
    For this simulation, it simply returns the content.
    """
    if uploaded_file is None:
        return None

    file_type = uploaded_file.type
    content = ""
    try:
        if "text/plain" in file_type:
            content = read_txt(uploaded_file)
        elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in file_type:
            content = read_docx(uploaded_file)
        elif "application/pdf" in file_type:
            content = read_pdf(uploaded_file)
        if content: # Only show success if content was actually read
            st.success("Document content extracted for RAG!")
        return content # Return content as a simple "vector store"
    except Exception as e:
        st.error(f"Error processing RAG document: {e}")
        return None

async def retrieve_context_for_query(query, document_content):
    """
    Simulates retrieving relevant context from a 'document_content' based on a query.
    In a real RAG system, this would query a vector database.
    For this simulation, it returns a snippet of the document or a general statement.
    """
    if not document_content:
        return "No document context available."
    
    # Simple simulation: return the first 200 characters of the document
    # or a more sophisticated LLM call to extract relevant parts.
    
    # For a more realistic simulation, use an LLM to extract relevant parts
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Given the following document content:\n\n{document_content}\n\nExtract the most relevant sentences or paragraphs that answer the question: '{query}'. If no direct answer is found, summarize relevant information. Keep the response concise."
    
    try:
        response = await asyncio.to_thread(lambda: model.generate_content(prompt))
        return response.text
    except Exception as e:
        st.warning(f"Could not retrieve RAG context using LLM: {e}. Returning simple snippet.")
        return document_content[:500] + "..." if len(document_content) > 500 else document_content


# --- Trend Monitoring Function (Simulated) ---
async def simulate_trend_monitoring(keywords):
    """
    Simulates trend monitoring using Gemini to generate trending topics
    based on provided keywords.
    """
    if not GOOGLE_API_KEY:
        return ["Google API Key not set. Cannot simulate trend monitoring."]

    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"Generate 3-5 simulated trending topics related to the following keywords: '{keywords}'. Each topic should be a concise phrase. Format as a comma-separated list. Do not include any conversational text, just the topics."
    
    try:
        response = await asyncio.to_thread(lambda: model.generate_content(prompt))
        # Split the response into a list of topics, handling potential extra spaces
        topics = [topic.strip() for topic in response.text.strip().split(',')]
        return topics
    except Exception as e:
        st.error(f"Error simulating trend monitoring: {e}")
        return ["Failed to retrieve simulated trends. Please try again."]

# --- Simulated Social Media API Functions ---
def post_to_facebook(access_token, message):
    """Simulates posting a message to Facebook."""
    # Simulate success if token is present (even if it's the placeholder)
    if access_token: # Check if it's not empty
        return {"status": "success", "message": f"Simulated Facebook post: '{message}'"}
    else:
        return {"status": "error", "message": "Facebook Access Token is not set."}

def send_instagram_dm(access_token, recipient_id, message):
    """Simulates sending an Instagram DM."""
    if access_token and recipient_id: # Check if both are present
        return {"status": "success", "message": f"Simulated Instagram DM to {recipient_id}: '{message}'"}
    else:
        return {"status": "error", "message": "Instagram Access Token or Recipient ID is not set."}

def post_to_twitter(api_key, api_secret, access_token, access_token_secret, message):
    """Simulates posting a tweet to X (Twitter)."""
    if all([api_key, api_secret, access_token, access_token_secret]): # Check if all are present
        return {"status": "success", "message": f"Simulated X (Twitter) tweet: '{message}'"}
    else:
        return {"status": "error", "message": "Twitter API credentials are not fully set."}

def send_whatsapp_message(phone_number_id, recipient_number, message, access_token):
    """Simulates sending a WhatsApp message."""
    if all([phone_number_id, recipient_number, message, access_token]): # Check if all are present
        return {"status": "success", "message": f"Simulated WhatsApp message to {recipient_number}: '{message}'"}
    else:
        return {"status": "error", "message": "WhatsApp API credentials or recipient number are not fully set."}


# --- Streamlit UI Configuration ---
st.set_page_config(page_title="AI Social Media Assistant", layout="wide")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }
    .stApp {
        background-color: #000000; /* Changed to black */
        padding: 20px;
    }
    .container {
        background-color: #ffffff;
        border-radius: 1.5rem;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        padding: 2.5rem;
        max-width: 900px;
        margin: 20px auto;
    }
    h1 {
        text-align: center;
        font-size: 2.5rem !important; /* text-4xl */
        font-weight: 800 !important; /* font-extrabold */
        margin-bottom: 2rem !important; /* mb-8 */
        background: linear-gradient(to right, #8b5cf6, #6366f1); /* from-purple-600 to-indigo-600 */
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    /* Ensure text is visible on black background */
    .stTextArea label, .stSelectbox label, .stFileUploader label,
    .stRadio label, .stTextInput label, .stMarkdown p, .stMarkdown li, .stMarkdown h3, .stMarkdown h4, .stInfo, .stSuccess, .stWarning, .stError, .stCheckbox label {
        color: #CCCCCC !important; /* Light gray for better contrast on black */
    }
    /* FIX: Input field styling for better visibility */
    .stTextArea textarea, .stSelectbox select, .stFileUploader input[type="file"], .stTextInput input[type="text"],
    .stTextInput input[type="password"] { /* Added password input */
        padding: 0.75rem; /* p-3 */
        border: 1px solid #4a5568; /* Darker border for contrast */
        border-radius: 0.5rem; /* rounded-lg */
        outline: none;
        box-shadow: none;
        background-color: #2d3748; /* Darker background for input */
        color: #e2e8f0; /* Light text for input */
    }
    /* FIX: Placeholder text color for input fields */
    .stTextArea textarea::placeholder, .stTextInput input::placeholder {
        color: #a0aec0; /* Lighter placeholder text */
    }

    .stButton > button {
        background: linear-gradient(to right, #8b5cf6, #6366f1); /* from-purple-600 to-indigo-600 */
        color: white;
        font-weight: 600; /* font-semibold */
        padding: 0.75rem 1.5rem; /* py-3 px-6 */
        border-radius: 0.75rem; /* rounded-xl */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* shadow-lg */
        transition: all 0.3s ease;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .stButton > button:hover {
        background: linear-gradient(to right, #7c3aed, #4f46e5); /* hover:from-purple-700 hover:to-indigo-700 */
        transform: scale(1.02); /* slight scale for hover effect */
    }
    .slide-card, .content-card, .chat-card {
        background-color: #2d3748; /* Darker background for output cards */
        border: 1px solid #4a5568; /* Darker border */
        border-radius: 0.75rem; /* rounded-xl */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); /* shadow-lg */
        padding: 1.5rem; /* p-6 */
        margin-bottom: 1.5rem; /* mb-6 */
        color: #e2e8f0; /* Light text for output cards */
    }
    .slide-card h3, .content-card h3, .chat-card h3 {
        font-size: 1.25rem; /* text-xl */
        font-weight: 700; /* font-bold */
        color: #e2e8f0; /* Light text for headings in cards */
        margin-bottom: 0.75rem; /* mb-3 */
    }
    .slide-card ul, .content-card ul {
        list-style-type: disc;
        padding-left: 1.25rem; /* list-inside */
        color: #cbd5e0; /* Slightly darker light text for lists */
    }
    .slide-card img, .content-card img {
        width: 100%;
        height: 12rem; /* h-48 */
        object-fit: cover;
        border-radius: 0.5rem; /* rounded-lg */
        margin-top: 1rem; /* mt-4 */
    }
    .chat-message-user {
        background-color: #4c51bf; /* User message bubble (indigo-700) */
        padding: 0.75rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        text-align: right;
        color: white; /* White text for user messages */
    }
    .chat-message-ai {
        background-color: #667eea; /* AI message bubble (indigo-500) */
        padding: 0.75rem;
        border-radius: 0.75rem;
        margin-bottom: 0.5rem;
        text-align: left;
        color: white; /* White text for AI messages */
    }
    </style>
""", unsafe_allow_html=True)

st.title("AI Social Media Assistant")

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Content Generator", "Engagement Assistant", "Analytics & Insights"], # Consolidated Sentiment Analysis into Analytics
    key="main_navigation"
)

# --- Content Generator Page ---
if page == "Content Generator":
    st.header("✍️ AI Social Media Content Generator")
    st.markdown("Create engaging posts, captions, and ad copy for your social media channels.")

    input_method = st.radio(
        "Choose your input method:",
        ("Text Input", "Upload Document", "YouTube Link"), # Added Upload Document
        key="content_input_method",
        horizontal=True
    )

    content_text = ""
    uploaded_file = None
    youtube_url = ""
    
    if input_method == "Text Input":
        content_text = st.text_area(
            "Enter your topic or content for social media:",
            placeholder="e.g., 'Benefits of AI in marketing' or paste your article here...",
            height=150,
            key="content_topic_text"
        )
    elif input_method == "Upload Document":
        uploaded_file = st.file_uploader("Upload your document (TXT, DOCX, PDF):", type=["txt", "docx", "pdf"], key="content_file_upload")
        if uploaded_file is not None:
            try:
                if "text/plain" in uploaded_file.type:
                    content_text = uploaded_file.read().decode("utf-8")
                elif "application/vnd.openxmlformats-officedocument.wordprocessingml.document" in uploaded_file.type:
                    content_text = read_docx(uploaded_file)
                elif "application/pdf" in file_type:
                    content_text = read_pdf(uploaded_file)
                st.success("File uploaded and content read.")
            except Exception as e:
                st.error(f"Error reading file: {e}. Please ensure it's a valid TXT, DOCX, or PDF.")
                content_text = ""
    elif input_method == "YouTube Link":
        youtube_url = st.text_input(
            "Enter YouTube Video URL:",
            placeholder="e.g., https://www.youtube.com/watch?v=dQw4w9WgXcQ",
            key="content_youtube_url_input"
        )
        if youtube_url:
            with st.spinner("Fetching YouTube transcript..."):
                transcript_content = get_youtube_transcript(youtube_url)
                if transcript_content:
                    content_text = transcript_content
                    st.success("YouTube transcript fetched successfully!")
                else:
                    st.error("Could not fetch YouTube transcript. Please ensure the video has public captions/subtitles or try another video.")
                    st.stop()
    
    # --- Content Generation Options ---
    st.markdown("---")
    st.subheader("Content Options")
    col1, col2 = st.columns(2)
    with col1:
        content_type = st.selectbox("Content Type:", ["Social Media Post", "Caption", "Ad Copy", "Tweet", "Presentation Slides"], key="content_type_select") # Added Presentation Slides
    with col2:
        social_platform = st.selectbox("Target Platform:", ["General", "Facebook", "Instagram", "X (Twitter)", "LinkedIn"], key="social_platform_select")
    
    col3, col4 = st.columns(2)
    with col3:
        content_tone = st.selectbox("Tone:", ["General", "Informative", "Promotional", "Casual", "Professional", "Humorous"], key="content_tone_select")
    with col4:
        content_audience = st.selectbox("Audience:", ["General Public", "Professionals", "Students", "Customers", "Investors"], key="content_audience_select")

    # New: Add a selectbox for content length
    content_length = st.selectbox(
        "Content Length:",
        ["short", "medium", "long"],
        index=0, # Default to "short"
        help="Choose the desired length for your generated content."
    )

    # RAG Integration for Content Generation (Advanced Feature)
    st.markdown("---")
    st.subheader("Advanced Content Options (RAG)")
    enable_rag_content = st.checkbox("Enable RAG for Content Generation (Use your documents for context)", key="enable_rag_content_checkbox")
    rag_document_content = None
    if enable_rag_content:
        rag_uploaded_file = st.file_uploader("Upload document for RAG context (TXT, DOCX, PDF):", type=["txt", "docx", "pdf"], key="rag_content_doc_upload")
        if rag_uploaded_file:
            with st.spinner("Processing document for RAG..."):
                rag_document_content = asyncio.run(process_document_for_rag(rag_uploaded_file))
                if rag_document_content:
                    st.success("Document loaded for RAG context!")
                else:
                    st.warning("Failed to load document for RAG.")
    
    if st.button("Generate Content", key="generate_content_button"):
        if not content_text:
            st.error("Please provide content via text, document, or YouTube link.")
        else:
            with st.spinner("Generating your content..."):
                final_topic_for_llm = content_text
                current_rag_context = None
                if enable_rag_content and rag_document_content:
                    current_rag_context = asyncio.run(retrieve_context_for_query(final_topic_for_llm, rag_document_content))
                    if current_rag_context:
                        st.info("RAG context retrieved for generation.")
                    else:
                        st.warning("Could not retrieve RAG context for the query.")

                if content_type == "Presentation Slides":
                    num_slides_pres = st.number_input("Number of Slides (for presentation):", min_value=1, value=5, key="num_slides_pres_input_hidden") # Use a unique key
                    generated_output = asyncio.run(
                        generate_presentation_slides(
                            final_topic_for_llm, 
                            num_slides_pres, 
                            "English", 
                            content_tone, 
                            content_audience, 
                            "General", 
                            rag_context=current_rag_context 
                        )
                    )
                    st.session_state['generated_presentation_slides_result'] = generated_output
                    st.session_state['generated_social_media_post_result'] = None # Clear social media post if generating presentation
                else:
                    generated_post_result = asyncio.run(
                        generate_social_media_post(
                            final_topic_for_llm, 
                            social_platform, 
                            content_tone, 
                            content_audience, 
                            length=content_length,
                            rag_context=current_rag_context 
                        )
                    )
                    st.session_state['generated_social_media_post_result'] = generated_post_result
                    st.session_state['generated_presentation_slides_result'] = None # Clear presentation if generating social media
            
            # Reset image states on new content generation
            st.session_state['generate_image_checked_state'] = False
            if 'generated_image_url' in st.session_state:
                del st.session_state['generated_image_url']
            if 'last_image_prompt' in st.session_state:
                del st.session_state['last_image_prompt']
            
            st.rerun() # Force rerun to display content and image options

    # --- Display Generated Content and Image Options (OUTSIDE the Generate Content button block) ---
    # Display Social Media Post
    if 'generated_social_media_post_result' in st.session_state and \
       st.session_state['generated_social_media_post_result'] and \
       st.session_state.get('content_type_select') != "Presentation Slides": # Check current content_type_select
        
        generated_post = st.session_state['generated_social_media_post_result']
        
        st.subheader(f"Generated {st.session_state.get('content_type_select', 'Content')}:")
        st.markdown(f"""
            <div class="content-card">
                <p><strong>Post Text:</strong> {generated_post.get('post_text', 'N/A')}</p>
                <p><strong>Hashtags:</strong> {', '.join(generated_post.get('hashtags', []))}</p>
                <p><strong>Call to Action:</strong> {generated_post.get('call_to_action', 'N/A')}</p>
                <p><strong>Tone Suggestions:</strong> {generated_post.get('tone_suggestions', 'N/A')}</p>
            </div>
        """, unsafe_allow_html=True)
        st.success("Content generated!")

        st.markdown("---")
        st.subheader("Fetch Image for Post (from Unsplash)")
        st.info("Note: Unsplash provides existing images based on your prompt.")
        
        # Initialize checkbox state if not already present
        if 'generate_image_checked_state' not in st.session_state:
            st.session_state['generate_image_checked_state'] = False

        generate_image_checked = st.checkbox(
            "Fetch a relevant image for this post?", 
            key="generate_post_image_checkbox_after_gen",
            value=st.session_state['generate_image_checked_state']
        )
        
        # Update session state based on checkbox value
        st.session_state['generate_image_checked_state'] = generate_image_checked

        if generate_image_checked:
            # Only fetch image if it's not already in session state or if the prompt has changed significantly
            # and if the checkbox is currently checked
            if 'generated_image_url' not in st.session_state or \
               st.session_state.get('last_image_prompt') != generated_post.get('post_text', ''):
                with st.spinner("Fetching image from Unsplash... This may take a moment."):
                    image_prompt = generated_post.get('post_text', '') 
                    generated_image_url = generate_image_with_unsplash(image_prompt)
                    st.session_state['generated_image_url'] = generated_image_url
                    st.session_state['last_image_prompt'] = image_prompt # Store prompt to detect changes
            
            if st.session_state.get('generated_image_url'):
                st.image(st.session_state['generated_image_url'], caption="Fetched Image for Post", use_container_width=True)
            else:
                st.error("Failed to fetch image.")
        else:
            # If checkbox is unchecked, clear the image from session state
            if 'generated_image_url' in st.session_state:
                del st.session_state['generated_image_url']
            if 'last_image_prompt' in st.session_state:
                del st.session_state['last_image_prompt']

    # Display Presentation Slides
    elif 'generated_presentation_slides_result' in st.session_state and \
         st.session_state['generated_presentation_slides_result'] and \
         st.session_state.get('content_type_select') == "Presentation Slides": # Check current content_type_select
        
        st.subheader("Generated Presentation Outline:")
        for i, slide in enumerate(st.session_state['generated_presentation_slides_result']):
            st.markdown(f"""
                <div class="slide-card">
                    <h3>Slide {i+1}: {slide['title']}</h3>
                    <ul>
                        {"".join([f"<li>{bp}</li>" for bp in slide['bullet_points']])}
                    </ul>
                </div>
            """, unsafe_allow_html=True)
        st.info("Presentation outline generated. PPTX download and image generation for slides would be next steps.")


# --- Engagement Assistant Page ---
elif page == "Engagement Assistant":
    st.header("🤖 AI Social Media Engagement Assistant")
    st.markdown("Interact intelligently with your audience using AI-powered chatbot features.")

    st.subheader("Chat with your AI Assistant")

    # Initialize chat history in session state (no persistence in this version)
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'<div class="chat-message-user">You: {message["content"]}</div>', unsafe_allow_html=True)
        elif message["role"] == "ai":
            st.markdown(f'<div class="chat-message-ai">AI: {message["content"]}</div>', unsafe_allow_html=True)

    user_input = st.chat_input("Type your message here...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.markdown(f'<div class="chat-message-user">You: {user_input}</div>', unsafe_allow_html=True)

        with st.spinner("AI is thinking..."):
            ai_response = asyncio.run(get_ai_response(user_input))
        
        st.session_state.messages.append({"role": "ai", "content": ai_response})
        st.markdown(f'<div class="chat-message-ai">AI: {ai_response}</div>', unsafe_allow_html=True)
        st.rerun() # Rerun to display new messages

    # --- Social Media Direct Integration (Simulated) ---
    st.markdown("---")
    st.subheader("Social Media Direct Integration (Simulated)")
    st.info("This section demonstrates where direct social media API integrations would be. Full functionality requires platform-specific API keys and OAuth setup.")
    
    platform_to_post = st.selectbox("Select platform for direct action:", ["None", "Facebook", "Instagram DM", "X (Twitter)", "WhatsApp"], key="direct_platform_select")
    
    if platform_to_post != "None":
        message_to_send = st.text_area(f"Message to send to {platform_to_post}:", key="direct_message_input")
        
        # Additional inputs for specific platforms
        recipient_id = ""
        whatsapp_recipient_number = ""

        if platform_to_post == "Instagram DM":
            recipient_id = st.text_input("Recipient Instagram User ID:", key="insta_recipient_id", help="e.g., user_id_123")
        elif platform_to_post == "WhatsApp":
            whatsapp_recipient_number = st.text_input("Recipient WhatsApp Number (e.g., +1234567890):", key="whatsapp_recipient_num")

        if st.button(f"Send to {platform_to_post} (Simulated)", key="send_direct_button"):
            if message_to_send:
                result = {"status": "error", "message": "Simulated failure. API not configured."} # Default error

                if platform_to_post == "Facebook":
                    # Placeholder for actual Facebook API call
                    result = post_to_facebook("YOUR_FACEBOOK_ACCESS_TOKEN", message_to_send)
                elif platform_to_post == "Instagram DM":
                    if recipient_id:
                        result = send_instagram_dm("YOUR_INSTAGRAM_ACCESS_TOKEN", recipient_id, message_to_send)
                    else:
                        st.warning("Please provide a recipient ID for Instagram DM.")
                        result = None # Prevent success message if recipient missing
                elif platform_to_post == "X (Twitter)":
                    # Placeholder for actual Twitter API call
                    result = post_to_twitter("YOUR_TWITTER_API_KEY", "YOUR_TWITTER_API_SECRET", "YOUR_TWITTER_ACCESS_TOKEN", "YOUR_TWITTER_ACCESS_TOKEN_SECRET", message_to_send)
                elif platform_to_post == "WhatsApp":
                    if whatsapp_recipient_number:
                        result = send_whatsapp_message("YOUR_WHATSAPP_PHONE_NUMBER_ID", whatsapp_recipient_number, message_to_send, "YOUR_WHATSAPP_ACCESS_TOKEN")
                    else:
                        st.warning("Please provide a recipient WhatsApp number.")
                        result = None # Prevent success message if recipient missing
                
                if result: # Only show status if result is not None (i.e., not a missing recipient warning)
                    if result['status'] == "success":
                        st.success(f"Simulated {platform_to_post} Status: {result['message']}")
                    else:
                        st.error(f"Simulated {platform_to_post} Error: {result['message']}")
            else:
                st.warning("Please enter a message to send.")


# --- Analytics & Insights Page ---
elif page == "Analytics & Insights":
    st.header("📊 Analytics & Insights")
    st.markdown("Track your content performance, analyze sentiment, and identify trends.")

    st.subheader("Content Performance Metrics (Simulated)")
    st.info("This section displays simulated content performance data. In a real app, this would be powered by actual social media API data.")

    # Simulated data for charts
    # Data for Content Performance Over Time
    dates = pd.to_datetime(pd.date_range(end=pd.Timestamp.now(), periods=30, freq='D'))
    simulated_likes = np.random.randint(50, 500, size=30) + np.arange(30) * 5
    simulated_comments = np.random.randint(5, 50, size=30) + np.arange(30) * 1
    
    performance_df = pd.DataFrame({
        'Date': dates,
        'Likes': simulated_likes,
        'Comments': simulated_comments
    }).set_index('Date')
    
    st.line_chart(performance_df)
    st.markdown("---")

    # Data for Engagement by Platform
    platform_engagement = pd.DataFrame({
        'Platform': ['Facebook', 'Instagram', 'X (Twitter)', 'LinkedIn'],
        'Engagement (Simulated)': [np.random.randint(1000, 5000), np.random.randint(2000, 7000), np.random.randint(800, 4000), np.random.randint(500, 3000)]
    })
    st.subheader("Engagement by Platform (Simulated)")
    st.bar_chart(platform_engagement.set_index('Platform'))
    st.markdown("---")

    # Data for Sentiment Distribution
    sentiment_data = pd.DataFrame({
        'Sentiment': ['Positive', 'Neutral', 'Negative', 'Mixed'],
        'Count': [np.random.randint(50, 150), np.random.randint(20, 80), np.random.randint(10, 60), np.random.randint(5, 20)]
    })
    st.subheader("Sentiment Distribution (Simulated)")
    # FIX: Changed st.pie_chart to st.bar_chart
    st.bar_chart(sentiment_data.set_index('Sentiment')) 
    st.markdown("---")

    st.subheader("Sentiment Analysis (Advanced Feature)")
    text_for_sentiment = st.text_area("Enter text to analyze sentiment:", height=150, key="sentiment_input")

    if st.button("Analyze Sentiment", key="analyze_sentiment_button"):
        if text_for_sentiment:
            with st.spinner("Analyzing sentiment..."):
                sentiment_result = asyncio.run(analyze_sentiment(text_for_sentiment))
                if sentiment_result:
                    st.success(f"Sentiment: **{sentiment_result}**")
                else:
                    st.error("Could not analyze sentiment. Please try again.")
        else:
            st.warning("Please enter text for sentiment analysis.")

    st.markdown("---")
    st.subheader("Generated Content History")
    # For now, we'll simulate fetching data. In a real app, this would come from a database.
    # We'll use a placeholder for `user` and `firestore_st_ref` as they are not defined in this snippet.
    # Assuming `user` and `firestore_st_ref` would be handled by the main app structure.

    # Dummy data for demonstration if Firestore is not set up
    dummy_generated_content = [
        {
            "timestamp": time.time() - 3600 * 24 * 3, # 3 days ago
            "type": "Social Media Post",
            "topic": "Benefits of AI in marketing",
            "platform": "LinkedIn",
            "tone": "Professional",
            "audience": "Professionals",
            "generated_text": "AI is revolutionizing marketing by enabling hyper-personalization and predictive analytics. Stay ahead of the curve!",
            "hashtags": ["#AIinMarketing", "#DigitalTransformation", "#FutureofMarketing"],
            "cta": "Learn more in our latest blog post!"
        },
        {
            "timestamp": time.time() - 3600 * 24 * 1, # 1 day ago
            "type": "Tweet",
            "topic": "Sustainable living tips",
            "platform": "X (Twitter)",
            "tone": "Informative",
            "audience": "General Public",
            "generated_text": "Small changes, big impact! 🌱 Reduce your carbon footprint with these easy sustainable living tips. #SustainableLiving #EcoFriendly",
            "hashtags": ["#SustainableLiving", "#EcoFriendly", "#GreenFuture"],
            "cta": "Read our guide!"
        }
    ]

    # Check if Firestore is initialized (using a dummy check for this isolated snippet)
    # In your full app.py, replace this with the actual check for firestore_st_ref.session_state.get('firestore_db_initialized')
    firestore_initialized_dummy = True # Set to False if you don't have Firestore setup

    if firestore_initialized_dummy: # Replace with actual firestore_st_ref check
        # Assuming get_generated_content is available and returns a list of dicts
        # For this example, we'll use dummy_generated_content
        user_content_history = dummy_generated_content # Replace with get_generated_content(user['uid']) if Firestore is active

        if user_content_history:
            st.write("Here's a history of your generated content:")
            
            # Convert to DataFrame for better display
            df_history = pd.DataFrame(user_content_history)
            
            # Format timestamp for readability
            df_history['timestamp'] = pd.to_datetime(df_history['timestamp'], unit='s').dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Reorder columns for better presentation
            display_columns = ['timestamp', 'type', 'platform', 'topic', 'generated_text', 'hashtags', 'cta']
            df_history = df_history[display_columns]

            st.dataframe(df_history, use_container_width=True)

            # Optional: Display individual entries in expanders for more detail
            st.markdown("---")
            st.subheader("Detailed Content Entries:")
            for i, entry in enumerate(user_content_history):
                with st.expander(f"Content Type: {entry.get('type', 'N/A')} | Topic: {entry.get('topic', 'N/A')[:50]}... (Click to expand)"):
                    st.json(entry) # Still show raw JSON for full detail in expander
        else:
            st.info("No generated content history found yet. Generate some content first!")
    else:
        st.warning("Firestore not initialized. Content history is not available.")

    st.markdown("---")
    st.subheader("Trend Monitoring (Future Feature)")
    st.info("This section is a placeholder for future trend monitoring capabilities. It would identify trending topics relevant to your industry.")
    
    trend_keywords = st.text_input("Enter keywords to monitor trends (e.g., 'AI marketing', 'sustainable tech'):", key="trend_keywords")
    if st.button("Start Trend Monitoring", key="start_trend_monitoring_button"): # Enabled the button
        if trend_keywords:
            with st.spinner("Simulating trend monitoring..."):
                simulated_trends = asyncio.run(simulate_trend_monitoring(trend_keywords))
                st.subheader("Simulated Trending Topics:")
                if simulated_trends:
                    for trend in simulated_trends:
                        st.markdown(f"- {trend}")
                else:
                    st.info("No simulated trends found for these keywords.")
        else:
            st.warning("Please enter keywords to start trend monitoring.")
