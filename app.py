# app.py
import streamlit as st
import torch
import pandas as pd
import numpy as np
import random
import os
from datetime import datetime
import io  # For handling audio bytes

# Import specific components
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
import textwrap

# --- Configuration ---
# Ensure paths are relative to the app.py file
BASE_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
HISTORY_CSV_PATH = 'excuse_history.csv'
FONT_PATH = "arial.ttf"  # Make sure arial.ttf is in the same folder as app.py
FONT_SIZE_MSG = 16
FONT_SIZE_INFO = 12
GENERATED_IMSG_PATH = 'generated_imessage.png' # Temp file path
GENERATED_EMERGENCY_SMS_PATH = 'generated_emergency_sms.png' # Temp file path

GENERIC_REPLIES = [
    "Okay, thanks for letting me know.", "Understood. Hope everything is alright.",
    "Got it. Please keep me updated.", "Alright, appreciate the heads up.",
    "Okay, take care.", "Noted.", "Received.", "Acknowledged."
]

# Check available device (GPU priority, fallback to CPU)
# Note: Streamlit Cloud free tier usually provides CPU.
if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu_available = True
    print("GPU detected. Attempting to load model on GPU.")
    # Configure quantization for GPU
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16, # Use float16 for wider compatibility
        bnb_4bit_use_double_quant=False,
    )
else:
    device = torch.device("cpu")
    gpu_available = False
    bnb_config = None # No quantization on CPU
    print("No GPU detected. Loading model on CPU (inference will be slow).")


# --- Model Loading (Cached) ---
# Use st.cache_resource to load model only once
@st.cache_resource
def load_model_and_tokenizer():
    print(f"Attempting to load model: {BASE_MODEL_ID}")
    try:
        if gpu_available and bnb_config:
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                quantization_config=bnb_config,
                device_map="auto", # Let accelerate handle placement on GPU
                trust_remote_code=True,
                token=st.secrets.get("HF_TOKEN") # Use token from secrets if available
            )
            print("Model loaded on GPU (quantized).")
        else:
            # Load on CPU without quantization
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_ID,
                trust_remote_code=True,
                token=st.secrets.get("HF_TOKEN")
                # No quantization_config, device_map defaults to CPU
            )
            print("Model loaded on CPU.")

        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_ID,
            trust_remote_code=True,
            token=st.secrets.get("HF_TOKEN")
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Use left padding for generation on GPU
        print("Tokenizer loaded.")
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model/tokenizer: {e}", icon="🚨")
        # Consider adding instructions here if it's a common auth error
        if "GatedRepoError" in str(e):
             st.error("This might be a gated model. Ensure you have accepted terms on Hugging Face and provided a valid HF_TOKEN in secrets.", icon="🔑")
        return None, None

# --- Backend Helper Functions ---
# (Adapted from Colab - Ensure they use the passed model/tokenizer)

def generate_response(model, tokenizer, prompt_instruction):
    """Generates text using the loaded LLM."""
    if not model or not tokenizer:
        return "Error: Model not available."

    # Format for TinyLlama Chat
    system_message = "You are an intelligent assistant that generates context-aware excuses and apologies."
    full_prompt = f"<|system|>\n{system_message}</s>\n<|user|>\n{prompt_instruction}</s>\n<|assistant|>\n"

    try:
        # Use model.device if model is on GPU, otherwise default device is CPU
        current_device = model.device if hasattr(model, 'device') else device
        inputs = tokenizer(full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=512).to(current_device)

        print("Generating response...")
        with torch.no_grad():
            # Ensure generation happens on the correct device
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        input_length = inputs['input_ids'].shape[1]
        generated_ids = outputs[0][input_length:]
        response_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        print("Generation complete.")
        # Clean output
        response_text = response_text.split('<|user|>')[0].split('<|assistant|>')[0].strip()
        response_text = response_text.replace("Dear [Name],", "").strip()
        response_text = response_text.replace("I hope this letter finds you", "").strip()
        if response_text.startswith("I am writing to inform you"): response_text = response_text[len("I am writing to inform you"):].strip()
        return response_text

    except Exception as e:
        st.error(f"Error during generation: {e}", icon="🔥")
        return "Error: Generation failed."

def generate_voice_output(text):
    """Generates MP3 audio bytes from text using gTTS."""
    try:
        tts = gTTS(text=text, lang='en')
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0) # Rewind buffer to the beginning
        return mp3_fp.read() # Return bytes
    except Exception as e:
        st.error(f"Error generating voice: {e}", icon="🔊")
        return None

# (iMessage screenshot function - adapted slightly for file handling if needed)
def generate_imessage_screenshot(excuse_text, reply_text, filename=GENERATED_IMSG_PATH):
    # ... (Keep the exact code from the last working Colab version for this function) ...
    # --- [Paste the full working generate_imessage_screenshot function here] ---
    try:
        # Load font
        try:
            font_msg = ImageFont.truetype(FONT_PATH, FONT_SIZE_MSG)
            font_info = ImageFont.truetype(FONT_PATH, FONT_SIZE_INFO)
        except IOError:
            print(f"WARNING: Font '{FONT_PATH}' not found. Using default PIL font.")
            font_msg = ImageFont.load_default(); font_info = ImageFont.load_default()
        # Parameters
        padding = 15; bubble_padding_h = 12; bubble_padding_v = 8; bubble_radius = 15
        img_width = 380; max_bubble_width_ratio = 0.75; inter_bubble_space = 10
        timestamp_space = 20; line_spacing = 4
        max_text_width = img_width * max_bubble_width_ratio
        # Prepare Text
        avg_char_width = font_msg.getlength("A") if hasattr(font_msg, "getlength") else 8
        wrap_width_chars = int(max_text_width / (avg_char_width * 0.6))
        excuse_wrapped = "\n".join(textwrap.wrap(excuse_text, width=wrap_width_chars, replace_whitespace=False))
        reply_wrapped = "\n".join(textwrap.wrap(reply_text, width=wrap_width_chars, replace_whitespace=False))
        # Calculate Text Sizes
        dummy_img = Image.new('RGB', (1, 1)); dummy_draw = ImageDraw.Draw(dummy_img)
        try:
            excuse_bbox = dummy_draw.multiline_textbbox((0, 0), excuse_wrapped, font=font_msg, spacing=line_spacing); excuse_text_w = excuse_bbox[2] - excuse_bbox[0]; excuse_text_h = excuse_bbox[3] - excuse_bbox[1]
            reply_bbox = dummy_draw.multiline_textbbox((0, 0), reply_wrapped, font=font_msg, spacing=line_spacing); reply_text_w = reply_bbox[2] - reply_bbox[0]; reply_text_h = reply_bbox[3] - reply_bbox[1]
        except AttributeError:
             print("Warning: Using older Pillow textsize fallback."); excuse_text_w, excuse_text_h = dummy_draw.multiline_textsize(excuse_wrapped, font=font_msg, spacing=line_spacing); reply_text_w, reply_text_h = dummy_draw.multiline_textsize(reply_wrapped, font=font_msg, spacing=line_spacing)
        # Calculate Bubble Sizes
        excuse_bubble_w = min(excuse_text_w + 2 * bubble_padding_h, img_width - 2 * padding); excuse_bubble_h = excuse_text_h + 2 * bubble_padding_v
        reply_bubble_w = min(reply_text_w + 2 * bubble_padding_h, img_width - 2 * padding); reply_bubble_h = reply_text_h + 2 * bubble_padding_v
        # Calculate Image Height
        img_height = padding + excuse_bubble_h + inter_bubble_space + reply_bubble_h + timestamp_space + padding
        # Create Image and Draw
        img = Image.new('RGB', (img_width, int(img_height)), color=(255, 255, 255)); draw = ImageDraw.Draw(img)
        blue_bubble_color = (0, 122, 255); grey_bubble_color = (229, 229, 234); text_color_blue = (255, 255, 255); text_color_grey = (0, 0, 0); time_color = (150, 150, 150)
        # Draw Excuse Bubble
        excuse_bubble_x = img_width - padding - excuse_bubble_w; excuse_bubble_y = padding
        draw.rounded_rectangle((excuse_bubble_x, excuse_bubble_y, excuse_bubble_x + excuse_bubble_w, excuse_bubble_y + excuse_bubble_h), radius=bubble_radius, fill=blue_bubble_color)
        draw.multiline_text((excuse_bubble_x + bubble_padding_h, excuse_bubble_y + bubble_padding_v), excuse_wrapped, font=font_msg, fill=text_color_blue, align="left", spacing=line_spacing)
        # Draw Reply Bubble
        reply_bubble_x = padding; reply_bubble_y = excuse_bubble_y + excuse_bubble_h + inter_bubble_space
        draw.rounded_rectangle((reply_bubble_x, reply_bubble_y, reply_bubble_x + reply_bubble_w, reply_bubble_y + reply_bubble_h), radius=bubble_radius, fill=grey_bubble_color)
        draw.multiline_text((reply_bubble_x + bubble_padding_h, reply_bubble_y + bubble_padding_v), reply_wrapped, font=font_msg, fill=text_color_grey, align="left", spacing=line_spacing)
        # Draw Timestamp
        timestamp_text = datetime.now().strftime("%I:%M %p")
        try: ts_bbox = font_info.getbbox(timestamp_text); ts_width = ts_bbox[2] - ts_bbox[0]
        except AttributeError: ts_width, _ = font_info.getsize(timestamp_text)
        ts_x = (img_width - ts_width) / 2; ts_y = reply_bubble_y + reply_bubble_h + (timestamp_space / 4)
        draw.text((ts_x, ts_y), timestamp_text, font=font_info, fill=time_color)
        # Save image temporarily
        img.save(filename)
        return filename # Return path to saved file
    except Exception as e:
        st.error(f"Error generating iMessage screenshot: {e}", icon="🖼️")
        # import traceback; traceback.print_exc() # For detailed debugging if needed
        return None

# (Location log function)
def generate_location_log(model, tokenizer, excuse_text, user_reason=""):
    """Generates fake location log text using the LLM."""
    print("\n--- Generating Location Log ---")
    if not model or not tokenizer: return "Error: Model not loaded."
    context_hint = f"The user needed an excuse because they '{user_reason}'. " if user_reason else ""
    log_prompt = (
        f"{context_hint}Generate ONLY a single fake location log entry in this format: "
        f"HH:MM AM/PM: Location - [General Area/Street]. Status: [Brief Status, e.g., Heavy Traffic, At Doctor's Office, En Route]. "
        f"Example: 09:15 AM: Location - Near Main St. Status: Heavy Traffic." )
    location_log_entry = generate_response(model, tokenizer, log_prompt)
    location_log_entry = location_log_entry.split('\n')[0] # Take first line
    print(f"Generated Location Log: {location_log_entry}")
    return location_log_entry

# (Emergency SMS function - adapted for file handling)
def generate_emergency_sms_screenshot(emergency_text, sender="Mom", filename=GENERATED_EMERGENCY_SMS_PATH):
    # ... (Keep the exact code from the Colab version for this function) ...
    # --- [Paste the full working generate_emergency_sms_screenshot function here] ---
    try:
        # Load font
        try:
            font_msg = ImageFont.truetype(FONT_PATH, FONT_SIZE_MSG); font_info = ImageFont.truetype(FONT_PATH, FONT_SIZE_INFO)
        except IOError: font_msg = ImageFont.load_default(); font_info = ImageFont.load_default()
        # Parameters
        padding = 15; bubble_padding_h = 12; bubble_padding_v = 8; bubble_radius = 15; max_bubble_width_ratio = 0.7; timestamp_space = 25; img_width = 380
        # Calculate Text Size
        max_text_width = img_width * max_bubble_width_ratio; sms_wrapped, sms_size = get_wrapped_text_and_size(emergency_text, font_msg, max_text_width) # Assume get_wrapped_text_and_size is defined above
        sms_bubble_w = min(sms_size[0] + 2 * bubble_padding_h, img_width - 2 * padding); sms_bubble_h = sms_size[1] + 2 * bubble_padding_v; img_height = padding + timestamp_space + sms_bubble_h + padding
        # Create Image and Draw
        img = Image.new('RGB', (img_width, int(img_height)), color=(255, 255, 255)); draw = ImageDraw.Draw(img); grey_bubble_color = (229, 229, 234); text_color_grey = (0, 0, 0); info_color = (100, 100, 100)
        # Sender/Timestamp Info
        info_text = f"{sender} - {datetime.now().strftime('%I:%M %p')}"
        try: info_bbox = font_info.getbbox(info_text); info_width = info_bbox[2] - info_bbox[0]
        except AttributeError: info_width, _ = font_info.getsize(info_text)
        info_x = (img_width - info_width) / 2; info_y = padding
        draw.text((info_x, info_y), info_text, font=font_info, fill=info_color)
        # SMS Bubble
        sms_bubble_x = padding; sms_bubble_y = info_y + timestamp_space - 5
        draw.rounded_rectangle((sms_bubble_x, sms_bubble_y, sms_bubble_x + sms_bubble_w, sms_bubble_y + sms_bubble_h), radius=bubble_radius, fill=grey_bubble_color)
        draw.multiline_text((sms_bubble_x + bubble_padding_h, sms_bubble_y + bubble_padding_v), sms_wrapped, font=font_msg, fill=text_color_grey, align="left", spacing=line_spacing) # Use multiline_text
        # Save image temporarily
        img.save(filename)
        return filename
    except Exception as e:
        st.error(f"Error generating Emergency SMS screenshot: {e}", icon="🆘")
        return None

# Helper function from image generation needed for SMS function
def get_wrapped_text_and_size(text, font, max_width):
    lines = textwrap.wrap(text, width=int(max_width / (font.getlength("A") / 1.8 if hasattr(font, "getlength") else 8)))
    if not lines: lines = [""]
    wrapped_text = "\n".join(lines)
    try:
        bbox = font.getbbox(wrapped_text); text_width = bbox[2] - bbox[0]; text_height = bbox[3] - bbox[1]
    except AttributeError:
        text_width, text_height = font.getsize(wrapped_text)
    text_width = max(text_width, 1); text_height = max(text_height, font.size if hasattr(font, "size") else FONT_SIZE_MSG)
    return wrapped_text, (text_width, text_height)


# --- History Management ---
# Load history into session state only once
def load_history():
    if os.path.exists(HISTORY_CSV_PATH):
        try:
            df = pd.read_csv(HISTORY_CSV_PATH)
            if 'user_reason' not in df.columns: df['user_reason'] = None
            df['effectiveness_rating'] = pd.to_numeric(df['effectiveness_rating'], errors='coerce')
            df['is_favorite'] = df['is_favorite'].astype(bool)
            return df
        except Exception as e:
            st.error(f"Error loading history: {e}")
            # Fallback to empty dataframe
            return pd.DataFrame(columns=['timestamp', 'scenario', 'urgency', 'believability', 'type', 'user_reason', 'generated_text', 'effectiveness_rating', 'is_favorite'])
    else:
        return pd.DataFrame(columns=['timestamp', 'scenario', 'urgency', 'believability', 'type', 'user_reason', 'generated_text', 'effectiveness_rating', 'is_favorite'])

# Initialize session state for history if it doesn't exist
if 'history_df' not in st.session_state:
    st.session_state.history_df = load_history()

def save_history_state():
    try:
        st.session_state.history_df.to_csv(HISTORY_CSV_PATH, index=False)
        print(f"History saved to {HISTORY_CSV_PATH}") # Log to console
    except Exception as e:
        st.warning(f"Could not save history: {e}") # Show warning in UI

def add_to_history_state(scenario, urgency, believability, gen_type, reason, text):
    new_entry = pd.DataFrame([{
        'timestamp': datetime.now(), 'scenario': scenario, 'urgency': urgency,
        'believability': believability, 'type': gen_type, 'user_reason': reason,
        'generated_text': text, 'effectiveness_rating': None, 'is_favorite': False # Use None for rating initially
    }])
    st.session_state.history_df = pd.concat([st.session_state.history_df, new_entry], ignore_index=True)
    save_history_state() # Auto-save

# --- Ranking Logic ---
def get_ranked_suggestion(scenario, urgency, believability, gen_type, reason=""):
    df = st.session_state.history_df # Use history from session state
    if df.empty or df['effectiveness_rating'].isna().all(): return []
    # Basic filtering (ignoring reason for now)
    similar_context_df = df[
        (df['scenario'] == scenario) & (df['urgency'] == urgency) &
        (df['believability'] == believability) & (df['type'] == gen_type) &
        (df['effectiveness_rating'].notna())
    ].copy()
    if similar_context_df.empty: return []
    ranked_df = similar_context_df.sort_values(by=['effectiveness_rating', 'timestamp'], ascending=[False, False])
    suggestions = ranked_df['generated_text'].unique().tolist()[:3] # Top 3 unique
    return suggestions # Return list of suggestions


# --- Streamlit UI ---
st.set_page_config(page_title="Excuse Generator", layout="wide")
st.title("🧠 Intelligent Excuse Generator")
st.markdown("Generate context-aware excuses or apologies with supporting proofs.")

# Load Model and Tokenizer (cached)
model, tokenizer = load_model_and_tokenizer()

if model and tokenizer:
    col1, col2 = st.columns([1, 2]) # Columns for inputs and outputs

    with col1:
        st.subheader("Configuration")
        scenario = st.selectbox("Scenario:", ['work', 'school', 'family', 'social'])
        urgency = st.select_slider("Urgency:", ['low', 'medium', 'high'], value='medium')
        believability = st.select_slider("Believability:", ['low', 'medium', 'high'], value='medium')
        gen_type = st.radio("Type:", ['excuse', 'apology'], horizontal=True)
        user_reason = st.text_input("Specific Reason:", placeholder="e.g., missed meeting, forgot birthday")

        generate_btn = st.button("Generate Excuse/Apology", type="primary", use_container_width=True)

        st.subheader("Simulate Emergency")
        emergency_btn = st.button("Trigger Fake Emergency SMS", use_container_width=True)

    with col2:
        st.subheader("Generated Output & Proofs")

        if generate_btn:
            if not user_reason:
                st.warning("Please provide a specific reason.", icon="⚠️")
            else:
                # --- Generation Workflow ---
                with st.spinner("Generating text... (This may take time on CPU)"):
                    # 1. Construct Prompt
                    prompt = (
                        f"Generate a short, informal text message style {gen_type} suitable for a '{scenario}' scenario. "
                        f"The specific reason is: '{user_reason}'. "
                        f"Make it sound {believability} believability and {urgency} urgency. "
                        f"Keep it concise. Do NOT use formal letter formatting." )

                    # 2. Generate Text
                    generated_text = generate_response(model, tokenizer, prompt)

                if generated_text and "Error" not in generated_text:
                    st.success("Generation Complete!", icon="✅")
                    st.text_area("Generated Text:", generated_text, height=100)

                    # 3. Add to History (before generating proofs)
                    add_to_history_state(scenario, urgency, believability, gen_type, user_reason, generated_text)

                    # 4. Generate & Display Proofs
                    st.markdown("---")
                    st.markdown("#### Generated Proofs:")

                    # Voice
                    with st.spinner("Generating voice..."):
                        audio_bytes = generate_voice_output(generated_text)
                    if audio_bytes:
                        st.audio(audio_bytes, format='audio/mp3')
                    else:
                        st.warning("Could not generate voice.", icon="🔇")

                    # iMessage Screenshot
                    with st.spinner("Generating chat screenshot..."):
                        selected_reply = random.choice(GENERIC_REPLIES)
                        img_path = generate_imessage_screenshot(generated_text, selected_reply)
                    if img_path and os.path.exists(img_path):
                        st.image(img_path, caption="Simulated iMessage")
                        # Clean up temp image file after display
                        # os.remove(img_path) # Careful with cleanup on shared platforms
                    else:
                         st.warning("Could not generate iMessage screenshot.", icon="🖼️")

                    # Location Log
                    with st.spinner("Generating location log..."):
                         location_log = generate_location_log(model, tokenizer, generated_text, user_reason)
                    if location_log and "Error" not in location_log:
                         st.info(f"**Simulated Location Log:** {location_log}", icon="📍")
                    else:
                         st.warning("Could not generate location log.", icon="🗺️")

                else:
                    st.error(f"Generation failed: {generated_text}", icon="❌")


        if emergency_btn:
            st.markdown("---")
            st.markdown("#### Simulated Emergency:")
            with st.spinner("Generating emergency SMS..."):
                emergency_prompt = "Generate ONLY a short, urgent-sounding fake emergency text message content (like 'Urgent! Call me ASAP.' or 'Family emergency, need you now.')."
                emergency_msg = generate_response(model, tokenizer, emergency_prompt)

            if emergency_msg and "Error" not in emergency_msg and len(emergency_msg) < 100:
                 sms_img_path = generate_emergency_sms_screenshot(emergency_msg)
                 if sms_img_path and os.path.exists(sms_img_path):
                      st.image(sms_img_path, caption="Simulated Emergency SMS")
                      # os.remove(sms_img_path) # Cleanup temp file
                 else:
                      st.warning("Could not generate emergency SMS screenshot.", icon="🆘")
            else:
                 st.error(f"Could not generate suitable emergency message text. Got: {emergency_msg}", icon="⚠️")


    # --- Display History ---
    st.divider()
    st.subheader("History")
    st.markdown("Recent excuses/apologies generated in this session.")

    if not st.session_state.history_df.empty:
        # Prepare DataFrame for display (select columns, maybe format timestamp)
        display_df = st.session_state.history_df[['timestamp', 'scenario', 'type', 'user_reason', 'generated_text', 'is_favorite']].copy()
        display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M') # Format timestamp
        st.dataframe(display_df, use_container_width=True)
        # TODO: Add buttons/select box for favoriting/rating rows in history if desired
    else:
        st.info("No history recorded in this session yet.")

else:
    st.error("Model or Tokenizer failed to load. Cannot start the application.", icon="🚫")
    st.info("Please check the console logs for detailed errors. Common issues include missing Hugging Face token (check secrets), network problems, or insufficient memory (especially on free tiers).")