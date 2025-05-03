# Intelligent Excuse Generator (Local Deployment Focus)

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)
[![Hugging Face Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Transformers-blue)](https://huggingface.co/docs/transformers/index)

A Streamlit web application powered by a local Large Language Model (LLM) to generate context-aware excuses and apologies, complete with simulated supporting proofs. This version is configured for local execution, primarily on CPU.

---

## 🚀 Core Features

*   **AI Generation:** Generates **Excuses** or **Apologies** using the `TinyLlama-1.1B-Chat-v1.0` language model loaded locally.
*   **Contextual Input:** Takes user input for:
    *   **Scenario:** `Work`, `School`, `Family`, `Social`
    *   **Urgency:** `Low`, `Medium`, `High` (Slider)
    *   **Believability:** `Low`, `Medium`, `High` (Slider)
    *   **Type:** `Excuse` or `Apology` (Radio Button)
    *   **Specific Reason:** User-defined reason (e.g., "missed meeting", "forgot anniversary") via text input.
*   **Proof Generation:** Creates simulated evidence to support the generated text:
    *   **Text-to-Speech:** Generates playable MP3 audio of the excuse/apology using `gTTS`.
    *   **iMessage Screenshot:** Creates a PNG image simulating an iMessage chat with the generated text and a generic reply using `Pillow`.
    *   **Location Log:** Generates a plausible single-line location log entry using the LLM based on context.
*   **Emergency Simulation:**
    *   Button to trigger a simulated emergency.
    *   Generates an urgent SMS message text via the LLM.
    *   Creates a PNG image simulating an incoming SMS using `Pillow`.
*   **History:** Tracks generated items (inputs, reason, output) within the user's session and persists across runs by saving/loading a local `excuse_history.csv` file.
*   **Ranking Suggestions:** Offers the top 3 highest-rated unique past generations from the history file for similar contexts (Scenario, Urgency, Believability, Type). *Note: Rating input is currently via the console when running locally.*

---

## 🛠️ Tech Stack (Local Setup)

*   **Language:** Python (3.9+ Recommended)
*   **Web Framework:** Streamlit
*   **LLM Interaction:** Hugging Face `transformers`, `torch` (CPU version specified), `accelerate`
*   **LLM Used:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (pre-trained, downloaded from Hub)
*   **Proof Generation:** `Pillow` (Image manipulation), `gTTS` (Text-to-Speech)
*   **Data Handling:** `pandas`, `numpy`
*   **Dependencies:** See `requirements.txt` for exact versions.

---

Excuse_Generator/ 
|
|--.streamlit/
   |-secrets.toml           
|
|-- images/                    
|   |-app_screenshot.png     
|   |-imessage_example.png   
|
|--.gitignore
|
|-- app.py                    
|-- requirements.txt         
|-- arial.ttf             
|-- excuse_history.csv         
|-- README.md 


## 💡 How to Run Locally (Requires Python 3.9+)

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/meet1027/Excuse_Generator 
    cd your-repo-name
    ```

2.  **Create and activate a virtual environment (Highly Recommended):**
    ```bash
    # Create environment
    python -m venv venv
    # Activate environment
    # Windows:
    venv\Scripts\activate
    # MacOS/Linux:
    source venv/bin/activate
    ```
    *(You should see `(venv)` at the start of your terminal prompt)*

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(This will install PyTorch CPU version and other libraries. May take some time.)*

4.  **Set up Hugging Face Token (Required for Model Download):**
    *   Make sure the `.streamlit` folder exists in the project directory.
    *   Inside `.streamlit`, create/edit the `secrets.toml` file.
    *   Add your Hugging Face Access Token (obtain from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens), requires 'read' permission):
        ```toml
        HF_TOKEN = "hf_YOUR_TOKEN_HERE"
        ```

5.  **Ensure Font File is Present:**
    *   Confirm that the font file specified in `app.py` (default `arial.ttf`) is present in the main project directory. Image generation will fail without it.

6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```

7.  **Access the App:** Open the local URL displayed in your terminal (usually `http://localhost:8501`) in your web browser.

    *(**Performance Note:** The initial model download may take time. Generation will run on your computer's CPU and will likely be noticeably slower than GPU-accelerated examples.)*

---

## ⚠️ Limitations & Known Issues

*   **CPU Performance:** Designed for local CPU execution, resulting in slower LLM generation times compared to GPU.
*   **Model Size:** Uses the smaller TinyLlama model for compatibility with limited local resources (especially RAM). Generation quality may be less sophisticated than larger models like Mistral-7B or Gemma-2B.
*   **No Fine-Tuning:** The model is used pre-trained. Fine-tuning was explored but infeasible on readily available hardware (Colab free tier, typical laptops) due to VRAM/RAM constraints. Fine-tuning would significantly improve task-specific quality.
*   **Proof Realism:** Screenshots are programmatically simulated and may have minor visual differences from native OS interfaces.
*   **Basic Ranking/History Interaction:** Rating/favoriting for ranking suggestions currently relies on console input during local execution.

---

## 🙏 Acknowledgements

*   [Streamlit Team](https://streamlit.io/)
*   [Hugging Face Team](https://huggingface.co/) & The `transformers` library developers
*   [TinyLlama Team](https://github.com/jzhang38/TinyLlama) & Contributors
*   Developers of Pillow, gTTS, Pandas, and other dependencies.
*   *(Add course instructors, mentors, etc. if applicable)*

---

## 📬 Contact

*   GitHub: [meet1027](https://github.com/meet1027) 
*   Email: meetck05@gmail.com

---

⭐ If you run into issues setting up or find this project useful, feel free to open an issue or star the repository!