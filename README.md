# Gnani AI Student Tutor Agent

An interactive, voice-first AI Student Tutor designed for Indian school and intermediate students. The agent engages students in real-time bilingual voice dialogue, teaching curriculum concepts across school classes and intermediate streams with adaptive pedagogical reasoning, automatic speech recognition (STT), and neural text-to-speech (TTS).

---

## Key Capabilities

- **Curriculum Scope**:
  - **School Classes 1 to 10**: State Board, CBSE, and ICSE aligned foundational concepts (Mathematics, General Science, Social Studies, English).
  - **Intermediate 1st & 2nd Year (Classes 11 & 12)**: Stream-specific instruction across **MPC** (Maths, Physics, Chemistry) and **BiPC** (Biology, Physics, Chemistry). *(Note: Commercial and Arts groups such as MEC, CEC, and HEC are not supported).*
- **Multilingual Dialogue (Telugu, English & Tanglish)**:
  - Native support for Telugu (`te-IN`), Indian English (`en-IN`), and conversational Tanglish (Telugu mixed with English words).
  - Seamless code-switching and script detection with automatic language routing.
- **Gnani Prisma STT Engine**:
  - Powered by Gnani's `prisma-v3.0` automatic speech recognition with `indic-auto` bilingual language detection and `['te-IN', 'en-IN']` candidate weighting.
  - Automatic fallback to `te-IN` on connection dips to ensure no query is dropped.
- **Google Gemini Pedagogical Reasoning**:
  - Powered by Google Gemini (`gemini-3.5-flash` / `gemini-3.5-flash-lite` / `gemini-3.8-flash`) with structured system instructions.
  - Implements age-appropriate explanations, Socratic questioning, real-world Indian examples, interactive quizzes, and concise speech-optimized answers (under 3 sentences per turn for natural spoken pace).
- **Gnani Timbre v2.5 Neural TTS**:
  - High-fidelity voice synthesis using `timbre-v2.5` with the **`Pranav`** neural voice and dynamic regional language routing (`te-IN` for Telugu/Tanglish, `en-IN` for English).
  - Fine-tuned speech rate (0.95x–1.0x) for clear, patient tutor delivery.
- **Smart Voice Activity Detection (VAD)**:
  - Adaptive ambient noise calibration, dynamic onset/continuation hysteresis, pre-roll buffering, and consonant dip hangover handling.
  - Intelligent silence detection with dedicated handling for extended quiz response pauses.

> [!NOTE]
> **Integration Notice**: This project uses **Gnani Prisma STT** and **Gnani Timbre v2.5 TTS**. **Gnani Evon v3.3 is NOT integrated** into this codebase. All conversational reasoning and pedagogical logic are handled by the Google Gemini API.

---

## Repository Structure

```text
gnani-voice-assistant/
├── voice_assistant.py                 # Core AI Student Tutor Agent & Voice Loop
├── mic_to_text.py                     # Standalone STT transcription diagnostic tool
├── requirements.txt                   # Verified Python dependencies
├── .env.example                       # Environment configuration template
├── .gitignore                         # Excludes secrets, .venv, wav recordings & cache
├── test_*.py                          # Comprehensive unit & verification test suites
└── README.md                          # Project documentation and setup guide
```

---

## Windows Setup Instructions

Follow these steps to set up and run the AI Student Tutor Agent on any Windows 10/11 laptop.

### 1. Prerequisites
- **Operating System**: Windows 10 or Windows 11 (64-bit).
- **Python**: Python 3.10 to 3.14 (Python 3.11, 3.12, or 3.14 recommended).
  - Download from [python.org](https://www.python.org/downloads/windows/).
  - **Important**: During installation, check the box **"Add python.exe to PATH"**.
- **Hardware**: Working microphone and speakers or headphones.
- **Git**: [Git for Windows](https://git-scm.com/download/win).

### 2. Clone the Repository
Open Windows PowerShell and clone the private repository:
```powershell
git clone <PRIVATE_REPOSITORY_URL>
cd gnani-voice-assistant
```

### 3. Create a Virtual Environment
Create a clean, isolated virtual environment to prevent dependency conflicts:
```powershell
python -m venv .venv
```

Activate the virtual environment:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If PowerShell shows an execution policy error, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` and retry).*

### 4. Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

> [!TIP]
> On Windows, `sounddevice` and `soundfile` automatically include their required PortAudio and libsndfile DLLs. No separate system libraries or compiler installations are required.

### 5. Configure API Credentials
Create your local `.env` file by copying the provided template:
```powershell
Copy-Item .env.example .env
```

Open `.env` in Notepad:
```powershell
notepad .env
```

Fill in your actual API keys:
```dotenv
GNANI_API_KEY=your_actual_gnani_api_key
GEMINI_API_KEY=your_actual_gemini_api_key
```
Save and close Notepad.

> [!CAUTION]
> Never commit your `.env` file or share your API keys. `.env` is ignored by `.gitignore`.

---

## Running the Application

### 1. Launch the AI Student Tutor Agent
Ensure your virtual environment is active, then run:
```powershell
python voice_assistant.py
```

### 2. Onboarding Workflow
1. **Welcome**: The tutor greets you in Telugu & English and asks for your Class/Grade.
2. **Student Profile**: Answer with your class (e.g., *"Class 7"*, *"10th class"*, *"Inter 1st year MPC"*, or in Telugu *"Nenu 9th class"*).
3. **Interactive Learning**:
   - Ask any concept doubt (e.g., *"Explain photosynthesis"*, *"What is Newton's second law"*, *"Trigonometry basics cheppandi"*).
   - The tutor answers concisely in voice and can quiz you on key points.
4. **Exit**: Say *"Stop"*, *"Quit"*, *"Bye"*, or *"Aapu"* to gracefully end the session.

### 3. Quick STT Diagnostic
To verify that your microphone and Gnani Prisma STT are communicating properly:
```powershell
python test_mic.py      # Records a 3-second test clip to mic_test.wav
python mic_to_text.py   # Transcribes the clip with Gnani Prisma
```

---

## Audio & Hardware Troubleshooting

### Microphone Not Detected or Wrong Input Used
1. Run the device diagnostic to list all connected audio hardware:
   ```powershell
   python -c "import sounddevice as sd; print(sd.query_devices())"
   ```
2. Find the index number of your active microphone.
3. Open `.env` and specify the index:
   ```dotenv
   MIC_DEVICE_INDEX=1
   ```
4. Check Windows Privacy Settings: Ensure **Start > Settings > Privacy & security > Microphone** has **"Microphone access"** and **"Let desktop apps access your microphone"** toggled **ON**.

### No Sound Output (TTS Playback Silent)
1. Verify Windows volume is unmuted and default output device (speakers or headphones) is selected.
2. Ensure you have an active internet connection to stream audio synthesis from `api.gnani.ai`.

### Background Noise or Cut-Off Speech
If your environment is noisy or your voice gets cut off early, tune the VAD variables in `.env`:
- Increase `SILENCE_TIMEOUT_SECONDS=1.4` (gives more time before stopping recording).
- Increase `INITIAL_SILENCE_TIMEOUT=8.0` (allows more time to start speaking).
- Adjust `VAD_ENERGY_THRESHOLD=0.015` if ambient background fan or street noise triggers false recordings.

---

## Running Test Verification
To verify all Phase 1–11.5.1 test suites pass in your environment:
```powershell
python -m unittest test_phase11_5_tutor_intelligence.py
python -m unittest test_phase11_4_educational_accuracy.py
python -m unittest test_phase11_3_live_reliability.py
python -m unittest test_phase11_2_verification.py
python -m unittest test_phase10_audit.py
```
*(All suites should report `OK` with zero errors).*
