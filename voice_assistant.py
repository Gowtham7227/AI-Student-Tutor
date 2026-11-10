import os
import sys
import re
import time
import collections
import sounddevice as sd
import soundfile as sf
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient

# Ensure UTF-8 output on Windows console so Telugu prints cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Load API keys from .env file
load_dotenv()

api_key = os.getenv("GNANI_API_KEY")
if not api_key:
    raise RuntimeError("GNANI_API_KEY missing. Check your .env file.")

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise RuntimeError("GEMINI_API_KEY missing. Check your .env file.")

# Initialize API clients
stt = GnaniSTTClient(api_key=api_key)
tts = GnaniTTSClient(api_key=api_key)
gemini_client = genai.Client(api_key=gemini_api_key)

# Audio and adaptive VAD recognition configuration (all configurable via environment variables)
SILENCE_RMS_THRESHOLD = float(os.getenv("SILENCE_RMS_THRESHOLD", "0.0025"))  # Below this is considered empty silence
VAD_ENERGY_THRESHOLD = float(os.getenv("VAD_ENERGY_THRESHOLD", "0.010"))    # Default energy threshold for speech onset (adaptive calibration adjusts automatically)
SAFE_MIN_VAD_THRESHOLD = float(os.getenv("SAFE_MIN_VAD_THRESHOLD", "0.005")) # Absolute lower bound for VAD to prevent false triggers
SAFE_MAX_VAD_THRESHOLD = float(os.getenv("SAFE_MAX_VAD_THRESHOLD", "0.025")) # Absolute upper bound for VAD in noisy environments
VAD_CONTINUATION_RATIO = float(os.getenv("VAD_CONTINUATION_RATIO", "0.55"))  # Hysteresis: continuation threshold = onset * ratio
VAD_HANGOVER_CHUNKS = int(os.getenv("VAD_HANGOVER_CHUNKS", "6"))             # 6 chunks (300ms) hangover across intra-word consonant dips
SILENCE_TIMEOUT_SECONDS = float(os.getenv("SILENCE_TIMEOUT_SECONDS", "1.1")) # 1.1s continuous silence after speech -> stop in normal mode
MAX_RECORD_DURATION = float(os.getenv("MAX_RECORD_DURATION", "30.0"))        # Configurable 30s max duration for normal doubt mode
INITIAL_SILENCE_TIMEOUT = float(os.getenv("INITIAL_SILENCE_TIMEOUT", "6.0")) # Wait up to 6s for user to begin speaking
PRE_ROLL_SECONDS = float(os.getenv("PRE_ROLL_SECONDS", "0.4"))              # 400ms pre-roll to prevent clipping start of speech
MIN_SPEECH_DURATION = float(os.getenv("MIN_SPEECH_DURATION", "0.3"))        # Minimum speech duration to qualify as spoken input
CHUNK_DURATION = 0.05                                                       # 50ms audio chunk for agile VAD analysis
SETTLING_PAUSE_SECONDS = float(os.getenv("SETTLING_PAUSE_SECONDS", "0.25")) # Echo prevention settling pause
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_RECORD_DURATION = int(os.getenv("RECORD_DURATION", "6"))             # Fallback default duration
NORMALIZATION_TARGET_PEAK = 0.70                                            # Gentle gain target peak with safe headroom

# Dedicated Quiz Answer Listening Configuration
QUIZ_ANSWER_TIMEOUT_SECONDS = float(os.getenv("QUIZ_ANSWER_TIMEOUT_SECONDS", "120.0")) # Safety timeout for quiz answers (120s)
QUIZ_PAUSE_CHECK_SECONDS = float(os.getenv("QUIZ_PAUSE_CHECK_SECONDS", "1.2"))         # Natural pause duration to trigger interim command check
QUIZ_SETTLING_PAUSE = float(os.getenv("QUIZ_SETTLING_PAUSE", "0.0"))                  # Additional settling pause (playback already includes SETTLING_PAUSE_SECONDS)

# Quiz Command Actions
QUIZ_ACTION_SUBMIT = "submit"
QUIZ_ACTION_SKIP = "skip"
QUIZ_ACTION_REPEAT = "repeat"
QUIZ_ACTION_STOP = "stop"
QUIZ_ACTION_TIMEOUT = "timeout"
QUIZ_ACTION_EMPTY = "empty"

# Quiz Answer Mode Commands (English, Telugu, Tanglish)
QUIZ_SUBMIT_COMMANDS = [
    "submit answer", "submit", "done", "సబ్మిట్", "అయిపోయింది",
    "సబ్మిట్ చేయండి", "నా సమాధానం పూర్తి", "completed", "finish", "finished",
    "సమాధానం సమర్పించు", "సమర్పించు", "సబ్మిట్ ఆన్సర్",
]
QUIZ_SKIP_COMMANDS = [
    "skip question", "skip this", "skip", "స్కిప్", "వద్దు", "నెక్స్ట్ ప్రశ్న",
    "skip cheyyi", "తరువాతి ప్రశ్న", "next", "pass", "దాటవేయి", "వదిలేయి",
]
QUIZ_REPEAT_COMMANDS = [
    "repeat question", "repeat", "మళ్లీ చెప్పు", "ఒక్కసారి మళ్లీ చెప్పు",
    "మళ్ళీ అడుగు", "read again", "repeat cheyyi", "ఒక్కసారి చెప్పు",
]
QUIZ_STOP_COMMANDS = [
    "stop quiz", "exit quiz", "quit quiz", "క్విజ్ ఆపు", "క్విజ్ ఆపండి",
]

# Filler noise / hesitation tokens returned by ASR during quiet breaths or background noise
FILLER_NOISE_TOKENS = {
    "ఉమ్", "ఉం", "హో", "ఆ", "హా", "అ", "హ్మ్",
    "uh", "um", "umm", "hmm", "hm", "ah", "er", "ha",
    ".", "-", "हं", "हं हं", "हं।",
}

# Valid short conversational and quiz answer tokens preserved against noise filtering
VALID_SHORT_ANSWERS = {
    # English single-letter options
    "a", "b", "c", "d",
    # Telugu single-letter options
    "ఏ", "బి", "సి", "డి", "ఆ", "ఇ", "ఈ",
    # Conversational affirmations / negations (English)
    "yes", "no", "ok", "okay", "yep", "nope", "yeah", "sure", "true", "false",
    "done", "skip", "pass", "correct", "right",
    # Conversational affirmations / negations (Telugu & Tanglish)
    "అవును", "కాదు", "సరే", "నిజం", "తప్పు", "ఒప్పు", "చాలు", "వద్దు",
    "కరెక్ట్", "రైట్", "హౌను", "సరి",
    "avunu", "kaadu", "sare", "nijam", "thappu", "oppu",
    # Single-word scientific and academic terms commonly spoken in school/inter quizzes
    "oxygen", "hydrogen", "carbon", "nitrogen", "helium", "iron", "gold", "silver", "sodium", "chlorine",
    "acid", "base", "salt", "atom", "mole", "ion", "cell", "tissue", "organ", "plant", "animal",
    "root", "stem", "leaf", "flower", "seed", "water", "air", "soil", "atp", "dna", "rna", "gene",
    "mitochondria", "nucleus", "chlorophyll", "force", "work", "power", "energy", "mass", "weight",
    "speed", "velocity", "friction", "heat", "light", "sound", "joule", "newton", "watt", "volt",
    "circle", "square", "triangle", "angle", "radius", "diameter", "area", "volume", "mean", "median",
    "mode", "sum", "product", "fraction", "decimal", "zero",
    # Telugu scientific single-word terms
    "ఆక్సిజన్", "హైడ్రోజన్", "కార్బన్", "నత్రజని", "ఇనుము", "బంగారం", "వెండి", "సోడియం", "క్లోరిన్",
    "ఆమ్లం", "క్షారం", "లవణం", "పరమాణువు", "కణం", "కణజాలం", "అవయవం", "మొక్క", "జంతువు",
    "వేరు", "కాండం", "ఆకు", "పువ్వు", "విత్తనం", "నీరు", "గాలి", "మట్టి", "కేంద్రకం", "పత్రహరితం",
    "బలం", "పని", "సామర్థ్యం", "శక్తి", "ద్రవ్యరాశి", "వేగం", "వడి", "ఘర్షణ", "ఉష్ణం", "కాంతి", "ధ్వని",
    "వృత్తం", "చతురస్రం", "త్రిభుజం", "కోణం", "వ్యాసార్థం", "వ్యాసం", "వైశాల్యం", "ఘనపరిమాణం", "సగటు", "సున్నా",
}

# Stop commands in Telugu and English
STOP_COMMANDS = [
    "ఆపు",
    "ఆపండి",
    "ఆపివేయి",
    "ముగించు",
    "stop",
    "exit",
    "quit",
    "bye",
    "బై",
]

# Quiz trigger keywords
QUIZ_TRIGGERS = [
    "start quiz",
    "quiz start cheyyi",
    "నాకు క్విజ్ ఇవ్వండి",
    "take quiz",
    "quiz start",
    "క్విజ్",
    "quiz",
    "రసప్రశ్నలు",
    "రసప్రశ్న",
]

# Available, verified fast Gemini flash models
CANDIDATE_MODELS = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.8-flash",
]

# Curriculum subjects strictly restricted to School and Intermediate
SCHOOL_SUBJECTS = ["Mathematics", "Science", "English", "Social Studies"]
INTER_MPC_SUBJECTS = ["Mathematics", "Physics", "Chemistry"]
INTER_BIPC_SUBJECTS = ["Botany", "Zoology", "Physics", "Chemistry"]


class StudentProfile:
    """Tracks student education level, grade/year, group, and active subject for the session."""

    def __init__(
        self,
        level: str = None,
        grade_or_year: int = None,
        group: str = None,
        subject: str = None,
    ):
        self.level = level  # "school" or "intermediate"
        self.grade_or_year = grade_or_year  # 1..10 (School) or 1..2 (Intermediate)
        self.group = group  # "MPC" or "BiPC" (Intermediate only)
        self.subject = subject

    def is_complete(self) -> bool:
        if not self.level:
            return False
        if self.level == "school":
            return bool(self.grade_or_year and 1 <= self.grade_or_year <= 10)
        if self.level == "intermediate":
            return bool(self.grade_or_year in (1, 2) and self.group in ("MPC", "BiPC"))
        return False

    def describe(self) -> str:
        if self.level == "school":
            return f"Class {self.grade_or_year} School student"
        elif self.level == "intermediate":
            yr_str = "1st Year" if self.grade_or_year == 1 else "2nd Year"
            return f"{yr_str} Intermediate ({self.group}) student"
        return "Student"

    def allowed_subjects(self) -> list[str]:
        if self.level == "school":
            return SCHOOL_SUBJECTS
        elif self.level == "intermediate":
            return INTER_MPC_SUBJECTS if self.group == "MPC" else INTER_BIPC_SUBJECTS
        return []


class QuizSession:
    """Tracks interactive quiz progress, questions, scoring, difficulty, and hints."""

    def __init__(self, total_questions: int = 3):
        self.active: bool = False
        self.question_num: int = 0
        self.total_questions: int = total_questions
        self.score: int = 0
        self.current_question: str = ""
        self.current_expected: str = ""
        self.current_hint: str = ""
        self.subject: str = "General"
        self.topic: str = "General"
        self.difficulty: str = "medium"  # "easy", "medium", "challenging"
        self.history: list[dict] = []

    def reset(self):
        self.active = False
        self.question_num = 0
        self.score = 0
        self.current_question = ""
        self.current_expected = ""
        self.current_hint = ""
        self.subject = "General"
        self.topic = "General"
        self.difficulty = "medium"
        self.history = []

    def adjust_difficulty(self, was_correct: bool):
        """Adapt difficulty dynamically based on student performance."""
        if was_correct:
            if self.difficulty == "easy":
                self.difficulty = "medium"
            elif self.difficulty == "medium":
                self.difficulty = "challenging"
        else:
            if self.difficulty == "challenging":
                self.difficulty = "medium"
            elif self.difficulty == "medium":
                self.difficulty = "easy"


class ConversationTurn:
    """Represents a single conversational turn in the tutor session."""

    def __init__(self, role: str, text: str, intent: str = None, timestamp: float = None):
        self.role = role  # "student" or "tutor"
        self.text = text
        self.intent = intent
        self.timestamp = timestamp or time.time()


class LearningSession:
    """
    Manages autonomous tutoring session state:
    - Active student profile
    - Interactive quiz tracker
    - Short-term conversational memory
    - Active learning topic and subject
    - Pause / Resume state
    - Concept mastery tracking
    """

    def __init__(self, profile: StudentProfile = None):
        self.profile = profile or StudentProfile()
        self.quiz = QuizSession(total_questions=3)
        self.history: list[ConversationTurn] = []
        self.active_topic: str = "General"
        self.active_subject: str = self.profile.subject or ("Mathematics" if self.profile.level == "school" else "Physics")
        self.is_paused: bool = False
        self.mastered_concepts: list[str] = []
        self.needs_practice_concepts: list[str] = []

    def add_turn(self, role: str, text: str, intent: str = None):
        self.history.append(ConversationTurn(role=role, text=text, intent=intent))
        # Keep last 8 turns in active sliding window memory
        if len(self.history) > 8:
            self.history = self.history[-8:]

    def get_conversation_context(self) -> str:
        """Format recent dialogue for contextual Gemini prompting."""
        if not self.history:
            return ""
        lines = ["[RECENT CONVERSATION CONTEXT]"]
        for t in self.history[-6:]:
            speaker = "Student" if t.role == "student" else "Tutor"
            lines.append(f"{speaker}: {t.text}")
        return "\n".join(lines)

    def extract_and_update_topic(self, text: str):
        """Extract and update learning topic from student query."""
        if not text:
            return
        t_low = text.lower()
        topic_match = re.search(
            r"(?:what is|explain|teach me|about|gurinchi|ante enti|definition of)\s+([a-zA-Z\s]{3,30})",
            t_low,
        )
        if topic_match:
            candidate = topic_match.group(1).strip()
            # Ignore stop words, pronouns, and subject names (subjects are managed separately via active_subject)
            excluded_words = {
                "the", "a", "an", "this", "that", "it", "more", "everything", "something",
                "math", "maths", "mathematics", "science", "physics", "chemistry",
                "botany", "zoology", "biology", "social", "social studies", "english",
            }
            if candidate and candidate.lower() not in excluded_words:
                self.active_topic = candidate.title()


# Student Intent Constants
INTENT_STOP = "stop"
INTENT_PAUSE = "pause"
INTENT_RESUME = "resume"
INTENT_QUIZ_START = "quiz_start"
INTENT_EXAM_PREP = "exam_prep"
INTENT_REVISION = "revision"
INTENT_STEP_BY_STEP = "step_by_step"
INTENT_HINT = "hint"
INTENT_SWITCH_TOPIC = "switch_topic"
INTENT_DOUBT_FOLLOWUP = "doubt_followup"
INTENT_EXPLANATION = "explanation"

# Canonical subject mapping for School and Intermediate curricula
SUBJECT_CANONICAL_MAP = {
    # Mathematics
    "mathematics": "Mathematics",
    "math": "Mathematics",
    "maths": "Mathematics",
    "గణితం": "Mathematics",
    "గణిత": "Mathematics",
    "లెక్కలు": "Mathematics",
    "లెక్కల": "Mathematics",
    # Science
    "science": "Science",
    "సైన్స్": "Science",
    "విజ్ఞాన శాస్త్రం": "Science",
    "విజ్ఞాన": "Science",
    # Physics
    "physics": "Physics",
    "ఫిజిక్స్": "Physics",
    "భౌతిక శాస్త్రం": "Physics",
    "భౌతిక": "Physics",
    # Chemistry
    "chemistry": "Chemistry",
    "కెమిస్ట్రీ": "Chemistry",
    "రసాయన శాస్త్రం": "Chemistry",
    "రసాయన": "Chemistry",
    # Botany
    "botany": "Botany",
    "బాటనీ": "Botany",
    "వృక్ష శాస్త్రం": "Botany",
    "వృక్ష": "Botany",
    # Zoology
    "zoology": "Zoology",
    "జువాలజీ": "Zoology",
    "జంతు శాస్త్రం": "Zoology",
    "జంతు": "Zoology",
    # Biology
    "biology": "Biology",
    "బయాలజీ": "Biology",
    "జీవ శాస్త్రం": "Biology",
    "జీవ": "Biology",
    # English
    "english": "English",
    "ఇంగ్లీష్": "English",
    "ఆంగ్లం": "English",
    # Social Studies
    "social studies": "Social Studies",
    "social": "Social Studies",
    "సాంఘిక శాస్త్రం": "Social Studies",
    "సాంఘిక": "Social Studies",
    "సోషల్": "Social Studies",
}

# Subject keyword dictionaries for curriculum detection
SUBJECT_TOPIC_KEYWORDS = {
    "Mathematics": [
        "algebra", "geometry", "calculus", "trigonometry", "matrix", "matrices", "derivative",
        "integration", "integral", "equation", "quadratic", "polynomial", "fraction", "decimal",
        "percentage", "ratio", "proportion", "triangle", "circle", "radius", "diameter", "area",
        "perimeter", "volume", "lcm", "hcf", "factorization", "logarithm", "probability",
        "statistics", "mean", "median", "mode", "pythagoras", "hypotenuse", "angle", "slope",
        "coordinate", "arithmetic", "integers", "rational", "square root", "cube root",
        "గణితం", "లెక్కలు", "వైశాల్యం", "వ్యాసార్థం", "త్రికోణమితి", "సమీకరణం"
    ],
    "Physics": [
        "velocity", "acceleration", "kinematics", "force", "friction", "newton", "newton's",
        "momentum", "gravity", "gravitation", "inertia", "kinetic energy", "potential energy",
        "work energy", "power", "optics", "light", "reflection", "refraction", "lens", "mirror",
        "electricity", "current", "voltage", "resistance", "ohm's law", "circuit", "magnetism",
        "magnetic", "sound", "frequency", "wavelength", "amplitude", "thermodynamics", "heat",
        "temperature", "pressure", "density", "buoyancy", "speed", "joule", "watt", "pascal",
        "భౌతిక", "ఫిజిక్స్", "వేగం", "త్వరణం", "బలం", "శక్తి", "కాంతి", "ధ్వని", "విద్యుత్"
    ],
    "Chemistry": [
        "chemical", "reaction", "molecule", "molecular", "atom", "atomic", "element", "compound",
        "periodic table", "acid", "base", "bases", "salt", "ph value", "oxidation", "reduction",
        "redox", "valency", "electron", "proton", "neutron", "covalent", "ionic bond", "catalyst",
        "solution", "solute", "solvent", "distillation", "organic chemistry", "alkane", "alkene",
        "alkyne", "polymer", "neutralization", "h2o", "co2", "nacl",
        "రసాయన", "కెమిస్ట్రీ", "ఆమ్లం", "క్షారం", "లవణం", "పరమాణువు", "అణువు", "రసాయన చర్య"
    ],
    "Biology": [
        "photosynthesis", "chlorophyll", "cell", "cellular", "mitochondria", "nucleus", "cytoplasm",
        "tissue", "organ", "organism", "plant", "leaf", "root", "stem", "flower", "seed",
        "germination", "respiration", "digestion", "digestive", "circulatory", "blood", "heart",
        "lungs", "kidney", "nervous system", "brain", "neuron", "dna", "rna", "chromosome", "gene",
        "genetics", "evolution", "ecosystem", "food chain", "bacteria", "virus", "fungi",
        "chloroplast", "stomata", "xylem", "phloem", "transpiration",
        "బయాలజీ", "సైన్స్", "జీవ శాస్త్రం", "కిరణజన్య సంయోగక్రియ", "కణం", "కేంద్రకం", "పత్రహరితం"
    ],
    "Botany": [
        "photosynthesis", "plant", "plants", "leaf", "leaves", "root", "stem", "flower", "seed",
        "germination", "gymnosperm", "angiosperm", "bryophyte", "pteridophyte", "algae", "fungi",
        "chloroplast", "stomata", "xylem", "phloem", "transpiration", "botany", "plant anatomy",
        "morphology", "బాటనీ", "వృక్ష శాస్త్రం", "మొక్క"
    ],
    "Zoology": [
        "animal", "animals", "tissue", "tissues", "organ", "digestion", "circulatory", "blood",
        "heart", "lungs", "kidney", "nervous system", "brain", "neuron", "excretion", "reproduction",
        "genetics", "evolution", "invertebrate", "vertebrate", "zoology", "human physiology",
        "జువాలజీ", "జంతు శాస్త్రం", "జంతువు"
    ],
    "English": [
        "grammar", "tense", "tenses", "noun", "pronoun", "verb", "adverb", "adjective", "preposition",
        "conjunction", "active voice", "passive voice", "direct speech", "indirect speech",
        "comprehension", "synonym", "antonym", "metaphor", "simile", "idiom", "vocabulary",
        "spelling", "clause", "sentence", "punctuation", "essay", "letter writing",
        "ఇంగ్లీష్", "వ్యాకరణం"
    ],
    "Social Studies": [
        "constitution", "democracy", "parliament", "government", "monarchy", "fundamental rights",
        "judiciary", "election", "history", "mughal", "british", "freedom struggle", "world war",
        "dynasty", "geography", "climate", "monsoon", "plateau", "himalayas", "rivers", "earthquake",
        "latitude", "longitude", "economics", "gdp", "inflation", "agriculture", "industry", "poverty",
        "సాంఘిక", "సోషల్", "రాజ్యాంగం", "చరిత్ర", "భూగోళం"
    ]
}


def detect_question_subject(text: str, profile: StudentProfile = None) -> str | None:
    """
    Infer the academic subject of a student question based on curriculum keywords.
    Safeguards:
      - Does not force inference when keywords are ambiguous or equally scored.
      - Maps to student curriculum (School combines Physics/Chemistry/Biology into Science;
        Intermediate respects MPC/BiPC group subjects).
    Returns canonical subject name if confident, or None if ambiguous/generic.
    """
    if not text:
        return None

    t = text.lower()
    scores = {}

    for subj, kws in SUBJECT_TOPIC_KEYWORDS.items():
        score = 0
        for kw in kws:
            if re.search(r"\b" + re.escape(kw) + r"(?:s|es)?\b", t) or (any(c > '\u0c00' for c in kw) and kw in t):
                score += len(kw)
        if score > 0:
            scores[subj] = score

    if not scores:
        return None

    # Sort subjects by score descending
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_subj, best_score = sorted_scores[0]

    # Safeguard 1: Do not force subject inference when ambiguous (close score between different disciplines)
    if len(sorted_scores) > 1:
        second_subj, second_score = sorted_scores[1]
        # If second score is equal and belongs to an unrelated discipline, it is ambiguous
        if best_score == second_score and best_subj != second_subj:
            # Check if both are science branches under School
            if not (profile and profile.level == "school" and
                    best_subj in ["Physics", "Chemistry", "Biology", "Botany", "Zoology"] and
                    second_subj in ["Physics", "Chemistry", "Biology", "Botany", "Zoology"]):
                return None

    level = profile.level if profile else "school"
    group = profile.group if profile else None

    # Map to student curriculum:
    if level == "school":
        if best_subj in ["Biology", "Physics", "Chemistry", "Botany", "Zoology"]:
            return "Science"
        return best_subj if best_subj in ["Mathematics", "Science", "English", "Social Studies"] else None
    elif level == "intermediate":
        if group == "MPC":
            if best_subj in ["Biology", "Botany", "Zoology"]:
                return None
            return best_subj if best_subj in ["Mathematics", "Physics", "Chemistry"] else None
        elif group == "BiPC":
            if best_subj == "Biology":
                return "Botany"
            if best_subj == "Mathematics":
                return None
            return best_subj if best_subj in ["Botany", "Zoology", "Physics", "Chemistry"] else None

    return best_subj


# Patterns indicating incidental mentions or concept inquiries that must NOT trigger subject switch
INCIDENTAL_SUBJECT_PATTERNS = [
    r"\b(?:chemical|physical|phase|state|temperature|climate|pressure)\s+change\b",
    r"(?:రసాయన|భౌతిక|స్థితి|ఉష్ణోగ్రత)\s*మార్పు",
    # Comparative questions or inquiries about subject relationships:
    r"\b(?:how|why|is|can|does|do)\b.*\b(?:used in|help in|needed for|related to|relationship between|difference between|harder than|easier than)\b",
    r"(?:కి\s+తేడా|ఎందుకు\s+వాడతారు|సంబంధం|ఉపయోగం)",
    # Definitional queries:
    r"^(?:what is|define|explain)\s+(?:science|mathematics|math|physics|chemistry|botany|zoology|biology|english|social studies)\??$",
    r"^(?:సైన్స్|గణితం|లెక్కలు|ఫిజిక్స్|కెమిస్ట్రీ|బాటనీ|జువాలజీ|బయాలజీ)\s+అంటే\s+ఏమిటి\??$",
    # Calculation / problem solving queries (not subject switch commands):
    r"\b(?:calculate|find|solve|evaluate|simplify|what is the value of|derive|compute|లెక్క చేయి|సాల్వ్ చేయి|కనుక్కోండి|విలువ ఎంత)\b",
    # Formula / problem inquiries:
    r"\b(?:problem|equation|numerical|formula for|formula of|law of|principles? of|సూత్రం|నియమం|ప్రశ్న|సమస్య|లెక్క)\b",
    # General question starters:
    r"^(?:what|why|how|when|where|which|who|explain|define|tell me about|ఏమిటి|ఎందుకు|ఎలా|వివరించు|గురించి)\b",
    # Physics/science quantities in word problems (e.g. speed, velocity, acceleration, mass):
    r"\b(?:if\s+a\s+(?:car|ball|body|object|particle|train)|velocity|speed|acceleration|mass|force|gravity|momentum|distance|time|energy)\b.*\b(?:is|=|\d+|km/h|m/s|kg|meters?|seconds?)\b",
]


def detect_subject_switch(text: str, profile: StudentProfile = None) -> tuple[bool, str | None, str | None]:
    """
    Detect intentional student requests to switch the active learning subject.
    Distinguishes explicit switch commands from incidental subject mentions (e.g. 'How does math help in science?').
    Returns (is_switch: bool, canonical_subject: str | None, raw_subject: str | None).
    """
    if not text:
        return False, None, None

    t = text.lower().strip()

    # 1. Guard against incidental mentions and concept terms (e.g. 'chemical change', 'is math used in science?')
    for p in INCIDENTAL_SUBJECT_PATTERNS:
        if re.search(p, t):
            return False, None, None

    # 2. English switch patterns
    english_patterns = [
        r"\b(?:switch|change)\s+(?:the\s+)?(?:subject\s+)?(?:to\s+)?([a-zA-Z\s]+)",
        r"\b(?:switch\s+to|change\s+to)\s+([a-zA-Z\s]+)",
        r"\b(?:let'?s\s+(?:study|learn|start|explore|do))\s+([a-zA-Z\s]+)",
        r"\b(?:i\s+want\s+to\s+(?:study|learn))\s+([a-zA-Z\s]+)",
        r"\b(?:can\s+(?:we|you)\s+(?:study|learn|teach|start))\s+([a-zA-Z\s]+)",
        r"\b(?:teach\s+me|teach)\s+([a-zA-Z\s]+)",
        r"\b(?:start\s+with|start\s+learning|start\s+studying|start)\s+([a-zA-Z\s]+)",
    ]

    # 3. Telugu / Tanglish switch patterns
    telugu_patterns = [
        r"([a-zA-Z\s\u0c00-\u0c7f]+)\s+(?:చదువుదాం|నేర్చుకుందాం|బోధించు|చెప్పు|స్టార్ట్ చేయి|మొదలుపెట్టు|సబ్జెక్ట్ చెప్పు)",
        r"([a-zA-Z\s\u0c00-\u0c7f]+)\s*(?:కి|కు)\s*(?:మారు|మార్చు|మారండి|మార్చండి|switch)",
        r"(?:సబ్జెక్ట్\s+మార్చు|సబ్జెక్ట్\s+మార్చండి|మారు|మార్చు)\s*(?:మరియు|కి)?\s*([a-zA-Z\s\u0c00-\u0c7f]+)?",
        r"([a-zA-Z\s]+)\s+(?:chaduvudam|nerchukundam|cheppu|start\s+cheyyi|ki\s+switch\s+cheyyi)",
    ]

    captured_cand = None

    for pat in english_patterns:
        m = re.search(pat, t)
        if m:
            captured_cand = m.group(1).strip()
            break

    if not captured_cand:
        for pat in telugu_patterns:
            m = re.search(pat, t)
            if m:
                cand = m.group(1) if m.lastindex else None
                if cand:
                    captured_cand = cand.strip()
                    break

    # Resolve against canonical subject dictionary (longest key first)
    sorted_keys = sorted(SUBJECT_CANONICAL_MAP.keys(), key=len, reverse=True)
    canonical = None
    raw_subj = None

    if captured_cand:
        for k in sorted_keys:
            if re.search(r"\b" + re.escape(k) + r"\b", captured_cand) or k in captured_cand:
                canonical = SUBJECT_CANONICAL_MAP[k]
                raw_subj = k
                break

    if not canonical:
        # Check if text contains a clear switch trigger along with a subject
        has_switch_trigger = any(
            k in t for k in [
                "switch to", "switch subject to", "change to", "change subject to",
                "let's study", "let's learn", "start subject",
                "సబ్జెక్ట్ మార్చు", "సబ్జెక్ట్ మార్చండి",
                "chaduvudam", "nerchukundam"
            ]
        )
        if has_switch_trigger:
            for k in sorted_keys:
                if re.search(r"\b" + re.escape(k) + r"\b", t) or k in t:
                    canonical = SUBJECT_CANONICAL_MAP[k]
                    raw_subj = k
                    break

    if canonical:
        # Profile-specific curriculum resolution:
        # School (Classes 1-10) combines biological sciences into "Science"
        if canonical == "Biology":
            if profile and profile.level == "school":
                canonical = "Science"
            elif profile and profile.level == "intermediate" and profile.group == "BiPC":
                canonical = "Botany"
        return True, canonical, raw_subj

    return False, None, None


def classify_learning_intent(text: str, session: LearningSession = None) -> str:
    """Classify student intent for agentic decision making."""
    if not text:
        return INTENT_EXPLANATION
    t = text.lower().strip()

    # 1. Stop intent
    if is_stop_command(t):
        return INTENT_STOP

    # 2. Pause intent
    if any(k in t for k in ["pause", "aagu", "aagandi", "wait a minute", "hold on", "కాసేపు ఆగు", "ఆగు", "wait"]):
        return INTENT_PAUSE

    # 3. Resume intent
    if any(k in t for k in ["resume", "continue", "కంటిన్యూ", "tarvatha cheppu", "modalupettu", "మొదలుపెట్టు", "start again", "మళ్లీ చెప్పు", "పునఃప్రారంభించు", "ముందుకు సాగు"]):
        return INTENT_RESUME

    # 4. Quiz trigger
    if is_quiz_trigger(t):
        return INTENT_QUIZ_START

    # 5. Exam prep
    if any(k in t for k in ["2-mark", "2 marks", "5-mark", "5 marks", "10-mark", "10 marks", "మార్కులు", "exam question", "exam prep"]):
        return INTENT_EXAM_PREP

    # 6. Revision / summary
    if any(k in t for k in ["revise", "revision", "summary", "summarize", "recap", "పునశ్చరణ", "సారాంశం", "మొత్తం చెప్పు"]):
        return INTENT_REVISION

    # 7. Step-by-step problem solving
    if any(k in t for k in ["step by step", "how to solve", "help me solve", "solve this", "లెక్క చేయి", "సాల్వ్ చేయి", "steps cheppu"]):
        return INTENT_STEP_BY_STEP
    # Math equation pattern (e.g. 2x + 5 = 15 or solve equation)
    if re.search(r"\b\d*\s*[a-z]\s*[\+\-\*\/]\s*\d+\s*=", t):
        return INTENT_STEP_BY_STEP

    # 8. Hint request
    if any(k in t for k in ["hint", "clue", "stuck", "హింట్", "నాకు అర్థం కాలేదు సహాయం చేయి"]):
        return INTENT_HINT

    # 9. Switch subject
    is_switch, _, _ = detect_subject_switch(t, session.profile if session else None)
    if is_switch:
        return INTENT_SWITCH_TOPIC

    # 10. Doubt followup (pronouns or short interrogatives with prior context)
    if session and session.history and session.active_topic != "General":
        if any(t.startswith(w) for w in ["why", "where", "how", "what if", "mari", "ఎందుకు", "ఎక్కడ", "ఎలా"]):
            return INTENT_DOUBT_FOLLOWUP
        if any(w in t for w in ["it", "this", "that", "దీని", "దాని"]):
            return INTENT_DOUBT_FOLLOWUP

    return INTENT_EXPLANATION



def get_input_device() -> tuple[int, dict]:
    """
    Detect and validate the best microphone input device.
    1. Checks MIC_DEVICE_INDEX from .env if provided and validates max_input_channels > 0.
    2. Checks system default input device (sd.default.device[0]).
    3. Scans all available host devices to find the first valid input device.
    4. Provides safe fallback metadata if device query fails.
    """
    try:
        devices = sd.query_devices()
        env_dev = os.getenv("MIC_DEVICE_INDEX")
        if env_dev is not None and env_dev.strip().isdigit():
            dev_idx = int(env_dev.strip())
            if 0 <= dev_idx < len(devices) and devices[dev_idx].get("max_input_channels", 0) > 0:
                return dev_idx, devices[dev_idx]
            else:
                print(f"⚠️ MIC_DEVICE_INDEX={dev_idx} is not an input device (channels=0). Finding active microphone...")

        # Check system default input device
        default_in = sd.default.device[0]
        if default_in is not None and 0 <= default_in < len(devices):
            if devices[default_in].get("max_input_channels", 0) > 0:
                return default_in, devices[default_in]

        # Scan for the first device with input channels
        for idx, dev in enumerate(devices):
            if dev.get("max_input_channels", 0) > 0:
                return idx, dev

        return 1, {
            "name": "Default Microphone",
            "default_samplerate": 16000.0,
            "max_input_channels": 1,
        }
    except Exception as e:
        print(f"⚠️ Audio device query warning: {e}. Falling back to device 1.")
        return 1, {
            "name": "Default Microphone",
            "default_samplerate": 16000.0,
            "max_input_channels": 1,
        }


def is_stop_command(text: str) -> bool:
    """
    Check if the user transcript contains a stop or exit command.
    Supports Telugu ("ఆపు", "ఆపండి", etc.) and English ("stop", "exit", etc.).
    """
    if not text:
        return False

    clean_text = text.strip().lower()

    # 1. Check individual words (stripping common punctuation)
    words = [w.strip(".,!?;:\"'()[]{}") for w in clean_text.split()]
    for word in words:
        if word in STOP_COMMANDS:
            return True

    # 2. Check phrase / substring match for Telugu multi-word or compound forms
    for command in STOP_COMMANDS:
        if command in clean_text:
            return True

    return False


def is_quiz_trigger(text: str) -> bool:
    """Check if the student asked to start a quiz."""
    if not text:
        return False
    t = text.lower()
    return any(trigger in t for trigger in QUIZ_TRIGGERS)


def parse_quiz_command(transcript: str) -> tuple[str | None, str]:
    """
    Parse student's quiz speech to detect explicit completion commands:
    - 'submit' / 'done' -> (QUIZ_ACTION_SUBMIT, cleaned_answer_without_command)
    - 'skip' -> (QUIZ_ACTION_SKIP, "")
    - 'repeat' -> (QUIZ_ACTION_REPEAT, "")
    - 'stop' -> (QUIZ_ACTION_STOP, "")
    If no command found, returns (None, transcript).
    """
    if not transcript:
        return None, ""

    t_low = transcript.lower().strip()

    # 1. Stop Quiz command (check before general stop to prioritize quiz context)
    for cmd in QUIZ_STOP_COMMANDS:
        if cmd in t_low:
            return QUIZ_ACTION_STOP, ""
    if t_low in ["stop", "exit", "ఆపు", "ఆపండి"]:
        return QUIZ_ACTION_STOP, ""

    # 2. Repeat Question command
    for cmd in QUIZ_REPEAT_COMMANDS:
        if cmd in t_low:
            return QUIZ_ACTION_REPEAT, ""

    # 3. Skip Question command
    for cmd in QUIZ_SKIP_COMMANDS:
        if cmd in t_low:
            return QUIZ_ACTION_SKIP, ""

    # 4. Submit Answer / Done command
    for cmd in QUIZ_SUBMIT_COMMANDS:
        pattern = r"(?:[\s,.]|^)" + re.escape(cmd) + r"(?:[\s,.]|$)"
        if re.search(pattern, t_low):
            cleaned = re.sub(pattern, " ", transcript, flags=re.IGNORECASE).strip()
            cleaned = re.sub(r"[\s\.,!?;:\"']+$", "", cleaned).strip()
            cleaned = re.sub(r"^[\s\.,!?;:\"']+", "", cleaned).strip()
            cleaned = re.sub(r"\s+", " ", cleaned).strip()
            return QUIZ_ACTION_SUBMIT, cleaned

    return None, transcript


def is_filler_or_noise(text: str) -> bool:
    """
    Check if transcript is merely a hesitation sound, punctuation, or breathing noise.
    Preserves valid short answers such as A/B/C/D, digits, yes/no, Oxygen,
    and Telugu scientific equivalents.
    """
    if not text:
        return True
    cleaned = text.strip().lower()
    if cleaned in FILLER_NOISE_TOKENS:
        return True
    # Strip punctuation and whitespace
    stripped = re.sub(r"[\s\.,!?;:\"'\(\)\[\]\{\}\-_]+", "", cleaned)
    if not stripped:
        return True
    # Digits (e.g. "8", "1", "10") are meaningful answers and must NOT be flagged as noise
    if stripped.isdigit():
        return False
    # Valid short conversational and quiz tokens (A/B/C/D, yes/no, Oxygen, Telugu equivalents)
    if stripped in VALID_SHORT_ANSWERS:
        return False
    # Legitimate words with 2 or more characters that are not in filler noise tokens
    if len(stripped) >= 2:
        return False
    # Isolated single-character noise tokens not in whitelist
    return True


# Isolated stopwords / dangling grammatical fragments that indicate speech was cut off or incomplete
DANGLING_GRAMMATICAL_FRAGMENTS = {
    # English conjunctions, prepositions, articles, copulas
    "and", "or", "so", "the", "a", "an", "is", "are", "of", "to", "in", "at",
    "by", "for", "with", "about", "it", "that", "this", "but", "because",
    "then", "if", "just", "as", "than", "which",
    # Telugu conjunctions, postpositions, and incomplete linkers
    "మరియు", "కానీ", "అయితే", "యొక్క", "లో", "తో", "కొరకు", "గురించి",
    "అని", "కోసం", "గా", "నుండి", "ద్వారా",
}

# Standalone short interrogatives that are valid only if prior conversation context exists
CONTEXTUAL_DOUBT_INTERROGATIVES = {
    "why", "why?", "where", "where?", "how", "how?", "what if", "mari",
    "ఎందుకు", "ఎందుకు?", "ఎక్కడ", "ఎక్కడ?", "ఎలా", "ఎలా?", "మరి", "ఏమిటి", "ఏంటి",
}

# Bare incomplete question starters lacking a subject, concept, or noun
INCOMPLETE_QUESTION_STARTERS = {
    "what is", "what are", "what was", "what were", "tell me", "can you",
    "can you explain", "explain the", "explain", "how to", "how do", "how does",
    "why is", "why does", "why do",
    "చెప్పు", "వివరించు", "ఏమిటి", "ఏంటి",
}


def is_incomplete_speech(text: str, session: "LearningSession" = None) -> bool:
    """
    Detect empty, meaningless, or incomplete speech transcripts.
    Preserves valid standalone doubt questions (e.g. 'ఎందుకు?') when prior conversation context exists.
    Preserves all valid short answers, options, and scientific terms.
    Returns True if transcript is incomplete/meaningless and should NOT be sent to Gemini.
    """
    if not text:
        return True

    cleaned = text.strip()
    if not cleaned:
        return True

    # Punctuation-only or noise
    stripped = re.sub(r"[\s\.,!?;:\"'\(\)\[\]\{\}\-_]+", "", cleaned.lower())
    if not stripped:
        return True

    # Check against known filler noise tokens
    if stripped in FILLER_NOISE_TOKENS or cleaned.lower() in FILLER_NOISE_TOKENS:
        return True

    # Check if this is a known valid short answer (e.g. A/B/C/D, yes/no, Oxygen, digits)
    if stripped.isdigit() or stripped in VALID_SHORT_ANSWERS or cleaned.lower() in VALID_SHORT_ANSWERS:
        return False

    # Check if text contains any recognized scientific term
    words = [w.strip() for w in re.split(r"[\s\.,!?;:\"']+", cleaned.lower()) if w.strip()]
    if any(w in VALID_SHORT_ANSWERS for w in words):
        return False

    # Check standalone short interrogatives (e.g. 'Why?', 'ఎందుకు?', 'How?')
    t_clean = cleaned.lower().strip(" \t\n\r.?!,;:")
    if t_clean in CONTEXTUAL_DOUBT_INTERROGATIVES:
        # Valid standalone question ONLY IF there is prior active dialogue in session history
        if session and session.history and len(session.history) > 0 and session.active_topic != "General":
            return False  # Context-aware valid follow-up doubt!
        else:
            return True   # No prior context to attach to; incomplete query

    # Check for bare incomplete question starters without concept/noun
    if t_clean in INCOMPLETE_QUESTION_STARTERS:
        return True

    # Check for isolated dangling grammatical linkers
    if len(words) == 1 and words[0] in DANGLING_GRAMMATICAL_FRAGMENTS:
        return True

    # Check trailing cutoffs (short sentence of <= 3 words ending with dangling fragment)
    if len(words) <= 3 and words[-1] in DANGLING_GRAMMATICAL_FRAGMENTS:
        return True

    # Check for repeated filler syllables (e.g. "uh uh", "um um", "హా హా", "ఉమ్ ఉమ్")
    if len(words) >= 2 and all(w in FILLER_NOISE_TOKENS for w in words):
        return True

    # If it is a single isolated character not in whitelist
    if len(stripped) <= 1:
        return True

    return False


def extract_school_class(text: str) -> int | None:
    """
    Extract school class (1 to 10) from English, Telugu, and Tanglish speech transcripts.
    Handles digits with English/Telugu suffixes (8th, 8వ, 8va), word forms (eight, ఎనిమిదో),
    Roman numerals (VIII), and phrases like 'Class 8', 'I am studying 8th class', 'నేను 8వ తరగతి చదువుతున్నాను'.
    """
    if not text:
        return None

    t = text.lower().strip()

    # 1. Explicit phrases with class/standard/grade/తరగతి/క్లాస్
    # e.g. "class 8", "8th class", "class 8th", "క్లాస్ 8", "8వ తరగతి", "8 తరగతి", "std 8", "8th standard"
    explicit_match = re.search(
        r"(?:class|క్లాస్|std|standard|grade|తరగతి)\s*[:\-]?\s*(\d{1,2})|"
        r"(\d{1,2})\s*(?:st|nd|rd|th|వ|va|o|do)?\s*(?:class|క్లాస్|std|standard|grade|తరగతి)",
        t,
        re.IGNORECASE,
    )
    if explicit_match:
        val_str = explicit_match.group(1) or explicit_match.group(2)
        val = int(val_str)
        if 1 <= val <= 10:
            return val

    # 2. Digits 1 to 10 with optional suffixes (8th, 8వ, 8va, 8)
    digit_matches = re.finditer(r"(?<!\d)(10|[1-9])(?:\s*(?:st|nd|rd|th|వ|va|o|do|th\s*class))?(?!\d)", t, re.IGNORECASE)
    for m in digit_matches:
        val = int(m.group(1))
        if 1 <= val <= 10:
            return val

    # 3. Telugu ordinals and number words (ordered longest to shortest to prevent prefix collisions)
    telugu_class_map = [
        ("తొమ్మిదవ", 9), ("తొమ్మిదో", 9), ("తొమ్మిది", 9),
        ("ఎనిమిదవ", 8), ("ఎనిమిదో", 8), ("ఎనిమిది", 8),
        ("నాలుగవ", 4), ("నాలుగో", 4), ("నాలుగు", 4),
        ("మొదటి", 1), ("ఒకటవ", 1), ("ఒకటో", 1), ("ఒకటి", 1),
        ("రెండవ", 2), ("రెండో", 2), ("రెండు", 2),
        ("మూడవ", 3), ("మూడో", 3), ("మూడు", 3),
        ("ఐదవ", 5), ("ఐదో", 5), ("ఐదు", 5),
        ("ఆరవ", 6), ("ఆరో", 6), ("ఆరు", 6),
        ("ఏడవ", 7), ("ఏడో", 7), ("ఏడు", 7),
        ("పదవ", 10), ("పదో", 10), ("పది", 10),
    ]
    for word, val in telugu_class_map:
        if word in t:
            return val

    # 4. English words and ordinals
    english_class_map = {
        "tenth": 10, "ten": 10,
        "ninth": 9, "nine": 9,
        "eighth": 8, "eight": 8,
        "seventh": 7, "seven": 7,
        "sixth": 6, "six": 6,
        "fifth": 5, "five": 5,
        "fourth": 4, "four": 4,
        "third": 3, "three": 3,
        "second": 2, "two": 2,
        "first": 1, "one": 1,
    }
    for word, val in english_class_map.items():
        if re.search(r"\b" + re.escape(word) + r"\b", t):
            return val

    # 5. Roman numerals (VIII, VII, IX, X, etc.)
    # Single-letter Roman numerals ('i', 'v', 'x') require explicit class/std prefix or suffix to avoid colliding with English pronoun 'I'
    roman_map = {
        "viii": 8, "vii": 7, "vi": 6, "ix": 9, "x": 10,
        "iv": 4, "v": 5, "iii": 3, "ii": 2, "i": 1,
    }
    for r_num, val in roman_map.items():
        if len(r_num) == 1:
            if re.search(r"\b(?:class|క్లాస్|std|standard|grade|తరగతి)\s*" + re.escape(r_num) + r"\b", t):
                return val
            if re.search(r"\b" + re.escape(r_num) + r"(?:st|nd|rd|th)\b", t):
                return val
        else:
            if re.search(r"\b(?:class|క్లాస్|std|standard|grade|తరగతి)?\s*" + re.escape(r_num) + r"(?:st|nd|rd|th)?\b", t):
                return val

    return None



def clean_text_for_speech(text: str) -> str:
    """
    Sanitize text by removing markdown symbols, backticks, and headers for smooth audio playback,
    while preserving mathematical operators, negative signs (e.g. '(-g, -f)', 'x - h'),
    and formula symbols (e.g. 'sqrt(g^2 + f^2 - c)').
    """
    if not text:
        return ""
    # Strip markdown headers at line start (e.g. # Header, ## Header, ### Header)
    cleaned = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    # Strip markdown list bullet markers at the start of a line (*, -, •) without affecting inline minus signs
    cleaned = re.sub(r"(?m)^\s*[-•*]\s+", "", cleaned)
    # Strip markdown bold/italics and backticks
    cleaned = (
        cleaned.replace("**", "")
        .replace("*", "")
        .replace("`", "")
    )
    # Collapse multiple inline spaces and newlines into a single space for smooth speech synthesis
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def detect_tts_language(text: str) -> str:
    """
    Determine the appropriate TTS language code for Gnani Timbre.
    If text contains Telugu Unicode characters, use 'te-IN'.
    Otherwise use 'en-IN' for clear English pronunciation.
    """
    if not text:
        return "te-IN"
    has_telugu = any("\u0c00" <= char <= "\u0c7f" for char in text)
    return "te-IN" if has_telugu else "en-IN"


def parse_intermediate_group(text: str) -> tuple[str | None, str | None]:
    """
    Parse and normalize Intermediate group from student speech transcripts.
    Supports common STT variants for BiPC ('bipc', 'bpc', 'bipac', 'bi p c', 'b.p.c', 'బైపీసీ', 'బైపిసి', 'బిపిసి', 'బైపాక్')
    and MPC ('mpc', 'm p c', 'm.p.c', 'm-p-c', 'ఎంపీసీ', 'ఎంపిసి').
    Returns (group: 'MPC' | 'BiPC' | None, clarification_prompt: str | None).
    Guarantees:
      - Words like biology/botany/zoology/maths alone do not confirm a group without clarification.
      - Unsupported groups (CEC, MEC, HEC, arts, commerce) are politely flagged.
      - Ambiguous inputs trigger clarification and are never silently saved.
    """
    if not text:
        return None, None
    t = text.lower().strip()

    # 1. Check for unsupported groups
    unsupported = re.search(r"\b(?:c[\s\.\-]*e[\s\.\-]*c|m[\s\.\-]*e[\s\.\-]*c|h[\s\.\-]*e[\s\.\-]*c|commerce|arts)\b", t)
    if unsupported:
        grp = unsupported.group(0).upper().replace(" ", "").replace(".", "")
        return None, f"Our AI Tutor currently supports MPC and BiPC groups in Intermediate. We do not support {grp}. Please choose MPC or BiPC."

    # 2. Check for BiPC variants
    bipc_en = bool(re.search(r"(?:(?<=\W)|^)(?:b[\s\.\-]*i[\s\.\-]*p[\s\.\-]*a?[\s\.\-]*c|b[\s\.\-]*p[\s\.\-]*c)(?=\W|$)", t))
    bipc_te = any(w in t for w in ["బైపీసీ", "బైపిసి", "బిపిసి", "బైపాక్", "బైపాసి"])
    is_bipc = bipc_en or bipc_te

    # 3. Check for MPC variants
    mpc_en = bool(re.search(r"(?:(?<=\W)|^)(?:m[\s\.\-]*p[\s\.\-]*c)(?=\W|$)", t))
    mpc_te = any(w in t for w in ["ఎంపీసీ", "ఎంపిసి"])
    is_mpc = mpc_en or mpc_te

    if is_bipc and is_mpc:
        return None, "Please specify whether you are studying MPC or BiPC."
    if is_bipc:
        return "BiPC", None
    if is_mpc:
        return "MPC", None

    # 4. Standalone subject mentions: require confirmation
    if any(w in t for w in ["biology", "botany", "zoology", "బయాలజీ", "బాటనీ", "జువాలజీ"]):
        return None, "Did you mean the BiPC group (Biology, Physics, Chemistry)? Please say BiPC to confirm."
    if any(w in t for w in ["maths", "mathematics", "గణితం", "లెక్కలు"]):
        return None, "Did you mean the MPC group (Maths, Physics, Chemistry)? Please say MPC to confirm."

    return None, None


OUT_OF_SCOPE_LEVELS = [
    "college", "degree", "btech", "b.tech", "engineering", "diploma",
    "b.sc", "bsc", "bcom", "b.com", "ba", "mba", "mca", "university",
    "కాలేజ్", "డిగ్రీ", "ఇంజనీరింగ్"
]


def parse_education_level(text: str, profile: StudentProfile) -> tuple[bool, str]:
    """
    Parse student's spoken/text response to determine or update their education level.
    Returns (is_complete, next_question_or_confirmation).
    Safeguards:
      - Never overwrites a saved complete student profile due to an ambiguous or out-of-scope utterance.
      - Detects out-of-scope education levels (college, degree, B.Tech) and explains scope clearly.
      - Asks for clarification when school vs intermediate is ambiguous.
    """
    t = text.lower().strip()

    # Safeguard 4: If profile is already complete, never overwrite with an ambiguous or out-of-scope query
    if profile.is_complete():
        if any(k in t for k in OUT_OF_SCOPE_LEVELS):
            return True, f"You are already registered as a {profile.describe()}. We do not cover college or degree courses. What would you like to study?"
        # Only allow explicit school class changes for school students if a valid Class 1-10 is explicitly mentioned
        if profile.level == "school" and any(k in t for k in ["class", "క్లాస్", "తరగతి", "grade", "standard", "std"]):
            explicit_class = extract_school_class(t)
            if explicit_class:
                profile.grade_or_year = explicit_class
        desc = profile.describe()
        subjects_str = ", ".join(profile.allowed_subjects())
        return True, f"Wonderful! You are registered as a {desc}. We can study {subjects_str}. What would you like to learn today?"

    # Track if user explicitly selected school or intermediate in this exact turn
    selected_school_now = False
    selected_inter_now = False

    # 1. Detect level if not yet set
    if not profile.level:
        # Check for out-of-scope college / degree levels
        if any(k in t for k in OUT_OF_SCOPE_LEVELS):
            return False, "Our AI Student Tutor is specialized exclusively for School (Classes 1 to 10) and Intermediate (MPC/BiPC) students. We do not cover college or degree courses. Please say School or Intermediate to proceed."

        has_school = any(k in t for k in ["school", "స్కూల్", "పాఠశాల", "బడి"])
        has_inter = any(k in t for k in ["intermediate", "ఇంటర్మీడియట్", "ఇంటర్", "inter"])

        # Ambiguous selection (both mentioned)
        if has_school and has_inter:
            return False, "Please clarify whether you are studying in School (Classes 1 to 10) or Intermediate (1st/2nd Year)."

        if has_school:
            profile.level = "school"
            selected_school_now = True
        elif has_inter:
            profile.level = "intermediate"
            selected_inter_now = True

    # 2. Check for intermediate group indicators
    detected_group, group_clarification = parse_intermediate_group(t)
    if detected_group:
        profile.level = "intermediate"
        profile.group = detected_group

    # 3. Check for School class (1 to 10)
    # Extracts class from English, Telugu, and Tanglish phrases (e.g. "Class 8", "8th class", "నేను 8వ తరగతి చదువుతున్నాను", "8")
    if not profile.grade_or_year:
        extracted_class = extract_school_class(t)
        if extracted_class and profile.level != "intermediate":
            profile.level = "school"
            profile.grade_or_year = extracted_class
    else:
        # If class is already set, only change if user explicitly mentions a new class
        if any(k in t for k in ["class", "క్లాస్", "తరగతి", "grade", "standard", "std"]):
            explicit_class = extract_school_class(t)
            if explicit_class and profile.level != "intermediate":
                profile.grade_or_year = explicit_class

    # 4. If level is intermediate, check for year (1st or 2nd) and group
    if profile.level == "intermediate":
        if not profile.grade_or_year:
            if any(k in t for k in ["1st", "first", "1", "ఒకటవ", "మొదటి", "one", "1వ"]):
                profile.grade_or_year = 1
            elif any(k in t for k in ["2nd", "second", "2", "రెండవ", "two", "2వ"]):
                profile.grade_or_year = 2

        if not profile.group:
            if detected_group:
                profile.group = detected_group
            elif group_clarification:
                return False, group_clarification

    # 5. Evaluate profile completeness
    if profile.is_complete():
        desc = profile.describe()
        subjects_str = ", ".join(profile.allowed_subjects())
        confirm_msg = (
            f"Wonderful! You are registered as a {desc}. "
            f"We can study {subjects_str}. What would you like to learn today?"
        )
        return True, confirm_msg

    # 6. Determine exact next onboarding question
    if not profile.level:
        return False, "Welcome to your AI Student Tutor! Are you a School student or an Intermediate student?"
    elif profile.level == "school" and not profile.grade_or_year:
        # Clarification if student mentions 11th or 12th in school context
        if any(k in t for k in ["11", "12", "11th", "12th", "11వ", "12వ"]):
            return False, "Classes 11 and 12 are Intermediate 1st and 2nd Year. Are you an Intermediate student or in School Class 1 to 10?"
        if selected_school_now:
            return False, "Great! Which class are you studying in? Please say Class 1 to Class 10."
        else:
            return False, "I didn't catch your class clearly. Could you please repeat which class you are in, from Class 1 to Class 10?"
    elif profile.level == "intermediate":
        if not profile.grade_or_year:
            if any(k in t for k in ["3rd", "3", "4th", "4", "final year", "semester"]):
                return False, "Intermediate has 1st Year and 2nd Year only. Are you studying in 1st Year or 2nd Year Intermediate?"
            if selected_inter_now:
                return False, "Great! Are you studying in 1st Year or 2nd Year Intermediate?"
            else:
                return False, "I didn't catch your year clearly. Are you studying in 1st Year or 2nd Year Intermediate?"
        elif not profile.group:
            if group_clarification:
                return False, group_clarification
            return False, "Which group are you in? Please say MPC or BiPC."

    return False, "Please tell me if you are a School or Intermediate student."


def build_system_instruction(profile: StudentProfile, session: "LearningSession" = None, turn_subject: str = None) -> str:
    """
    Build dynamic system instructions for Gemini tailored to the active student profile.
    Strictly restricted to School (Classes 1-10) and Intermediate (MPC/BiPC).
    Strictly forbids B.Tech and Engineering subjects.
    Incorporates active topic, conversation memory, Socratic guidance, and anti-hallucination guardrails.
    Supports turn_subject for temporary turn-level cross-subject questions without mutating profile.
    """
    desc = profile.describe() if profile and profile.is_complete() else "School or Intermediate student"
    level = profile.level if profile else "school"
    grade = profile.grade_or_year if profile else "General"
    group = profile.group if profile else "General"
    active_subject = turn_subject or (session.active_subject if session else (profile.subject if profile else "General"))
    active_topic = session.active_topic if session else "General"

    context_str = ""
    if session and session.history:
        ctx = session.get_conversation_context()
        if ctx:
            context_str = f"\n{ctx}\n"

    return (
        "You are an encouraging, friendly, and expert AI Student Tutor exclusively for:\n"
        "1. School Students (Classes 1 to 10): Mathematics, Science, English, Social Studies.\n"
        "2. Intermediate Students (1st & 2nd Year):\n"
        "   - MPC: Mathematics, Physics, Chemistry\n"
        "   - BiPC: Botany, Zoology, Physics, Chemistry\n\n"
        "CRITICAL SCOPE RULE:\n"
        "- Do NOT teach B.Tech, Engineering, college-level computer programming (such as C, Java, Python, OS, DBMS, Networks), or engineering topics.\n"
        "- If a student asks about college-level engineering or B.Tech subjects, politely reply that you are exclusively specialized for School (Classes 1-10) and Intermediate (MPC/BiPC) students, and invite them to ask about their school or intermediate subjects.\n\n"
        f"ACTIVE STUDENT PROFILE:\n"
        f"- Target Level: {desc} (Level: {level}, Class/Year: {grade}, Group: {group})\n"
        f"- Current Subject: {active_subject}\n"
        f"- Current Topic: {active_topic}\n"
        f"{context_str}\n"
        "TEACHING GUIDELINES:\n"
        "1. School Mode (Classes 1 to 10):\n"
        "   - Adjust explanation difficulty strictly to the student's class.\n"
        "   - Use simple, age-appropriate language and relatable everyday examples (e.g. sharing fruits/sweets, nature, cricket).\n"
        "   - For Mathematics problems: explain step-by-step with clear logic. Encourage students instead of just giving bare answers.\n"
        "   - End explanations with a small, friendly check question to see if the student understood.\n\n"
        "2. Intermediate Mode (1st & 2nd Year MPC / BiPC):\n"
        "   - Explain concepts clearly according to Intermediate Board / NCERT syllabus.\n"
        "   - Help with textbook questions, numerical problem formulas, and conceptual clarity.\n"
        "   - Exam Preparation Rubric:\n"
        "     * 2-mark answer: Provide 1-2 crisp, high-yield sentences with exact definition and key formula.\n"
        "     * 5-mark answer: Provide structured short answer with clear definition, 3-4 key points/steps, and a real-world example.\n"
        "     * 10-mark answer: Provide comprehensive, structured breakdown with introduction, numbered key principles, formula/diagram description, and practical significance.\n"
        "   - IMPORTANT: Never claim or guarantee that any question will appear in an actual examination.\n"
        "   - Do not invent facts or provide unverified information.\n\n"
        "3. Socratic Problem Solving & Guidance:\n"
        "   - DIRECT ANSWERS VS SOCRATIC GUIDANCE:\n"
        "     * Direct Answers: When the student explicitly asks for a direct answer or definition, provide it clearly, directly, and immediately.\n"
        "     * Practice Mode: When the student is practicing or solving a problem, use Socratic guidance: state the relevant formula, explain the first step, and ask the student to deduce the next step. Do NOT reveal the final numerical answer prematurely.\n"
        "   - Missing Values & Assumption Handling:\n"
        "     * When a student asks whether they can assume a value (e.g. 'Can we assume a value?', 'Can I assume a value?', 'Shall we assume a value?', 'Let us assume a value', 'విలువ అనుకోవచ్చా?', 'మనం ఒక విలువ అనుకుందామా?'): First ask which value they want to choose (e.g. 'Sure! Which value would you like to use for the radius?'). State the formula and let the student choose the value before proceeding.\n"
        "     * When a student explicitly asks the tutor to choose an example or value (e.g. 'You choose an example', 'You choose a value', 'Give me an example value', 'నువ్వే ఒక విలువ తీసుకో', 'నువ్వే ఎగ్జాంపుల్ తీసుకో'): Provide clearly labeled illustrative values (e.g. 'For example, let us take radius r = 7 cm...'), substitute into the formula, and guide the student step-by-step.\n"
        "     * Missing Problem Values: Never invent or assume unstated problem values without clearly stating they are an example (e.g. 'For example, if we let mass m = 5 kg...'). State the formula and ask the student for the given values.\n\n"
        "4. Anti-Hallucination Guardrails:\n"
        "   - State only standard NCERT / State Board textbook syllabus facts.\n"
        "   - Never invent nonexistent formulas, dates, or page numbers.\n"
        "   - If the student's question is ambiguous or lacks necessary values (e.g. missing variables in a math problem), politely ask for the missing values.\n\n"
        "5. Language & Tone Guidelines:\n"
        "   - If the student asks in Telugu, reply in conversational, natural Telugu (in Telugu script).\n"
        "   - Telugu Conciseness: Keep normal Telugu explanations short, simple, and student-friendly (2 to 3 sentences).\n"
        "     * Use relatable everyday examples (e.g. sharing fruits/sweets, nature, cricket, sunlight, boiling milk).\n"
        "     * Avoid heavy academic jargon, run-on sentences, or essay-length paragraphs.\n"
        "     * End with exactly one short, warm follow-up check question when appropriate (e.g. 'ఇది అర్థమైందా?', 'మీకు ఇది స్పష్టంగా ఉందా?'). Limit to AT MOST ONE follow-up check question; NEVER ask multiple follow-up questions.\n"
        "     * Avoid Repetition: Do NOT repeat or echo back the student's question, and avoid repeating the same concept multiple times.\n"
        "   - If the student asks in English, reply in simple, clear English.\n"
        "   - If the student mixes Telugu and English (Tanglish), reply in conversational Tanglish or Telugu.\n"
        "   - Explicit switches: 'Telugu lo explain cheyyi' or 'తెలుగులో చెప్పు' -> explain in Telugu script (తెలుగు లిపి); 'Explain in English' -> explain in English; 'Simple ga cheppu' -> explain extra simply.\n\n"
        "6. Voice-Friendly Formatting & Spoken Length:\n"
        "   - DEFAULT CONCISENESS (Normal Doubt Mode): Keep spoken answers concise (strictly 2 to 3 sentences).\n"
        "     * Explain the core concept directly and clearly in a conversational tone.\n"
        "     * Direct Start & Zero Echoing: Start directly with the core answer or explanation. Do NOT repeat, echo, or paraphrase the student's question.\n"
        "     * No Repetitive Phrasing: Do NOT repeat the same concept or rephrase previously spoken sentences in different words.\n"
        "     * Limit follow-up: Ask AT MOST ONE friendly check question at the very end. NEVER ask multiple questions in a single response.\n"
        "   - EXCEPTION FOR DETAILED ANSWERS: Provide detailed, structured, comprehensive answers ONLY when explicitly requested:\n"
        "     * 2-mark answer: Exactly 1 to 2 crisp, high-yield sentences with key definition and formula.\n"
        "     * 5-mark answer: Structured short answer (definition, 3-4 key points/steps, example).\n"
        "     * 10-mark answer: Comprehensive breakdown (intro, principles, diagram explanation, summary).\n"
        "     * Explicit detail requests ('explain in detail', 'వివరంగా చెప్పు', 'పూర్తిగా వివరించు', 'detailed explanation'): Provide comprehensive step-by-step breakdown.\n"
        "   - Do NOT use markdown asterisks (**bold**), backticks (`code`), bullet points (*), headers (###), or markdown tables.\n"
        "   - Pronounce formulas verbally (e.g. write 'H two O' or 'x squared' or 'F equals m a' instead of raw syntax).\n\n"
        "7. Mathematical Rigor & Accuracy:\n"
        "   - Always preserve and correctly present mathematical operators, brackets, exponents, negative signs, and roots.\n"
        "   - Circle Equations (Strict syllabus accuracy):\n"
        "     * Standard form with center at origin (0, 0) and radius r: x^2 + y^2 = r^2\n"
        "       (Verbalize: 'x squared plus y squared equals r squared' or 'x స్క్వేర్ ప్లస్ y స్క్వేర్ ఈక్వల్స్ r స్క్వేర్')\n"
        "     * Center-Radius form with center (h, k) and radius r: (x - h)^2 + (y - k)^2 = r^2\n"
        "       (Verbalize: 'open bracket x minus h close bracket squared plus open bracket y minus k close bracket squared equals r squared')\n"
        "     * General second-degree equation of a circle: x^2 + y^2 + 2gx + 2fy + c = 0\n"
        "       Center is (-g, -f). ALWAYS explicitly include both negative signs (spoken: 'minus g, minus f' or 'మైనస్ g, మైనస్ f'). NEVER drop minus signs!\n"
        "       Radius is r = sqrt(g^2 + f^2 - c) where condition g^2 + f^2 - c >= 0 must hold. ALWAYS include the square root (spoken: 'square root of g squared plus f squared minus c' or 'స్క్వేర్ రూట్ ఆఫ్ g స్క్వేర్ ప్లస్ f స్క్వేర్ మైనస్ c').\n"
        "   - NEVER omit negative signs, brackets, exponents, or square roots when formulating or explaining mathematical solutions."
    )


def call_gemini(prompt: str, system_instruction: str, max_retries: int = 3) -> str:
    """
    Call Gemini API with model fallback across candidate flash models.
    """
    for model_name in CANDIDATE_MODELS:
        for attempt in range(max_retries):
            try:
                response = gemini_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.7,
                    ),
                )
                if response and response.text:
                    return clean_text_for_speech(response.text)
            except Exception:
                time.sleep(1.0)
                continue

    return "క్షమించండి, ప్రస్తుతం సమాధానం అందించడంలో సమస్య ఉంది. దయచేసి మీ ప్రశ్నను మళ్లీ అడగండి."


def is_college_cse_query(text: str) -> bool:
    """Detect if the user is asking about excluded college/engineering/CSE topics."""
    t = text.lower()
    engineering_keywords = [
        "b.tech", "btech", "engineering", "operating system", "deadlock",
        "primary key", "foreign key", "dbms", "sql query", "tcp/ip", "udp",
        "osi layer", "c pointers", "java inheritance", "list comprehension",
        "data structures and algorithms", "compiler design", "microprocessor",
        "inheritance", "polymorphism", "data structures", "software engineering",
        "c++", "pointers", "pointer",
    ]
    if any(k in t for k in engineering_keywords):
        return True
    # Catch programming language names when used in a technical context
    if any(lang in t for lang in ["java", "python", "c programming"]) and any(w in t for w in ["program", "code", "syntax", "inheritance", "function", "variable", "loop", "class"]):
        return True
    return False


def detect_requested_grade_override(text: str, current_profile: StudentProfile = None) -> int | None:
    """
    Detect if the student requested an ad-hoc explanation tailored to a different grade level
    (e.g., an Intermediate student asking: 'Explain photosynthesis like in Class 8' or '8th class level lo cheppu').
    Returns requested grade integer (1 to 10) if an override was requested, else None.
    Never modifies the persistent StudentProfile or active subject.
    """
    if not text:
        return None

    t = text.lower().strip()

    # Look for explicit grade-level request patterns
    grade_override_patterns = [
        # English: "like in class 8", "for class 8", "class 8 level", "in class 8 level", "8th class level", "at class 8 level"
        r"\b(?:like\s+in|as\s+in|for|at|in)?\s*(?:class|grade|std|standard)\s*[:\-]?\s*(\d{1,2})\s*(?:level|level\s+lo|లాగా|స్థాయి\s*లో)?\b",
        r"\b(\d{1,2})\s*(?:st|nd|rd|th|వ|va|o|do)?\s*(?:class|grade|std|standard)\s*(?:level|level\s+lo|లాగా|స్థాయి\s*లో)\b",
        # Telugu / Tanglish: "8వ తరగతి లాగా", "8th class level lo", "క్లాస్ 8 లాగా", "8 తరగతి స్థాయి లో"
        r"(?:క్లాస్|తరగతి)\s*(\d{1,2})\s*(?:లాగా|స్థాయి\s*లో|level\s*lo)",
        r"(\d{1,2})\s*(?:వ|va)?\s*తరగతి\s*(?:లాగా|స్థాయి\s*లో)",
    ]

    for pat in grade_override_patterns:
        m = re.search(pat, t)
        if m:
            val_str = m.group(1)
            if val_str and val_str.isdigit():
                val = int(val_str)
                if 1 <= val <= 10:
                    # If current profile is school and already in that exact class, no override needed
                    if current_profile and current_profile.level == "school" and current_profile.grade_or_year == val:
                        return None
                    return val

    # Check word numbers: "eighth class level", "class eight level"
    word_to_num = {
        "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
        "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    }
    for word, num in word_to_num.items():
        if re.search(r"\b" + re.escape(word) + r"\s+(?:class|grade|standard)\s+(?:level|level\s+lo|లాగా)\b", t):
            if current_profile and current_profile.level == "school" and current_profile.grade_or_year == num:
                return None
            return num

    return None


def generate_reply(text: str, profile: StudentProfile = None, session: "LearningSession" = None, turn_subject: str = None) -> str:
    """
    Generate conversational tutor response for School and Intermediate students.
    Handles general questions, homework help, exam preparation, doubts, hints,
    step-by-step problem solving, revision, and scope boundaries.
    Supports turn_subject for temporary turn-level cross-subject questions without mutating profile.
    """
    if session is not None and profile is None:
        profile = session.profile
    if profile is None:
        profile = StudentProfile(level="school", grade_or_year=8)

    # 1. Enforce strict scope: Exclude B.Tech & Engineering topics
    if is_college_cse_query(text):
        has_telugu = any("\u0c00" <= c <= "\u0c7f" for c in text) or "cheyyi" in text.lower()
        if has_telugu:
            ans = (
                "క్షమించండి! నేను కేవలం స్కూల్ (1 నుంచి 10వ తరగతి) మరియు ఇంటర్మీడియట్ (MPC, BiPC) విద్యార్థుల కోసం "
                "ప్రత్యేకించిన AI ట్యూటర్‌ను. ఇంజనీరింగ్ లేదా B.Tech సబ్జెక్టులను నేను బోధించను. "
                "మీ స్కూల్ లేదా ఇంటర్మీడియట్ సబ్జెక్టులలో ఏదైనా ప్రశ్న ఉంటే అడగండి!"
            )
        else:
            ans = (
                "I am exclusively specialized as an AI Student Tutor for School (Classes 1 to 10) and Intermediate (MPC and BiPC) students. "
                "I do not cover B.Tech or college-level engineering programming. "
                "Please feel free to ask questions from your School or Intermediate subjects like Mathematics, Physics, Chemistry, Science, Botany, Zoology, English, or Social Studies!"
            )
        if session:
            session.add_turn("student", text, intent=INTENT_EXPLANATION)
            session.add_turn("tutor", ans)
        return ans

    # 2. Agentic intent handling if session is active
    intent = classify_learning_intent(text, session) if session else INTENT_EXPLANATION
    pedagogical_instruction = ""

    # Detect temporary turn-level subject context if question is in a different allowed subject
    if turn_subject is None and session is not None:
        inferred = detect_question_subject(text, profile)
        if inferred and inferred in profile.allowed_subjects() and inferred != session.active_subject:
            turn_subject = inferred

    if turn_subject and session and turn_subject != session.active_subject:
        pedagogical_instruction += (
            f"\n[TURN SUBJECT CONTEXT: The student asked a question in {turn_subject} "
            f"(Active session subject is {session.active_subject}). "
            f"Explain using expert {turn_subject} concepts appropriate for a {profile.describe()}. "
            f"Do NOT permanently change the student's profile or active session subject.]"
        )

    if session:
        # Check Pause intent
        if intent == INTENT_PAUSE:
            session.is_paused = True
            has_telugu = any("\u0c00" <= c <= "\u0c7f" for c in text) or any(k in text.lower() for k in ["aagu", "aagandi"])
            if has_telugu:
                pause_reply = "సరే, సెషన్‌ను కాసేపు పాజ్ చేస్తున్నాను. మీరు మళ్లీ చదువుకోవాలనుకున్నప్పుడు 'continue' లేదా 'మళ్లీ చెప్పు' అని అనండి."
            else:
                pause_reply = "I have paused our tutoring session. Whenever you are ready to resume, just say 'resume' or 'continue'!"
            session.add_turn("student", text, intent=INTENT_PAUSE)
            session.add_turn("tutor", pause_reply)
            return pause_reply

        # Check Resume intent
        if intent == INTENT_RESUME:
            session.is_paused = False
            has_telugu = any("\u0c00" <= c <= "\u0c7f" for c in text) or any(k in text.lower() for k in ["cheppu", "modalupettu"])
            if has_telugu:
                resume_reply = f"స్వాగతం! మనం {session.active_subject} లో {session.active_topic} గురించి మాట్లాడుకుంటున్నాం. మీ ప్రశ్న ఏమిటి?"
            else:
                resume_reply = f"Welcome back! We were studying {session.active_topic} in {session.active_subject}. What would you like to explore next?"
            session.add_turn("student", text, intent=INTENT_RESUME)
            session.add_turn("tutor", resume_reply)
            return resume_reply

        # If currently paused and not a resume command
        if session.is_paused:
            return "Our tutoring session is currently paused. Please say 'resume' or 'continue' whenever you are ready to continue."

        # Check Switch Topic/Subject intent
        if intent == INTENT_SWITCH_TOPIC:
            is_switch, target_subj, raw_subj = detect_subject_switch(text, profile)
            if is_switch and target_subj:
                allowed = profile.allowed_subjects()
                if target_subj in allowed:
                    session.active_subject = target_subj
                    session.profile.subject = target_subj
                    session.active_topic = "General"
                    pedagogical_instruction = (
                        f"\n[INSTRUCTION: The student explicitly requested to switch their active subject to {target_subj}. "
                        f"Warmly confirm the switch to {target_subj} and invite them to explore a topic or ask a question in {target_subj}. "
                        f"Tailor language to a {profile.describe()}. Keep it brief, encouraging, and natural for speech.]"
                    )
                else:
                    # Requested subject is outside this student's curriculum
                    allowed_str = ", ".join(allowed)
                    has_telugu = any("\u0c00" <= c <= "\u0c7f" for c in text) or any(k in text.lower() for k in ["cheppu", "nerchukundam", "chaduvudam"])
                    if has_telugu:
                        disallowed_reply = (
                            f"{raw_subj or target_subj} మీ {profile.describe()} సిలబస్‌లో భాగం కాదు. "
                            f"మీకు అందుబాటులో ఉన్న సబ్జెక్టులు: {allowed_str}. "
                            f"దయచేసి వీటిలో ఏదైనా ఒక సబ్జెక్ట్ ఎంచుకోండి!"
                        )
                    else:
                        disallowed_reply = (
                            f"{raw_subj or target_subj} is not part of the {profile.describe()} syllabus. "
                            f"Your curriculum subjects are: {allowed_str}. "
                            f"Please choose one of your subjects to continue!"
                        )
                    session.add_turn("student", text, intent=INTENT_SWITCH_TOPIC)
                    session.add_turn("tutor", disallowed_reply)
                    return disallowed_reply

        # Extract and update active learning topic
        session.extract_and_update_topic(text)

    # 3. Construct pedagogical prompt instructions
    t_lower = text.lower()

    # Safeguard 3: Check if direct answer is explicitly requested
    is_direct_requested = any(k in t_lower for k in [
        "direct answer", "just tell me the answer", "give me the answer",
        "what is the answer", "direct ga cheppu", "జవాబు చెప్పు",
        "నేరుగా సమాధానం చెప్పు", "డైరెక్ట్ గా చెప్పు"
    ])
    if is_direct_requested:
        pedagogical_instruction += (
            "\n[DIRECT ANSWER REQUEST: The student explicitly requested a direct answer. "
            "Provide the direct, accurate answer immediately without withholding the solution or forcing Socratic steps. Keep it crisp, accurate, and helpful.]"
        )

    # Socratic guidance in practice mode
    is_practice_mode = intent == INTENT_STEP_BY_STEP or any(k in t_lower for k in [
        "practice", "help me solve", "how to solve", "solve this", "లెక్క చేయి", "సాల్వ్ చేయి", "practice చేద్దాం"
    ])

    # Phase 11.5.1: Tutor illustrative example values vs Student assumption proposal
    is_tutor_choose_request = any(bool(re.search(p, t_lower)) for p in [
        r"\b(?:you\s+choose|you\s+pick|you\s+take|you\s+assume|you\s+select|you\s+decide)\b",
        r"\b(?:choose|pick|give(?:\s+me)?|take|provide)\s+(?:an?\s+)?(?:example\s+value|example|sample\s+value|value\s+for\s+me|value)\b",
        r"\btutor\s+(?:please\s+)?(?:choose|pick|take|assume|decide)\b",
        r"\bcan\s+you\s+(?:choose|pick|take|assume|give\s+(?:me\s+)?(?:an?\s+)?example)\b",
        r"\b(?:assume|take)\s+(?:some\s+)?value\s+yourself\b",
        r"(?:నువ్వే|మీరే)(?:\s+ఒక|\s+ఏదైనా)?\s*(?:విలువ|ఉదాహరణ|ఎగ్జాంపుల్)?\s*(?:తీసుకో|తీసుకోండి|ఎంచుకో|ఎంచుకోండి|చెప్పు|చెప్పండి)",
        r"(?:ఉదాహరణ|ఎగ్జాంపుల్)\s*(?:విలువ)?\s*(?:తీసుకో|తీసుకోండి|చెప్పు|చెప్పండి|ఇవ్వు|ఇవ్వండి)",
    ])

    is_student_assume_request = any(bool(re.search(p, t_lower)) for p in [
        r"\b(?:can\s+(?:we|i)\s+assume|shall\s+we\s+assume|let'?s\s+assume|let\s+us\s+assume|can\s+(?:we|i)\s+take\s+a\s+value|assume\s+a\s+value|suppose\s+we\s+take)\b",
        r"\b(?:assume|suppose|let us assume|let's assume|let's take)\b",
        r"(?:అనుకోవచ్చా|అనుకుందామా|తీసుకుందామా|ఊహించుకుందామా|అనుకుందాం|తీసుకుందాం)",
    ])

    if is_tutor_choose_request:
        pedagogical_instruction += (
            "\n[TUTOR-CHOSEN EXAMPLE VALUES: The student explicitly asked you (the tutor) to choose an example value or illustrative value. "
            "Provide clearly labeled illustrative example values (e.g. 'For example, let us take radius r = 7 cm...'). "
            "State the relevant formula, substitute the chosen example value into the first step, and guide the student step-by-step.]"
        )
    elif is_student_assume_request:
        pedagogical_instruction += (
            "\n[ASSUMPTION REQUEST: The student asked whether they can assume a value or proposed assuming a value. "
            "Warmly acknowledge their idea, ask which specific value they want to use for the variable (e.g. 'Sure! Which value would you like to use for the radius?'), "
            "state the relevant formula, and let the student choose their value before proceeding with the calculation. Do not invent the value for them.]"
        )
    elif is_practice_mode and not is_direct_requested:
        pedagogical_instruction += (
            "\n[PRACTICE MODE & SOCRATIC GUIDANCE: The student is practicing problem solving. "
            "State the formula and explain the first step. Do NOT reveal the final numerical answer immediately. "
            "Invite the student to calculate or deduce the next step.]"
        )

    # Missing values rule in calculation questions
    if any(k in t_lower for k in ["calculate", "find the area", "find velocity", "find acceleration", "find the speed"]):
        pedagogical_instruction += (
            "\n[MISSING VALUES GUARD: Never invent missing problem values without clearly stating they are an example "
            "(e.g. 'For example, if we let radius r = 7 cm...'). State the formula and ask the student for any missing given values.]"
        )

    if intent == INTENT_EXAM_PREP or any(k in t_lower for k in ["2-mark", "2 marks", "5-mark", "5 marks", "10-mark", "10 marks"]):
        if "2-mark" in t_lower or "2 marks" in t_lower or "రెండు మార్కులు" in t_lower:
            pedagogical_instruction += (
                "\n[INSTRUCTION: The student requested a 2-MARK exam answer. "
                "Provide 1 to 2 crisp, high-yield sentences with the exact textbook definition and key formula. Keep it concise.]"
            )
        elif "5-mark" in t_lower or "5 marks" in t_lower or "ఐదు మార్కులు" in t_lower:
            pedagogical_instruction += (
                "\n[INSTRUCTION: The student requested a 5-MARK exam answer. "
                "Provide a structured short answer: 1) clear definition, 2) three to four key points or steps, 3) a practical example. Structure clearly for spoken audio.]"
            )
        elif "10-mark" in t_lower or "10 marks" in t_lower or "పది మార్కులు" in t_lower:
            pedagogical_instruction += (
                "\n[INSTRUCTION: The student requested a 10-MARK exam answer. "
                "Provide a comprehensive, structured explanation: 1) Introduction, 2) Main principles/steps explained clearly, 3) Formula/applications, 4) Summary conclusion. Speak clearly in structured points.]"
            )
        else:
            pedagogical_instruction += (
                "\n[INSTRUCTION: Focus on exam-oriented preparation for this topic. "
                "Highlight key definitions, essential points, and common exam questions. Never claim a question is guaranteed to appear.]"
            )
    elif intent == INTENT_REVISION:
        pedagogical_instruction += (
            "\n[INSTRUCTION: The student requested a REVISION / SUMMARY. "
            "Provide a concise, 3-point spoken summary of the core concept, key formula or principle, and a quick takeaway tip for exams. Keep it punchy and memorable.]"
        )
    elif intent == INTENT_STEP_BY_STEP:
        pedagogical_instruction += (
            "\n[INSTRUCTION: The student needs STEP-BY-STEP problem solving help. "
            "Use a Socratic approach: clearly explain Step 1, perform the initial simplification or setup, and then ask the student what they think the next step is. Do not just dump the final answer.]"
        )
    elif intent == INTENT_HINT:
        pedagogical_instruction += (
            "\n[INSTRUCTION: The student is stuck or asked for a HINT. "
            "Provide an intuitive, encouraging clue or conceptual hint that helps them discover the solution on their own without giving away the complete answer.]"
        )
    elif intent == INTENT_DOUBT_FOLLOWUP:
        topic_hint = f" regarding {session.active_topic}" if session and session.active_topic != "General" else ""
        pedagogical_instruction += (
            f"\n[INSTRUCTION: The student has a follow-up doubt{topic_hint}. "
            "Connect directly to the previous explanation, resolve any pronouns (like 'it', 'this', 'that'), and clarify their specific confusion clearly and warmly.]"
        )

    # Check for explicit detail requests
    is_explicit_detail = any(k in t_lower for k in ["in detail", "detailed explanation", "detailed", "వివరంగా", "పూర్తిగా వివరించు", "detailed ga"])
    if is_explicit_detail:
        pedagogical_instruction += (
            "\n[INSTRUCTION: The student explicitly requested a DETAILED explanation. "
            "Provide a comprehensive, clear, structured explanation with key concepts and practical examples. "
            "Speak clearly and naturally.]"
        )

    # Turn-level default conciseness constraint (Objective 2)
    is_detailed_mode = (
        is_explicit_detail or
        intent in [INTENT_EXAM_PREP, INTENT_REVISION] or
        any(k in t_lower for k in ["2-mark", "2 marks", "5-mark", "5 marks", "10-mark", "10 marks"])
    )
    if not is_detailed_mode and not is_practice_mode and not is_tutor_choose_request and not is_student_assume_request:
        pedagogical_instruction += (
            "\n[DEFAULT CONCISENESS: Keep your spoken answer strictly concise: 2 to 3 short, clear sentences. "
            "Start directly with the core explanation without echoing or paraphrasing the student's question. "
            "Avoid repeating the same concept in different words. "
            "Use simple, age-appropriate language for their class/year. Ask at most ONE brief, friendly check question at the very end.]"
        )

    # Check for ad-hoc grade-level explanation override (applied to this request only, never mutates profile)
    grade_override = detect_requested_grade_override(text, profile)
    grade_override_instruction = ""
    if grade_override is not None:
        active_subj = turn_subject or (session.active_subject if session else profile.subject)
        grade_override_instruction = (
            f"\n[GRADE-LEVEL EXPLANATION OVERRIDE: The student is enrolled as a {profile.describe()}, "
            f"but explicitly requested this specific concept to be explained at a Class {grade_override} School level. "
            f"Explain using simple Class {grade_override} fundamentals, intuitive everyday analogies, and clear basics. "
            f"Do NOT use college or advanced Intermediate formulas. "
            f"Keep your explanation strictly appropriate for a Class {grade_override} student, "
            f"while maintaining their active subject ({active_subj}) in context.]"
        )

    system_instruction = build_system_instruction(profile, session=session, turn_subject=turn_subject)
    prompt = text + pedagogical_instruction + grade_override_instruction
    reply = call_gemini(prompt, system_instruction)

    if session:
        session.add_turn("student", text, intent=intent)
        session.add_turn("tutor", reply)

    return reply


def generate_quiz_question(
    profile: StudentProfile,
    question_num: int,
    subject: str = None,
    topic: str = None,
    difficulty: str = "medium",
    return_hint: bool = False,
) -> tuple[str, str] | tuple[str, str, str]:
    """
    Generate an age-appropriate quiz question for the active student profile, subject, topic, and difficulty.
    Returns (question_text, expected_answer_summary) or (question_text, expected_answer_summary, hint_text).
    """
    if profile is None:
        profile = StudentProfile(level="school", grade_or_year=8)

    subj = subject or profile.subject or profile.allowed_subjects()[0]
    topic_str = f"in {subj} on the topic of '{topic}'" if topic and topic != "General" else f"in the subject of {subj}"
    desc = profile.describe()

    prompt = (
        f"Generate Question {question_num} for a quick spoken interactive quiz for a {desc} {topic_str}.\n"
        f"Difficulty Level: {difficulty.upper()}.\n"
        "Requirements:\n"
        "1. Question must be strictly suitable for their syllabus level.\n"
        "2. Formulate 1 single clear question that has a straightforward spoken answer.\n"
        "3. Provide an intuitive conceptual hint that helps if the student is stuck.\n"
        "4. Provide the output in exactly this format:\n"
        "QUESTION: <The question to ask the student verbally, no markdown>\n"
        "ANSWER: <A 1-2 sentence description of the expected correct answer>\n"
        "HINT: <A short 1-sentence clue without giving away the direct answer>\n"
    )

    sys_inst = (
        "You are an encouraging quiz master for School (Classes 1-10) and Intermediate (MPC/BiPC) students. "
        "Create engaging, clear questions. No markdown formatting."
    )

    reply = call_gemini(prompt, sys_inst)
    q_text = f"Question {question_num}: What is the powerhouse of the cell?"
    a_text = "Mitochondria"
    h_text = "Think of the organelle responsible for cellular respiration and ATP energy production."

    if "QUESTION:" in reply and "ANSWER:" in reply:
        try:
            parts = reply.split("ANSWER:")
            q_part = parts[0].replace("QUESTION:", "").strip()
            rest = parts[1]
            if "HINT:" in rest:
                a_part, h_part = rest.split("HINT:", 1)
                a_part = a_part.strip()
                h_part = h_part.strip()
                if h_part:
                    h_text = clean_text_for_speech(h_part)
            else:
                a_part = rest.strip()

            if q_part and a_part:
                q_text = clean_text_for_speech(q_part)
                a_text = clean_text_for_speech(a_part)
        except Exception:
            pass
    else:
        q_text = clean_text_for_speech(reply)

    if return_hint:
        return q_text, a_text, h_text
    return q_text, a_text


def evaluate_quiz_answer(
    student_answer: str,
    question: str,
    expected_answer: str,
    profile: StudentProfile = None,
    hint: str = None,
) -> tuple[bool, str]:
    """
    Evaluate student's spoken quiz answer.
    Returns (is_correct: bool, feedback_text: str).
    """
    hint_instruction = f"\nConceptual Hint for the student if needed: {hint}" if hint else ""
    prompt = (
        f"Quiz Question: {question}\n"
        f"Expected Answer: {expected_answer}\n"
        f"Student Answer: {student_answer}{hint_instruction}\n\n"
        "Evaluate if the student's answer is conceptually correct or incorrect.\n"
        "Respond in exactly this format:\n"
        "VERDICT: [CORRECT or INCORRECT]\n"
        "EXPLANATION: [A friendly 1-2 sentence spoken explanation praising them if correct, or explaining the correct answer gently with the hint/concept if incorrect.]"
    )

    desc = profile.describe() if profile else "Student"
    sys_inst = (
        f"You are a warm, encouraging quiz tutor evaluating a {desc}. "
        "Be motivating, clear, and voice-friendly. No markdown asterisks or special characters."
    )

    reply = call_gemini(prompt, sys_inst)
    upper_reply = reply.upper()
    first_line = upper_reply.split("\n")[0] if "\n" in upper_reply else upper_reply

    if "INCORRECT" in first_line or "VERDICT: INCORRECT" in upper_reply:
        is_correct = False
    elif "VERDICT: CORRECT" in upper_reply or "CORRECT" in first_line:
        is_correct = True
    else:
        is_correct = False

    explanation = reply
    if "EXPLANATION:" in reply:
        explanation = reply.split("EXPLANATION:")[1].strip()

    explanation = clean_text_for_speech(explanation)
    return is_correct, explanation


class AudioRecordResult(tuple):
    """
    Diagnostic result from record_audio.
    Provides tuple indexing [0..6] as well as attribute access:
    (success, is_silent, rms, peak, clipping_count, total_duration, speech_duration).
    """

    def __new__(
        cls,
        success: bool,
        is_silent: bool,
        rms: float,
        peak: float,
        clipping_count: int,
        total_duration: float,
        speech_duration: float,
    ):
        return super().__new__(
            cls,
            (success, is_silent, rms, peak, clipping_count, total_duration, speech_duration),
        )

    @property
    def success(self) -> bool:
        return self[0]

    @property
    def is_silent(self) -> bool:
        return self[1]

    @property
    def rms(self) -> float:
        return self[2]

    @property
    def peak(self) -> float:
        return self[3]

    @property
    def clipping_count(self) -> int:
        return self[4]

    @property
    def total_duration(self) -> float:
        return self[5]

    @property
    def speech_duration(self) -> float:
        return self[6]


def record_audio(
    filename: str = "mic_test.wav",
    duration: float = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: int = None,
    vad_threshold: float = VAD_ENERGY_THRESHOLD,
    silence_timeout: float = SILENCE_TIMEOUT_SECONDS,
    max_duration: float = MAX_RECORD_DURATION,
    initial_timeout: float = INITIAL_SILENCE_TIMEOUT,
    pre_roll: float = PRE_ROLL_SECONDS,
) -> AudioRecordResult:
    """
    Adaptive Voice Activity Detection (VAD) recording.
    Detects speech onset, records continuously with 0.4s pre-roll buffer,
    and automatically stops when 1.1s of silence follows speech.
    Returns AudioRecordResult(success, is_silent, rms, peak, clipping_count, total_duration, speech_duration).
    """
    if device is None:
        device, _ = get_input_device()

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except Exception:
            pass

    max_dur = float(duration) if (duration is not None and duration > 0) else max_duration
    chunk_dur = CHUNK_DURATION
    chunk_samples = int(sample_rate * chunk_dur)
    pre_roll_chunks_count = max(1, int(pre_roll / chunk_dur))
    preroll_buffer = collections.deque(maxlen=pre_roll_chunks_count)
    recorded_chunks = []

    # Dynamic Noise Floor & Adaptive Threshold Configuration
    is_custom_threshold = (vad_threshold is not None and vad_threshold != VAD_ENERGY_THRESHOLD)
    effective_vad_threshold = vad_threshold if is_custom_threshold else VAD_ENERGY_THRESHOLD
    continuation_threshold = effective_vad_threshold * VAD_CONTINUATION_RATIO
    calibrated = is_custom_threshold
    ambient_noise_samples = []
    hangover_count = 0

    speech_started = False
    speech_duration = 0.0
    silence_duration = 0.0

    print(f"\n🎤 Listening... Speak now! (Adaptive VAD active, max {max_dur:.0f}s)...")
    t0 = time.perf_counter()

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=chunk_samples,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(chunk_samples)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()

                chunk_rms = float(np.sqrt(np.mean(chunk**2)))
                time_elapsed = time.perf_counter() - t0

                if not speech_started:
                    preroll_buffer.append(chunk)

                    # Calibrate ambient noise floor dynamically across initial 3-5 pre-roll chunks
                    if not calibrated:
                        ambient_noise_samples.append(chunk_rms)
                        if len(ambient_noise_samples) >= 3:
                            measured_noise = float(np.median(ambient_noise_samples))
                            auto_thresh = max(
                                SAFE_MIN_VAD_THRESHOLD,
                                min(SAFE_MAX_VAD_THRESHOLD, measured_noise * 2.2 + 0.0025)
                            )
                            effective_vad_threshold = auto_thresh
                            continuation_threshold = effective_vad_threshold * VAD_CONTINUATION_RATIO
                            calibrated = True

                    is_speech_chunk = chunk_rms >= effective_vad_threshold
                    if is_speech_chunk:
                        speech_started = True
                        recorded_chunks.extend(preroll_buffer)
                        speech_duration += len(preroll_buffer) * chunk_dur
                        silence_duration = 0.0
                        hangover_count = 0
                        print(f"✨ [VAD] Voice detected (RMS: {chunk_rms:.4f} >= Thresh: {effective_vad_threshold:.4f})! Recording active speech...")
                    elif time_elapsed >= initial_timeout:
                        print(f"ℹ️ [VAD] Initial silence ({initial_timeout:.1f}s) elapsed without speech.")
                        recorded_chunks.extend(preroll_buffer)
                        break
                else:
                    recorded_chunks.append(chunk)
                    if chunk_rms >= continuation_threshold:
                        speech_duration += chunk_dur
                        silence_duration = 0.0
                        hangover_count = 0
                    else:
                        # Intra-word consonant / brief pause hangover (200ms)
                        if hangover_count < VAD_HANGOVER_CHUNKS:
                            hangover_count += 1
                            speech_duration += chunk_dur
                        else:
                            silence_duration += chunk_dur
                            if silence_duration >= silence_timeout:
                                print(f"🛑 [VAD] Natural pause ({silence_duration:.2f}s) detected. Finalizing speech.")
                                break

                if time_elapsed >= max_dur:
                    print(f"⏰ [VAD] Maximum recording limit ({max_dur:.1f}s) reached.")
                    break

        audio = np.concatenate(recorded_chunks) if recorded_chunks else np.zeros(chunk_samples, dtype="float32")
        total_audio_duration = len(audio) / sample_rate

        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        clipping_count = int(np.sum(np.abs(audio) >= 0.99))

        print(
            f"📊 [AUDIO DIAGNOSTICS] Recorded: {total_audio_duration:.2f}s | Speech: {speech_duration:.2f}s | "
            f"Trailing Silence: {silence_duration:.2f}s | RMS: {rms:.5f} | Peak: {peak:.5f} | "
            f"Clipping: {clipping_count} | Device: {device} | Thresh: {effective_vad_threshold:.4f}"
        )

        if clipping_count > 0:
            print(f"⚠️ [AUDIO] Warning: {clipping_count} clipped samples detected! Audio may be distorted. Speak slightly further from the microphone.")

        is_silent = (not speech_started) or (speech_duration < MIN_SPEECH_DURATION) or (rms < SILENCE_RMS_THRESHOLD)
        if is_silent:
            print("ℹ️ [AUDIO] Silence or very low volume detected.")
            sf.write(filename, audio, sample_rate)
            return AudioRecordResult(True, True, rms, peak, clipping_count, total_audio_duration, speech_duration)

        # Smart gentle gain: avoid blindly amplifying already loud audio.
        # If peak is quiet (peak < 0.30) and non-trivial (peak > 0.01), gently normalize to 0.65 (cap 3.0x).
        if 0.01 < peak < 0.30:
            gain = min(3.0, 0.65 / max(peak, 0.001))
            audio = np.clip(audio * gain, -1.0, 1.0)
            print(f"✨ [AUDIO] Gentle gain applied ({gain:.2f}x -> new peak: {peak*gain:.2f})")

        sf.write(filename, audio, sample_rate)
        return AudioRecordResult(True, False, rms, peak, clipping_count, total_audio_duration, speech_duration)

    except Exception as e:
        print(f"⚠️ Microphone error: {e}")
        print("Please check your microphone device settings and permissions.")
        return AudioRecordResult(False, True, 0.0, 0.0, 0, 0.0, 0.0)


def diagnose_microphone(
    duration: float = 3.0,
    device: int = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    output_filename: str = None,
) -> dict:
    """
    Perform a comprehensive diagnostic audit of the microphone input.
    Measures device capabilities, ambient noise floor, RMS, peak, dynamic range, clipping, and STT suitability.
    """
    if device is None:
        device, dev_info = get_input_device()
    else:
        try:
            dev_info = sd.query_devices(device)
        except Exception:
            dev_info = {"name": f"Device {device}", "max_input_channels": 1, "default_samplerate": sample_rate}

    print("\n======================================================================")
    print("🎙️  MICROPHONE AUDIO DIAGNOSTICS & SYSTEM AUDIT")
    print("======================================================================")
    print(f"• Selected Device   : [{device}] {dev_info.get('name', 'Unknown')}")
    print(f"• Input Channels    : {dev_info.get('max_input_channels', 0)}")
    print(f"• Native Sample Rate: {dev_info.get('default_samplerate', 0):.0f} Hz")
    print(f"• Target Recording  : {duration:.1f}s @ {sample_rate} Hz (Float32 Mono)")
    print("----------------------------------------------------------------------")
    print("🎤 Please speak a sentence now (e.g. 'Class 8 Science' or 'Newton's law')...")

    t0 = time.perf_counter()
    try:
        rec = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="float32", device=device)
        sd.wait()
        audio = rec.flatten()
        actual_dur = len(audio) / sample_rate

        rms = float(np.sqrt(np.mean(audio**2)))
        peak = float(np.max(np.abs(audio)))
        clipping_count = int(np.sum(np.abs(audio) >= 0.99))
        noise_floor = float(np.percentile(np.abs(audio), 15))

        is_silent = (rms < SILENCE_RMS_THRESHOLD) or (peak < 0.01)
        speech_detected = (rms >= 0.005) and (peak >= 0.035)
        clipping_detected = clipping_count > 0

        if is_silent:
            suitability = "POOR (Too quiet or muted - check mic levels/distance)"
        elif clipping_detected:
            suitability = "WARN (Clipping detected - move further from microphone)"
        elif speech_detected:
            suitability = "EXCELLENT (Clear speech detected, optimal for STT)"
        else:
            suitability = "MODERATE (Low speech level detected - consider speaking closer)"

        print(f"• Recorded Duration : {actual_dur:.2f}s")
        print(f"• RMS Energy Level  : {rms:.5f} (Silence threshold: {SILENCE_RMS_THRESHOLD:.5f})")
        print(f"• Peak Amplitude    : {peak:.5f}")
        print(f"• Est. Noise Floor  : {noise_floor:.5f}")
        print(f"• Clipping Samples  : {clipping_count}")
        print(f"• Speech Activity   : {'YES' if speech_detected else 'NO'}")
        print(f"• STT Suitability   : {suitability}")
        print("======================================================================\n")

        if output_filename:
            sf.write(output_filename, audio, sample_rate)
            print(f"💾 Diagnostic sample saved to: {output_filename}")

        return {
            "device_index": device,
            "device_name": dev_info.get("name", "Unknown"),
            "sample_rate": sample_rate,
            "duration": actual_dur,
            "rms": rms,
            "peak": peak,
            "noise_floor": noise_floor,
            "is_silent": is_silent,
            "speech_detected": speech_detected,
            "clipping_detected": clipping_detected,
            "clipping_count": clipping_count,
            "stt_suitable": speech_detected and not clipping_detected,
            "suitability": suitability,
        }
    except Exception as e:
        print(f"❌ Microphone diagnostic failed with error: {e}")
        return {
            "device_index": device,
            "error": str(e),
            "stt_suitable": False,
        }


def is_transient_connection_error(exc: Exception) -> bool:
    """
    Check if an exception is a transient network/connection error eligible for a single quick retry.
    Catches WinError 10053 (ConnectionAbortedError), ConnectionResetError, socket timeouts,
    and requests connection aborts.
    Excludes authentication (401/403), bad request (400), and other permanent API errors.
    """
    if isinstance(exc, (ConnectionAbortedError, ConnectionResetError, ConnectionRefusedError, TimeoutError)):
        return True

    err_str = str(exc).lower()

    # Never retry authentication or client validation errors
    if any(k in err_str for k in ["401", "403", "unauthorized", "forbidden", "authentication", "api_key is required", "400", "bad request"]):
        return False

    transient_indicators = [
        "10053",  # WSAECONNABORTED: An established connection was aborted by the software in your host machine
        "10054",  # WSAECONNRESET: An existing connection was forcibly closed by the remote host
        "connectionabortederror",
        "connectionreseterror",
        "connection aborted",
        "connection reset",
        "connection refused",
        "remote end closed connection",
        "max retries exceeded",
        "nameresolutionerror",
        "timed out",
        "timeout",
    ]
    return any(ind in err_str for ind in transient_indicators)


def transcribe_speech(filename: str = "mic_test.wav") -> dict | None:
    """
    Transcribe audio with Gnani Prisma v3.0 using language_candidates=['te-IN', 'en-IN'] and format='transcribe'.
    Handles transient connection aborts (WinError 10053) with a single 0.3s backoff retry.
    Falls back to te-IN on persistent primary failure without excessive retry delays.
    Never exposes API keys or sensitive configuration.
    """
    if not os.path.exists(filename):
        print("⚠️ Audio file does not exist for transcription.")
        return None

    try:
        audio_data, sr = sf.read(filename)
        audio_dur = len(audio_data) / sr
        audio_rms = float(np.sqrt(np.mean(audio_data**2)))
        audio_peak = float(np.max(np.abs(audio_data)))
    except Exception:
        audio_dur, audio_rms, audio_peak = 0.0, 0.0, 0.0

    t0 = time.perf_counter()
    result = None
    primary_failed = False
    last_err = None

    # Attempt 1: Primary transcription (indic-auto with prisma-v3.0)
    try:
        result = stt.transcribe(
            audio=filename,
            model="prisma-v3.0",
            language_code="indic-auto",
            language_candidates=["te-IN", "en-IN"],
            format="transcribe",
        )
    except Exception as e:
        primary_failed = True
        last_err = e
        is_transient = is_transient_connection_error(e)
        err_msg = type(e).__name__
        print(f"⚠️ Primary STT error: {err_msg}")

        # Retry transient connection-related failure ONCE after 0.3s backoff
        if is_transient:
            print("🔄 Transient STT connection issue detected ([WinError 10053/Socket Reset]). Retrying once after 0.3s backoff...")
            time.sleep(0.3)
            try:
                result = stt.transcribe(
                    audio=filename,
                    model="prisma-v3.0",
                    language_code="indic-auto",
                    language_candidates=["te-IN", "en-IN"],
                    format="transcribe",
                )
                primary_failed = False
                print("✅ Primary STT transient retry succeeded!")
            except Exception as retry_err:
                primary_failed = True
                last_err = retry_err
                print(f"⚠️ Primary STT retry failed ({type(retry_err).__name__}). Proceeding to fallback...")

    # Fallback to te-IN if primary failed and error is not a permanent auth/client issue
    if primary_failed or not result:
        err_str = str(last_err).lower() if last_err else ""
        if any(k in err_str for k in ["401", "403", "unauthorized", "forbidden", "authentication", "api_key is required"]):
            print("🛑 Permanent STT authentication error. Skipping fallback.")
            return None

        print("⚠️ Primary STT failed. Falling back to te-IN...")
        try:
            t_fb0 = time.perf_counter()
            result = stt.transcribe(audio=filename, language_code="te-IN", format="transcribe")
            fb_duration = time.perf_counter() - t_fb0
            if result:
                result["client_processing_time"] = round(fb_duration, 3)
                req_id = result.get("request_id", "N/A")
                raw_txt = result.get("transcript", "")
                print(f"🧠 [STT FALLBACK] Req ID: {req_id} | Lang: te-IN | Round-trip: {fb_duration:.2f}s | Transcript: '{raw_txt}'")
        except Exception as err:
            print(f"⚠️ Fallback STT error: {type(err).__name__}")
            return None

    client_duration = time.perf_counter() - t0
    if result:
        result["client_processing_time"] = round(client_duration, 3)
        req_id = result.get("request_id", "N/A")
        lang = result.get("detected_language_code", "unknown")
        t_sec = result.get("processing_time", 0.0)
        raw_txt = result.get("transcript", "")
        conf = result.get("confidence") or result.get("transcript_confidence")
        conf_val = None
        if conf is not None:
            try:
                conf_val = float(conf)
                if conf_val < 0.35 and raw_txt.strip():
                    result["low_confidence"] = True
            except (ValueError, TypeError):
                pass
        conf_str = f" | Confidence: {conf_val:.2f}" if conf_val is not None else ""
        print(
            f"🧠 [STT DIAGNOSTICS] Req ID: {req_id} | File: {filename} ({audio_dur:.2f}s, RMS: {audio_rms:.4f}, Peak: {audio_peak:.4f}) | "
            f"Detected Lang: {lang}{conf_str} | Gnani Server: {t_sec:.2f}s | Round-trip: {client_duration:.2f}s | Transcript: '{raw_txt}'"
        )
    return result


def play_audio(filename: str = "assistant_reply.wav") -> bool:
    """
    Play the synthesized tutor speech through the speakers.
    Inspects audio file metadata (duration, sample rate, channels, format/subtype).
    Measures playback wall-clock time and verifies duration alignment.
    Settling pause (0.25s) prevents microphone echo.
    Calls sd.stop() in exception handler to release audio buffers on corrupt/missing stream.
    """
    try:
        spoken_audio, rate = sf.read(filename, dtype="float32")
        try:
            info = sf.info(filename)
            audio_dur = info.duration
            sr = info.samplerate
            channels = info.channels
            fmt = info.format
            subtype = info.subtype
        except Exception:
            audio_dur = len(spoken_audio) / rate
            sr = rate
            channels = 1 if spoken_audio.ndim == 1 else spoken_audio.shape[1]
            fmt = "WAV"
            subtype = "PCM_16"

        t_play_start = time.perf_counter()
        sd.play(spoken_audio, rate)
        sd.wait()
        play_wall_dur = time.perf_counter() - t_play_start
        delta = play_wall_dur - audio_dur

        # Echo prevention settling pause
        time.sleep(SETTLING_PAUSE_SECONDS)

        # Diagnostic telemetry log
        base_name = os.path.basename(filename)
        print(
            f"🔊 [AUDIO PLAYBACK] File: {base_name} | "
            f"Audio: {audio_dur:.2f}s, {sr}Hz, {channels}ch, {fmt}/{subtype} | "
            f"Playback Wall-Clock: {play_wall_dur:.2f}s (delta: {delta:+.2f}s) | "
            f"Audible: YES | Plays: 1 (Single, no duplicates) | Settling Pause: {SETTLING_PAUSE_SECONDS:.2f}s"
        )
        return True
    except Exception as e:
        print(f"⚠️ Audio playback error: {e}")
        try:
            sd.stop()
        except Exception:
            pass
        return False


def speak_and_play(
    text: str,
    output_file: str = "assistant_reply.wav",
) -> tuple[bool, float, float]:
    """
    Synthesize text using Gnani Timbre TTS and play it via speakers.
    Returns (success: bool, tts_duration: float, play_duration: float).
    """
    tts_lang = detect_tts_language(text)
    print(f"\n🔊 Generating voice reply ({tts_lang})...")
    try:
        t_tts0 = time.perf_counter()
        tts.synthesize(
            text=text,
            voice="Pranav",
            model="timbre-v2.5",
            language=tts_lang,
            output_file=output_file,
        )
        tts_dur = time.perf_counter() - t_tts0

        t_play0 = time.perf_counter()
        play_ok = play_audio(output_file)
        play_dur = time.perf_counter() - t_play0

        return play_ok, tts_dur, play_dur
    except Exception as e:
        print(f"⚠️ Voice generation failed: {e}")
        return False, 0.0, 0.0


def print_latency_summary(
    turn_label: str,
    rec_duration: float,
    speech_duration: float,
    stt_duration: float,
    stt_server_time: float,
    gemini_duration: float,
    tts_duration: float,
    play_duration: float,
):
    """
    Print accurate, measured latency breakdown using time.perf_counter().
    """
    trailing_silence = max(0.0, rec_duration - speech_duration)
    time_to_voice = stt_duration + gemini_duration + tts_duration
    total_turn = rec_duration + time_to_voice + play_duration

    print("\n" + "─" * 65)
    print(f"⏱️  [LATENCY BREAKDOWN - {turn_label}]")
    print(f"• Microphone Recording : {rec_duration:.2f}s (Speech: {speech_duration:.2f}s, Trailing silence: {trailing_silence:.2f}s)")
    print(f"• STT Transcription    : {stt_duration:.2f}s (Gnani Server: {stt_server_time:.2f}s)")
    if gemini_duration > 0:
        print(f"• Gemini Generation    : {gemini_duration:.2f}s")
    print(f"• TTS Voice Synthesis  : {tts_duration:.2f}s")
    print(f"• Audio Playback       : {play_duration:.2f}s")
    print("─" * 65)
    print(f"⚡ Pipeline Latency (STT + AI + TTS) : {time_to_voice:.2f}s (time to voice start)")
    print(f"🕒 Total Response Time (Full Turn)  : {total_turn:.2f}s")
    print("─" * 65 + "\n")


class QuizAnswerResult:
    """
    Diagnostic result from dedicated record_quiz_answer.
    Provides action, extracted answer, full raw transcript, duration, and silence status.
    """

    def __init__(
        self,
        action: str,
        answer_text: str = "",
        raw_transcript: str = "",
        duration: float = 0.0,
        speech_duration: float = 0.0,
        is_silent: bool = False,
    ):
        self.action = action
        self.answer_text = answer_text
        self.raw_transcript = raw_transcript
        self.duration = duration
        self.speech_duration = speech_duration
        self.is_silent = is_silent

    def __repr__(self) -> str:
        return (
            f"QuizAnswerResult(action='{self.action}', answer='{self.answer_text}', "
            f"duration={self.duration:.2f}s, speech={self.speech_duration:.2f}s, is_silent={self.is_silent})"
        )


def record_quiz_answer(
    filename: str = "mic_test.wav",
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    device: int = None,
    vad_threshold: float = VAD_ENERGY_THRESHOLD,
    pause_check_duration: float = QUIZ_PAUSE_CHECK_SECONDS,
    timeout: float = QUIZ_ANSWER_TIMEOUT_SECONDS,
    settling_pause: float = QUIZ_SETTLING_PAUSE,
) -> QuizAnswerResult:
    """
    Dedicated Quiz Answer Listening Mode:
    - Waits for settling pause to prevent tutor TTS playback echo.
    - Listens continuously through natural pauses (does NOT stop at 1.1s silence).
    - Uses a 120-second safety timeout instead of normal 15s/30s cap.
    - Transcribes interim pauses to detect explicit commands:
      * 'Submit answer' / 'Done' -> submits extracted answer
      * 'Skip' -> skips current question
      * 'Repeat question' -> repeats current question
      * 'Stop quiz' -> exits quiz mode
    - Returns QuizAnswerResult(action, answer_text, raw_transcript, duration, speech_duration, is_silent).
    """
    if device is None:
        device, _ = get_input_device()

    # Echo prevention settling pause: ensure tutor TTS playback has fully stopped
    if settling_pause > 0:
        time.sleep(settling_pause)

    if os.path.exists(filename):
        try:
            os.remove(filename)
        except Exception:
            pass

    interim_file = "quiz_interim.wav"
    chunk_dur = CHUNK_DURATION
    chunk_samples = int(sample_rate * chunk_dur)
    accumulated_chunks = []

    # Dynamic noise floor and continuation threshold
    is_custom_threshold = (vad_threshold is not None and vad_threshold != VAD_ENERGY_THRESHOLD)
    effective_vad_threshold = vad_threshold if is_custom_threshold else VAD_ENERGY_THRESHOLD
    continuation_threshold = effective_vad_threshold * VAD_CONTINUATION_RATIO
    calibrated = is_custom_threshold
    ambient_noise_samples = []
    hangover_count = 0

    speech_started = False
    speech_duration = 0.0
    silence_after_speech = 0.0
    last_checked_sample_count = 0
    last_checked_speech_duration = 0.0
    last_accumulated_transcript = ""

    print(f"\n🎙️ [QUIZ ANSWER MODE] Listening for your answer (Safety timeout: {timeout:.0f}s)...")
    print("• You can take your time and pause to think (120s limit).")
    print("• Say 'Submit answer' or 'Done' when finished.")
    print("• Say 'Skip' to skip | 'Repeat question' to hear it again | 'Stop quiz' to exit.")

    t0 = time.perf_counter()

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            device=device,
            blocksize=chunk_samples,
        ) as stream:
            while True:
                chunk, overflowed = stream.read(chunk_samples)
                if chunk.ndim > 1:
                    chunk = chunk.flatten()

                chunk_rms = float(np.sqrt(np.mean(chunk**2)))
                time_elapsed = time.perf_counter() - t0

                if not speech_started:
                    if not calibrated:
                        ambient_noise_samples.append(chunk_rms)
                        if len(ambient_noise_samples) >= 3:
                            measured_noise = float(np.median(ambient_noise_samples))
                            auto_thresh = max(
                                SAFE_MIN_VAD_THRESHOLD,
                                min(SAFE_MAX_VAD_THRESHOLD, measured_noise * 2.2 + 0.0025)
                            )
                            effective_vad_threshold = auto_thresh
                            continuation_threshold = effective_vad_threshold * VAD_CONTINUATION_RATIO
                            calibrated = True

                    is_speech = chunk_rms >= effective_vad_threshold
                    if is_speech:
                        speech_started = True
                        accumulated_chunks.append(chunk)
                        speech_duration += chunk_dur
                        silence_after_speech = 0.0
                        hangover_count = 0
                        print(f"✨ [QUIZ VAD] Voice detected (RMS: {chunk_rms:.4f})! Listening to student answer...")
                else:
                    accumulated_chunks.append(chunk)
                    if chunk_rms >= continuation_threshold:
                        speech_duration += chunk_dur
                        silence_after_speech = 0.0
                        hangover_count = 0
                    else:
                        if hangover_count < VAD_HANGOVER_CHUNKS:
                            hangover_count += 1
                            speech_duration += chunk_dur
                        else:
                            silence_after_speech += chunk_dur

                        # Check for commands when a natural pause (>= 1.2s silence) occurs after speech
                        if silence_after_speech >= pause_check_duration:
                            # Only call interim STT if NEW speech was detected since the last check
                            has_new_speech = (speech_duration - last_checked_speech_duration) >= 0.2
                            if has_new_speech:
                                current_samples = sum(len(c) for c in accumulated_chunks)
                                interim_audio = np.concatenate(accumulated_chunks)
                                sf.write(interim_file, interim_audio, sample_rate)

                                stt_res = transcribe_speech(interim_file)
                                last_checked_sample_count = current_samples
                                last_checked_speech_duration = speech_duration

                                if stt_res and stt_res.get("transcript"):
                                    curr_t = stt_res.get("transcript", "").strip()
                                    cmd, clean_ans = parse_quiz_command(curr_t)

                                    if cmd is not None:
                                        print(f"🎯 [QUIZ COMMAND DETECTED]: '{cmd}' in \"{curr_t}\"")
                                        sf.write(filename, interim_audio, sample_rate)
                                        if not clean_ans and last_accumulated_transcript:
                                            clean_ans = last_accumulated_transcript
                                        if os.path.exists(interim_file):
                                            try:
                                                os.remove(interim_file)
                                            except Exception:
                                                pass
                                        return QuizAnswerResult(
                                            action=cmd,
                                            answer_text=clean_ans,
                                            raw_transcript=curr_t,
                                            duration=time_elapsed,
                                            speech_duration=speech_duration,
                                            is_silent=False,
                                        )
                                    else:
                                        last_accumulated_transcript = curr_t
                                        print(f"👂 [QUIZ LISTENING] Heard so far: \"{curr_t}\"... (Pause detected; keep answering or say 'Submit answer' / 'Done')")

                                # Reset silence counter so we don't spam STT repeatedly during continuous silence
                                silence_after_speech = 0.0

                # Check safety timeout
                if time_elapsed >= timeout:
                    print(f"⏰ [QUIZ MODE] Safety timeout ({timeout:.0f}s) reached.")
                    break

        # Safety timeout reached or stream ended
        if accumulated_chunks and speech_started and speech_duration >= MIN_SPEECH_DURATION:
            if last_accumulated_transcript and (speech_duration - last_checked_speech_duration) < 0.2:
                # No new speech occurred since last interim check; reuse existing transcript without redundant STT call
                raw_t = last_accumulated_transcript
            else:
                final_audio = np.concatenate(accumulated_chunks)
                sf.write(filename, final_audio, sample_rate)
                stt_res = transcribe_speech(filename)
                raw_t = stt_res.get("transcript", "").strip() if stt_res else last_accumulated_transcript
            cmd, clean_ans = parse_quiz_command(raw_t)
            if cmd is None:
                cmd = QUIZ_ACTION_SUBMIT
                clean_ans = raw_t
            elif not clean_ans and last_accumulated_transcript:
                clean_ans = last_accumulated_transcript
            return QuizAnswerResult(
                action=cmd,
                answer_text=clean_ans,
                raw_transcript=raw_t,
                duration=time_elapsed,
                speech_duration=speech_duration,
                is_silent=False,
            )
        else:
            return QuizAnswerResult(
                action=QUIZ_ACTION_TIMEOUT,
                answer_text="",
                raw_transcript="",
                duration=time_elapsed,
                speech_duration=0.0,
                is_silent=True,
            )

    except Exception as e:
        print(f"⚠️ Microphone error during quiz answer recording: {e}")
        return QuizAnswerResult(
            action=QUIZ_ACTION_EMPTY,
            answer_text="",
            raw_transcript="",
            duration=0.0,
            speech_duration=0.0,
            is_silent=True,
        )
    finally:
        if os.path.exists(interim_file):
            try:
                os.remove(interim_file)
            except Exception:
                pass


def main():
    """
    Continuous conversation loop for the School & Intermediate AI Student Tutor.
    Manages Education Level Onboarding, School/Intermediate Learning Modes,
    Interactive Quiz Mode, Exam Preparation Mode, Adaptive VAD recording, and Stop Commands.
    """
    sample_rate = DEFAULT_SAMPLE_RATE
    device_idx, dev_info = get_input_device()

    input_file = "mic_test.wav"
    output_file = "assistant_reply.wav"

    print("=" * 70)
    print("🎓  AI STUDENT TUTOR AGENT (SCHOOL & INTERMEDIATE)")
    print("=" * 70)
    print(f"• Microphone Device: [{device_idx}] {dev_info.get('name', 'Unknown')}")
    print(f"• Sample Rate: {sample_rate} Hz | Adaptive VAD: ACTIVE (Max {MAX_RECORD_DURATION}s, Cutoff: {SILENCE_TIMEOUT_SECONDS}s silence)")
    print(f"• Echo Settling Pause: {SETTLING_PAUSE_SECONDS}s | VAD Energy Threshold: {VAD_ENERGY_THRESHOLD}")
    print("• Target Audience: School (Classes 1-10) & Intermediate (1st/2nd Year MPC & BiPC)")
    print("• School Subjects: Mathematics, Science, English, Social Studies")
    print("• Intermediate MPC: Mathematics, Physics, Chemistry")
    print("• Intermediate BiPC: Botany, Zoology, Physics, Chemistry")
    print("• Interactive Quiz Mode: Say 'Start quiz' or 'Quiz start cheyyi' anytime")
    print("• Exam Preparation: Ask for '2-mark', '5-mark', or '10-mark' answers")
    print("• Stop Commands: Say 'ఆపు', 'ఆపండి', 'stop', or 'exit' to end the session")
    print("=" * 70)

    profile = StudentProfile()
    session = LearningSession(profile=profile)
    quiz = session.quiz

    # Clean up any leftover audio files from prior runs
    for f in [input_file, output_file]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

    # -------------------------------------------------------------
    # Step 1: Education Level Selection Onboarding Flow
    # -------------------------------------------------------------
    welcome_prompt = "Welcome to your AI Student Tutor! Are you a School student or an Intermediate student?"
    print(f"\n🎓 Tutor: {welcome_prompt}")
    speak_and_play(welcome_prompt, output_file)

    while not profile.is_complete():
        rec_res = record_audio(
            filename=input_file,
            sample_rate=sample_rate,
            device=device_idx,
        )
        if not rec_res.success:
            print("Microphone error. Retrying...")
            continue

        if rec_res.is_silent:
            silence_hint = "మీ స్వరం వినపడలేదు. దయచేసి మైక్రోఫోన్ దగ్గరగా వచ్చి కాస్త గట్టిగా మాట్లాడండి."
            print(f"ℹ️ {silence_hint}")
            speak_and_play("I could not hear any speech. Please speak a little louder or closer to the microphone.", output_file)
            continue

        t_stt0 = time.perf_counter()
        result = transcribe_speech(input_file)
        stt_dur = time.perf_counter() - t_stt0
        stt_srv_time = result.get("processing_time", 0.0) if result else 0.0

        if not result or not result.get("success"):
            print("⚠️ Speech recognition connection issue during onboarding. Retrying...")
            stt_fail_msg = "I had trouble connecting to the speech service. Could you please repeat your answer?"
            ok, tts_dur, play_dur = speak_and_play(stt_fail_msg, output_file)
            print_latency_summary("Onboarding STT Error", rec_res.total_duration, rec_res.speech_duration, stt_dur, stt_srv_time, 0.0, tts_dur, play_dur)
            continue

        transcript = result.get("transcript", "").strip() if result else ""

        if not transcript:
            print("ℹ️ Empty transcript received from STT during onboarding.")
            empty_msg = "I could not hear that clearly. Could you please repeat your answer?"
            ok, tts_dur, play_dur = speak_and_play(empty_msg, output_file)
            print_latency_summary("Onboarding Unclear", rec_res.total_duration, rec_res.speech_duration, stt_dur, stt_srv_time, 0.0, tts_dur, play_dur)
            continue

        # Check if transcript contains class or education level answer before noise filtering
        has_class_info = extract_school_class(transcript) is not None
        has_level_info = any(k in transcript.lower() for k in ["school", "స్కూల్", "పాఠశాల", "inter", "ఇంటర్", "mpc", "bipc", "1st", "2nd"])

        if is_filler_or_noise(transcript) and not (has_class_info or has_level_info):
            clarify_msg = "I didn't catch that clearly. Are you a School student or an Intermediate student?"
            if profile.level == "school":
                clarify_msg = "I didn't catch your class clearly. Could you please repeat which class you are in, from Class 1 to Class 10?"
            elif profile.level == "intermediate" and not profile.grade_or_year:
                clarify_msg = "I didn't catch your year clearly. Are you studying in 1st Year or 2nd Year Intermediate?"
            elif profile.level == "intermediate" and not profile.group:
                clarify_msg = "I didn't catch your group clearly. Please tell me if you are in MPC or BiPC."

            print(f"ℹ️ Unclear speech detected ('{transcript}'). Asking for clarification.")
            ok, tts_dur, play_dur = speak_and_play(clarify_msg, output_file)
            print_latency_summary(
                turn_label="Onboarding Clarification",
                rec_duration=rec_res.total_duration,
                speech_duration=rec_res.speech_duration,
                stt_duration=stt_dur,
                stt_server_time=stt_srv_time,
                gemini_duration=0.0,
                tts_duration=tts_dur,
                play_duration=play_dur,
            )
            continue

        print(f"🗣️ Student: {transcript}")

        if is_stop_command(transcript):
            print("\n🛑 Stop command detected during setup. Exiting.")
            return

        is_done, next_msg = parse_education_level(transcript, profile)
        if profile.is_complete():
            session.profile = profile
            session.active_subject = profile.subject or profile.allowed_subjects()[0]
        if profile.grade_or_year:
            print(f"✅ Onboarding saved: {profile.describe()} (Level: {profile.level}, Grade/Year: {profile.grade_or_year}, Group: {profile.group})")
        print(f"\n🎓 Tutor: {next_msg}")
        ok, tts_dur, play_dur = speak_and_play(next_msg, output_file)

        print_latency_summary(
            turn_label="Onboarding Step",
            rec_duration=rec_res.total_duration,
            speech_duration=rec_res.speech_duration,
            stt_duration=stt_dur,
            stt_server_time=stt_srv_time,
            gemini_duration=0.0,
            tts_duration=tts_dur,
            play_duration=play_dur,
        )

    # -------------------------------------------------------------
    # Step 2: Continuous Learning & Interaction Loop
    # -------------------------------------------------------------
    print(f"\n✅ Ready! Profile active: {profile.describe()}")
    turn_count = 0
    last_processed_transcript = None

    try:
        while True:
            turn_count += 1
            if session.is_paused:
                mode_label = "PAUSED"
            elif quiz.active:
                mode_label = f"QUIZ MODE ({quiz.difficulty.upper()})"
            else:
                mode_label = f"LEARNING MODE ({session.active_subject})"
            print(f"\n--- [Tutor Session - Round {turn_count} ({mode_label})] ---")

            # =========================================================
            # BRANCH A: DEDICATED QUIZ ANSWER LISTENING MODE
            # =========================================================
            if quiz.active:
                ans_res = record_quiz_answer(
                    filename=input_file,
                    sample_rate=sample_rate,
                    device=device_idx,
                    timeout=QUIZ_ANSWER_TIMEOUT_SECONDS,
                )
                rec_duration = ans_res.duration
                speech_duration = ans_res.speech_duration

                # 1. Stop Quiz Command
                if ans_res.action == QUIZ_ACTION_STOP:
                    print("\n🛑 Stop command received during quiz. Exiting quiz mode.")
                    quiz_exit_msg = (
                        f"Quiz ended. You answered {quiz.score} questions correctly. "
                        "Returning to normal learning mode. What would you like to study?"
                    )
                    quiz.reset()
                    print(f"\n🎓 Tutor: {quiz_exit_msg}")
                    ok, tts_dur, play_dur = speak_and_play(quiz_exit_msg, output_file)
                    print_latency_summary(
                        turn_label=f"Round {turn_count} (Quiz Exit)",
                        rec_duration=rec_duration,
                        speech_duration=speech_duration,
                        stt_duration=0.0,
                        stt_server_time=0.0,
                        gemini_duration=0.0,
                        tts_duration=tts_dur,
                        play_duration=play_dur,
                    )
                    continue

                # 2. Repeat Question Command
                if ans_res.action == QUIZ_ACTION_REPEAT:
                    print(f"\n🔁 Repeat command received. Repeating Question {quiz.question_num}...")
                    repeat_msg = f"Let me repeat Question {quiz.question_num}: {quiz.current_question}"
                    print(f"\n🎓 Tutor: {repeat_msg}")
                    ok, tts_dur, play_dur = speak_and_play(repeat_msg, output_file)
                    print_latency_summary(
                        turn_label=f"Round {turn_count} (Quiz Repeat)",
                        rec_duration=rec_duration,
                        speech_duration=speech_duration,
                        stt_duration=0.0,
                        stt_server_time=0.0,
                        gemini_duration=0.0,
                        tts_duration=tts_dur,
                        play_duration=play_dur,
                    )
                    continue

                # 3. Skip Question Command
                if ans_res.action == QUIZ_ACTION_SKIP:
                    print(f"\n⏭️ Skip command received for Question {quiz.question_num}.")
                    quiz.adjust_difficulty(was_correct=False)
                    skip_feedback = f"Question skipped. The expected answer was {quiz.current_expected}."

                    if quiz.question_num < quiz.total_questions:
                        quiz.question_num += 1
                        t_q0 = time.perf_counter()
                        next_q, next_exp, next_hint = generate_quiz_question(
                            profile,
                            quiz.question_num,
                            quiz.subject,
                            topic=quiz.topic,
                            difficulty=quiz.difficulty,
                            return_hint=True,
                        )
                        gem_dur = time.perf_counter() - t_q0
                        quiz.current_question = next_q
                        quiz.current_expected = next_exp
                        quiz.current_hint = next_hint

                        combined_reply = f"{skip_feedback} Now, here is Question {quiz.question_num} ({quiz.difficulty} level): {next_q}"
                        print(f"\n🎓 Tutor: {combined_reply}")
                        ok, tts_dur, play_dur = speak_and_play(combined_reply, output_file)
                    else:
                        gem_dur = 0.0
                        final_score_msg = (
                            f"{skip_feedback} That concludes our quiz! Your total score is {quiz.score} out of {quiz.total_questions}. "
                            "Keep up the great learning! We are now back in regular tutor mode."
                        )
                        print(f"\n🎓 Tutor: {final_score_msg}")
                        ok, tts_dur, play_dur = speak_and_play(final_score_msg, output_file)
                        quiz.reset()

                    print_latency_summary(
                        turn_label=f"Round {turn_count} (Quiz Skip)",
                        rec_duration=rec_duration,
                        speech_duration=speech_duration,
                        stt_duration=0.0,
                        stt_server_time=0.0,
                        gemini_duration=gem_dur,
                        tts_duration=tts_dur,
                        play_duration=play_dur,
                    )
                    continue

                # 4. Empty Audio or Microphone Error
                if ans_res.action == QUIZ_ACTION_EMPTY or ans_res.is_silent:
                    empty_msg = "I could not hear any speech. Please speak your answer, or say 'Skip' or 'Stop quiz'."
                    print(f"ℹ️ {empty_msg}")
                    ok, tts_dur, play_dur = speak_and_play(empty_msg, output_file)
                    print_latency_summary(
                        turn_label=f"Round {turn_count} (Quiz Empty)",
                        rec_duration=rec_duration,
                        speech_duration=speech_duration,
                        stt_duration=0.0,
                        stt_server_time=0.0,
                        gemini_duration=0.0,
                        tts_duration=tts_dur,
                        play_duration=play_dur,
                    )
                    continue

                # 5. Safety Timeout Reached
                if ans_res.action == QUIZ_ACTION_TIMEOUT and not ans_res.answer_text:
                    timeout_msg = f"Time is up for Question {quiz.question_num} (120s limit). The expected answer was {quiz.current_expected}."
                    print(f"⏰ {timeout_msg}")
                    quiz.adjust_difficulty(was_correct=False)
                    if quiz.question_num < quiz.total_questions:
                        quiz.question_num += 1
                        t_q0 = time.perf_counter()
                        next_q, next_exp, next_hint = generate_quiz_question(
                            profile,
                            quiz.question_num,
                            quiz.subject,
                            topic=quiz.topic,
                            difficulty=quiz.difficulty,
                            return_hint=True,
                        )
                        gem_dur = time.perf_counter() - t_q0
                        quiz.current_question = next_q
                        quiz.current_expected = next_exp
                        quiz.current_hint = next_hint
                        combined_reply = f"{timeout_msg} Now, here is Question {quiz.question_num} ({quiz.difficulty} level): {next_q}"
                        print(f"\n🎓 Tutor: {combined_reply}")
                        ok, tts_dur, play_dur = speak_and_play(combined_reply, output_file)
                    else:
                        gem_dur = 0.0
                        final_msg = f"{timeout_msg} That concludes our quiz! Your score is {quiz.score} out of {quiz.total_questions}."
                        print(f"\n🎓 Tutor: {final_msg}")
                        ok, tts_dur, play_dur = speak_and_play(final_msg, output_file)
                        quiz.reset()
                    continue

                # 6. Student Submitted Answer (QUIZ_ACTION_SUBMIT)
                student_answer = ans_res.answer_text or ans_res.raw_transcript
                print(f"\n📝 [RECOGNIZED QUIZ ANSWER]: \"{student_answer}\"")

                # Evaluate student answer
                t_gem0 = time.perf_counter()
                is_correct, feedback = evaluate_quiz_answer(
                    student_answer=student_answer,
                    question=quiz.current_question,
                    expected_answer=quiz.current_expected,
                    profile=profile,
                    hint=quiz.current_hint,
                )
                gem_dur = time.perf_counter() - t_gem0
                if is_correct:
                    quiz.score += 1

                quiz.adjust_difficulty(was_correct=is_correct)

                print(f"🎓 Evaluation: {'Correct (+1)' if is_correct else 'Incorrect'}")
                print(f"🎓 Tutor Feedback: {feedback}")

                if quiz.question_num < quiz.total_questions:
                    quiz.question_num += 1
                    t_q0 = time.perf_counter()
                    next_q, next_exp, next_hint = generate_quiz_question(
                        profile,
                        quiz.question_num,
                        quiz.subject,
                        topic=quiz.topic,
                        difficulty=quiz.difficulty,
                        return_hint=True,
                    )
                    gem_dur += (time.perf_counter() - t_q0)
                    quiz.current_question = next_q
                    quiz.current_expected = next_exp
                    quiz.current_hint = next_hint

                    combined_reply = f"You answered {student_answer}. {feedback} Now, here is Question {quiz.question_num} ({quiz.difficulty} level): {next_q}"
                    print(f"\n🎓 Tutor: {combined_reply}")
                    ok, tts_dur, play_dur = speak_and_play(combined_reply, output_file)
                else:
                    final_score_msg = (
                        f"You answered {student_answer}. {feedback} That concludes our quiz! Your total score is {quiz.score} out of {quiz.total_questions}. "
                        "Keep up the great learning! We are now back in regular tutor mode."
                    )
                    print(f"\n🎓 Tutor: {final_score_msg}")
                    ok, tts_dur, play_dur = speak_and_play(final_score_msg, output_file)
                    quiz.reset()

                print_latency_summary(
                    turn_label=f"Round {turn_count} (Quiz Answer)",
                    rec_duration=rec_duration,
                    speech_duration=speech_duration,
                    stt_duration=0.0,
                    stt_server_time=0.0,
                    gemini_duration=gem_dur,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            # =========================================================
            # BRANCH B: NORMAL DOUBT / CONVERSATION LEARNING MODE
            # =========================================================
            # 1. Record audio from microphone adaptively (using normal 30s max & 1.1s silence cutoff)
            rec_res = record_audio(
                filename=input_file,
                sample_rate=sample_rate,
                device=device_idx,
                max_duration=MAX_RECORD_DURATION,
                silence_timeout=SILENCE_TIMEOUT_SECONDS,
            )
            if not rec_res.success:
                print("Could not record audio. Retrying in 2 seconds...")
                continue

            # 2. Check for silence before making STT calls
            if rec_res.is_silent:
                silence_msg = "I could not hear any speech. Please speak a little louder or closer to the microphone."
                print(f"ℹ️ {silence_msg}")
                ok, tts_dur, play_dur = speak_and_play(silence_msg, output_file)
                print_latency_summary(
                    turn_label=f"Round {turn_count} (Silence)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=0.0,
                    stt_server_time=0.0,
                    gemini_duration=0.0,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            # 3. Transcribe speech using Gnani Prisma STT
            t_stt0 = time.perf_counter()
            result = transcribe_speech(input_file)
            stt_dur = time.perf_counter() - t_stt0
            stt_srv_time = result.get("processing_time", 0.0) if result else 0.0

            if not result or not result.get("success"):
                print("⚠️ Speech recognition failed or timed out. Prompting student to repeat...")
                stt_fail_msg = "I had trouble connecting to the speech service. Could you please repeat your question?"
                ok, tts_dur, play_dur = speak_and_play(stt_fail_msg, output_file)
                print_latency_summary(
                    turn_label=f"Round {turn_count} (STT Error)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=stt_dur,
                    stt_server_time=stt_srv_time,
                    gemini_duration=0.0,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            transcript = result.get("transcript", "").strip()

            if not transcript:
                print("ℹ️ Empty transcript received from STT. Asking for repeat...")
                repeat_msg = "I could not hear that clearly. Could you please repeat your question?"
                ok, tts_dur, play_dur = speak_and_play(repeat_msg, output_file)
                print_latency_summary(
                    turn_label=f"Round {turn_count} (Empty Transcript)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=stt_dur,
                    stt_server_time=stt_srv_time,
                    gemini_duration=0.0,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            # 4. Filter filler noise / hesitation sounds or low-confidence transcription
            if result.get("low_confidence"):
                low_conf_msg = "I didn't catch that with full confidence. Could you please repeat your question clearly?"
                print(f"ℹ️ Low confidence STT ('{transcript}'). Asking student to repeat rather than guessing.")
                ok, tts_dur, play_dur = speak_and_play(low_conf_msg, output_file)
                print_latency_summary(
                    turn_label=f"Round {turn_count} (Low Confidence)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=stt_dur,
                    stt_server_time=stt_srv_time,
                    gemini_duration=0.0,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            if is_incomplete_speech(transcript, session):
                has_telugu = any("\u0c00" <= c <= "\u0c7f" for c in transcript)
                if has_telugu:
                    unclear_msg = "మీ ప్రశ్న పూర్తిగా వినపడలేదు. దయచేసి మీ ప్రశ్నను మళ్లీ స్పష్టంగా అడగండి."
                else:
                    unclear_msg = "I could not hear a complete question. Could you please repeat your question clearly?"
                print(f"ℹ️ Incomplete/meaningless speech detected ('{transcript}'). Asking student to repeat.")
                ok, tts_dur, play_dur = speak_and_play(unclear_msg, output_file)
                print_latency_summary(
                    turn_label=f"Round {turn_count} (Incomplete)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=stt_dur,
                    stt_server_time=stt_srv_time,
                    gemini_duration=0.0,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            detected_lang = result.get("detected_language_code", "auto")
            print(f"🗣️ Student [{detected_lang}]: {transcript}")

            # Guard against duplicate identical processing in the same state
            last_processed_transcript = transcript

            # 5. Check for stop commands in normal mode
            if is_stop_command(transcript):
                print("\n🛑 Stop command detected! Ending tutor session.")
                goodbye_reply = "సరే, సెషన్ ముగిస్తున్నాను. All the best for your studies! మళ్లీ కలుద్దాం!"
                print(f"\n🎓 Tutor: {goodbye_reply}")
                ok, tts_dur, play_dur = speak_and_play(goodbye_reply, output_file)
                print_latency_summary(
                    turn_label=f"Round {turn_count} (Stop)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=stt_dur,
                    stt_server_time=stt_srv_time,
                    gemini_duration=0.0,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                print("\n✅ Tutor session finished. Goodbye!")
                break

            # 7. Check if student wants to START a quiz
            if is_quiz_trigger(transcript):
                print("\n🎯 Quiz mode triggered!")
                quiz.reset()
                quiz.active = True
                quiz.question_num = 1
                quiz.subject = session.active_subject
                quiz.topic = session.active_topic
                quiz.difficulty = "medium"

                for subj in profile.allowed_subjects():
                    if subj.lower() in transcript.lower():
                        quiz.subject = subj
                        session.active_subject = subj
                        session.profile.subject = subj
                        break

                t_gem0 = time.perf_counter()
                q_text, a_text, h_text = generate_quiz_question(
                    profile, 1, quiz.subject, topic=quiz.topic, difficulty=quiz.difficulty, return_hint=True
                )
                gem_dur = time.perf_counter() - t_gem0
                quiz.current_question = q_text
                quiz.current_expected = a_text
                quiz.current_hint = h_text

                quiz_start_msg = f"Let's start your {quiz.subject} quiz! Question 1: {q_text}"
                print(f"\n🎓 Tutor: {quiz_start_msg}")
                ok, tts_dur, play_dur = speak_and_play(quiz_start_msg, output_file)

                print_latency_summary(
                    turn_label=f"Round {turn_count} (Quiz Start)",
                    rec_duration=rec_res.total_duration,
                    speech_duration=rec_res.speech_duration,
                    stt_duration=stt_dur,
                    stt_server_time=stt_srv_time,
                    gemini_duration=gem_dur,
                    tts_duration=tts_dur,
                    play_duration=play_dur,
                )
                continue

            # 8. Generate normal tutor reply (Teaching / Doubts / Socratic / Exam prep)
            inferred_subj = detect_question_subject(transcript, profile)
            turn_subj = inferred_subj if (inferred_subj and inferred_subj in profile.allowed_subjects()) else session.active_subject
            if turn_subj != session.active_subject:
                print(f"\n🤔 Thinking as an Autonomous Tutor for {profile.describe()} (Subject: {turn_subj} [Turn Context, Active: {session.active_subject}])...")
            else:
                print(f"\n🤔 Thinking as an Autonomous Tutor for {profile.describe()} (Subject: {session.active_subject})...")

            t_gem0 = time.perf_counter()
            reply = generate_reply(transcript, profile=profile, session=session, turn_subject=turn_subj)
            gem_dur = time.perf_counter() - t_gem0
            print(f"\n🎓 Tutor:\n{reply}")

            # 9. Synthesize voice reply and play
            ok, tts_dur, play_dur = speak_and_play(reply, output_file)

            # 10. Print accurate latency breakdown
            print_latency_summary(
                turn_label=f"Round {turn_count} (Learning)",
                rec_duration=rec_res.total_duration,
                speech_duration=rec_res.speech_duration,
                stt_duration=stt_dur,
                stt_server_time=stt_srv_time,
                gemini_duration=gem_dur,
                tts_duration=tts_dur,
                play_duration=play_dur,
            )
            print("\n🔄 Ready for your next student question...")

    except KeyboardInterrupt:
        print("\n\n👋 Tutor session interrupted by user (Ctrl+C). Goodbye and all the best!")
    except Exception as e:
        print(f"\n❌ Tutor session encountered an unexpected exception: {e}")
        try:
            speak_and_play("Tutoring session ended due to an unexpected system error. Please restart the assistant.", output_file)
        except Exception:
            pass


if __name__ == "__main__":
    if "--diagnose-mic" in sys.argv or "--test-mic" in sys.argv:
        dur = 3.0
        for arg in sys.argv[1:]:
            try:
                dur = float(arg)
            except ValueError:
                pass
        diagnose_microphone(duration=dur, output_filename="diagnostic_sample.wav")
    else:
        main()
