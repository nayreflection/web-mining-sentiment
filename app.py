import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import warnings
warnings.filterwarnings('ignore')

from collections import Counter

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="News Sentiment Analysis · Agro Bromo",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@500;600;700&family=Manrope:wght@400;500;600&display=swap');
@import url('https://fonts.googleapis.com/icon?family=Material+Icons');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined');

:root {
    --bg: #0D0D0E;
    --surface: #161618;
    --border: #242426;

    --accent: #FF6A2A;
    --accent-soft: #FF8A50;

    --positive: #3EC97A;
    --neutral: #F59E0B;
    --negative: #EF4444;

    --text: #F5F5F5;
    --muted: #9A9A9A;
}

/* ===== BASE ===== */
html, body, .stApp {
    background: var(--bg);
    color: var(--text);
    font-family: 'Manrope', sans-serif;
}

/* ===== ICON ===== */
.material-icons,
.material-icons-outlined,
.material-icons-round,
.material-icons-sharp,
span[class*="material-icons"],
i[class*="material-icons"] {
    font-family: 'Material Icons' !important;
    font-style: normal !important;
    font-weight: normal !important;
    display: inline-block !important;
}
.material-symbols-outlined {
    font-family: 'Material Symbols Outlined' !important;
    font-weight: normal !important;
    font-style: normal !important;
    font-size: 24px !important;
    display: inline-block !important;
    line-height: 1 !important;
}
/* ===== HEADING ===== */
.hero-title,
.section-title,
.metric-value,
.result-sentiment-label {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
}

/* ===== SIDEBAR FIX TOTAL ===== */
section[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
    min-width: 260px !important;
    max-width: 260px !important;
}

/* paksa font sidebar tanpa rusak icon */
section[data-testid="stSidebar"] *:not(.material-icons) {
    font-family: 'Manrope', sans-serif !important;
    color: var(--text) !important;
}

/* ===== HIDE BRANDING ===== */
#MainMenu, footer { visibility: hidden; }

/* ===== CARD ===== */
.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px 24px;
    transition: 0.2s;
}
.metric-card:hover {
    border-color: var(--accent);
    transform: translateY(-2px);
}

/* ===== TEXT ===== */
.metric-label {
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--muted);
}
.metric-sub {
    font-size: 12px;
    color: var(--muted);
}

/* ===== BADGE ===== */
.badge {
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
.badge-positive { background: rgba(62,201,122,0.15); color: #3EC97A; }
.badge-neutral  { background: rgba(245,158,11,0.15); color: #F59E0B; }
.badge-negative { background: rgba(239,68,68,0.15); color: #EF4444; }

/* ===== SECTION ===== */
.section-title {
    font-size: 22px;
    margin: 24px 0 6px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 10px;
}

/* ===== INPUT ===== */
.stTextArea textarea,
.stTextInput input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 12px !important;
}
.stTextArea textarea:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 2px rgba(255,106,42,0.2) !important;
}

/* ===== SELECT ===== */
div[data-baseweb="select"] * {
    font-family: 'Manrope', sans-serif !important;
}

/* ===== BUTTON ===== */
.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border-radius: 12px !important;
    border: none !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background: var(--accent-soft) !important;
    box-shadow: 0 6px 18px rgba(255,106,42,0.3);
}
.stButton > button * {
    background: transparent !important;
}

/* ===== PROGRESS ===== */
.stProgress > div > div > div {
    background: var(--accent) !important;
}

/* ===== HERO ===== */
.hero-title {
    font-size: 48px;
}
.hero-accent { color: var(--accent); }
.hero-sub {
    font-size: 15px;
    color: var(--muted);
}

/* ===== RESULT ===== */
.result-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
}

/* ===== TAB ===== */
button[data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ===== CLEAN BACKGROUND ===== */
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"] {
    background: transparent !important;
}

/* ===== DIVIDER ===== */
.divider {
    border-top: 1px solid var(--border);
    margin: 24px 0;
}

</style>
""", unsafe_allow_html=True)


# ── NLP Setup ────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nlp():
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    nltk.download('stopwords', quiet=True)
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    return word_tokenize, stopwords

@st.cache_resource(show_spinner=False)
def load_stemmer():
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    factory = StemmerFactory()
    return factory.create_stemmer()

@st.cache_resource(show_spinner=False)
def load_roberta():
    from transformers import pipeline as hf_pipeline
    model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"
    clf = hf_pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
        device=-1,
    )
    return clf

@st.cache_resource(show_spinner=False)
def load_indobert():
    from transformers import pipeline as hf_pipeline
    model_name = "mdhugol/indonesia-bert-sentiment-classification"
    clf = hf_pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name,
        truncation=True,
        max_length=512,
        device=-1,
    )
    return clf


# ── Constants ────────────────────────────────────────────────
SLANG_DICT = {

    # ── Negasi ────────────────────────────────────────────────
    'gk':'tidak', 'ga':'tidak', 'gak':'tidak', 'nggak':'tidak',
    'ngga':'tidak', 'ndak':'tidak', 'enggak':'tidak', 'kagak':'tidak',
    'gakk':'tidak', 'gaada':'tidak ada', 'gada':'tidak ada',
    'nda':'tidak', 'ndk':'tidak', 'tak':'tidak', 'tdk':'tidak',

    # ── Intensifier / Penguat ─────────────────────────────────
    'bgt':'sangat', 'banget':'sangat', 'bngt':'sangat', 'bngtt':'sangat',
    'bgtt':'sangat', 'bnget':'sangat', 'parah':'sangat', 'pol':'sangat',
    'abis':'sangat', 'abiss':'sangat', 'pisan':'sangat',
    'bet':'sangat', 'bett':'sangat',

    # ── Kata Ganti Orang ──────────────────────────────────────
    'gue':'saya', 'gw':'saya', 'gwe':'saya', 'gweh':'saya',
    'aku':'saya', 'ak':'saya', 'akku':'saya', 'sy':'saya',
    'w':'saya', 'q':'saya', 'aq':'saya',
    'lo':'kamu', 'lu':'kamu', 'loe':'kamu', 'elo':'kamu',
    'elu':'kamu', 'u':'kamu', 'lw':'kamu',
    'doi':'dia', 'dy':'dia', 'dya':'dia', 'doski':'dia',
    'mereka':'mereka', 'mrk':'mereka', 'kita':'kita', 'kt':'kita',
    'kalian':'kalian', 'klian':'kalian',

    # ── Kata Umum / Singkatan ─────────────────────────────────
    'yg':'yang', 'yng':'yang', 'yag':'yang',
    'dr':'dari', 'dri':'dari',
    'dgn':'dengan', 'dg':'dengan', 'sama':'dengan',
    'aja':'saja', 'aj':'saja', 'doang':'saja', 'tok':'saja',
    'klo':'kalau', 'kl':'kalau', 'kalo':'kalau', 'klw':'kalau',
    'klu':'kalau', 'klaw':'kalau',
    'krn':'karena', 'karna':'karena', 'krna':'karena', 'krn':'karena',
    'soalnya':'karena', 'soale':'karena', 'soal':'karena',
    'jd':'jadi', 'jdi':'jadi', 'jadinya':'jadi',
    'udh':'sudah', 'sdh':'sudah', 'udah':'sudah', 'dah':'sudah',
    'uda':'sudah', 'udag':'sudah',
    'blm':'belum', 'blum':'belum', 'belm':'belum',
    'lg':'lagi', 'lgi':'lagi', 'msh':'masih', 'masi':'masih',
    'trs':'terus', 'trus':'terus', 'truss':'terus', 'terusin':'teruskan',
    'jg':'juga', 'juga':'juga', 'jg':'juga',
    'tp':'tapi', 'tpi':'tapi', 'tapii':'tapi', 'tapi':'tapi',
    'pd':'pada', 'kpd':'kepada', 'utk':'untuk', 'tuk':'untuk', 'buat':'untuk',
    'sm':'sama', 'ama':'sama', 'brsama':'bersama',
    'spy':'supaya', 'biar':'supaya', 'bnya':'supaya',
    'dl':'dulu', 'dlu':'dulu', 'duluan':'dulu',
    'skrg':'sekarang', 'skrang':'sekarang', 'skr':'sekarang', 'skg':'sekarang',
    'ntar':'nanti', 'tar':'nanti', 'nanti':'nanti', 'ntr':'nanti',
    'emg':'memang', 'emang':'memang', 'memg':'memang', 'mmg':'memang',
    'bener':'benar', 'beneran':'benar', 'bnr':'benar', 'bnar':'benar',
    'org':'orang', 'orng':'orang', 'orag':'orang',
    'bnyk':'banyak', 'byk':'banyak', 'bnk':'banyak',
    'dikit':'sedikit', 'skit':'sedikit', 'cuma':'hanya', 'cm':'hanya',
    'cma':'hanya', 'cmn':'hanya', 'doang':'hanya',
    'bs':'bisa', 'bsa':'bisa', 'biss':'bisa',
    'hrs':'harus', 'msti':'mesti', 'musti':'mesti',
    'knp':'kenapa', 'knapaa':'kenapa', 'ngapa':'kenapa', 'ngapain':'kenapa',
    'gmn':'gimana', 'gmna':'gimana', 'gimna':'gimana', 'bgmn':'bagaimana',
    'bgmna':'bagaimana', 'kayak':'seperti', 'kyk':'seperti', 'kek':'seperti',
    'sprti':'seperti', 'spt':'seperti',
    'ttp':'tetap', 'tetep':'tetap', 'ttep':'tetap',
    'ktnya':'katanya', 'ktny':'katanya', 'konon':'katanya',
    'bth':'butuh', 'perlu':'butuh',
    'liat':'lihat', 'lht':'lihat', 'lh':'lihat',
    'bilang':'berkata', 'blng':'berkata', 'ngomong':'berkata', 'omong':'berkata',
    'dtg':'datang', 'dateng':'datang',
    'prgi':'pergi', 'pgi':'pergi',
    'krj':'kerja', 'kerjaan':'pekerjaan', 'kerja':'bekerja',
    'makan':'makan', 'mkn':'makan',
    'minum':'minum', 'mnm':'minum',
    'tidur':'tidur', 'tdr':'tidur',
    'tau':'tahu', 'tw':'tahu', 'taw':'tahu',
    'donlot':'unduh', 'download':'unduh', 'upload':'unggah',
    'chat':'pesan', 'dm':'pesan langsung',

    # ── Kata Sifat / Penilaian ────────────────────────────────
    'keren':'bagus', 'kereen':'bagus', 'kerenn':'bagus',
    'jos':'bagus', 'josss':'bagus', 'mantap':'bagus',
    'mantul':'bagus', 'mantabs':'bagus', 'mantab':'bagus',
    'goks':'luar biasa', 'gokil':'luar biasa', 'gilak':'gila',
    'gila':'gila', 'gilee':'gila', 'gileeee':'gila',
    'kocak':'lucu', 'ngakak':'tertawa', 'ngkak':'tertawa',
    'lucu':'lucu', 'lucuu':'lucu',
    'sedih':'sedih', 'sdih':'sedih', 'sediih':'sedih',
    'marah':'marah', 'mrh':'marah', 'kesel':'kesal', 'ksl':'kesal',
    'nyebelin':'menjengkelkan', 'nyebel':'menjengkelkan',
    'menyebalkan':'menjengkelkan', 'sebel':'kesal',
    'capek':'lelah', 'cape':'lelah', 'cpk':'lelah',
    'susah':'sulit', 'sussh':'sulit', 'ribet':'rumit',
    'gampang':'mudah', 'gmpng':'mudah',
    'jelek':'buruk', 'jlek':'buruk', 'buruk':'buruk',
    'bagus':'bagus', 'bgus':'bagus',
    'pinter':'pintar', 'pnter':'pintar', 'cerdas':'pintar',
    'bego':'bodoh', 'bgu':'bodoh', 'tolol':'bodoh', 'dungu':'bodoh',
    'bahaya':'berbahaya', 'berbahaya':'berbahaya', 'bhya':'berbahaya',

    # ── Ekspresi / Interjeksi ─────────────────────────────────
    'wkwk':'tertawa', 'wkwkwk':'tertawa', 'wkkwk':'tertawa',
    'wkwkwkwk':'tertawa', 'kwkwk':'tertawa',
    'haha':'tertawa', 'hahaha':'tertawa', 'hahahaha':'tertawa',
    'hehe':'tertawa', 'hehehe':'tertawa',
    'hihi':'tertawa', 'huhu':'menangis', 'huhuu':'menangis',
    'huft':'kecewa', 'hufft':'kecewa', 'huff':'kecewa',
    'hmm':'ragu', 'hmmm':'ragu', 'hmmmm':'ragu',
    'duh':'kecewa', 'aduh':'kecewa', 'aduhh':'kecewa',
    'astaga':'terkejut', 'astaghfirullah':'terkejut',
    'ya allah':'terkejut', 'yaAllah':'terkejut',
    'hadeh':'kecewa', 'haduh':'kecewa', 'hadeuh':'kecewa',
    'anjir':'ekspresi', 'anjay':'ekspresi', 'anjing':'ekspresi',
    'asem':'ekspresi', 'asu':'ekspresi',
    'kampret':'ekspresi', 'sialan':'ekspresi', 'kurang ajar':'ekspresi',
    'sial':'ekspresi', 'brengsek':'ekspresi',
    'busyet':'terkejut', 'buset':'terkejut',
    'wow':'kagum', 'woow':'kagum', 'waw':'kagum',
    'ih':'ekspresi', 'ihh':'ekspresi', 'iih':'ekspresi',
    'eh':'ekspresi', 'ehh':'ekspresi',
    'kok':'ekspresi', 'loh':'ekspresi', 'lho':'ekspresi',
    'dong':'ekspresi', 'deh':'ekspresi', 'sih':'ekspresi',
    'nih':'ini', 'tuh':'itu', 'gini':'begini', 'gitu':'begitu',
    'gtr':'begitu', 'gnr':'begini',

    # ── Konteks Kecelakaan / Berita Kereta ───────────────────
    'krl':'kereta rel listrik', 'ka':'kereta api', 'kt':'kereta',
    'masinis':'masinis', 'rel':'rel', 'stasiun':'stasiun',
    'gerbong':'gerbong', 'lok':'lokomotif',
    'PT KAI':'PT Kereta Api Indonesia', 'kai':'kereta api indonesia',
    'kemenhub':'kementerian perhubungan',
    'menhub':'menteri perhubungan',
    'rem':'rem', 'tabrakan':'tabrakan', 'tabrak':'menabrak',
    'senggol':'menabrak', 'adu':'bertabrakan',
    'korban':'korban', 'meninggal':'meninggal', 'wafat':'meninggal',
    'innalillahi':'duka cita', 'inalillahi':'duka cita',
    'rip':'beristirahat dalam damai',

    # ── Ekspresi Media Sosial ─────────────────────────────────
    'subscribe':'berlangganan', 'sub':'berlangganan', 'subs':'berlangganan',
    'share':'bagikan', 'komen':'komentar', 'comment':'komentar',
    'like':'suka', 'likes':'suka', 'dislike':'tidak suka',
    'viral':'viral', 'trending':'tren', 'fyp':'untuk kamu',
    'fypシ':'untuk kamu', 'xyzbca':'untuk kamu',
    'oot':'keluar topik', 'otw':'sedang menuju',
    'info':'informasi', 'infoin':'informasikan',
    'update':'perbarui', 'up':'perbarui',
    'numpang':'menumpang', 'numpang lewat':'menumpang lewat',

    # ── Singkatan Umum Lainnya ────────────────────────────────
    'yth':'yang terhormat', 'bpk':'bapak', 'ibu':'ibu',
    'pak':'bapak', 'bu':'ibu', 'kak':'kakak', 'bang':'abang',
    'mas':'mas', 'mbak':'mbak', 'dik':'adik',
    'tmn':'teman', 'tmen':'teman', 'sobat':'teman',
    'bro':'saudara', 'sis':'saudara', 'gaes':'teman-teman',
    'guys':'teman-teman', 'gan':'teman', 'sis':'saudari',
    'pemerintah':'pemerintah', 'pmrntah':'pemerintah', 'pmntah':'pemerintah',
    'negara':'negara', 'nkri':'negara', 'indonesia':'indonesia',
    'rakyat':'rakyat', 'masyarakat':'masyarakat', 'msyrkt':'masyarakat',
    'media':'media', 'berita':'berita', 'brita':'berita', 'brta':'berita',
    'wartawan':'wartawan', 'jurnalis':'wartawan',
    'hoax':'berita bohong', 'hoaks':'berita bohong', 'bohong':'bohong',
    'fitnah':'fitnah', 'hasut':'hasutan',
    'duit':'uang', 'fulus':'uang', 'duwit':'uang',
    'kerjaan':'pekerjaan', 'kerjaan':'pekerjaan',
    'rumah':'rumah', 'rmh':'rumah',
    'jln':'jalan', 'jalan':'jalan',
    'anak':'anak', 'anaknya':'anaknya', 'keluarga':'keluarga', 'klrga':'keluarga',
    'hidup':'hidup', 'hdp':'hidup', 'mati':'mati',
    'selamat':'selamat', 'slmt':'selamat',
    'terima kasih':'terima kasih', 'makasih':'terima kasih',
    'makasi':'terima kasih', 'tengkyu':'terima kasih',
    'thanks':'terima kasih', 'thx':'terima kasih', 'tq':'terima kasih',
    'oke':'baik', 'ok':'baik', 'oks':'baik', 'okee':'baik',
    'siap':'siap', 'sip':'siap',
    'lanjut':'lanjut', 'lnjt':'lanjut', 'next':'selanjutnya',
    'hampir':'hampir', 'hmpir':'hampir',
    'selalu':'selalu', 'sllu':'selalu',
    'pasti':'pasti', 'pst':'pasti',
    'kayanya':'sepertinya', 'kyknya':'sepertinya', 'kyaknya':'sepertinya',
    'mungkin':'mungkin', 'mgkn':'mungkin', 'mgkin':'mungkin',
    'harusnya':'seharusnya', 'harusnye':'seharusnya',
    'seharusnya':'seharusnya', 'seharsnya':'seharusnya',
    'padahal':'padahal', 'pdhl':'padahal',
    'ternyata':'ternyata', 'trnyata':'ternyata', 'trnyta':'ternyata',
    'akhirnya':'akhirnya', 'akhrnya':'akhirnya',
    'langsung':'langsung', 'lngsng':'langsung',
    'cepat':'cepat', 'cpat':'cepat', 'cpt':'cepat',
    'lambat':'lambat', 'lmbt':'lambat', 'lelet':'lambat',
    'lama':'lama', 'lmaa':'lama',
}

SENTIMENT_COLORS = {
    'positive': '#3EC97A',
    'neutral':  '#E8C547',
    'negative': '#E85454',
}

LABEL_MAP_ROBERTA = {
    'positive': 'positive', 'POSITIVE': 'positive', 'Positive': 'positive',
    'negative': 'negative', 'NEGATIVE': 'negative', 'Negative': 'negative',
    'neutral':  'neutral',  'NEUTRAL':  'neutral',  'Neutral':  'neutral',
}
LABEL_MAP_INDOBERT = {
    'LABEL_0': 'positive', 'LABEL_1': 'neutral', 'LABEL_2': 'negative',
    'Positif': 'positive',  'Netral': 'neutral',  'Negatif': 'negative',
    'positive': 'positive', 'neutral': 'neutral', 'negative': 'negative',
}


# ── Text Processing ──────────────────────────────────────────
def clean_text(text):
    if pd.isna(text): return ''
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'@\w+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'\d+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'(.)\1{2,}', r'\1\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def normalize_slang(text):
    return ' '.join(
        SLANG_DICT.get(w, w)
        for w in text.split()
    )


# Load NLP
word_tokenize, stopwords = load_nlp()

# Stopwords Indonesia
indonesia_sw = stopwords.words('indonesian')

# Pertahankan kata negasi
stop_words = set(indonesia_sw) - {
    'tidak',
    'bukan',
    'kurang',
    'tak'
}

# Custom stopwords
CUSTOM_SW = {
    'innalillahi','wainnailaihirojiun','rojiun',
    'moga','semoga','aamiin','amiin',
    'alfatihah','allahuakbar','subhanallah',
    'allahumma','doa','doakan','almarhum',
    'surga','wa','yaa','si','pak','bu',
    'bang','kak','mas','mbak','nih',
    'sih','lah','dong','deh','tuh','kok',
}

stop_words |= CUSTOM_SW

# Stemmer
stemmer = load_stemmer()


def preprocess(text):

    # clean text
    clean = clean_text(text)

    # normalize slang
    normalized = normalize_slang(clean)

    # tokenisasi
    tokens_word = word_tokenize(normalized)

    # remove stopwords
    tokens_no_stop = [
        w for w in tokens_word
        if w not in stop_words
        and len(w) > 2
        and w.isalpha()
    ]

    # stemming
    tokens_stem = [
        stemmer.stem(w)
        for w in tokens_no_stop
    ]

    # final text
    final_text = ' '.join(tokens_stem)

    return {
        'clean_text': clean,
        'normalized_text': normalized,
        'tokens_word': tokens_word,
        'tokens_no_stop': tokens_no_stop,
        'tokens_stem': tokens_stem,
        'final_text': final_text
    }


def analyze_sentiment(text, model_name, clf_roberta, clf_indobert):

    # preprocessing
    prep = preprocess(text)

    # text final hasil preprocessing
    processed = prep['final_text']

    # fallback jika kosong
    if not processed.strip():

        return {
            'label': 'neutral',
            'score': 1.0,
            'scores': {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0
            },
            **prep
        }

    try:

        # ── RoBERTa ─────────────────────
        if model_name == "RoBERTa":

            result = clf_roberta(processed[:512])[0]

            label = LABEL_MAP_ROBERTA.get(
                result['label'],
                result['label']
            )

            score = result['score']

            # ambil semua score
            all_results = clf_roberta(
                processed[:512],
                top_k=None
            )

            scores = {}

            if isinstance(all_results, list):

                for r in all_results:

                    mapped = LABEL_MAP_ROBERTA.get(
                        r['label'],
                        r['label']
                    )

                    scores[mapped] = r.get('score', 0.0)

            else:

                scores = {
                    label: score,
                    **{
                        k: (1-score)/2
                        for k in [
                            'positive',
                            'neutral',
                            'negative'
                        ]
                        if k != label
                    }
                }

        # ── IndoBERT ────────────────────
        else:

            result = clf_indobert(processed[:512])[0]

            label = LABEL_MAP_INDOBERT.get(
                result['label'],
                result['label']
            )

            score = result['score']

            scores = {
                label: score,
                **{
                    k: (1-score)/2
                    for k in [
                        'positive',
                        'neutral',
                        'negative'
                    ]
                    if k != label
                }
            }

        # normalize score
        total = sum(scores.values())

        if total > 0:

            scores = {
                k: v / total
                for k, v in scores.items()
            }

        # pastikan semua label ada
        for key in ['positive', 'neutral', 'negative']:
            scores.setdefault(key, 0.0)

        # feature engineering
        char_length = len(str(text))

        word_count = len(prep['tokens_no_stop'])

        has_emoji = int(
            bool(
                re.search(r'[^\x00-\x7F]', str(text))
            )
        )

        # return hasil lengkap
        return {

            # hasil sentimen
            'label': label,
            'score': score,
            'scores': scores,

            # preprocessing
            'clean_text': prep['clean_text'],
            'normalized_text': prep['normalized_text'],
            'tokens_word': prep['tokens_word'],
            'tokens_no_stop': prep['tokens_no_stop'],
            'tokens_stem': prep['tokens_stem'],
            'final_text': prep['final_text'],

            # fitur tambahan
            'char_length': char_length,
            'word_count': word_count,
            'has_emoji': has_emoji
        }

    except Exception as e:

        return {
            'label': 'neutral',
            'score': 1.0,
            'scores': {
                'positive': 0.0,
                'neutral': 1.0,
                'negative': 0.0
            },
            'error': str(e),
            **prep
        }


# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 24px;">
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.12em; color:#888; margin-bottom:8px;">Project</div>
        <div style="font-family:'Instrument Serif',serif; font-size:20px; line-height:1.2; color:#F0EFEA;">
            Web Mining &<br>Analisis Sentimen
        </div>
        <div style="font-size:12px; color:#888; margin-top:6px;">Kecelakaan KA Agro Bromo</div>
    </div>
    <hr style="border:none; border-top:1px solid #242426; margin: 0 0 20px;">
    """, unsafe_allow_html=True)

    page = st.selectbox(
        "Navigasi",
        ["Dashboard", "Analisis Sentimen", "Statistik", "Batch Prediksi", "Tentang"],
        label_visibility="collapsed"
    )

    st.markdown("<hr style='border:none; border-top:1px solid #242426; margin: 20px 0;'>", unsafe_allow_html=True)

    st.markdown('<div style="font-size:11px; text-transform:uppercase; letter-spacing:0.1em; color:#888; margin-bottom:12px;">Pengaturan Model</div>', unsafe_allow_html=True)

    model_choice = st.selectbox("Model Sentimen", ["RoBERTa", "IndoBERT"], key="model_sel")
    show_preprocessing = st.toggle("Tampilkan Preprocessing", value=False)

    st.markdown("<hr style='border:none; border-top:1px solid #242426; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:11px; color:#666; line-height:1.8;">
        <div style="color:#888; font-weight:600; margin-bottom:6px; font-size:11px; text-transform:uppercase; letter-spacing:0.1em;">Pipeline</div>
        ① Web Crawling<br>
        ② Preprocessing<br>
        ③ NLP (Stemming)<br>
        ④ Sentiment Labeling<br>
        ⑤ ML Classification
    </div>
    """, unsafe_allow_html=True)


# PAGE: Dashboard
if "Dashboard" in page:
    st.markdown("""
    <div style="padding: 40px 0 20px;">
        <div class="hero-title">
            News Sentiment Analysis <span class="hero-accent">Argo Bromo</span>
        </div>
        <div class="hero-sub">
            Analisis sentimen berbasis transformer terhadap komentar YouTube dari TVONE, KOMPAS, dan METROTV
            mengenai kecelakaan Kereta Api Argo Bromo.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Stats overview
    col2, col3, col4 = st.columns(3)  
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Labeling Pre-trained</div>
            <div class="metric-value" style="color:#3EC97A;">IndoBERT & RoBERTa</div>
        </div>""", unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Model ML</div>
            <div class="metric-value" style="color:#3EC97A;">RF & XGBoost</div>
        </div>""", unsafe_allow_html=True)
        
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Sentimen Dominan</div>
            <div class="metric-value" style="color:#E85454;">Negatif</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)
    
    # Channel info
    st.markdown('<div class="section-title">Channel YouTube</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    channels = [
        (
            "TVONE",
            "#E63946",
            "tag-tvone",
            "qHr-ky9Iwik",
            "https://youtu.be/qHr-ky9Iwik?si=n_8RgtEXaNOuyo-R"
        ),
        (
            "KOMPAS",
            "#2196F3",
            "tag-kompas",
            "-lvgdiR6z1g",
            "https://www.youtube.com/watch?v=-lvgdiR6z1g"
        ),
        (
            "METROTV",
            "#4CAF50",
            "tag-metrotv",
            "5EHTgRAyyMw",
            "https://youtu.be/5EHTgRAyyMw?si=bTVzlnY3NgC9Q8Y-"
        ),
    ]

    for col, (name, color, tag_cls, vid_id, url) in zip([col1, col2, col3], channels):
        with col:
    
            st.markdown(f"""
                <div class="metric-card" style="border-left: 3px solid {color};">
                    <span class="channel-tag {tag_cls}">{name}</span>
                </div>
                """, unsafe_allow_html=True)
            st.link_button(
                "▶ Buka Video",
                url,
                use_container_width=True
            )
    
    st.markdown('<br>', unsafe_allow_html=True)

# PAGE: Analisis Sentimen (Single Input)
elif "Analisis Sentimen" in page:
    st.markdown("""
    <div style="padding: 24px 0 8px;">
        <div class="hero-title" style="font-size:40px;">Analisis <span class="hero-accent">Sentimen</span></div>
        <div class="hero-sub">Input komentar YouTube untuk dianalisis sentimennya secara real-time.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Quick examples
    st.markdown('<div style="font-size:12px; color:#888; margin-bottom:8px; text-transform:uppercase; letter-spacing:0.08em;">Contoh Komentar Cepat</div>', unsafe_allow_html=True)
    ex_cols = st.columns(3)
    examples = [
        ("Negatif", "Ini sungguh tragedi yang sangat menyedihkan. Pemerintah harus segera bertindak dan memperbaiki sistem keselamatan kereta api di Indonesia!"),
        ("Netral", "Saya melihat laporan ini dari Metro TV, tampaknya ada masalah teknis pada sistem pengereman. Perlu investigasi lebih lanjut."),
        ("Positif", "Terima kasih kepada seluruh tim penyelamat yang sudah bekerja keras membantu korban. Semoga para korban cepat sembuh."),
    ]
    for col, (label, text) in zip(ex_cols, examples):
        with col:
            if st.button(label, use_container_width=True):
                st.session_state['input_text'] = text

    st.markdown('<br>', unsafe_allow_html=True)

    # Main input
    user_input = st.text_area(
        "Masukkan komentar YouTube:",
        value=st.session_state.get('input_text', ''),
        height=130,
        placeholder="Tulis atau tempel komentar YouTube di sini...",
        label_visibility="collapsed",
        key="main_input"
    )

    col_info, col_btn = st.columns([3, 1])
    with col_info:
        if user_input:
            word_count = len(user_input.split())
            char_count = len(user_input)
            st.markdown(f'<div style="font-size:12px; color:#666; margin-top:6px;">{word_count} kata · {char_count} karakter</div>', unsafe_allow_html=True)
    with col_btn:
        analyze_btn = st.button("Analisis →", use_container_width=True)

    # Show preprocessing
    if show_preprocessing and user_input:
    
        prep = preprocess(user_input)
    
        with st.expander("🔧 Hasil Preprocessing"):
    
            st.markdown("""
            <div style="
                font-size:12px;
                color:#888;
                margin-bottom:18px;
                text-transform:uppercase;
                letter-spacing:0.08em;
            ">
            Tahapan preprocessing NLP
            </div>
            """, unsafe_allow_html=True)
    
            # ROW 1
            col1, col2 = st.columns(2)
    
            with col1:
                st.markdown("### 🧹 Clean Text")
                st.code(prep['clean_text'], language=None)
    
            with col2:
                st.markdown("### 🔄 Normalized Text")
                st.code(prep['normalized_text'], language=None)
    
            st.markdown("<br>", unsafe_allow_html=True)
    
            # ROW 2
            col3, col4 = st.columns(2)
    
            with col3:
                st.markdown("### ✂️ Tokenized")
                st.code(prep['tokens_word'], language=None)
    
            with col4:
                st.markdown("### 🚫 Remove Stopwords")
                st.code(prep['tokens_no_stop'], language=None)
    
            st.markdown("<br>", unsafe_allow_html=True)
    
            # ROW 3
            col5, col6 = st.columns(2)
    
            with col5:
                st.markdown("### 🌱 Stemming")
                st.code(prep['tokens_stem'], language=None)
    
            with col6:
                st.markdown("### ✅ Final Text")
                st.code(prep['final_text'], language=None)
    
            # Statistik tambahan
            st.markdown("<hr>", unsafe_allow_html=True)
    
            s1, s2, s3 = st.columns(3)
    
            with s1:
                st.metric(
                    "Jumlah Token",
                    len(prep['tokens_word'])
                )
    
            with s2:
                st.metric(
                    "Setelah Stopword",
                    len(prep['tokens_no_stop'])
                )
    
            with s3:
                st.metric(
                    "Panjang Final Text",
                    len(prep['final_text'].split())
                )
    # Analysis result
    if analyze_btn and user_input.strip():
        with st.spinner("Menganalisis sentimen..."):
            try:
                if model_choice == "RoBERTa":
                    clf = load_roberta()
                else:
                    clf = load_indobert()

                result = analyze_sentiment(
                    user_input, model_choice,
                    load_roberta() if model_choice == "RoBERTa" else None,
                    load_indobert() if model_choice == "IndoBERT" else None
                )

                label = result['label']
                score = result['score']
                scores = result['scores']

                color_map = {'positive': '#3EC97A', 'neutral': '#E8C547', 'negative': '#E85454'}
                icon_map  = {'positive': '😊', 'neutral': '😐', 'negative': '😔'}
                label_id  = {'positive': 'Positif', 'neutral': 'Netral', 'negative': 'Negatif'}
                color = color_map.get(label, '#888')
                icon  = icon_map.get(label, '🤔')

                st.markdown(f"""
                <div class="result-card" style="border-left: 4px solid {color};">
                    <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.1em; color:#888;">Hasil Analisis · {model_choice}</div>
                    <div style="display:flex; align-items:center; gap:16px; margin: 12px 0;">
                        <div style="font-size:48px;">{icon}</div>
                        <div>
                            <div style="font-family:'Instrument Serif',serif; font-size:36px; color:{color}; line-height:1;">
                                {label_id.get(label, label.capitalize())}
                            </div>
                            <div style="font-size:13px; color:#888; margin-top:4px;">Konfiden: {score*100:.1f}%</div>
                        </div>
                    </div>
                    <hr style="border:none; border-top:1px solid #242426; margin: 16px 0;">
                    <div style="font-size:12px; text-transform:uppercase; letter-spacing:0.08em; color:#888; margin-bottom:12px;">Skor Per Kelas</div>
                """, unsafe_allow_html=True)

                for sent_key, sent_label in [('positive','Positif'), ('neutral','Netral'), ('negative','Negatif')]:
                    s = scores.get(sent_key, 0.0) * 100
                    c = color_map[sent_key]
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px;">
                            <span style="color:#C0BFBA;">{sent_label}</span>
                            <span style="color:{c}; font-weight:600;">{s:.1f}%</span>
                        </div>
                        <div class="score-bar-container">
                            <div class="score-bar" style="width:{s}%; background:{c};"></div>
                        </div>
                    </div>""", unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {e}\n\nPastikan model terinstal: `pip install transformers torch`")

    elif analyze_btn:
        st.warning("Silakan masukkan teks terlebih dahulu.")


# PAGE: Statistik 
elif "Statistik" in page:
    st.markdown("""
    <div style="padding: 24px 0 8px;">
        <div class="hero-title" style="font-size:40px;">Statistik <span class="hero-accent">&amp; EDA</span></div>
        <div class="hero-sub">Eksplorasi distribusi dan pola data komentar YouTube.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.info("Halaman ini menampilkan visualisasi berbasis data crawling yang sudah dijalankan. Upload file CSV hasil crawling untuk melihat statistik real.")

    uploaded = st.file_uploader("Upload CSV hasil analisis (opsional)", type=["csv"])

    if uploaded:
        import io
        df = pd.read_csv(io.BytesIO(uploaded.getvalue()))
        st.success(f"Dataset loaded: {len(df):,} baris, {len(df.columns)} kolom")

        if 'sentiment' in df.columns and 'Source' in df.columns:
            st.markdown('<div class="section-title">Distribusi Sentimen</div>', unsafe_allow_html=True)

            try:
                import matplotlib.pyplot as plt
                import matplotlib
                matplotlib.rcParams.update({
                    'figure.facecolor': '#161618',
                    'axes.facecolor': '#161618',
                    'text.color': '#F0EFEA',
                    'axes.labelcolor': '#888887',
                    'xtick.color': '#888887',
                    'ytick.color': '#888887',
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                    'axes.spines.left': False,
                    'axes.spines.bottom': True,
                    'axes.edgecolor': '#242426',
                    'grid.color': '#242426',
                    'axes.grid': True,
                })

                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                fig.patch.set_facecolor('#161618')

                COLORS = {'positive':'#3EC97A','neutral':'#E8C547','negative':'#E85454'}
                sent_counts = df['sentiment'].value_counts()
                sent_order = [s for s in ['positive','neutral','negative'] if s in sent_counts.index]
                colors = [COLORS.get(s,'#888') for s in sent_order]

                axes[0].bar(sent_order, [sent_counts.get(s,0) for s in sent_order],
                            color=colors, edgecolor='#0D0D0E', linewidth=1.5, width=0.5)
                axes[0].set_title('Distribusi Sentimen', color='#F0EFEA', fontsize=13, pad=12)
                axes[0].set_ylabel('Jumlah', color='#888')

                if 'Source' in df.columns:
                    ct = df.groupby(['Source','sentiment']).size().unstack(fill_value=0)
                    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
                    bottom = np.zeros(len(ct_pct))
                    for sent in sent_order:
                        if sent in ct_pct.columns:
                            axes[1].bar(ct_pct.index, ct_pct[sent].values,
                                        bottom=bottom, label=sent.capitalize(),
                                        color=COLORS.get(sent,'#888'), alpha=0.85, edgecolor='#0D0D0E')
                            bottom += ct_pct[sent].values
                    axes[1].set_title('Sentimen per Channel (%)', color='#F0EFEA', fontsize=13, pad=12)
                    axes[1].legend(fontsize=9, labelcolor='#C0BFBA', facecolor='#161618', edgecolor='#242426')

                plt.tight_layout()
                st.pyplot(fig)
            except ImportError:
                st.warning("Install matplotlib untuk visualisasi: `pip install matplotlib`")

        # Raw data table
        st.markdown('<div class="section-title">Preview Data</div>', unsafe_allow_html=True)
        st.dataframe(
            df.head(50),
            use_container_width=True,
            height=300
        )

    else:
        # Sample/demo charts
        st.markdown('<div class="section-title">Demo Visualisasi (Data Sampel)</div>', unsafe_allow_html=True)

        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.rcParams.update({
                'figure.facecolor': '#161618', 'axes.facecolor': '#161618',
                'text.color': '#F0EFEA', 'axes.labelcolor': '#888887',
                'xtick.color': '#888887', 'ytick.color': '#888887',
                'axes.spines.top': False, 'axes.spines.right': False,
                'axes.edgecolor': '#242426', 'grid.color': '#1E1E20',
                'axes.grid': True, 'grid.alpha': 0.5,
            })

            # Simulated data
            np.random.seed(42)
            n = 3200
            channels = np.random.choice(['TVONE','KOMPAS','METROTV'], n, p=[0.35,0.3,0.35])
            sentiments = np.random.choice(['positive','neutral','negative'], n, p=[0.22,0.31,0.47])
            df_demo = pd.DataFrame({'Source': channels, 'sentiment': sentiments,
                                    'word_count': np.random.poisson(15, n)})

            COLORS = {'positive':'#3EC97A','neutral':'#E8C547','negative':'#E85454'}
            CHANNEL_COLORS = {'TVONE':'#E63946','KOMPAS':'#2196F3','METROTV':'#4CAF50'}

            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            fig.patch.set_facecolor('#161618')

            # Chart 1: Sentiment distribution
            sent_counts = df_demo['sentiment'].value_counts().reindex(['positive','neutral','negative'], fill_value=0)
            bars = axes[0].bar(sent_counts.index, sent_counts.values,
                               color=[COLORS[s] for s in sent_counts.index],
                               edgecolor='#0D0D0E', linewidth=1.5, width=0.5)
            for bar, val in zip(bars, sent_counts.values):
                pct = val / n * 100
                axes[0].text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                             f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold', color='#C0BFBA')
            axes[0].set_title('Distribusi Sentimen', color='#F0EFEA', fontsize=13, pad=12)

            # Chart 2: Per channel stacked
            ct = df_demo.groupby(['Source','sentiment']).size().unstack(fill_value=0)
            ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
            bottom = np.zeros(len(ct_pct))
            for sent in ['positive','neutral','negative']:
                if sent in ct_pct.columns:
                    axes[1].bar(ct_pct.index, ct_pct[sent].values,
                                bottom=bottom, label=sent.capitalize(),
                                color=COLORS[sent], alpha=0.85, edgecolor='#0D0D0E', linewidth=0.5)
                    for i,(v,b) in enumerate(zip(ct_pct[sent].values, bottom)):
                        if v > 6:
                            axes[1].text(i, b+v/2, f'{v:.0f}%', ha='center', va='center',
                                         fontsize=9, fontweight='bold', color='#0D0D0E')
                    bottom += ct_pct[sent].values
            axes[1].set_title('Sentimen per Channel (%)', color='#F0EFEA', fontsize=13, pad=12)
            axes[1].legend(fontsize=9, labelcolor='#C0BFBA', facecolor='#161618', edgecolor='#242426')

            # Chart 3: Word count distribution
            for src, color in CHANNEL_COLORS.items():
                data = df_demo[df_demo['Source']==src]['word_count']
                axes[2].hist(data, bins=20, alpha=0.6, label=src, color=color, edgecolor='none')
            axes[2].set_title('Distribusi Jumlah Kata', color='#F0EFEA', fontsize=13, pad=12)
            axes[2].set_xlabel('Jumlah Kata', color='#888')
            axes[2].legend(fontsize=9, labelcolor='#C0BFBA', facecolor='#161618', edgecolor='#242426')

            plt.tight_layout()
            st.pyplot(fig)
            st.caption("*Data simulasi untuk demo. Upload CSV hasil crawling untuk data asli.*")

        except ImportError:
            st.warning("Install matplotlib: `pip install matplotlib`")


# PAGE: Batch Prediksi

elif "Batch" in page:
    st.markdown("""
    <div style="padding: 24px 0 8px;">
        <div class="hero-title" style="font-size:40px;">Batch <span class="hero-accent">Prediksi</span></div>
        <div class="hero-sub">Analisis sentimen untuk banyak komentar sekaligus.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Input Manual", "Upload CSV"])

    with tab1:
        bulk_text = st.text_area(
            "Masukkan komentar (satu baris = satu komentar):",
            height=200,
            placeholder="Komentar pertama...\nKomentar kedua...\nKomentar ketiga...",
            label_visibility="collapsed"
        )

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            max_batch = st.slider("Maks komentar diproses", 5, 50, 10)
        with col_opt2:
            st.markdown('<br>', unsafe_allow_html=True)
            batch_btn = st.button("Jalankan Batch Analisis", use_container_width=True)

        if batch_btn and bulk_text.strip():
            lines = [l.strip() for l in bulk_text.strip().split('\n') if l.strip()][:max_batch]

            st.markdown(f'<div style="font-size:13px; color:#888; margin: 16px 0 8px;">Memproses {len(lines)} komentar...</div>', unsafe_allow_html=True)

            progress_bar = st.progress(0)
            results = []

            try:
                if model_choice == "RoBERTa":
                    clf_r = load_roberta()
                    clf_i = None
                else:
                    clf_r = None
                    clf_i = load_indobert()

                for i, line in enumerate(lines):
                    res = analyze_sentiment(line, model_choice, clf_r, clf_i)
                    results.append({
                        'Komentar': line[:80] + ('...' if len(line)>80 else ''),
                        'Sentimen': res['label'].capitalize(),
                        'Konfiden': f"{res['score']*100:.1f}%",
                        'Positif':  f"{res['scores'].get('positive',0)*100:.1f}%",
                        'Netral':   f"{res['scores'].get('neutral',0)*100:.1f}%",
                        'Negatif':  f"{res['scores'].get('negative',0)*100:.1f}%",
                    })
                    progress_bar.progress((i+1)/len(lines))

                df_results = pd.DataFrame(results)
                st.markdown('<div class="section-title">Hasil</div>', unsafe_allow_html=True)
                st.dataframe(df_results, use_container_width=True)

                # Summary
                sent_summary = df_results['Sentimen'].value_counts()
                cols = st.columns(3)
                icons = {'Positive':'😊','Neutral':'😐','Negative':'😔','Positif':'😊','Netral':'😐','Negatif':'😔'}
                for i, (label, count) in enumerate(sent_summary.items()):
                    pct = count / len(df_results) * 100
                    color = {'Positive':'#3EC97A','Neutral':'#E8C547','Negative':'#E85454',
                             'Positif':'#3EC97A','Netral':'#E8C547','Negatif':'#E85454'}.get(label,'#888')
                    with cols[i % 3]:
                        st.markdown(f"""
                        <div class="metric-card" style="text-align:center;">
                            <div style="font-size:28px;">{icons.get(label,'🤔')}</div>
                            <div style="font-weight:600; color:{color}; font-size:16px;">{label}</div>
                            <div style="font-family:'Instrument Serif',serif; font-size:28px;">{count}</div>
                            <div style="color:#888; font-size:12px;">{pct:.1f}%</div>
                        </div>""", unsafe_allow_html=True)

                # Download
                csv_out = df_results.to_csv(index=False)
                st.download_button("Download Hasil CSV", csv_out,
                                   file_name="batch_sentiment_results.csv", mime="text/csv")

            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        uploaded_csv = st.file_uploader("Upload CSV dengan kolom 'Comment' atau 'comment'", type=["csv"])
        if uploaded_csv:
            import io
            df_up = pd.read_csv(io.BytesIO(uploaded_csv.getvalue()))
            comment_col = next((c for c in df_up.columns if c.lower() in ['comment','komentar','text','teks']), None)

            if comment_col:
                st.success(f"{len(df_up):,} baris ditemukan. Kolom teks: '{comment_col}'")
                n_process = st.slider("Jumlah baris diproses", 10, min(500, len(df_up)), min(50, len(df_up)))

                if st.button("Jalankan pada CSV", use_container_width=True):
                    sample = df_up[comment_col].dropna().head(n_process).tolist()
                    prog = st.progress(0)

                    try:
                        clf_r = load_roberta() if model_choice == "RoBERTa" else None
                        clf_i = load_indobert() if model_choice == "IndoBERT" else None
                        labels = []
                        for idx, text in enumerate(sample):
                            r = analyze_sentiment(str(text), model_choice, clf_r, clf_i)
                            labels.append(r['label'])
                            prog.progress((idx+1)/len(sample))

                        df_out = df_up.head(n_process).copy()
                        df_out['predicted_sentiment'] = labels
                        st.dataframe(df_out, use_container_width=True, height=300)
                        csv_dl = df_out.to_csv(index=False)
                        st.download_button("Download CSV", csv_dl, file_name="predicted_sentiments.csv", mime="text/csv")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Kolom 'Comment', 'comment', 'text', atau 'teks' tidak ditemukan. Pastikan CSV memiliki header yang sesuai.")


# PAGE: Tentang
elif "Tentang" in page:
    st.markdown("""
    <div style="padding: 24px 0 8px;">
        <div class="hero-title" style="font-size:40px;">Tentang <span class="hero-accent">Proyek</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    st.markdown("""
    <div class="metric-card" style="margin-bottom: 16px;">
        <div style="font-family:'Instrument Serif',serif; font-size:18px; margin-bottom:12px; color:#E8C547;"> Deskripsi</div>
        <div style="font-size:14px; color:#C0BFBA; line-height:1.8;">
            Proyek ini merupakan implementasi <strong style="color:#F0EFEA;">Web Mining dan Analisis Sentimen</strong> terhadap komentar YouTube 
            pada video pemberitaan kecelakaan Kereta Api Agro Bromo. Data dikumpulkan dari tiga channel berita besar 
            Indonesia: <strong style="color:#E63946;">TVONE</strong>, <strong style="color:#2196F3;">KOMPAS TV</strong>, 
            dan <strong style="color:#4CAF50;">Metro TV</strong>.
        </div>
    </div>

    <div class="metric-card" style="margin-bottom: 16px;">
        <div style="font-family:'Instrument Serif',serif; font-size:18px; margin-bottom:12px; color:#E8C547;">Tools</div>
        <div style="font-size:14px; color:#C0BFBA; line-height:2;">
            <strong style="color:#F0EFEA;">Data Collection:</strong> YouTube Data API v3<br>
            <strong style="color:#F0EFEA;">Preprocessing:</strong> Sastrawi Stemmer, NLTK, Custom Slang Dict (~200+ kata)<br>
            <strong style="color:#F0EFEA;">Sentiment Labeling:</strong> IndoBERT & RoBERTa (Indonesian fine-tuned)<br>
            <strong style="color:#F0EFEA;">ML Classification:</strong> TF-IDF + Random Forest + XGBoost<br>
            <strong style="color:#F0EFEA;">Topic Modelling:</strong> LDA (Gensim)<br>
            <strong style="color:#F0EFEA;">Web App:</strong> Streamlit
        </div>
    </div>

    <div class="metric-card" style="margin-bottom: 16px;">
        <div style="font-family:'Instrument Serif',serif; font-size:18px; margin-bottom:12px; color:#E8C547;">Pipeline</div>
        <div style="font-size:14px; color:#C0BFBA; line-height:2;">
            [1] Install & Import → [2] Web Crawling (YouTube API) → [3] Preprocessing (clean, normalize, stem)<br>
            [4] EDA (distribusi, wordcloud, timeline) → [5] Sentiment Analysis (IndoBERT / RoBERTa)<br>
            [6] Web Mining (n-gram, topik) → [7] ML Classification (RF & XGBoost) → [8] Insight
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Cara Instalasi</div>', unsafe_allow_html=True)
    st.code("""# 1. Clone / siapkan file
pip install streamlit transformers torch sastrawi nltk
pip install scikit-learn xgboost gensim wordcloud
pip install google-api-python-client matplotlib seaborn

# 2. Jalankan
streamlit run app.py
""", language="bash")

    st.markdown('<div class="section-title">Catatan Model</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-card">
        <div style="font-size:13px; color:#C0BFBA; line-height:1.8;">
            • <strong style="color:#F0EFEA;">RoBERTa</strong>: <code>w11wo/indonesian-roberta-base-sentiment-classifier</code><br>
            • <strong style="color:#F0EFEA;">IndoBERT</strong>: <code>mdhugol/indonesia-bert-sentiment-classification</code><br>
        </div>
    </div>
    """, unsafe_allow_html=True)
