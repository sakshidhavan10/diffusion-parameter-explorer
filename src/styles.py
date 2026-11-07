CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=JetBrains+Mono&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.stApp { background: radial-gradient(1200px 600px at 10% -10%, #1a1040 0%, transparent 60%),
         radial-gradient(900px 500px at 100% 0%, #06222e 0%, transparent 55%), #07070d; }
h1.hero { font-size: 2.4rem; font-weight: 700; margin: 0;
  background: linear-gradient(90deg,#7c5cff,#00e5ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.sub { color:#9aa0c3; margin-bottom: 1rem; }
.card { background: rgba(255,255,255,0.04); border: 1px solid rgba(124,92,255,.25); border-radius: 16px;
  padding: 14px 16px; backdrop-filter: blur(10px); transition: all .25s ease; }
.card:hover { border-color:#00e5ff; box-shadow: 0 0 24px rgba(0,229,255,.25); transform: translateY(-2px); }
.card .k { color:#9aa0c3; font-size:.72rem; letter-spacing:.08em; text-transform:uppercase; }
.card .v { font-size:1.5rem; font-weight:700; color:#fff; font-family:'JetBrains Mono',monospace; }
.badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.72rem; margin:4px 6px 0 0;
  border:1px solid; font-family:'JetBrains Mono',monospace; }
.badge.ok { color:#00e5a0; border-color:#00e5a0; background:rgba(0,229,160,.08); }
.badge.bad { color:#ff4d6d; border-color:#ff4d6d; background:rgba(255,77,109,.1); }
.badge.info { color:#00e5ff; border-color:#00e5ff; background:rgba(0,229,255,.08); }
.badge.vio { color:#b39dff; border-color:#7c5cff; background:rgba(124,92,255,.12); }
.stButton > button, .stDownloadButton > button { background: linear-gradient(90deg,#7c5cff,#00b8ff); color:#fff; border:0;
  border-radius:12px; font-weight:600; transition: all .2s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { box-shadow:0 0 22px rgba(124,92,255,.7); transform: translateY(-1px); }
.stTabs [data-baseweb="tab-list"] { gap: 6px; flex-wrap: wrap; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,.04); border-radius: 10px; padding: 6px 14px; }
.stTabs [aria-selected="true"] { background: rgba(124,92,255,.25); }
.stImage img { border-radius: 14px; border: 1px solid rgba(124,92,255,.3); }
@media (max-width: 768px) { h1.hero { font-size: 1.6rem; } .card .v { font-size: 1.1rem; } }
</style>
"""
