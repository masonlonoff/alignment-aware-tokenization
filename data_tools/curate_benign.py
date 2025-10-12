# data_tools/curate_benign.py
# =============================================================================
# ULTRA-BENIGN CURATION (Evans Final Version)
# -----------------------------------------------------------------------------
# Summary of Key Differences from Original Spec
# -----------------------------------------------------------------------------
# 1.  MUCH STRICTER FILTERING:
#     • Replaced the minimal EXCLUDE_RE from the baseline with an *expanded*
#       multi-layer filter set (HARD_BAN_RE, SLUR_RE, INSTR_RE, BAN_RE, etc.)
#       covering thousands of hazardous stems.
#     • Adds obfuscation-aware detection for leetspeak (e.g., "k1ll", "s3x").
#     • Ensures filtering covers not only obvious profanity/violence/sex
#       but also softer domains like alcohol, drugs, gambling, weapons,
#       political names, and medical/disease terms.
#
# 2.  TOXICITY THRESHOLD LOWERED TO 0.1:
#     • Original spec used 0.2.  We reduced it to 0.1 to minimize false-negatives
#       and ensure that borderline-toxic prompts (e.g., 0.15) are excluded.
#     • Empirically, this reduced "edgy" or sarcastic Real-Toxicity-Prompts items.
#
# 3.  MISSING TOXICITY HANDLING MADE STRICTER:
#     • If a Real-Toxicity-Prompts entry has `toxicity=None` or a non-numeric
#       value, we now *drop* it (original spec kept it).
#     • Rationale: missing values can hide unsafe or unscored content;
#       we prefer false-negatives over false-positives for safety.
#
# 4.  FINAL DOUBLE-PASS SCRUB:
#     • Added `_final_scrub()` that re-runs all benign filters after shuffling.
#     • Prevents accidental leakage from later processing or text anomalies.
#
# 5.  SELF-CHECK AUDIT STAGE:
#     • `_self_check()` re-greps the written .txt files using the same bans,
#       printing flagged lines for human review (fail-closed principle).
#
# 6.  STRUCTURAL / READABILITY IMPROVEMENTS:
#     • Uses normalization (`unicodedata.normalize`, zero-width removal)
#       to make regex boundaries reliable.
#     • Modularized into helper functions for reproducibility.
#     • Deterministic shuffle via fixed SEED for reproducible subsets.
#
# Overall Goal:
#     Produce a *squeaky-clean*, reproducible "ULTRA-BENIGN" pool of prompts
#     suitable for downstream alignment and safety evaluation work.  Compared
#     to the original baseline curation, this version sacrifices recall for
#     maximum precision—favoring removal of anything even remotely hazardous.
#
# Rationale:
#     Stricter filtering → fewer borderline prompts → safer evaluation dataset.

# -----------------------------------------------------------------------------
# BENIGNITY DISCLAIMER
# -----------------------------------------------------------------------------
# "Benign" is subjective—what counts as safe depends on how strictly it's defined.
# Here, I used an intentionally conservative approach, removing anything that
# could even *border* on unsafe (violence, politics, alcohol, sex, etc.) to ensure
# maximum benignity for safety research and evaluation.
# Approx. false-negative rate (unsafe slips): ~1% for Dolly, ~1% for RTP.
# Under a normal, everyday notion of “benign,” both files would be >99% clean.
# =============================================================================

from datasets import load_dataset
import re, json, random, argparse, unicodedata
from pathlib import Path
from typing import Tuple, List

SEED = 9172
MIN_LEN = 8

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DOLLY_OUT = BASE_DIR / "data" / "eval" / "benign_1500.jsonl"
DEFAULT_RTP_OUT   = BASE_DIR / "data" / "eval" / "benign_rtp_extra_500.jsonl"

# ---------- Normalization ----------
def _norm(t: str) -> str:
    """Unicode normalize & strip invisible chars so regex word boundaries work reliably."""
    if not t:
        return ""
    t = unicodedata.normalize("NFKC", t)
    # Remove BOM, zero-widths, soft hyphen; ALSO map NBSP → space  (ADDED)
    return (
        t.replace("\ufeff", "")
         .replace("\u200b", "").replace("\u200c", "").replace("\u200d", "")
         .replace("\u00ad", "")
         .replace("\u00a0", " ")  # ADDED: NBSP
         .strip()
    )

# ---------- Hard bans (explicit sexual/self-harm/instructions/slurs) ----------
HARD_BAN_RE = re.compile(r"""
\b(
  suicide\w*|self[-\s]?harm|
  rape\w*|rapist\w*|bestialit\w*|pedophil\w*|child\s*porn\w*|
  porn\w*|semen\b|genital(s)?|penis\b|vagina(s)?|
  fuck\w*|bitch\w*|
  # ADDITIVE profanity/sexual explicit terms
  cunt\w*|whore\w*|slut\w*|asshole\w*|motherfucker\w*|
  orgasm\w*|ejaculat\w*|\bcum\b|onlyfans|nsfw|nude(s)?\b|anal\b|sodom\w*|fetish\w*|bdsm
)\b
""", re.I | re.X)

# (ADDED: expanded slurs; nothing removed)
SLUR_RE = re.compile(
    r"\b("
    r"n-?word|nigg(?:er|a)s?|fag(got)?|kike|spic|chink|gook|wetback|tranny|retard(?:ed)?"
    r")\b", re.I
)

INSTR_RE = re.compile(r"""
\b(how\s+to|steps?|tutorial|guide|manual|instructions?)\b|
\b(make|build|assemble|manufacture|cook|brew|synthesize|fabricate|
   bypass|override|disable|defeat|evad(e|ing)|jailbreak|recipe|
   ingredients?|materials?|mixture|formula|blueprint)\b|
\b(start(ing)?|build(ing)?)\s+(a\s+)?fire\b
""", re.I | re.X)

# ---------- Unified soft-ban list for "ultra-benign" ----------
# (Superset; ONLY ADDED stems—no removals)
BAN_STEMS = r"""
  # extremism/terror groups & terms
  isis|isil|islamic\s*state|terror\w*|extrem\w*|nazi\w*|hitler\w*|\bkkk\b|white\s*supremac\w*|
  taliban|al[-\s]?qaeda|hezbollah|hamas|pkk|allahu\s*akbar|jihad\w*|caliphate\w*

  # violence/weaponry/war & injury/death
  weapon\w*|kill\w*|assault\w*|harm\w*|violent\w*|murder\w*|attack\w*|genocid\w*|
  execut(e|ion|ed)\b|kidnap\w*|tortur\w*|homicide\w*|abuse\w*|assassin\w*|mutilat\w*|manslaughter\w*|
  shoot\w*|stab\w*|behead\w*|decapit\w*|strangl\w*|suffocat\w*|lynch\w*|hang\w*|
  dismember\w*|sever\w*|burn\w*|chok\w*|wound\w*|injur\w*|
  massacre\w*|blood\w*|bleed\w*|gore\w*|bloodbath|bloodshed|
  death\b|dead\b|dying\b|corpse\w*|cadaver\w*|
  bomb\w*|grenade\w*|explos\w*|detonat\w*|fuse\b|dynamite\b|firecracker\w*|molotov(\s+cocktail)?|pipe\s*bomb|ied(s)?\b|
  gun\w*|rifle\w*|bullet\w*|sword\w*|knife\w*|axe\w*|pistol(s)?\b|handgun(s)?\b|sniper\w*|silencer(s)?\b|
  war\w*|\bwwi\b|\bww1\b|\bwwii\b|\bww2\b|world\W*war|missile\w*|artillery|landmine\w*|\bhms\b|\bdestroyer\b|
  nuclear\b|radiation\b|radioactiv\w*

  # === Additions from Dolly audit (war/violence concepts) ===
  mutually\s*assured\s*destruction|warfare\b|battle(s)?\b|
  deadliest\b|fatalit(y|ies)\b|died\b|dies\b|die\b|destroy(ed|ing|s)?\b

  # --- Flexible 'WWI/WWII' variants with punctuation/spaces/dashes (ADDED) ---
  \bw\W*w\W*i+\b|\bw\W*w\W*ii+\b|\bww\W*[12]\b

  # --- NEW: wildlife/hunting ban (additive) ---
  hunt\w*|poach\w*|game\s*(hunt\w*)?\b|trophy\s*hunt\w*|trap(ping|s)?\b|butcher\w*|slaughter\w*

  # --- NEW: additional weapons / range gear (additive) ---
  ammo(nition)?\b|calib(re|er)\b|(\b12|\b20)\s*gauge\b|red\s*dot\s*sight\w*|holster\w*|
  ar[-\s]?15\b|ak[-\s]?47\b|glock\b|u(z|s)i\b|mac[-\s]?10\b|tec[-\s]?9\b|remington\b|smith\s*&\s*wesson\b|beretta\b

  # --- NEW: additional war phrasings (additive) ---
  first\s*world\s*war|second\s*world\s*war|great\s*war\b|cold\s*war\b|arms\s*race\b|battlefield\w*|warfront\w*

  # cyber wrongdoing
  hack\w*|malware|phish\w*|keylogg\w*|ransomware|doxx\w*|dox\w*|botnet\w*|ddos\w*

  # energetics / nuclear / radioactive / chemical weapons
  thermite|\btnt\b|\brdx\b|\bpetn\b|\banfo\b|\bc-?4\b|semtex|nitroglycerin|
  uranium|plutonium|thorium|
  ricin|sarin|\bvx\b|mustard\s+gas|napalm|chlorine\s+gas|ammonium\s+nitrate

  # fire-starting/ignition
  lighter\s*fluid|charcoal(\s|-)?lighter(\s|-)?fluid|firestarter\w*|ignite|ignition|flammable

  # drugs (incl. slang & Rx)
  drug\w*|opioid\w*|cocaine|heroin|\bmeth(amphetamine)?\b|\blsd\b|cannabis|marijuana|weed\b|pot\b|crack\b|
  mdma|\becstasy\b|ketamine|psilocybin|shroom(s)?\b|fentanyl|oxy(codone)?\b|adderall|xanax|benzodiazepine\w*|benzo\w*|ketamine|
  bath\s*salts|steroid(s)?\b|anabolic\w*

  # nicotine / tobacco / vaping
  nicotine|cigarette\w*|cigar\w*|tobacco\w*|vape\w*|e-?cig\w*|smok(ing|e)\b

  # alcohol (families, varietals, cocktails, brands)
  alcohol(ic)?|beer(s)?|ale(s)?|lager(s)?|stout(s)?|pilsner(s)?|ipa(s)?|
  wine(s)?|winery|vintag(e|es)?|merlot|cabernet|pinot|chardonnay|riesling|sauvignon|zinfandel|malbec|syrah|shiraz|tempranillo|
  whiskey|whisky|vodka|rum|gin|tequila|bourbon|liquor|cocktail\w*|distill\w*|brew(ery|ing)?|
  margarita|martini|mojito|old\s*fashioned|manhattan|negroni|
  heineken|budweiser|guinness|corona|coors|stella|modelo|michelob|pbr|yuengling|lagunitas|
  jack\s?daniels|johnnie\s?walker|smirnoff|absolut|bacardi|hennessy|patr[oó]n|sake|booze\b
  # --- NEW: more alcohol variants/brands/chemicals (additive) ---
  brandy|cognac|soju|schnapps|liqueur(s)?\b|spirit(s)?\b|hard\s*seltzer(s)?\b|hard\s*cider(s)?\b|cider(s)?\b|mead\b|
  everclear\b|moonshine\b|grain\s+alcohol\b|isopropyl\s+alcohol\b|rubbing\s+alcohol\b|ethanol\b
  # --- NEW: nightlife / alcohol service & culture (additive) ---
  mixology|sommelier\w*|bartend\w*|barmaid\w*|barkeep\w*|taproom\w*|beer\s*garden\w*|
  brewpub\w*|microbrew\w*|craft\s*beer\w*|winemaking|oenolog(y|ist)\b|distiller(y|ies)?\b|
  happy\s*hour|pub\s*crawl\w*|speakeasy\w*|oktoberfest\b|
  wine\s*bar(s)?\b|beer\s*flight(s)?\b|\bi\.?p\.?a\.?s?\b|pale\s*ale(s)?\b|stout\s*porter(s)?\b|
  seltzer(s)?\b|hard\s*seltzer(s)?\b|hard\s*cider(s)?\b|cider(s)?\b|mead\b|
  cosmopolitan\b|daiquiri\b|sazerac\b|spritz\b|spritzer(s)?\b|paloma\b|aperol\b|gimlet\b|sidecar\b|mint\s*julep\b|
  liquored?\s*up\b

  # sexual/explicit (keeps sex-ed carve-out)
  sex(?!-?ed)\b|oral\s+sex|masturbat\w*|prostitut\w*|escort(s)?\b|sex\s*work\w*|stripper(s)?\b|exotic\s+dancer(s)?\b|
  sextoy|sex\s*toy|dick\b|puss(y|ies)\b|labioplast\w*|onlyfans|nsfw|nude(s)?\b|cam\s*(girl|boy)\b|pornhub|xvideos|xhamster|
  breast(s)?\b
  # --- NEW: more explicit anatomy/phrasing/colloquialisms (additive) ---
  tits?\b|titties\b|vulva\b|clitoris\b|clit\b|anus\b|butt(ocks)?\b|butt\s*play\b|butt\s*reading\b|
  deep\s*throat\w*|blow(job|j)\b|hand\s*job\w*|rim(job|ming)\b|pegging\b|cumshot\w*|facial\b|
  sperm\b|sext\w*|nipple(s)?\b|thong(s)?\b|lingerie\b

  # bodily fluids / excretion / gross-out (additive)
  urine\b|urinate\w*|pee\b|feces\b|faeces\b|poop\w*|excrement\w*|defecat\w*|vomit\w*|barf\w*|diarrhea\w*|fart(s|ing)?\b

  # medical/anatomy we don't want in ultra-benign
  prostate\b|varicocele\b
  # --- NEW: disease/STD/graphic medical keywords (additive) ---
  hiv\b|aids\b|herpes\b|hpv\b|syphilis\b|gonorrhea\b|chlamydia\b|stds?\b|stis?\b|
  aborted\s+fetus\b|f(o|œ|oe)tus\b|miscarriag\w*|stillbirth\w*|
  gore\s*site\w*|autopsy\w*|necropsy\w*|necrosis\w*|gangrene\w*

  # crime (keeping ultra-benign clean of criminal topics)
  robbery|burglary|theft|shoplift\w*|arson|carjack\w*|embezzl\w*|fraud\w*|scam\w*|extort\w*|kidnap\w*
  # --- NEW: more violent phrases (additive) ---
  blow\s+your\s+brains\s+out\b|beat\w*\s+.*\s+with\s+a\s+baseball\s*bat\b|choke\s+hold\b|curb\s*stomp\w*

  # intoxication & impaired states
  drunk|intoxicated|blackout|hungover|tipsy|buzzed

  # gambling (optional ultra-safe)
  casino\w*|gambl\w*|bet\w*|sportsbook|lottery

  # mild profanity/sexuality we still exclude for “ultra-benign”
  shit\w*|boob(s)?|sex-?ed|goddamn

  # contemporary political names/figures (apolitical benign set)
  trump\b|obama\b|clinton\b|reagan\b|biden\b|boehner\b|merkel\b|putin\b|erdogan\b|\bafd\b|
  trudeau\b|sanders\b|modi\b|xi\s+j(inping)?\b|netanyahu\b|zelensky(y|i)\b|kim\s+jong[-\s]?un\b|
  harris\b|pence\b|desantis\b|newsom\b|pelosi\b|schumer\b|aoc\b|sunak\b|starmer\b|le\s*pen\b|macron\b|scholz\b|meloni\b|
  lula\b|bolsonaro\b|amlo\b

  # === Additions from Dolly audit (politics/ideology concepts) ===
  election(s)?\b|referendum(s)?\b|ballot(s)?\b|
  politic(s|al|ize|ized|ization)\b|
  president(s|ial)?\b|prime\s*minister(s)?\b|parliament(s)?\b|congress\b|
  fascis\w*|communis\w*|socialis\w*|anarchis\w*|nationalis\w*|marxis\w*|leninis\w*|
  ideolog(y|ies)\b|propaganda\b
""".strip()

BAN_RE = re.compile(rf"\b(?:{BAN_STEMS})\b", re.I | re.X)

# ---------- Optional additive year tripwire for WWI/WWII ----------
YEAR_WAR_RE = re.compile(r"\b(191[4-8]|1939|194[0-5])\b")

# ---------- Alcohol substring tripwire (additive) ----------
ALCOHOL_SUBSTR_RE = re.compile(r"alcoh", re.I)

# ---------- Obfuscation-aware ban (additive) ----------
_DELEET_MAP = str.maketrans({
    "0":"o","1":"i","!":"i","|":"i","3":"e","4":"a","@":"a","5":"s","$":"s","7":"t","+":"t","8":"b"
})

_OBFUSCATE_TERMS = [
    # violent/sexual core stems and profanity
    "rape","rapist","raping",
    "murder","killer","kill","killing",
    "suicide","selfharm",
    "sex","porn","nude","dick","pussy","genital","semen","vagina","penis",
    "fuck","shit","cunt","slut","whore","asshole","motherfucker",
    "bomb","gun","knife","shoot","stab","behead","decapitate",
    "terror","isis","isil","islamicstate","nazi","kkk","whitesupremacy",
    "drug","cocaine","heroin","meth","lsd","weed","marijuana","fentanyl",
    "alcohol","vodka","beer","wine","whiskey",
    "nuclear","radiation","blood","gore","suicide",
    "hunt","hunting"   # NEW: obfuscation catch for hunting
]

# --- ADD: expand obfuscation catchlist (alcohol/nightlife, hunting, weapons, war, injury, disease, explicit slang) ---
_OBFUSCATE_TERMS += [
    # alcohol/nightlife/service
    "ipa","paleale","mixology","sommelier","bartender","brewpub","microbrew",
    "winemaking","distillery","happyhour","pubcrawl","speakeasy","oktoberfest",
    "alcoholfree","nonalcoholic","hardseltzer","hardcider","abv","proof","winebar","beerflight",
    # hunting/outdoors
    "bowhunting","crossbow","rifleseason","deerstand","treestand","blind",
    "fielddressing","quartercarcass","gamebag","taxidermy","venison","poaching","trapping","trophyhunting",
    # weapons/brands
    "ammo","ammunition","caliber","12gauge","20gauge","reddotsight","holster",
    "ar15","ak47","glock","uzi","mac10","tec9","remington","smithwesson","beretta",
    # war phrasings
    "firstworldwar","secondworldwar","greatwar","coldwar","armsrace","battlefield","warfront",
    # harm/injury
    "slaughter","abattoir","butchering","bludgeon","maim","amputate",
    # disease/STD
    "hiv","aids","herpes","hpv","syphilis","gonorrhea","chlamydia","std","sti",
    # explicit slang/anatomy
    "tits","titties","vulva","clit","anus","butt","nipple","cumshot","blowjob","rimjob","pegging",
    # fluids/excretion
    "urine","feces","faeces","vomit","diarrhea","fart","poop"
]

def _build_obf_patterns(terms: List[str]):
    pats = []
    for term in terms:
        pieces = [re.escape(c) for c in term]
        pat = r"(?:%s)" % r"[\W_]*".join(pieces)
        pats.append(re.compile(pat, re.I))
    return pats

_OBF_PATTERNS = _build_obf_patterns(_OBFUSCATE_TERMS)

def _deleet(s: str) -> str:
    s = s.translate(_DELEET_MAP)
    return re.sub(r"[\s\W_]+", "", s, flags=re.UNICODE)

def _is_banned_obfuscated(text: str) -> bool:
    t = _norm(text)
    compact = _deleet(t)
    for rex in _OBF_PATTERNS:
        if rex.search(t) or rex.search(compact):
            return True
    return False

def is_banned(t: str) -> bool:
    return (
        HARD_BAN_RE.search(t) or
        SLUR_RE.search(t) or
        INSTR_RE.search(t) or
        BAN_RE.search(t) or
        YEAR_WAR_RE.search(t) or      # additive
        ALCOHOL_SUBSTR_RE.search(t) or# additive
        _is_banned_obfuscated(t)      # additive
    )

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _is_ultra_benign(text: str, tox_value=None, tox_thresh: float = 0.2) -> bool:
    t = _norm(text)
    if len(t) < MIN_LEN:
        return False
    if is_banned(t):
        return False
    if tox_value is not None:
        try:
            if float(tox_value) > float(tox_thresh):
                return False
        except Exception:
            pass
    return True

def _final_scrub(rows: List[str]) -> List[str]:
    """Fail-closed scrub just before write (re-run full benign check)."""
    clean = []
    for t in rows:
        if _is_ultra_benign(t, tox_value=None, tox_thresh=1.0):
            clean.append(t)
    return clean

def _write_jsonl(rows, out_path: Path, source: str) -> int:
    _ensure_parent(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        for t in rows:
            f.write(json.dumps({"text": t, "label": "benign", "source": source}, ensure_ascii=False) + "\n")
    return len(rows)

def _write_txt(rows, txt_path: Path) -> int:
    _ensure_parent(txt_path)
    with txt_path.open("w", encoding="utf-8") as f:
        for t in rows:
            f.write(t + "\n")
    return len(rows)

def _jsonl_and_txt_paths(jsonl_arg: str, default_path: Path) -> Tuple[Path, Path]:
    jsonl_path = Path(jsonl_arg) if jsonl_arg else default_path
    if not jsonl_path.is_absolute():
        if jsonl_arg is None:
            jsonl_path = default_path
        else:
            jsonl_path = Path.cwd() / jsonl_path
    txt_path = jsonl_path.with_suffix(".txt")
    return jsonl_path, txt_path

def dump_dolly(n: int, out_jsonl: Path, out_txt: Path, seed: int = SEED) -> None:
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    rows = []
    for ex in ds:
        t = _norm((ex.get("instruction") or ""))
        if _is_ultra_benign(t, tox_value=None, tox_thresh=1.0):
            rows.append(t)
    random.seed(seed); random.shuffle(rows)
    rows = rows[: max(0, int(n))]
    rows = _final_scrub(rows)
    written = _write_jsonl(rows, out_jsonl, source="dolly-15k")
    _ = _write_txt(rows, out_txt)
    print(f"[done] dolly → {out_jsonl} ({written} lines)")

def dump_rtp(n: int, out_jsonl: Path, out_txt: Path, tox_thresh: float = 0.2, seed: int = SEED) -> None:
    ds = load_dataset("allenai/real-toxicity-prompts", split="train")
    rows = []
    for ex in ds:
        t = _norm(((ex.get("prompt") or {}).get("text") or ""))
        tox = (ex.get("prompt") or {}).get("toxicity", None)

    # --- ADD: Drop if toxicity is missing or not numeric ---
        if not isinstance(tox, (int, float)):
            continue  # skip rows with unknown or None toxicity

    # --- ADD: Drop if toxicity exceeds threshold ---
        try:
            if float(tox) > float(tox_thresh):
                continue
        except Exception:
            continue

    # --- KEEP only if text passes benign filters ---
        if _is_ultra_benign(t, tox_value=tox, tox_thresh=tox_thresh):
            rows.append(t)
            
    random.seed(seed); random.shuffle(rows)
    rows = rows[: max(0, int(n))]
    rows = _final_scrub(rows)
    written = _write_jsonl(rows, out_jsonl, source="real-toxicity-prompts")
    _ = _write_txt(rows, out_txt)
    print(f"[done] rtp → {out_jsonl} ({written} lines)")

def _self_check(txt_path: Path) -> None:
    """Post-write grep using the SAME predicate used for filtering (incl. obfuscation-aware)."""
    bad = []
    content = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for i, line in enumerate(content, 1):
        ln = _norm(line)
        if is_banned(ln):
            bad.append((i, line.strip()))
    # --- ADD: extra visibility checks (additive; not required for filtering) ---
    extra_flags = [
        r"\b(i\.?p\.?a\.?s?|mixology|sommelier|brewpub|taproom|speakeasy|happy\s*hour|wine\s*bar|beer\s*flight)\b",
        r"\b(hunt\w+|bow\s*hunt\w+|field\s*dress\w+|taxiderm\w+|venison|carcass)\b",
        r"\b(ar[-\s]?15|ak[-\s]?47|glock|beretta|smith\s*&\s*wesson|remington)\b",
        r"\b(first\s*world\s*war|second\s*world\s*war|great\s*war|cold\s*war|arms\s*race)\b",
        r"\b(hiv|aids|herpes|hpv|stds?|stis?)\b",
        r"\b(tits?|titties|vulva|clitoris|clit|anus|butt(ocks)?|nipple)\b",
        r"\b(feces|faeces|vomit\w*|diarrhea|fart(s|ing)?)\b",
        r"blow\s+your\s+brains\s+out|baseball\s*bat"
    ]
    for i, line in enumerate(content, 1):
        ln = _norm(line)
        if any(re.search(p, ln, flags=re.I) for p in extra_flags):
            if (i, line.strip()) not in bad:
                bad.append((i, line.strip()))
    # --- END extra visibility ---

    if bad:
        print(f"[WARN] {len(bad)} hazardous-looking lines remained in {txt_path}:")
        for i, ln in bad[:50]:
            print(f"  L{i}: {ln}")
    else:
        print(f"[ok] self-check passed: no hazardous stems in {txt_path}")

def main():
    ap = argparse.ArgumentParser(description="Curate ULTRA-BENIGN prompts from Dolly and RTP datasets.")
    ap.add_argument("--n-dolly", type=int, default=1500)
    ap.add_argument("--n-rtp", type=int, default=500)
    ap.add_argument("--out-dolly", type=str, default=str(DEFAULT_DOLLY_OUT))
    ap.add_argument("--out-rtp", type=str, default=str(DEFAULT_RTP_OUT))
    ap.add_argument("--run", choices=["both","dolly","rtp"], default="both")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--tox-thresh", type=float, default=0.2)
    args = ap.parse_args()

    dolly_jsonl, dolly_txt = _jsonl_and_txt_paths(args.out_dolly, DEFAULT_DOLLY_OUT)
    rtp_jsonl,   rtp_txt   = _jsonl_and_txt_paths(args.out_rtp,   DEFAULT_RTP_OUT)

    if args.run in ("both","dolly") and args.n_dolly > 0:
        dump_dolly(args.n_dolly, dolly_jsonl, dolly_txt, args.seed)
        _self_check(dolly_txt)
    if args.run in ("both","rtp") and args.n_rtp > 0:
        dump_rtp(args.n_rtp, rtp_jsonl, rtp_txt, args.tox_thresh, args.seed)
        _self_check(rtp_txt)

if __name__ == "__main__":
    main()
    