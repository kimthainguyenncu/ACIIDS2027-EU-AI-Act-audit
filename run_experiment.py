"""Same System, Different Verdict – experiment runner (local version for VS Code).

Usage:
  python run_experiment.py --list-models     # show model IDs your keys can use
  python run_experiment.py --pilot           # 20 requests per provider
  python run_experiment.py --full            # all 2,112 requests per provider (resumable)
  python run_experiment.py --summary [--pilot]
API keys are read from the file .env in this folder (see .env.example).
"""
import os, sys, json, random, re, time, threading, datetime, argparse
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace"); sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(HERE, ".env"))
except ImportError:
    sys.exit("Thiếu thư viện. Chạy: python -m pip install -r requirements.txt")
import pandas as pd

CFG = json.load(open("config.json", encoding="utf-8"))
PROVIDERS = CFG["providers"]
DESC = pd.read_csv("descriptions_960.csv")
PROMPTS = json.load(open("prompts.json", encoding="utf-8"))
SYSTEM = PROMPTS["system"]
assert "EXCERPTS FROM REGULATION" in SYSTEM and len(DESC) == 960
LABELS = {"prohibited", "high_risk", "transparency_only", "minimal"}
KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "gemini": "GEMINI_API_KEY", "xai": "XAI_API_KEY",
        "groq": "GROQ_API_KEY", "nvidia": "NVIDIA_API_KEY", "mistral": "MISTRAL_API_KEY", "deepseek": "DEEPSEEK_API_KEY", "openrouter": "OPENROUTER_API_KEY", "ollama": None}
# OpenAI-compatible endpoints
BASE_URLS = {"xai": "https://api.x.ai/v1", "groq": "https://api.groq.com/openai/v1", "nvidia": "https://integrate.api.nvidia.com/v1",
             "mistral": "https://api.mistral.ai/v1", "deepseek": "https://api.deepseek.com", "openrouter": "https://openrouter.ai/api/v1",
             "ollama": "http://localhost:11434/v1"}
def API(p): return PROVIDERS[p].get("api", p)   # several entries may share one API (e.g. two Gemini models)
os.makedirs("results", exist_ok=True)

# ---------- job list (identical for every provider) ----------
rng = random.Random(CFG["seed"])
BASE = [(r.desc_id, arm) for r in DESC.itertuples() for arm in ("direct", "checker")]
REP = set(rng.sample(BASE, int(len(BASE) * CFG["repeat_share"])))
JOBS = [f"{d}|{a}|r0" for d, a in BASE] + [f"{d}|{a}|r1" for d, a in sorted(REP)]
rng.shuffle(JOBS)
DTEXT = dict(zip(DESC.desc_id, DESC.description))

def user_msg(job):
    d, arm, _ = job.split("|")
    return PROMPTS[arm].replace("<<DESCRIPTION>>", DTEXT[d])

# ---------- provider clients ----------
DROP = {p: set() for p in PROVIDERS}
STOP = {p: False for p in PROVIDERS}   # set when a daily quota is exhausted

def key_for(p):
    if KEYS[API(p)] is None: return "local"
    k = os.environ.get(KEYS[API(p)], "").strip()
    if not k:
        raise SystemExit(f"Chưa có key cho {p}: hãy điền {KEYS[API(p)]}= trong file .env")
    return k

def make_client(p):
    key = key_for(p); a = API(p)
    if a == "ollama": return None   # local server, called directly in call()
    if a in BASE_URLS:
        from openai import OpenAI
        hdr = {"X-Title": "ACIIDS 2027 conjoint audit"} if a == "openrouter" else None
        return OpenAI(api_key=key, base_url=BASE_URLS[a], max_retries=0, timeout=600, default_headers=hdr)
    if a == "anthropic":
        import anthropic; return anthropic.Anthropic(api_key=key, max_retries=0)
    if a == "openai":
        from openai import OpenAI; return OpenAI(api_key=key, max_retries=0)
    if a == "xai":
        from openai import OpenAI; return OpenAI(api_key=key, base_url="https://api.x.ai/v1", max_retries=0)
    if a == "gemini":
        from google import genai; return genai.Client(api_key=key)

def call(p, client, model, user):
    a = API(p)
    if a == "anthropic":
        kw = dict(model=model, max_tokens=1500, messages=[{"role": "user", "content": user}],
                  system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}])
        if "temperature" not in DROP[p]: kw["temperature"] = 0
        r = client.messages.create(**kw); u = r.usage
        return ("".join(b.text for b in r.content if getattr(b, "type", "") == "text"), r.model,
                {"in": u.input_tokens, "out": u.output_tokens, "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_write": getattr(u, "cache_creation_input_tokens", 0) or 0})
    if a == "ollama":   # local model: native API so the context window can be set (default is too small for the 7k-token prompt)
        import urllib.request
        body = {"model": model, "stream": False, "format": "json", "keep_alive": "30m",
                "options": {"num_ctx": 16384, **({} if "temperature" in DROP[p] else {"temperature": 0})},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]}
        req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=900) as resp: r = json.loads(resp.read())
        return (r.get("message", {}).get("content", ""), r.get("model", model),
                {"in": r.get("prompt_eval_count"), "out": r.get("eval_count"), "cache_read": None})
    if a == "openai" or a in BASE_URLS:
        kw = dict(model=model, messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}])
        kw["max_completion_tokens" if a in ("openai", "groq") else "max_tokens"] = 6000
        if "temperature" not in DROP[p]: kw["temperature"] = 0
        if "response_format" not in DROP[p]: kw["response_format"] = {"type": "json_object"}
        r = client.chat.completions.create(**kw); u = r.usage; det = getattr(u, "prompt_tokens_details", None)
        return (r.choices[0].message.content or "", r.model,
                {"in": u.prompt_tokens, "out": u.completion_tokens, "cache_read": (getattr(det, "cached_tokens", 0) or 0) if det else 0})
    if a == "gemini":
        from google.genai import types
        if model.startswith("gemma") or "system_instruction" in DROP[p]:
            # Gemma models on the Gemini API accept no system instruction or JSON mode: same text, sent as one message
            cfg = dict(max_output_tokens=6000); contents = SYSTEM + "\n\n---\n\n" + user
        else:
            cfg = dict(system_instruction=SYSTEM, max_output_tokens=6000); contents = user
            if "response_format" not in DROP[p]: cfg["response_mime_type"] = "application/json"
        if "temperature" not in DROP[p]: cfg["temperature"] = 0
        r = client.models.generate_content(model=model, contents=contents, config=types.GenerateContentConfig(**cfg))
        um = r.usage_metadata
        return (r.text or "", getattr(r, "model_version", None) or model,
                {"in": um.prompt_token_count, "out": um.candidates_token_count, "cache_read": getattr(um, "cached_content_token_count", 0) or 0})

def parse(text):
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m: return None
    try: obj = json.loads(m.group(0))
    except Exception: return None
    return obj if obj.get("risk_class") in LABELS else None

def call_with_retry(p, client, model, user):
    for attempt in range(7):
        try:
            return call(p, client, model, user)
        except Exception as e:
            msg = str(e).lower()
            if ("developer instruction" in msg or "system instruction" in msg) and "system_instruction" not in DROP[p]:
                DROP[p].add("system_instruction"); print(f"[{p}] gộp system prompt vào tin nhắn"); continue
            if ("json mode" in msg or "response_mime_type" in msg) and "response_format" not in DROP[p]:
                DROP[p].add("response_format"); print(f"[{p}] bỏ chế độ JSON"); continue
            for prm in ("temperature", "response_format"):
                if prm in msg and ("unsupported" in msg or "not support" in msg or "invalid" in msg) and prm not in DROP[p]:
                    DROP[p].add(prm); print(f"[{p}] bỏ tham số {prm}"); break
            else:
                if any(t in msg for t in ("per day", "perday", "per-day", "daily")):
                    STOP[p] = True; raise RuntimeError("DAILY_QUOTA: " + str(e)[:200])
                if any(s in msg for s in ("rate", "429", "overloaded", "529", "500", "502", "503", "timeout", "temporarily", "resource_exhausted")):
                    time.sleep(min(60, 2 ** attempt + random.random())); continue
                raise
    raise RuntimeError("Hết số lần thử")

def out_file(p, pilot): return os.path.join("results", f"{'pilot_' if pilot else ''}{p}.jsonl")

def run_provider(p, pilot):
    model = PROVIDERS[p]["model"]
    if model.startswith("PASTE"): raise SystemExit(f"Chưa điền tên model cho {p} trong config.json")
    out = out_file(p, pilot); done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            try:
                rec = json.loads(line)
                if not rec.get("error"): done.add(rec["job_id"])   # errored requests are retried on the next run
            except Exception: pass
    jobs = [j for j in (JOBS[:CFG["pilot_n"]] if pilot else JOBS) if j not in done]
    print(f"[{p}] {model}: còn {len(jobs)} lượt (đã có {len(done)})")
    client = make_client(p); lock = threading.Lock()
    def one(job):
        if STOP[p]: return None
        user = user_msg(job); rec = {"provider": p, "model_requested": model, "job_id": job}
        try:
            obj = None
            for attempt in (1, 2):
                text, mret, usage = call_with_retry(p, client, model, user)
                obj = parse(text)
                rec.update({"attempts": attempt, "raw": text, "model_returned": mret, "usage": usage})
                if obj: break
            rec.update({"valid": obj is not None, "parsed": obj})
        except Exception as e:
            if STOP[p] and "DAILY_QUOTA" in str(e): return None
            rec.update({"valid": False, "error": repr(e)[:500]})
        rec["ts"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        with lock:
            with open(out, "a", encoding="utf-8") as f: f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return rec.get("valid")
    ok = 0
    with ThreadPoolExecutor(CFG["workers"]) as ex:
        for i, fut in enumerate(as_completed([ex.submit(one, j) for j in jobs]), 1):
            ok += bool(fut.result())
            if i % 100 == 0: print(f"[{p}] {i}/{len(jobs)}")
    if STOP[p]:
        print(f"[{p}] Đã hết hạn mức trong ngày. Mai chạy lại đúng lệnh này, máy sẽ chạy tiếp từ chỗ dừng.")
    print(f"[{p}] xong: {ok}/{len(jobs)} hợp lệ")

def run(pilot):
    active = [p for p, c in PROVIDERS.items() if c["on"]]
    with ThreadPoolExecutor(len(active)) as ex:
        futs = {ex.submit(run_provider, p, pilot): p for p in active}
        for f in as_completed(futs):
            try: f.result()
            except BaseException as e: print("LỖI", futs[f], e)
    summary(pilot)

def summary(pilot):
    rows = []
    for p in PROVIDERS:
        fn = out_file(p, pilot)
        if not os.path.exists(fn): continue
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line); d, arm, rep = r["job_id"].split("|"); o = r.get("parsed") or {}; u = r.get("usage") or {}
            rows.append({"provider": p, "model_requested": r["model_requested"], "model_returned": r.get("model_returned"),
                         "job_id": r["job_id"], "desc_id": d, "arm": arm, "rep": rep, "valid": bool(r.get("valid")),
                         "attempts": r.get("attempts"), "risk_class": o.get("risk_class"),
                         **{k: o.get(k) for k in ("q1_prohibited", "q2_annex_I", "q3_annex_III", "q4_art6_3_derogation", "q5_art50")},
                         "justification": o.get("justification"), "error": r.get("error"),
                         "in_tok": u.get("in"), "out_tok": u.get("out"), "cache_read": u.get("cache_read"), "ts": r.get("ts")})
    if not rows: print("Chưa có kết quả."); return
    R = pd.DataFrame(rows)
    R["_err"] = R["error"].notna()
    R = R.sort_values(["provider", "job_id", "_err", "ts"], ascending=[True, True, True, False]).drop_duplicates(["provider", "job_id"]).drop(columns="_err")
    R = R.merge(DESC.drop(columns=["description"]), on="desc_id", how="left")
    R["correct"] = R.risk_class == R.reference
    fn = os.path.join("results", f"{'pilot_' if pilot else ''}results_all.csv")
    R.to_csv(fn, index=False, encoding="utf-8")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print("\nĐã lưu", fn, "-", len(R), "dòng\n")
    print(R.groupby("provider").agg(luot=("job_id", "size"), hop_le=("valid", "mean"), dung=("correct", "mean"),
          tb_token_vao=("in_tok", "mean"), tb_token_ra=("out_tok", "mean"), tb_token_cache=("cache_read", "mean")).round(2))
    print("\nModel trả về:", {k: sorted(map(str, set(v))) for k, v in R.groupby("provider").model_returned})
    bad = R[~R.valid]
    if len(bad): print("\nLượt lỗi/không hợp lệ:\n", bad[["provider", "job_id", "error"]].head(20).to_string())

def list_models():
    seen = set()
    for p in PROVIDERS:
        a = API(p)
        if a in seen: continue
        seen.add(a); print(f"\n===== {a} =====")
        try:
            c = make_client(p)
            if a == "gemini":
                names = [m.name.replace("models/", "") for m in c.models.list()
                         if "generateContent" in (getattr(m, "supported_actions", None) or ["generateContent"])]
            else:
                names = [m.id for m in c.models.list()]
                if a == "openrouter":
                    names = [n for n in names if n.endswith(":free")]
                    print("  (chỉ hiện model miễn phí, đuôi :free)")
            for n in sorted(names): print(" ", n)
        except BaseException as e:
            print("  Không lấy được danh sách:", e)

def check_key():
    """Show OpenRouter key status: credit purchased, daily free limit tier."""
    import urllib.request
    k = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not k: raise SystemExit("Chưa có OPENROUTER_API_KEY trong .env")
    req = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": f"Bearer {k}"})
    with urllib.request.urlopen(req, timeout=30) as r: d = json.loads(r.read()).get("data", {})
    for f in ("label", "is_free_tier", "usage", "limit", "limit_remaining", "rate_limit"):
        if f in d: print(f"{f}: {d[f]}")
    print("\nGhi chú: is_free_tier = True nghĩa là tài khoản chưa từng nạp đủ 10 USD -> chỉ khoảng 50 lượt model miễn phí mỗi ngày.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Chạy thí nghiệm ACIIDS 2027")
    ap.add_argument("--list-models", action="store_true", help="liệt kê model dùng được với key của bạn")
    ap.add_argument("--pilot", action="store_true", help="chạy thử 20 lượt mỗi hãng (hoặc tóm tắt file chạy thử)")
    ap.add_argument("--full", action="store_true", help="chạy toàn bộ")
    ap.add_argument("--summary", action="store_true", help="in bảng tóm tắt và gộp file kết quả")
    ap.add_argument("--check-key", action="store_true", help="xem tình trạng key OpenRouter (hạn mức)")
    a = ap.parse_args()
    if a.check_key: check_key()
    elif a.list_models: list_models()
    elif a.summary: summary(a.pilot)
    elif a.full: run(False)
    elif a.pilot: run(True)
    else: ap.print_help()
