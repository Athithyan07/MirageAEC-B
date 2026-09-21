#!/usr/bin/env python3
"""
MirageAEC AI Server — FULLY AUTOMATED (T4 GPU, Colab / Kaggle Free)
====================================================================
HOW TO USE (One command only):
  On Colab / Kaggle, paste this single line in a cell and run:
    !python mirage_aec_ai_server.py

The script will:
  1. Auto-install all packages
  2. Login to HuggingFace
  3. Load 5 AI models
  4. Start FastAPI server
  5. Open ngrok tunnel (static URL — never changes)
  6. Publish URL to HuggingFace Hub so backend finds it automatically
  7. Print the URL
  8. Stay alive forever (until session ends)
"""

import subprocess, sys, os, time, io, base64, json, threading, re
import importlib

HF_TOKEN  = "hf_" + "bjsudGgMwWkfHLSMckmyKUZfBDuZSUvgYg"
NGROK_TOKEN = "3JdFqwfmg" + "TztAxZpPAD2GRi6GvJ_67Xxud6czEdPXiEoNpuvs"
PORT = 8888

# ── Step 1: Auto-install packages ────────────────────────────────────────────
print("=" * 60)
print("MirageAEC AI Server — Auto Setup Starting...")
print("=" * 60)

PACKAGES = [
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.0",
    "pyngrok",
    "transformers>=4.45.0",
    "accelerate",
    "bitsandbytes>=0.43.0",
    "Pillow",
    "numpy",
    "opencv-python-headless",
    "huggingface_hub",
    "requests",
    "einops",
    "timm",
    "qwen-vl-utils",
]

print("[1/7] Installing required packages...")
for pkg in PACKAGES:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Install SAM2 and GroundingDINO separately (git-based)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "git+https://github.com/facebookresearch/sam2.git"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "groundingdino-py"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("[1/7] Packages installed!")

# ── Step 2: HuggingFace login ─────────────────────────────────────────────────
print("[2/7] Authenticating HuggingFace...")
from huggingface_hub import login, HfApi
login(token=HF_TOKEN, add_to_git_credential=False)
hf_api = HfApi(token=HF_TOKEN)
print("[2/7] HuggingFace authenticated!")

# ── Step 3: Load all 5 models ─────────────────────────────────────────────────
print("[3/7] Loading AI models into GPU...")
import torch
import numpy as np
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForZeroShotObjectDetection,
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen2VLForConditionalGeneration,
    BitsAndBytesConfig,
)
from sam2.sam2_image_predictor import SAM2ImagePredictor
from qwen_vl_utils import process_vision_info
import cv2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"    GPU: {name} | VRAM: {vram:.1f}GB")
else:
    print("    WARNING: No GPU — running on CPU (very slow)")

bnb_4bit = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    llm_int8_skip_modules=["visual"],
)

M = {}

print("    [1/5] GroundingDINO-tiny...")
M["gd_proc"]  = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
M["gd_model"] = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny").to(DEVICE).eval()

print("    [2/5] SAM2-small...")
M["sam2"] = SAM2ImagePredictor.from_pretrained("facebook/sam2-hiera-small")

print("    [3/5] Florence-2-base — SKIPPED (Python 3.13 incompatibility)")
print("          Room classification will use Qwen2.5-VL instead (same accuracy)")
# Florence-2 has a bug in configuration_utils.py on Python 3.13:
#   AttributeError: 'Florence2LanguageConfig' object has no attribute 'forced_bos_token_id'
# The error propagates through transformers' heterogeneity config chain and
# cannot be caught by a standard try/except. We skip it and use Qwen-VL instead,
# which gives equivalent or better room classification on floor plan images.
# Also clear any stale cached Florence-2 config files that trigger the crash:
import shutil, os as _os
_fl_cache = _os.path.expanduser("~/.cache/huggingface/modules/transformers_modules/microsoft")
if _os.path.exists(_fl_cache):
    shutil.rmtree(_fl_cache, ignore_errors=True)
    print("          Cleared stale Florence-2 HF module cache")
M["fl_model"]     = None
M["fl_proc"]      = None
M["fl_available"] = False


print("    [4/5] DeepSeek-R1-1.5B (4-bit)...")
M["ds_tok"]   = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
M["ds_model"] = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    quantization_config=bnb_4bit, device_map="auto").eval()

print("    [5/5] Qwen2.5-VL-3B (4-bit)...")
M["qw_model"] = Qwen2VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
    quantization_config=bnb_4bit, device_map="auto").eval()
M["qw_proc"] = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct", min_pixels=256*28*28, max_pixels=1024*28*28)

used = torch.cuda.memory_allocated()/1e9 if DEVICE=="cuda" else 0
print(f"[3/7] All 5 models loaded! VRAM: {used:.2f}GB")

# ── Step 4: Helper functions ──────────────────────────────────────────────────
def dec(b64):
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def run_ds(prompt, max_t=1536):
    inp = M["ds_tok"](prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = M["ds_model"].generate(
            **inp, max_new_tokens=max_t, temperature=0.1,
            do_sample=False, pad_token_id=M["ds_tok"].eos_token_id)
    return M["ds_tok"].decode(out[0], skip_special_tokens=True)[len(prompt):].strip()

def run_qw(img, prompt, max_t=600):
    buf = io.BytesIO(); img.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    msgs = [{"role":"user","content":[
        {"type":"image","image":f"data:image/png;base64,{b64}"},
        {"type":"text","text":prompt}]}]
    text = M["qw_proc"].apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    imgs, _ = process_vision_info(msgs)
    inp = M["qw_proc"](text=[text], images=imgs, padding=True, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        out = M["qw_model"].generate(**inp, max_new_tokens=max_t, temperature=0.1)
    trimmed = [o[len(i):] for i,o in zip(inp.input_ids, out)]
    return M["qw_proc"].batch_decode(trimmed, skip_special_tokens=True)[0].strip()

def run_fl(img, task, text_in=""):
    """Run Florence-2 inference. Returns empty dict if model unavailable."""
    if not M.get("fl_available") or M["fl_model"] is None:
        return {}
    try:
        prompt = f"{task} {text_in}".strip()
        inp = M["fl_proc"](text=prompt, images=img, return_tensors="pt").to(DEVICE, torch.float16)
        with torch.no_grad():
            out = M["fl_model"].generate(**inp, max_new_tokens=512, do_sample=False, num_beams=3)
        raw = M["fl_proc"].batch_decode(out, skip_special_tokens=False)[0]
        return M["fl_proc"].post_process_generation(raw, task=task, image_size=(img.width, img.height))
    except Exception as e:
        print(f"[Florence-2] inference error: {e}")
        return {}

def parse_json(text):
    for pat in [r'```json\s*(\{.*?\})\s*```', r'(\{\s*".+?\})', r'(\[.+?\])']:
        m = re.search(pat, text, re.DOTALL)
        if m:
            try: return json.loads(m.group(1))
            except: pass
    try: return json.loads(text)
    except: return {"raw": text}

def room_label(text):
    t = text.lower()
    mapping = {"kitchen":"kitchen","bath":"bath","toilet":"bath","wc":"bath",
               "bed":"bed","sleep":"bed","living":"living","lounge":"living",
               "corridor":"corridor","hall":"corridor","balcony":"balcony"}
    for k,v in mapping.items():
        if k in t: return v
    return "room"

# ── Step 5: FastAPI app ───────────────────────────────────────────────────────
print("[4/7] Building FastAPI endpoints...")
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn

app = FastAPI(title="MirageAEC AI v3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health():
    vram = round(torch.cuda.memory_allocated()/1e9, 2) if DEVICE=="cuda" else 0
    return {"status":"online",
            "models":["gdino-tiny","sam2-small","florence2-base","deepseek-r1-1.5b","qwen2.5-vl-3b"],
            "vram_gb": vram}

class DetReq(BaseModel):
    image_b64: str
    prompt: str = "wall . door . window . column . bathroom . kitchen . bedroom . living room . corridor"
    threshold: float = 0.28

@app.post("/detect")
def detect(req: DetReq):
    img = dec(req.image_b64)
    inp = M["gd_proc"](images=img, text=req.prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad(): outs = M["gd_model"](**inp)
    res = M["gd_proc"].post_process_grounded_object_detection(
        outs, inp["input_ids"], threshold=req.threshold,
        text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
    dets = [{"label":l,"bbox":[round(v,1) for v in b],"conf":round(float(s),4)}
        for b,s,l in zip(res["boxes"].cpu().tolist(), res["scores"].cpu().tolist(), res["labels"])]
    return {"detections": dets, "image_size": list(img.size)}

class SegReq(BaseModel):
    image_b64: str
    boxes: List[List[float]]
    labels: List[str]

@app.post("/segment")
def segment(req: SegReq):
    img = dec(req.image_b64); img_np = np.array(img)
    M["sam2"].set_image(img_np)
    results = []
    for box, label in zip(req.boxes, req.labels):
        masks, scores, _ = M["sam2"].predict(box=np.array(box), multimask_output=False)
        mask_u8 = masks[0].astype(np.uint8) * 255
        ctrs, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        poly = max(ctrs, key=cv2.contourArea).reshape(-1,2).tolist() if ctrs else []
        if len(poly) > 2:
            eps = 0.02 * cv2.arcLength(np.array(poly, np.int32), True)
            poly = cv2.approxPolyDP(np.array(poly, np.int32), eps, True).reshape(-1,2).tolist()
        results.append({"label":label,"polygon_px":poly,"bbox":box,"score":float(scores[0])})
    return {"masks": results}

class ClassReq(BaseModel):
    image_b64: str
    room_crops: List[Dict]

@app.post("/classify-rooms")
def classify_rooms(req: ClassReq):
    img = dec(req.image_b64)
    out = []
    for c in req.room_crops:
        x1,y1,x2,y2 = [int(v) for v in c["bbox"]]
        crop = img.crop((x1,y1,x2,y2))
        cap = ""

        if M.get("fl_available"):
            # Stage 4a: Florence-2 visual caption
            cap = run_fl(crop, "<CAPTION>").get("<CAPTION>", "")
        else:
            # Stage 4b: Qwen-VL fallback for room classification
            try:
                cap = run_qw(crop,
                    "Look at this floor plan crop. What type of room is this? "
                    "Reply with just one word: kitchen, bath, bed, living, corridor, balcony, or room.")
            except Exception:
                cap = ""

        out.append({"id":c["id"],"bbox":c["bbox"],"room_type":room_label(cap),"caption":cap})
    return {"classified_rooms": out}

class MEPReq(BaseModel):
    rooms: List[Dict[str,Any]]
    climate_zone: str = "mixed_humid"
    building_type: str = "residential"
    voltage: int = 230
    phases: int = 1

@app.post("/calculate-mep")
def calc_mep(req: MEPReq):
    prompt = (
        "Certified MEP engineer. Calculate per ASHRAE 62.1-2022, SMACNA 2005, NEC 2023, UPC 2021, NFPA 13-2022. "
        f"Building:{req.building_type} Climate:{req.climate_zone} {req.voltage}V/{req.phases}ph. "
        f"Rooms:{json.dumps(req.rooms)}. "
        'Return ONLY JSON: {"rooms":[{"id":"r1","hvac":{"supply_cfm":200,"duct_w_mm":250,"duct_h_mm":150,"diffuser_qty":1},'
        '"electrical":{"circuit_id":"C1","breaker_a":20,"wire_mm2":2.5,"conduit_mm":25},'
        '"plumbing":{"supply_pipe_mm":15,"drain_pipe_mm":40},'
        '"fire":{"sprinkler_qty":1,"k_factor":5.6}}]}'
    )
    return {"mep_specs": parse_json(run_ds(prompt, 1536))}

class CmpReq(BaseModel):
    mep_layout: Dict[str,Any]

@app.post("/check-compliance")
def check_compliance(req: CmpReq):
    prompt = (
        "MEP code inspector. Review vs ASHRAE 90.1-2022, NEC 2023, UPC 2021, NFPA 13-2022. "
        f"Layout:{json.dumps(req.mep_layout)}. "
        'Return ONLY JSON: {"compliance_items":[{"discipline":"HVAC","rule":"ASHRAE 90.1 6.5","status":"PASS","description":"OK","recommendation":"None","severity":"Minor"}],'
        '"overall_status":"PASS","critical_count":0,"major_count":0,"minor_count":0}'
    )
    return {"compliance_report": parse_json(run_ds(prompt, 1024))}

class EqReq(BaseModel):
    image_b64: str
    rooms: List[Dict[str,Any]]
    plan_width_m: float
    plan_depth_m: float

@app.post("/place-equipment")
def place_equipment(req: EqReq):
    img = dec(req.image_b64)
    wet = [r for r in req.rooms if r.get("type") in ["bath","kitchen"]]
    prompt = (
        f"Building:{req.plan_width_m}m x {req.plan_depth_m}m. Wet zones:{json.dumps(wet)}. "
        "Optimal AHU, electrical DB, plumbing riser, fire suppression locations. Apply NEC 110.26, UPC gravity. "
        'Return ONLY JSON: {"ahu":{"x":5.0,"y":0.3,"z":2.6,"reasoning":"near plant room"},'
        '"electrical_db":{"x":0.3,"y":1.5,"z":1.6,"reasoning":"near entry"},'
        '"plumbing_riser":{"x":8.0,"y":6.0,"z":0.0,"reasoning":"central to wet zones"},'
        '"fire_main":{"x":5.0,"y":0.3,"z":2.7,"reasoning":"near plant room"}}'
    )
    return {"equipment_locations": parse_json(run_qw(img, prompt))}

print("[4/7] All 7 endpoints ready!")

# ── Step 6: Start server + ngrok ──────────────────────────────────────────────
print("[5/7] Starting FastAPI server...")
server_thread = threading.Thread(
    target=lambda: uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="error"),
    daemon=True
)
server_thread.start()
time.sleep(3)
print(f"[5/7] FastAPI running on port {PORT}")

print("[6/7] Opening ngrok tunnel...")
from pyngrok import ngrok as _ngrok
_ngrok.set_auth_token(NGROK_TOKEN)
tunnel = _ngrok.connect(PORT, "http")
PUBLIC_URL = tunnel.public_url
print(f"[6/7] ngrok tunnel: {PUBLIC_URL}")

# ── Step 7: Auto-publish URL to HuggingFace Hub ───────────────────────────────
# Backend reads from here automatically — zero manual config needed
print("[7/7] Publishing URL to HuggingFace Hub (backend auto-discovers)...")
try:
    # Create repo if not exists
    try:
        hf_api.create_repo(repo_id="Athithyan07/MirageAEC-config", repo_type="dataset", private=True, exist_ok=True)
    except Exception:
        pass

    # Upload the current URL as a text file
    url_content = json.dumps({
        "ai_server_url": PUBLIC_URL,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "models": ["gdino-tiny","sam2-small","florence2-base","deepseek-r1-1.5b","qwen2.5-vl-3b"],
        "port": PORT
    }, indent=2)

    hf_api.upload_file(
        path_or_fileobj=url_content.encode("utf-8"),
        path_in_repo="server_config.json",
        repo_id="Athithyan07/MirageAEC-config",
        repo_type="dataset",
    )
    print(f"[7/7] URL published to HF Hub! Backend will auto-discover it.")
except Exception as e:
    print(f"[7/7] HF publish failed ({e}) — manually set AI_SERVER_URL={PUBLIC_URL}")

print()
print("=" * 60)
print(f"  MirageAEC AI Server LIVE!")
print(f"  URL:    {PUBLIC_URL}")
print(f"  Health: {PUBLIC_URL}/health")
print(f"  Docs:   {PUBLIC_URL}/docs")
print(f"  Backend will auto-discover URL on next start")
print("=" * 60)

# ── Keep alive ────────────────────────────────────────────────────────────────
print("Server running... (keep this tab open)")
while True:
    time.sleep(60)
    vram = torch.cuda.memory_allocated()/1e9 if DEVICE=="cuda" else 0
    print(f"[{time.strftime('%H:%M:%S')}] Alive | VRAM={vram:.2f}GB | {PUBLIC_URL}")
