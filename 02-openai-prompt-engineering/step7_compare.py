import time
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

MODELS = ["gpt-4.1-mini", "gpt-4.1", "o4-mini"]   # three picks


def ask_model(model_name, messages):
    """Run one prompt through one model. Return (answer, seconds)."""
    start = time.time()
    response = client.responses.create(model=model_name, input=messages)
    elapsed = time.time() - start
    return response.output_text, elapsed


def compare(messages, label):
    """Run one prompt through ALL models, print side by side."""
    print(f"\n{'='*60}\n  {label}\n{'='*60}")
    for model in MODELS:                      
        try:
            answer, elapsed = ask_model(model, messages)
            print(f"\n--- {model}  ({elapsed:.1f}s) ---")
            print(answer)
        except Exception as e:
            print(f"\n--- {model} --- ERROR: {e}")


# ---- Experiments: just call compare() with different prompts ----

# Experiment 1
compare(
    [{"role": "user", "content": "What is GPU occupancy in one sentence?"}],
    "Experiment 1: Simple question"
)

# Experiment 2
compare(
    [{"role": "user", "content": "A SYCL kernel processes a 4096x4096 matrix. Work-group size is 256. Each work-item does 12 FLOPs. The kernel runs in 0.8ms. Estimate achieved GFLOPS, and explain whether this is memory-bound or compute-bound for an Intel Arc B580 (peak ~13 TFLOPS FP32, ~456 GB/s bandwidth)."}],  
    "Experiment 2: Hard reasoning problem"
)

# Experiment 3A — heavy instructions
HEAVY = """Follow these exact steps. Step 1: identify the symptom. 
Step 2: list three possible causes. Step 3: for each cause, explain 
how to confirm it. Step 4: rank them by likelihood. Step 5: give the 
single most likely fix. Do not skip steps. Number every step."""  

compare(
    [{"role": "developer", "content": HEAVY},
     {"role": "user", "content": "My SYCL kernel gives correct results on CPU but wrong on Arc GPU."}],
    "Experiment 3A: Heavy instructions"
)

# Experiment 3B — light instructions
LIGHT = """You are a GPU debugging expert. Diagnose the issue and 
recommend the most likely fix."""   

compare(
    [{"role": "developer", "content": LIGHT},
     {"role": "user", "content": "My SYCL kernel gives correct results on CPU but wrong on Arc GPU."}],
    "Experiment 3B: Light instructions"
)

print("\n✅ All experiments complete.")