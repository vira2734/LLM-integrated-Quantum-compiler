import json, os, asyncio, re
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tqdm import tqdm

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found. Put it in .env")

MODEL = "gpt-4o-mini"
PARAS_PER = 2          # 2 paraphrases per input -> total 3 variants including original
BATCH_SIZE = 20        # 20 inputs per API call (big speedup)
CONCURRENCY = 8        # increase until you hit 429s; then lower
TEMPERATURE = 0.8

SYSTEM = (
    "You rewrite instructions. You MUST preserve every qubit number exactly. "
    "Do NOT add or remove qubit ids. Do NOT change the requested shape."
)

def extract_ints(s: str):
    return re.findall(r"\d+", s)

def valid_para(orig: str, para: str) -> bool:
    # ensure exact same multiset of integers appears (order can differ)
    return sorted(extract_ints(orig)) == sorted(extract_ints(para))

async def paraphrase_batch(inputs):
    # Ask for strict JSON for machine parsing
    user = {
        "role": "user",
        "content": (
            f"Generate {PARAS_PER} paraphrases for each item in the list. "
            f"Return ONLY valid JSON with this exact format:\n"
            f'{{"paraphrases":[["p1","p2"],["p1","p2"], ...]}}\n'
            f"List length = {len(inputs)}; each inner list must have exactly {PARAS_PER} strings.\n\n"
            f"INPUTS:\n" + "\n".join([f"{i}. {t}" for i, t in enumerate(inputs)])
        )
    }
    resp = await client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM}, user],
        temperature=TEMPERATURE,
    )
    txt = resp.choices[0].message.content.strip()
    return json.loads(txt)["paraphrases"]

async def worker(sema, batch_inputs):
    async with sema:
        # basic retry loop for transient rate limits
        for attempt in range(6):
            try:
                return await paraphrase_batch(batch_inputs)
            except Exception as e:
                # exponential backoff
                await asyncio.sleep(min(2 ** attempt, 30))
        raise RuntimeError("Failed batch after retries")

async def main():
    # load base
    with open("base.jsonl", "r", encoding="utf-8") as f:
        base = [json.loads(line) for line in f]

    # batch inputs
    batches = [base[i:i+BATCH_SIZE] for i in range(0, len(base), BATCH_SIZE)]
    sema = asyncio.Semaphore(CONCURRENCY)

    expanded = []
    tasks = []
    for b in batches:
        batch_inputs = [x["input"] for x in b]
        tasks.append(asyncio.create_task(worker(sema, batch_inputs)))

    # gather with progress
    for b, t in tqdm(list(zip(batches, tasks)), total=len(tasks), desc="Paraphrasing"):
        paras = await t  # list of [p1,p2] per item
        for item, pp in zip(b, paras):
            # always keep original
            expanded.append(item)
            # add validated paraphrases
            kept = 0
            for p in pp:
                p = p.strip()
                if p and valid_para(item["input"], p):
                    expanded.append({"input": p, "output": item["output"]})
                    kept += 1
            # if the model messed up, you can fall back to keeping fewer paras;
            # dataset still works.
    with open("expanded.jsonl", "w", encoding="utf-8") as f:
        for s in expanded:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("Expanded size:", len(expanded))

if __name__ == "__main__":
    asyncio.run(main())