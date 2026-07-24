"""
collect_data.py
================
Runs on YOUR machine (network + Helius API key required: https://helius.dev).
Not runnable in this sandbox — no network here.

Scope note: the brief states ~16K bot buys vs ~5M total candidate launches.
That's the full pump.fun launch history over the bot's active window — likely
several months, not 30 days. Step 0 below finds the bot's real first/last
trade timestamps so you can size the pull correctly.

Checkpointed and resumable — a 5M-row pull takes hours and will hit rate limits.

pip install requests --break-system-packages
"""

import requests, time, json, csv, os, sys
from datetime import datetime, timezone

HELIUS_API_KEY = "YOUR_HELIUS_API_KEY_HERE"
RPC_URL = f"https://mainnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"
ENHANCED_TX_URL = f"https://api.helius.xyz/v0/transactions/?api-key={HELIUS_API_KEY}"

BOT_WALLET = "5brv79eFZ2rGprXNvqgVJBkBptkkw8GJX1XydJyZLyAr"
PUMPFUN_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"

OUT_DIR = "pumpbot_data"
os.makedirs(OUT_DIR, exist_ok=True)


def rpc_call(method, params, retries=5):
    for attempt in range(retries):
        try:
            r = requests.post(RPC_URL, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=30)
            r.raise_for_status()
            j = r.json()
            if "error" in j:
                raise RuntimeError(j["error"])
            return j["result"]
        except Exception as e:
            wait = 2 ** attempt
            print(f"  rpc_call retry {attempt}: {e} (sleep {wait}s)", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError(f"rpc_call failed after {retries} retries: {method}")


def find_bot_first_last_trade():
    """Oldest signature = bot's first trade. Tells you the real window needed for ~16K buys."""
    oldest = None
    before = None
    newest = None
    count = 0
    while True:
        params = [BOT_WALLET, {"limit": 1000}]
        if before:
            params[1]["before"] = before
        batch = rpc_call("getSignaturesForAddress", params)
        if not batch:
            break
        if newest is None:
            newest = batch[0]
        oldest = batch[-1]
        before = batch[-1]["signature"]
        count += len(batch)
        print(f"  scanned {count} sigs, oldest so far: "
              f"{datetime.fromtimestamp(oldest.get('blockTime', 0), tz=timezone.utc)}", file=sys.stderr)
        time.sleep(0.1)
    return newest, oldest, count


def get_all_signatures(address, checkpoint_file):
    sigs = []
    before = None
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            sigs = json.load(f)
        before = sigs[-1]["signature"] if sigs else None
        print(f"Resuming from checkpoint: {len(sigs)} sigs", file=sys.stderr)

    while True:
        params = [address, {"limit": 1000}]
        if before:
            params[1]["before"] = before
        batch = rpc_call("getSignaturesForAddress", params)
        if not batch:
            break
        sigs.extend(batch)
        before = batch[-1]["signature"]
        if len(sigs) % 5000 < 1000:
            with open(checkpoint_file, "w") as f:
                json.dump(sigs, f)
            print(f"  checkpoint: {len(sigs)} signatures", file=sys.stderr)
        time.sleep(0.1)

    with open(checkpoint_file, "w") as f:
        json.dump(sigs, f)
    return sigs


def enhanced_parse_batch(signatures, cache_file):
    """Decode via Helius Enhanced API, 100 at a time, cached to disk (append-only JSONL)."""
    done_sigs = set()
    if os.path.exists(cache_file):
        with open(cache_file) as f:
            for line in f:
                try:
                    done_sigs.add(json.loads(line)["signature"])
                except Exception:
                    pass

    todo = [s["signature"] for s in signatures if s["signature"] not in done_sigs]
    print(f"{len(todo)} transactions left to decode ({len(done_sigs)} already cached)", file=sys.stderr)

    with open(cache_file, "a") as f:
        for i in range(0, len(todo), 100):
            chunk = todo[i:i + 100]
            for attempt in range(5):
                try:
                    r = requests.post(ENHANCED_TX_URL, json={"transactions": chunk}, timeout=60)
                    r.raise_for_status()
                    for tx in r.json():
                        f.write(json.dumps(tx) + "\n")
                    f.flush()
                    break
                except Exception as e:
                    print(f"  enhanced parse retry: {e}", file=sys.stderr)
                    time.sleep(2 ** attempt)
            if i % 1000 == 0:
                print(f"  decoded {i}/{len(todo)}", file=sys.stderr)
            time.sleep(0.15)


def extract_bot_buys(cache_file, out_csv):
    """
    Filters bot's txs for pump.fun program interaction where slot == mint's
    creation slot (zero-block entries).
    NOTE: verify field names against a live sample first — print one decoded
    record before trusting this at scale, Helius's `type`/`instructions`
    shape can shift between API versions.
    """
    rows = []
    with open(cache_file) as f:
        for line in f:
            tx = json.loads(line)
            if tx.get("transactionError"):
                continue
            touches_pumpfun = any(
                ix.get("programId") == PUMPFUN_PROGRAM
                for ix in tx.get("instructions", [])
            )
            if not touches_pumpfun:
                continue
            for tt in tx.get("tokenTransfers", []):
                if tt.get("toUserAccount") == BOT_WALLET:
                    rows.append({
                        "signature": tx.get("signature"),
                        "slot": tx.get("slot"),
                        "block_time": tx.get("timestamp"),
                        "mint": tt.get("mint"),
                        "token_amount": tt.get("tokenAmount"),
                        "fee": tx.get("fee"),
                        "fee_payer": tx.get("feePayer"),
                        "type": tx.get("type"),
                        "source": tx.get("source"),
                    })

    fieldnames = list(rows[0].keys()) if rows else \
        ["signature", "slot", "block_time", "mint", "token_amount", "fee", "fee_payer", "type", "source"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} candidate buy rows to {out_csv} — de-dup/clean by mint before use", file=sys.stderr)
    return rows


def get_block_launches(slot):
    """All pump.fun program-touching txs in a slot -> candidate launch pool.
    TODO: filter to specifically the CREATE instruction using the current
    pump.fun IDL discriminator bytes — confirm against one known launch tx
    before trusting this at scale, since this currently returns ALL
    pump.fun-touching txs in the slot (creates + buys + sells)."""
    block = rpc_call("getBlock", [slot, {
        "maxSupportedTransactionVersion": 0,
        "transactionDetails": "full",
        "rewards": False,
    }])
    launches = []
    if not block:
        return launches
    for tx in block.get("transactions", []):
        msg = tx["transaction"]["message"]
        account_keys = msg.get("accountKeys", [])
        for ix in msg.get("instructions", []):
            program_idx = ix.get("programIdIndex")
            if program_idx is not None and program_idx < len(account_keys):
                if account_keys[program_idx] == PUMPFUN_PROGRAM:
                    launches.append({
                        "slot": slot,
                        "signature": tx["transaction"].get("signatures", [None])[0],
                        "fee_payer": account_keys[0] if account_keys else None,
                        "raw_accounts": account_keys,
                    })
    return launches


def main():
    print("=== Step 0: bot active window ===", file=sys.stderr)
    newest, oldest, total = find_bot_first_last_trade()
    print(f"Bot has {total} total signatures. "
          f"First: {oldest.get('blockTime')}, Last: {newest.get('blockTime')}", file=sys.stderr)
    print("^ Use this to size your real collection window (brief implies months, not 30 "
          "days, given ~16K buys).", file=sys.stderr)

    print("=== Step 1: bot signature history ===", file=sys.stderr)
    sig_checkpoint = os.path.join(OUT_DIR, "bot_signatures.json")
    bot_sigs = get_all_signatures(BOT_WALLET, sig_checkpoint)

    print("=== Step 2: decode bot transactions ===", file=sys.stderr)
    tx_cache = os.path.join(OUT_DIR, "bot_txs_decoded.jsonl")
    enhanced_parse_batch(bot_sigs, tx_cache)

    print("=== Step 3: extract clean buy rows ===", file=sys.stderr)
    buys_csv = os.path.join(OUT_DIR, "bot_trades.csv")
    rows = extract_bot_buys(tx_cache, buys_csv)

    print("=== Step 4: pull launch cohorts for bot's slots ===", file=sys.stderr)
    bot_slots = sorted(set(r["slot"] for r in rows if r.get("slot")))
    launches_csv = os.path.join(OUT_DIR, "launch_controls.csv")
    all_launches = []
    for i, slot in enumerate(bot_slots):
        all_launches.extend(get_block_launches(slot))
        if i % 200 == 0:
            print(f"  {i}/{len(bot_slots)} slots processed, {len(all_launches)} launches found", file=sys.stderr)
            with open(launches_csv, "w", newline="") as f:
                if all_launches:
                    w = csv.DictWriter(f, fieldnames=["slot", "signature", "fee_payer", "raw_accounts"])
                    w.writeheader()
                    for row in all_launches:
                        row = dict(row)
                        row["raw_accounts"] = json.dumps(row["raw_accounts"])
                        w.writerow(row)
        time.sleep(0.1)

    print("Done with bot-slot cohorts (your positives + same-slot negatives). For the full "
          "~5M-launch negative universe, extend Step 4 to also sample random slots across "
          "the whole window — see README.", file=sys.stderr)


if __name__ == "__main__":
    main()
