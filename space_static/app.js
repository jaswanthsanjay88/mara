import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.all.bundle.min.mjs";

const BASE_URL = "https://huggingface.co/jaswanthsanjay88/mara-small/resolve/main/";
const TOKENIZER_URL = BASE_URL + "tokenizer.json";
const HAS_WEBGPU = typeof navigator.gpu !== "undefined";
const MODEL_URL = BASE_URL + (HAS_WEBGPU ? "model_fp16.onnx" : "model_int8.onnx");
const MAX_SEQ = 512;
const EOS_ID = 0;

const statusEl = document.getElementById("status");
const outEl = document.getElementById("out");
const goBtn = document.getElementById("go");
const promptEl = document.getElementById("prompt");
const maxtokEl = document.getElementById("maxtok");
const tempEl = document.getElementById("temp");

maxtokEl.oninput = () => (document.getElementById("maxtokv").textContent = maxtokEl.value);
tempEl.oninput = () => (document.getElementById("tempv").textContent = Number(tempEl.value).toFixed(2));

function bytesToUnicode() {
  const charToByte = new Map();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    const printable = (b > 32 && b < 127) || (b > 160 && b < 173) || (b > 173 && b < 255);
    if (printable) {
      charToByte.set(String.fromCharCode(b), b);
    } else {
      charToByte.set(String.fromCharCode(256 + n), b);
      n++;
    }
  }
  return charToByte;
}

const CHAR_TO_BYTE = bytesToUnicode();
const BYTE_TO_CHAR = new Map([...CHAR_TO_BYTE].map(([c, b]) => [b, c]));

function utf8Bytes(str) {
  return Array.from(new TextEncoder().encode(str));
}

let tokenizer, session, provider;

async function loadTokenizer() {
  const tj = await (await fetch(TOKENIZER_URL)).json();
  const vocab = new Map(Object.entries(tj.model.vocab).map(([t, id]) => [id, t]));
  const mergesRaw = tj.model.merges.map((m) => (Array.isArray(m) ? m.join(" ") : m));
  const ranks = new Map(mergesRaw.map((m, i) => [m, i]));

  function bpe(word) {
    if (word.length < 2) return word ? [word] : [];
    let symbols = [...word];
    while (true) {
      let bestRank = Infinity, bestIdx = -1;
      for (let i = 0; i < symbols.length - 1; i++) {
        const r = ranks.get(symbols[i] + symbols[i + 1]);
        if (r !== undefined && r < bestRank) { bestRank = r; bestIdx = i; }
      }
      if (bestIdx === -1) break;
      symbols.splice(bestIdx, 2, symbols[bestIdx] + symbols[bestIdx + 1]);
    }
    return symbols;
  }

  const pattern = /'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu;

  return {
    encode(text) {
      const ids = [];
      for (const piece of text.match(pattern) || []) {
        const mapped = utf8Bytes(piece).map((b) => String.fromCharCode(b)).join("");
        for (const tok of bpe(mapped)) {
          const id = tj.model.vocab[tok];
          if (id !== undefined) ids.push(id);
        }
      }
      return ids;
    },
    decode(ids) {
      let s = "";
      for (const id of ids) {
        if (id === EOS_ID) break;
        const tok = vocab.get(id);
        if (tok === undefined) continue;
        for (const ch of tok) {
          const b = CHAR_TO_BYTE.has(ch) ? CHAR_TO_BYTE.get(ch) : ch.codePointAt(0);
          s += String.fromCharCode(b);
        }
      }
      const bytes = new Uint8Array([...s].map((c) => c.charCodeAt(0)));
      return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
    },
  };
}

function sample(logits, temperature) {
  const k = Math.min(50, logits.length);
  const idx = Array.from(logits.keys()).sort((a, b) => logits[b] - logits[a]).slice(0, k);
  const scaled = idx.map((i) => logits[i] / temperature);
  const max = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  let r = Math.random() * sum;
  for (let i = 0; i < idx.length; i++) {
    r -= exps[i];
    if (r <= 0) return idx[i];
  }
  return idx[idx.length - 1];
}

async function init() {
  try {
    tokenizer = await loadTokenizer();
    session = await ort.InferenceSession.create(MODEL_URL, {
      executionProviders: HAS_WEBGPU ? ["webgpu"] : ["wasm"],
    });
    provider = HAS_WEBGPU ? "webgpu (fp16)" : "wasm (int8)";
    statusEl.innerHTML = `ready · backend: <b>${provider}</b> · model: ${HAS_WEBGPU ? "fp16" : "int8"}`;
    goBtn.disabled = false;
  } catch (e) {
    statusEl.textContent = "failed to load model: " + e.message;
  }
}

async function generate() {
  goBtn.disabled = true;
  outEl.textContent = "";
  statusEl.textContent = "generating…";

  const ids = tokenizer.encode(promptEl.value);
  const maxNew = Number(maxtokEl.value);
  const temp = Number(tempEl.value);
  const generated = [];

  for (let step = 0; step < maxNew; step++) {
    const ctx = ids.slice(-MAX_SEQ);
    const input = new ort.Tensor("int64", BigInt64Array.from(ctx.map(BigInt)), [1, ctx.length]);
    const results = await session.run({ input_ids: input });
    const logits = results.logits.data;
    const seqLen = results.logits.dims[1];
    const V = results.logits.dims[2];
    const last = logits.slice((seqLen - 1) * V, seqLen * V);
    const nextId = sample(last, temp);
    if (nextId === EOS_ID) break;
    ids.push(nextId);
    generated.push(nextId);
    if (step % 4 === 0 || step === maxNew - 1) {
      outEl.textContent = promptEl.value + tokenizer.decode(generated);
    }
  }
  outEl.textContent = promptEl.value + tokenizer.decode(generated);
  statusEl.innerHTML = `done · ${generated.length} tokens · backend: <b>${provider}</b>`;
  goBtn.disabled = false;
}

goBtn.onclick = () => { if (!goBtn.disabled) generate(); };
init();
