"use client";

import React, { useState, useEffect } from "react";
import Link from "next/link";
import Header from "../components/Header";
import { ArrowLeft, Rss, ExternalLink, Cpu, Zap, ShieldCheck, CheckCircle2, AlertTriangle, Play } from "lucide-react";

export default function ResearchBlogPage() {
  const [theme, setTheme] = useState("dark");

  useEffect(() => {
    const saved = localStorage.getItem("mara-theme") || "dark";
    setTheme(saved);
    document.documentElement.setAttribute("data-theme", saved);
  }, []);

  const toggleTheme = () => {
    const next = theme === "dark" ? "light" : "dark";
    setTheme(next);
    localStorage.setItem("mara-theme", next);
    document.documentElement.setAttribute("data-theme", next);
    if (next === "dark") {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  return (
    <div className="min-h-[100dvh] flex flex-col bg-[var(--bg)] text-[var(--fg)] transition-colors">
      <Header theme={theme} onToggleTheme={toggleTheme} />

      {/* Main Blog Container - Measure strictly bounded to 720px for optimal reading */}
      <main className="flex-1 w-full max-w-[760px] mx-auto px-4 sm:px-6 py-10 sm:py-16">
        {/* Navigation Breadcrumb */}
        <div className="flex items-center justify-between gap-4 mb-8">
          <Link
            href="/"
            className="inline-flex items-center gap-1.5 text-[13px] font-medium text-[var(--fg-2)] hover:text-[var(--fg)] transition-colors group"
          >
            <ArrowLeft size={15} className="transition-transform group-hover:-translate-x-1" />
            <span>Interactive Playground</span>
          </Link>

          <a
            href="/feed.xml"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[6px] text-[11px] font-mono text-[#F26522] bg-[var(--fill-1)] border border-[var(--line)] hover:border-[#F26522]/40 transition-colors"
          >
            <Rss size={12} />
            <span>Subscribe via RSS</span>
          </a>
        </div>

        {/* Article Header */}
        <header className="space-y-4 pb-8 border-b border-[var(--line)]">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-full bg-[var(--fill-1)] border border-[var(--line)] text-[11px] font-mono text-[var(--fg-3)] uppercase tracking-wider">
            <Cpu size={12} className="text-emerald-500" />
            <span>Systems & Neural Architecture</span>
          </div>

          <h1 className="text-[32px] sm:text-[42px] leading-[1.15] font-[600] tracking-[-0.03em] text-[var(--fg)]">
            Thinking in Micro-Weights: Why We Built Mara, a 692k Parameter Atomic Function Model
          </h1>

          <p className="text-[17px] sm:text-[19px] leading-[1.5] text-[var(--fg-2)] font-normal">
            On the autoregressive tax, why JSON token generation is an anti-pattern for embedded dispatch, and what happens when you compare specialist against specialist from first principles.
          </p>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-2 pt-2 text-[13px] font-mono text-[var(--fg-3)]">
            <span className="text-[var(--fg)] font-medium font-sans">Jaswanth Sanjay</span>
            <span>•</span>
            <time dateTime="2026-09-21">September 21, 2026</time>
            <span>•</span>
            <span>11 min read</span>
          </div>
        </header>

        {/* Article Body */}
        <article className="mt-10 space-y-8 text-[16px] leading-[1.75] text-[var(--fg)] selection:bg-[var(--fg)] selection:text-[var(--bg)] font-normal">
          {/* Intro Section */}
          <section className="space-y-4">
            <p className="text-[18px] leading-[1.7] text-[var(--fg)]">
              A few weeks ago, I found myself watching a modern local voice assistant handle a routine instruction: <em className="text-[var(--fg)] font-medium">“Turn off the kitchen lights and lock the front door.”</em>
            </p>
            <p>
              I watched the profiler waterfall. The query went through an HTTP socket, was encoded into a 4,096-dimensional prompt embedding, fed through a 32-layer decoder-only LLM, and then sampled token-by-token for 140 sequential autoregressive steps:
            </p>
            
            <div className="p-4 rounded-[8px] bg-[var(--fill-1)] border border-[var(--line)] font-mono text-[13px] leading-[1.6] text-[var(--fg-2)] overflow-x-auto">
              <code>{`{\n  "name": "set_lights",\n  "arguments": {\n    "room": "kitchen",\n    "on": false\n  }\n}`}</code>
            </div>

            <p>
              After emitting this text, a JSON parser deserialized the string, validated it against a schema validator, and then executed a local IPC call. Total latency: <strong>480 milliseconds</strong>. Process memory: <strong>8 gigabytes</strong>. Compute consumed: trillions of floating-point operations.
            </p>
            <p>
              And you have to ask yourself: <em>what on earth are we doing?</em>
            </p>
            <p>
              Why are we spinning up massive general-intelligence monoliths capable of writing Elizabethan poetry, and then forcing them to slowly spit out curly braces, quotation marks, and indentation tokens just to flip a boolean register on an electrical switch?
            </p>
            <p>
              This is what I call the <strong>Autoregressive Tax</strong>.
            </p>
          </section>

          {/* Section 1 */}
          <section className="space-y-4 pt-6 border-t border-[var(--line)]">
            <h2 className="text-[24px] sm:text-[28px] leading-[1.25] font-[600] tracking-[-0.02em]">
              1. The Autoregressive Tax vs. First Principles
            </h2>
            <p>
              When you want to calculate the trajectory of a basketball falling to the ground, you do not invoke general relativity and solve the Einstein field equations over curved spacetime. You use <em>F = ma</em>. You pick the simplest mathematical formulation that accurately captures the physical regime.
            </p>
            <p>
              In machine learning, software engineers have accidentally conflated <em>language understanding</em> with <em>token-by-token generative sampling</em>.
            </p>
            <p>
              Consider what tool calling actually is. Given a user utterance <em>x</em> and a set of candidate tools <em>T = [t<sub>1</sub>, t<sub>2</sub>, ..., t<sub>k</sub>]</em>:
            </p>
            <ul className="list-disc list-inside space-y-2 text-[var(--fg-2)] pl-2">
              <li>It is a <strong>routing classification</strong> problem: select tool <em>t* &isin; T &cup; &#123;&empty;&#125;</em>.</li>
              <li>It is a <strong>slot filling</strong> problem: extract categorical choices (<em>room &isin; &#123;living room, kitchen, ...&#125;</em>) and bounded scalar values (<em>temperature &isin; [15, 30]</em>).</li>
            </ul>
            <p>
              Neither of these operations requires autoregressive loop generation. Autoregression has \(O(N)\) sequential memory bandwidth bottlenecks—each token requires a full read of every parameter weight from RAM into the CPU cache. If you emit 100 tokens, you must stream the entire model through cache 100 consecutive times.
            </p>
            <p>
              If, instead, you formulate tool calling as a <strong>single-pass discriminative representation</strong>, you pass the weights through cache exactly <em>once</em>.
            </p>
          </section>

          {/* Section 2 */}
          <section className="space-y-4 pt-6 border-t border-[var(--line)]">
            <h2 className="text-[24px] sm:text-[28px] leading-[1.25] font-[600] tracking-[-0.02em]">
              2. What is an Atomic Function Model (AFM)?
            </h2>
            <p>
              We set out to build the absolute minimal architecture that could reliably execute smart home tool calling with zero hallucination. We call this family an <strong>Atomic Function Model (AFM)</strong>.
            </p>
            <p>
              Here are the exact design constraints we chose:
            </p>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 my-4">
              <div className="p-3.5 rounded-[8px] bg-[var(--fill-1)] border border-[var(--line)] space-y-1">
                <span className="font-mono text-[11px] text-[var(--fg-3)]">PARAMETERS</span>
                <p className="text-[18px] font-mono font-[600]">692,000</p>
                <p className="text-[12px] text-[var(--fg-2)]">0.69M params (not 7B or 70B)</p>
              </div>
              <div className="p-3.5 rounded-[8px] bg-[var(--fill-1)] border border-[var(--line)] space-y-1">
                <span className="font-mono text-[11px] text-[var(--fg-3)]">ONNX GRAPH SIZE</span>
                <p className="text-[18px] font-mono font-[600]">153 KB</p>
                <p className="text-[12px] text-[var(--fg-2)]">Fits in L2 cache of any modern CPU</p>
              </div>
              <div className="p-3.5 rounded-[8px] bg-[var(--fill-1)] border border-[var(--line)] space-y-1">
                <span className="font-mono text-[11px] text-[var(--fg-3)]">LATENCY (CPU)</span>
                <p className="text-[18px] font-mono font-[600]">0.72 ms</p>
                <p className="text-[12px] text-[var(--fg-2)]">Sub-millisecond single forward pass</p>
              </div>
            </div>

            <p>
              The architecture is surprisingly pure. It consists of:
            </p>
            <ol className="list-decimal list-inside space-y-2 text-[var(--fg-2)] pl-2">
              <li><strong>Subword BPE Tokenizer</strong>: 512-entry compact vocabulary tuned for natural voice instructions.</li>
              <li><strong>2-Layer Transformer Encoder</strong>: Hidden dimension <em>d</em><sub>model</sub> = 128, 4 attention heads (<em>d</em><sub>k</sub> = 32), feed-forward dimension <em>d</em><sub>ff</sub> = 512.</li>
              <li><strong>Classification Routing Head</strong>: Linear projection with calibrated temperature for tool confidence scoring <em>P</em>(tool | <em>x</em>).</li>
              <li><strong>Direct Slot Decoders</strong>: Point-wise softmax distributions over bounded categorical slots (<em>room</em>, <em>device</em>, <em>door</em>, <em>action</em>).</li>
            </ol>

            {/* Visual ASCII / Architecture Diagram */}
            <div className="p-5 rounded-[8px] bg-[var(--fill-1)] border border-[var(--line)] font-mono text-[12px] leading-[1.5] text-[var(--fg)] overflow-x-auto my-6 select-none">
              <div className="text-[var(--fg-3)] mb-2">// MARA AFM ARCHITECTURE (692k PARAMETERS)</div>
{`   [User Query: "kill the kitchen lights"]
               │
      [512-Token Compact BPE]
               │
    [Embeddings: 128-dim + Pos]
               │
    ┌──────────────────────────┐
    │ Transformer Layer 1 (x4) │  <-- Self-Attention (128d) + MLP (512d)
    └──────────┬───────────────┘
               │
    ┌──────────────────────────┐
    │ Transformer Layer 2 (x4) │  <-- Self-Attention (128d) + MLP (512d)
    └──────────┬───────────────┘
               │  (Single Forward Pass, 0.72 ms)
     ┌─────────┴───────────────────────┐
     ▼                                 ▼
[Tool Classification Head]    [Slot Decoding Heads]
P(set_lights)      = 0.98     room       -> "kitchen" (0.99)
P(set_thermostat)  = 0.01     state      -> false (0.97)
P(none / refuse)   = 0.01     brightness -> 0% (inferred)`}
            </div>

            <p>
              Because the slot heads project directly onto valid enumerated candidates, <strong>it is mathematically impossible for Mara to emit malformed JSON or fabricate a non-existent parameter name</strong>. Syntax errors are eliminated by construction.
            </p>
          </section>

          {/* Section 3 */}
          <section className="space-y-4 pt-6 border-t border-[var(--line)]">
            <h2 className="text-[24px] sm:text-[28px] leading-[1.25] font-[600] tracking-[-0.02em]">
              3. The Honest Head-to-Head: Mara vs. Needle 3
            </h2>
            <p>
              In engineering, if you don't audit yourself ruthlessly, reality will audit you in production.
            </p>
            <p>
              In our initial experiments, we pitted Mara against Cactus Needle 3, and Mara looked 100× superior in every chart. But a rigorous technical review called us out on three real flaws:
            </p>
            <ol className="list-decimal list-inside space-y-2 text-[var(--fg-2)] pl-2">
              <li>Needle was tested in a stateful loop where conversational history polluted subsequent turns.</li>
              <li>The tool schemas had overlapping docstrings (`control_device` vs `set_lights`).</li>
              <li>We were comparing a domain-specialized Mara against Needle Base without fine-tuning.</li>
            </ol>
            <p>
              So we tore down the old benchmark and rebuilt it properly. We gave Needle fair, disjoint `typing.Literal` definitions, fine-tuned a Needle 3 Specialist using rank-16 LoRA on the exact same 1,650 domain training prompts, and evaluated both on a frozen 250-sample test suite with <strong>zero n-gram collisions</strong>.
            </p>
            <p>
              Here are the verbatim results from our isolated whole-process benchmark:
            </p>

            {/* Data Comparison Table */}
            <div className="overflow-x-auto my-4">
              <table className="w-full text-[13px] border-collapse text-left font-mono">
                <thead>
                  <tr className="border-b border-[var(--line)] text-[var(--fg-3)]">
                    <th className="py-2.5 pr-4 font-medium">METRIC</th>
                    <th className="py-2.5 px-4 font-medium text-[var(--fg)]">MARA AFM</th>
                    <th className="py-2.5 px-4 font-medium">NEEDLE 3 BASE</th>
                    <th className="py-2.5 pl-4 font-medium text-[var(--fg)]">NEEDLE 3 SPECIALIST</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--line)] text-[var(--fg-2)]">
                  <tr>
                    <td className="py-2.5 pr-4 font-medium font-sans">Model Size on Disk</td>
                    <td className="py-2.5 px-4 text-emerald-500 font-bold">0.15 MB (153 KB)</td>
                    <td className="py-2.5 px-4">35.34 MB</td>
                    <td className="py-2.5 pl-4">63.44 MB</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-medium font-sans">Median Latency (CPU)</td>
                    <td className="py-2.5 px-4 text-emerald-500 font-bold">0.72 ms</td>
                    <td className="py-2.5 px-4">1,002.5 ms</td>
                    <td className="py-2.5 pl-4">1,543.8 ms</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-medium font-sans">Speedup Factor</td>
                    <td className="py-2.5 px-4 text-emerald-500 font-bold">2,143× faster</td>
                    <td className="py-2.5 px-4">1.0×</td>
                    <td className="py-2.5 pl-4">0.65×</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-medium font-sans">Frozen 250 OEM (+ Plan)</td>
                    <td className="py-2.5 px-4 text-[var(--fg)] font-bold">72.4% [66.6–77.6%]</td>
                    <td className="py-2.5 px-4">55.2% [49.0–61.2%]</td>
                    <td className="py-2.5 pl-4">65.2% [59.1–70.8%]</td>
                  </tr>
                  <tr>
                    <td className="py-2.5 pr-4 font-medium font-sans">False Positive Rate (Refusal)</td>
                    <td className="py-2.5 px-4 text-emerald-500">12.0% (6/50)</td>
                    <td className="py-2.5 px-4">20.0% (10/50)</td>
                    <td className="py-2.5 pl-4">12.0% (6/50)</td>
                  </tr>
                </tbody>
              </table>
            </div>

            <h3 className="text-[17px] font-[600] text-[var(--fg)] pt-2">The Real Conclusion</h3>
            <p>
              Notice something very important in those numbers: <strong>Mara is not a magic bullet on colloquial phrasing</strong>. On independently authored paraphrases featuring heavy slang (<em>“nuke the salon spotlights”</em>, <em>“kill the chill in the master”</em>), Mara’s accuracy drops to 44.0%. A general-purpose 70B LLM with extensive pretraining can parse slang easily because it has seen the entire internet.
            </p>
            <p>
              However, on the operational axis, Mara operates in a completely different universe:
            </p>
            <ul className="list-disc list-inside space-y-2 text-[var(--fg-2)] pl-2">
              <li>It is <strong>2,143× faster</strong> than the LoRA fine-tuned specialist.</li>
              <li>Its ONNX runtime footprint is <strong>153 kilobytes</strong>—smaller than the favicon on most web pages.</li>
              <li>It executes deterministically in steady-state memory with zero allocation spikes.</li>
            </ul>
          </section>

          {/* Section 4 */}
          <section className="space-y-4 pt-6 border-t border-[var(--line)]">
            <h2 className="text-[24px] sm:text-[28px] leading-[1.25] font-[600] tracking-[-0.02em]">
              4. Running 100% Client-Side in the Browser (WebAssembly)
            </h2>
            <p>
              One of the most liberating consequences of building micro-models is that <strong>you do not need a backend server</strong>.
            </p>
            <p>
              In our live web playground, when you type an instruction or press an example chip:
            </p>
            <ol className="list-decimal list-inside space-y-2 text-[var(--fg-2)] pl-2">
              <li>The browser streams <code>mara.onnx</code> directly from our Hugging Face model repository (<code>jaswanthsanjay88/mara</code>) via the Fetch API and <code>ReadableStream</code>.</li>
              <li>The binary is cached in the browser's persistent <code>CacheStorage</code>. Repeated page reloads load from disk in under 20 milliseconds with zero network requests.</li>
              <li>Inference runs inside the user's browser tab using <code>onnxruntime-web</code> compiled to WebAssembly.</li>
              <li>The complete inference forward pass executes in <strong>2 to 4 milliseconds</strong> on a single thread.</li>
            </ol>
            <p>
              No cloud compute bills. No GPUs spinning in a datacenter. No telemetry or privacy concerns, because the user's audio and text never leave their browser sandbox.
            </p>
          </section>

          {/* Section 5 */}
          <section className="space-y-4 pt-6 border-t border-[var(--line)]">
            <h2 className="text-[24px] sm:text-[28px] leading-[1.25] font-[600] tracking-[-0.02em]">
              5. The PyTorch Implementation in 40 Lines
            </h2>
            <p>
              Here is the conceptual core of how the AFM forward pass is implemented in pure PyTorch:
            </p>

            <div className="p-4 rounded-[8px] bg-[var(--fill-1)] border border-[var(--line)] font-mono text-[12.5px] leading-[1.65] text-[var(--fg-2)] overflow-x-auto">
              <pre><code>{`import torch
import torch.nn as nn

class MaraAFM(nn.Module):
    def __init__(self, vocab_size=512, d_model=128, nhead=4, num_tools=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, 64, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512,
            batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Classification head for tool selection
        self.tool_head = nn.Linear(d_model, num_tools + 1)
        
        # Explicit slot decoding heads (bounded categorical spaces)
        self.room_head = nn.Linear(d_model, 5)        # living, kitchen, bed, bath, garage
        self.action_head = nn.Linear(d_model, 4)      # on, off, open, close
        self.temp_head = nn.Linear(d_model, 1)        # continuous regression scalar
        
    def forward(self, input_ids):
        # input_ids: [batch_size, seq_len]
        x = self.embedding(input_ids) + self.pos_encoder[:, :input_ids.size(1), :]
        features = self.transformer(x)
        
        # Mean pool over sequence length for global representation
        pooled = features.mean(dim=1)
        
        tool_logits = self.tool_head(pooled)
        room_logits = self.room_head(pooled)
        action_logits = self.action_head(pooled)
        temp_val = self.temp_head(pooled)
        
        return tool_logits, room_logits, action_logits, temp_val`}</code></pre>
            </div>
          </section>

          {/* Section 6 - What's Next */}
          <section className="space-y-4 pt-6 border-t border-[var(--line)]">
            <h2 className="text-[24px] sm:text-[28px] leading-[1.25] font-[600] tracking-[-0.02em]">
              6. What's Next & The Road Ahead
            </h2>
            <p>
              Small models are not dead. In fact, for edge devices, automobiles, IoT hubs, and microcontrollers, <strong>micro-transformers are the only path to deterministic, zero-latency software</strong>.
            </p>
            <p>
              Our roadmap for Mara AFM includes:
            </p>
            <ul className="list-disc list-inside space-y-2 text-[var(--fg-2)] pl-2">
              <li><strong>Learned Clause Segmentation</strong>: Replacing our rule-based planner with a 40k parameter sub-network to parse conjunctions like <em>“before you lock up, kill the fan”</em> in true execution order.</li>
              <li><strong>INT8 & W4 Quantization</strong>: Compressing the 153 KB graph down to <strong>42 KB</strong> to execute comfortably on ARM Cortex-M microcontrollers.</li>
              <li><strong>Streaming Acoustic AFM</strong>: Connecting the encoder directly to raw audio filterbank features, bypassing text tokenization altogether for true end-to-end voice-to-device actuation.</li>
            </ul>

            <div className="p-6 rounded-[10px] bg-[var(--fill-1)] border border-[var(--line)] space-y-3 mt-8">
              <h4 className="text-[15px] font-[600] text-[var(--fg)]">Try It Live in Your Browser</h4>
              <p className="text-[14px] leading-[1.6] text-[var(--fg-2)]">
                The full Mara AFM model is open-weights and available on Hugging Face. You can experiment with custom tool JSONs, test compound instructions, and watch the 2D floor plan actuate in real-time.
              </p>
              <div className="flex flex-wrap items-center gap-3 pt-2">
                <Link
                  href="/"
                  className="inline-flex items-center gap-2 px-4 py-2 rounded-[6px] text-[13px] font-[500] bg-[var(--fg)] text-[var(--bg)] hover:opacity-90 transition-opacity"
                >
                  <Play size={14} fill="currentColor" />
                  <span>Open Interactive Playground</span>
                </Link>
                <a
                  href="https://huggingface.co/jaswanthsanjay88/mara"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1.5 px-4 py-2 rounded-[6px] text-[13px] font-[500] border border-[var(--line)] text-[var(--fg)] hover:bg-[var(--fill-2)] transition-colors"
                >
                  <span>Model on Hugging Face</span>
                  <ExternalLink size={13} />
                </a>
              </div>
            </div>
          </section>
        </article>

        {/* Footer */}
        <footer className="mt-16 pt-8 border-t border-[var(--line)] flex flex-col sm:flex-row items-center justify-between gap-4 text-[13px] text-[var(--fg-3)] font-mono">
          <div>Mara Research • Released under Apache 2.0</div>
          <div className="flex items-center gap-4">
            <a href="/feed.xml" className="hover:text-[var(--fg)] transition-colors">RSS</a>
            <span>•</span>
            <a href="https://github.com/jaswanthsanjay88/mara" target="_blank" rel="noopener noreferrer" className="hover:text-[var(--fg)] transition-colors">GitHub</a>
            <span>•</span>
            <a href="https://huggingface.co/jaswanthsanjay88/mara" target="_blank" rel="noopener noreferrer" className="hover:text-[var(--fg)] transition-colors">Hugging Face</a>
          </div>
        </footer>
      </main>
    </div>
  );
}
