import { useState, useEffect, useRef, useCallback } from "react";

// ── Complexity metrics (JS port of Python framework) ──────────────────────
function rowEntropy(row) {
  const p1 = row.reduce((s, v) => s + v, 0) / row.length;
  const p0 = 1 - p1;
  if (p1 <= 0 || p0 <= 0) return 0;
  return -(p1 * Math.log2(p1) + p0 * Math.log2(p0));
}

function computeMetrics(history, burnin, window_) {
  const T = history.length, W = history[0].length;
  if (T < burnin + window_) return null;

  // entropy stats
  const Hs = [];
  for (let t = burnin; t < burnin + window_; t++) {
    Hs.push(rowEntropy(history[t]));
  }
  const meanH = Hs.reduce((a, b) => a + b, 0) / Hs.length;
  const stdH = Math.sqrt(Hs.map(h => (h - meanH) ** 2).reduce((a, b) => a + b, 0) / Hs.length);

  // opacity upward H(global|local)
  const joint = {}, marg = {};
  const stride = Math.max(1, Math.floor(W / 200));
  for (let t = burnin; t < burnin + window_; t++) {
    const row = history[t];
    const density = row.reduce((s, v) => s + v, 0) / W;
    const gbin = Math.min(Math.floor(density * 8), 7);
    for (let x = 0; x < W; x += stride) {
      const p0 = row[(x - 1 + W) % W], p1 = row[x], p2 = row[(x + 1) % W];
      const pk = `${p0}${p1}${p2}`;
      const jk = `${pk}|${gbin}`;
      joint[jk] = (joint[jk] || 0) + 1;
      marg[pk] = (marg[pk] || 0) + 1;
    }
  }
  const total = Object.values(joint).reduce((a, b) => a + b, 0);
  const lt = Object.values(marg).reduce((a, b) => a + b, 0);
  let hj = 0, hl = 0;
  for (const c of Object.values(joint)) if (c > 0) hj -= (c / total) * Math.log2(c / total);
  for (const c of Object.values(marg)) if (c > 0) hl -= (c / lt) * Math.log2(c / lt);
  const opUp = Math.max(0, Math.min(1, (hj - hl) / Math.log2(8)));

  // tcomp
  let tcSum = 0;
  for (let x = 0; x < W; x++) {
    let flips = 0;
    for (let t = burnin + 1; t < burnin + window_; t++) {
      if (history[t][x] !== history[t - 1][x]) flips++;
    }
    tcSum += Math.max(0, 1 - (1 + flips) / window_);
  }
  const tcomp = tcSum / W;

  // gzip approximation via entropy-based estimate
  // (true zlib not available in browser — use spatial entropy as proxy)
  const spatialEntropy = meanH;
  const gzipApprox = spatialEntropy * 0.16; // rough linear approximation

  // weights
  const gauss = (x, mu, sig) => Math.exp(-0.5 * ((x - mu) / sig) ** 2);
  const wH = Math.tanh(50 * meanH) * Math.tanh(50 * (1 - meanH)) * (1 + gauss(stdH, 0.012, 0.008));
  const wOP = gauss(opUp, 0.14, 0.10);
  const wT = Math.max(gauss(tcomp, 0.58, 0.08), gauss(tcomp, 0.73, 0.08));
  const wG = gauss(gzipApprox, 0.10, 0.05);
  const score = wH * wOP * wT * wG;

  return { meanH, stdH, opUp, tcomp, gzipApprox, wH, wOP, wT, wG, score };
}

// ── Simulation ────────────────────────────────────────────────────────────
function stepLife(grid, size, birth, survive) {
  const next = new Uint8Array(size * size);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      let nb = 0;
      for (let dy = -1; dy <= 1; dy++)
        for (let dx2 = -1; dx2 <= 1; dx2++) {
          if (dy === 0 && dx2 === 0) continue;
          const nx = (x + dx2 + size) % size;
          const ny = (y + dy + size) % size;
          nb += grid[ny * size + nx];
        }
      const alive = grid[y * size + x];
      next[y * size + x] = alive ? (survive.has(nb) ? 1 : 0) : (birth.has(nb) ? 1 : 0);
    }
  }
  return next;
}

function randomGrid(size, density = 0.35, seed = 42) {
  let s = seed;
  const lcg = () => { s = (s * 1664525 + 1013904223) & 0xffffffff; return (s >>> 0) / 0xffffffff; };
  const g = new Uint8Array(size * size);
  for (let i = 0; i < size * size; i++) g[i] = lcg() < density ? 1 : 0;
  return g;
}

function parseRule(str) {
  const [bPart, sPart] = str.split("/");
  const birth = new Set((bPart.replace(/^B/i, "")).split("").map(Number).filter(n => !isNaN(n)));
  const survive = new Set((sPart.replace(/^S/i, "")).split("").map(Number).filter(n => !isNaN(n)));
  return { birth, survive };
}

// ── Preset rules ──────────────────────────────────────────────────────────
const PRESETS = [
  { name: "Conway's Life", rule: "B3/S23", class: "C4", note: "The classic. Supports gliders, guns, universal computation." },
  { name: "HighLife", rule: "B36/S23", class: "C4", note: "Like Life but with a replicator pattern." },
  { name: "Morley", rule: "B368/S245", class: "C4", note: "Complex glider-rich dynamics." },
  { name: "Day & Night", rule: "B3678/S34678", class: "C4", note: "Symmetric — dead and alive are interchangeable." },
  { name: "34 Life", rule: "B34/S34", class: "C3", note: "Chaotic explosion. No stable structures." },
  { name: "Gnarl", rule: "B1/S1", class: "C3", note: "Maximally chaotic. Every cell is reactive." },
  { name: "Maze", rule: "B3/S12345", class: "C2", note: "Grows labyrinthine stable mazes. Our known edge case." },
  { name: "Seeds", rule: "B2/S", class: "C2", note: "Everything is born, nothing survives. Explosive." },
  { name: "Coral", rule: "B3/S45678", class: "C2", note: "Grows coral-like structures that freeze in place." },
  { name: "Survey #19", rule: "B015678/S013467", class: "?", note: "Top candidate from our 1000-rule survey. Holds up at full resolution." },
  { name: "Survey #10", rule: "B013468/S013457", class: "?", note: "Second-best survey candidate at full resolution." },
  { name: "Survey #2", rule: "B257/S02478", class: "?", note: "High survey score. Degrades at larger grids." },
];

const CLASS_COLOR = { C4: "#2ecc71", C3: "#e74c3c", C2: "#3498db", C1: "#95a5a6", "?": "#f39c12" };

// ── Gauge component ───────────────────────────────────────────────────────
function Gauge({ label, value, max = 1, color = "#2ecc71" }) {
  const pct = Math.min(Math.max(value / max, 0), 1);
  return (
    <div style={{ marginBottom: 6 }}>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 10, color: "#888", marginBottom: 2, fontFamily: "'DM Mono', monospace" }}>
        <span>{label}</span>
        <span style={{ color: "#ccc" }}>{value.toFixed(4)}</span>
      </div>
      <div style={{ height: 4, background: "#1a1a2a", borderRadius: 2, overflow: "hidden" }}>
        <div style={{ height: "100%", width: `${pct * 100}%`, background: color, borderRadius: 2, transition: "width 0.3s ease" }} />
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────
export default function LifeExplorer() {
  const SIZE = 80;
  const BURNIN = 15;
  const WINDOW = 40;

  const [ruleStr, setRuleStr] = useState("B3/S23");
  const [inputStr, setInputStr] = useState("B3/S23");
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(8);
  const [density, setDensity] = useState(0.35);
  const [seed, setSeed] = useState(42);
  const [generation, setGeneration] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [history, setHistory] = useState([]);
  const [selectedPreset, setSelectedPreset] = useState(0);
  const [customMode, setCustomMode] = useState(false);
  const [colorMode, setColorMode] = useState("heat"); // heat | mono | fade

  const gridRef = useRef(null);
  const animRef = useRef(null);
  const stateRef = useRef({ grid: null, history: [], generation: 0 });
  const canvasRef = useRef(null);

  const { birth, survive } = parseRule(ruleStr);

  const reset = useCallback((rule = ruleStr, d = density, s = seed) => {
    const g = randomGrid(SIZE, d, s);
    stateRef.current = { grid: g, history: [Array.from(g)], generation: 0 };
    setGeneration(0);
    setMetrics(null);
    setHistory([Array.from(g)]);
  }, [ruleStr, density, seed]);

  useEffect(() => { reset(ruleStr, density, seed); }, []);

  // Canvas rendering
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !stateRef.current.grid) return;
    const ctx = canvas.getContext("2d");
    const cellSize = canvas.width / SIZE;
    const grid = stateRef.current.grid;
    const gen = stateRef.current.generation;

    ctx.fillStyle = "#0a0a12";
    ctx.fillRect(0, 0, canvas.width, canvas.height);

    for (let y = 0; y < SIZE; y++) {
      for (let x = 0; x < SIZE; x++) {
        const v = grid[y * SIZE + x];
        if (v === 0) continue;
        if (colorMode === "mono") {
          ctx.fillStyle = "#e0e0ff";
        } else if (colorMode === "heat") {
          // color based on local density
          let nb = 0;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++) {
              if (dy === 0 && dx === 0) continue;
              nb += grid[((y + dy + SIZE) % SIZE) * SIZE + (x + dx + SIZE) % SIZE];
            }
          const h = nb / 8;
          const r = Math.floor(46 + h * 209);
          const g2 = Math.floor(204 - h * 170);
          const b2 = Math.floor(113 + h * 30);
          ctx.fillStyle = `rgb(${r},${g2},${b2})`;
        } else {
          // fade — older = dimmer (not easily available w/o per-cell age tracking)
          ctx.fillStyle = `rgba(160,200,255,0.9)`;
        }
        ctx.fillRect(x * cellSize, y * cellSize, cellSize - 0.5, cellSize - 0.5);
      }
    }
  });

  // Simulation loop
  useEffect(() => {
    if (!running) { cancelAnimationFrame(animRef.current); return; }
    let last = 0;
    const interval = 1000 / speed;
    const loop = (ts) => {
      animRef.current = requestAnimationFrame(loop);
      if (ts - last < interval) return;
      last = ts;
      const { birth: b, survive: s } = parseRule(ruleStr);
      const next = stepLife(stateRef.current.grid, SIZE, b, s);
      const hist = stateRef.current.history;
      hist.push(Array.from(next));
      if (hist.length > BURNIN + WINDOW + 5) hist.shift();
      stateRef.current.grid = next;
      stateRef.current.generation++;
      setGeneration(g => g + 1);

      if (hist.length >= BURNIN + WINDOW) {
        const m = computeMetrics(hist, BURNIN, WINDOW);
        setMetrics(m);
      }
    };
    animRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animRef.current);
  }, [running, speed, ruleStr]);

  const applyRule = () => {
    try {
      parseRule(inputStr);
      setRuleStr(inputStr);
      reset(inputStr, density, seed);
      setRunning(false);
    } catch {}
  };

  const selectPreset = (i) => {
    setSelectedPreset(i);
    const r = PRESETS[i].rule;
    setRuleStr(r);
    setInputStr(r);
    reset(r, density, seed);
    setRunning(false);
    setCustomMode(false);
  };

  const stepOnce = () => {
    const { birth: b, survive: s } = parseRule(ruleStr);
    const next = stepLife(stateRef.current.grid, SIZE, b, s);
    const hist = stateRef.current.history;
    hist.push(Array.from(next));
    if (hist.length > BURNIN + WINDOW + 5) hist.shift();
    stateRef.current.grid = next;
    stateRef.current.generation++;
    setGeneration(g => g + 1);
    if (hist.length >= BURNIN + WINDOW) {
      setMetrics(computeMetrics(hist, BURNIN, WINDOW));
    }
  };

  // B/S toggles
  const birthSet = new Set(ruleStr.split("/")[0].replace(/^B/i,"").split("").map(Number).filter(n=>!isNaN(n)));
  const survSet  = new Set(ruleStr.split("/")[1]?.replace(/^S/i,"").split("").map(Number).filter(n=>!isNaN(n)) || []);

  const toggleBit = (set, n, type) => {
    const s = new Set(set);
    s.has(n) ? s.delete(n) : s.add(n);
    const bStr = "B" + [...(type==="birth"?s:birthSet)].sort().join("");
    const sStr = "S" + [...(type==="survive"?s:survSet)].sort().join("");
    const r = `${bStr}/${sStr}`;
    setRuleStr(r); setInputStr(r);
    reset(r, density, seed); setRunning(false); setCustomMode(true);
  };

  const preset = PRESETS[selectedPreset];
  const clsColor = CLASS_COLOR[customMode ? "?" : preset?.class] || "#888";

  return (
    <div style={{
      minHeight: "100vh", background: "#07070f",
      fontFamily: "'DM Mono', 'Courier New', monospace",
      color: "#ccc", display: "flex", flexDirection: "column",
    }}>
      {/* Header */}
      <div style={{
        borderBottom: "1px solid #1a1a2e", padding: "14px 24px",
        display: "flex", alignItems: "center", gap: 20, background: "#09091a",
      }}>
        <div>
          <div style={{ fontSize: 11, color: "#555", letterSpacing: 3, textTransform: "uppercase" }}>complexity framework</div>
          <div style={{ fontSize: 18, color: "#e0e0ff", fontWeight: 600, letterSpacing: -0.5 }}>
            2D Life-like Rule Explorer
          </div>
        </div>
        <div style={{ flex: 1 }} />
        <div style={{ fontSize: 10, color: "#444", textAlign: "right", lineHeight: 1.6 }}>
          <div>262,144 possible rules</div>
          <div>outer-totalistic B/S notation</div>
        </div>
      </div>

      <div style={{ display: "flex", flex: 1, minHeight: 0 }}>

        {/* Left: Presets panel */}
        <div style={{
          width: 220, borderRight: "1px solid #1a1a2e",
          background: "#080814", overflowY: "auto", padding: 12,
        }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 2, textTransform: "uppercase", marginBottom: 10 }}>
            preset rules
          </div>
          {PRESETS.map((p, i) => (
            <div key={i}
              onClick={() => selectPreset(i)}
              style={{
                padding: "8px 10px", marginBottom: 4, borderRadius: 4,
                cursor: "pointer", border: "1px solid",
                borderColor: (!customMode && selectedPreset === i) ? clsColor : "#1a1a2e",
                background: (!customMode && selectedPreset === i) ? "#0d0d22" : "transparent",
                transition: "all 0.15s",
              }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <span style={{ fontSize: 11, color: "#ddd" }}>{p.name}</span>
                <span style={{
                  fontSize: 9, padding: "1px 5px", borderRadius: 3,
                  background: CLASS_COLOR[p.class] + "22",
                  color: CLASS_COLOR[p.class], border: `1px solid ${CLASS_COLOR[p.class]}44`
                }}>{p.class}</span>
              </div>
              <div style={{ fontSize: 9, color: "#555", marginTop: 2, fontFamily: "monospace" }}>{p.rule}</div>
            </div>
          ))}
          <div style={{ height: 1, background: "#1a1a2e", margin: "12px 0" }} />
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>
            survey top candidates
          </div>
          <div style={{ fontSize: 9, color: "#555", lineHeight: 1.7 }}>
            1,000 random rules sampled from 262,144 space. Top candidates by complexity score.
          </div>
        </div>

        {/* Center: Canvas + controls */}
        <div style={{ flex: 1, display: "flex", flexDirection: "column", alignItems: "center", padding: 20, gap: 14 }}>

          {/* Rule display */}
          <div style={{ display: "flex", alignItems: "center", gap: 12, width: "100%", maxWidth: 560 }}>
            <div style={{
              flex: 1, background: "#0d0d1e", border: "1px solid #2a2a4a",
              borderRadius: 6, padding: "8px 14px",
              display: "flex", alignItems: "center", gap: 10,
            }}>
              <span style={{ fontSize: 10, color: "#555" }}>RULE</span>
              <input value={inputStr} onChange={e => setInputStr(e.target.value.toUpperCase())}
                onKeyDown={e => e.key === "Enter" && applyRule()}
                style={{
                  background: "none", border: "none", outline: "none",
                  color: "#e0e0ff", fontSize: 16, fontFamily: "inherit",
                  letterSpacing: 1, flex: 1,
                }} />
              <button onClick={applyRule} style={{
                background: "#1a1a3a", border: "1px solid #3a3a6a",
                color: "#aaa", borderRadius: 4, padding: "3px 10px",
                cursor: "pointer", fontSize: 10,
              }}>APPLY</button>
            </div>
            <div style={{
              padding: "6px 12px", borderRadius: 4,
              background: clsColor + "18", border: `1px solid ${clsColor}44`,
              fontSize: 11, color: clsColor, minWidth: 50, textAlign: "center",
            }}>
              {customMode ? "CUSTOM" : preset?.class}
            </div>
          </div>

          {/* B/S toggles */}
          <div style={{ display: "flex", gap: 20, width: "100%", maxWidth: 560 }}>
            {[["Birth (B)", birthSet, "birth"], ["Survive (S)", survSet, "survive"]].map(([label, set, type]) => (
              <div key={type} style={{ flex: 1 }}>
                <div style={{ fontSize: 9, color: "#555", letterSpacing: 2, textTransform: "uppercase", marginBottom: 6 }}>{label}</div>
                <div style={{ display: "flex", gap: 5 }}>
                  {[0,1,2,3,4,5,6,7,8].map(n => (
                    <div key={n} onClick={() => toggleBit(set, n, type)}
                      style={{
                        width: 28, height: 28, borderRadius: 4,
                        display: "flex", alignItems: "center", justifyContent: "center",
                        fontSize: 11, cursor: "pointer", border: "1px solid",
                        borderColor: set.has(n) ? clsColor : "#1a1a2e",
                        background: set.has(n) ? clsColor + "25" : "#0a0a16",
                        color: set.has(n) ? clsColor : "#444",
                        transition: "all 0.1s",
                        userSelect: "none",
                      }}>{n}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Canvas */}
          <div style={{
            border: "1px solid #1a1a2e", borderRadius: 8, overflow: "hidden",
            boxShadow: "0 0 40px rgba(46,204,113,0.04)",
            position: "relative",
          }}>
            <canvas ref={canvasRef} width={480} height={480}
              style={{ display: "block" }} />
            <div style={{
              position: "absolute", bottom: 8, left: 10,
              fontSize: 9, color: "#334", fontFamily: "monospace",
            }}>gen {generation}</div>
          </div>

          {/* Transport controls */}
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <button onClick={() => { reset(ruleStr, density, seed); setRunning(false); }}
              style={btnStyle("#1a1a2e", "#888")}>RESET</button>
            <button onClick={stepOnce} disabled={running}
              style={btnStyle("#1a1a2e", running ? "#333" : "#aaa")}>STEP</button>
            <button onClick={() => setRunning(r => !r)}
              style={btnStyle(running ? "#2ecc7122" : "#1a1a2e", running ? "#2ecc71" : "#aaa")}>
              {running ? "⏸ PAUSE" : "▶ RUN"}
            </button>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 10 }}>
              <span style={{ fontSize: 9, color: "#444" }}>SPEED</span>
              <input type="range" min={1} max={30} value={speed}
                onChange={e => setSpeed(+e.target.value)}
                style={{ width: 70, accentColor: "#2ecc71" }} />
              <span style={{ fontSize: 9, color: "#666" }}>{speed}/s</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 9, color: "#444" }}>COLOR</span>
              <select value={colorMode} onChange={e => setColorMode(e.target.value)}
                style={{ background: "#0d0d1e", border: "1px solid #2a2a4a", color: "#aaa", fontSize: 10, borderRadius: 4, padding: "2px 4px" }}>
                <option value="heat">heat</option>
                <option value="mono">mono</option>
                <option value="fade">blue</option>
              </select>
            </div>
          </div>

          {/* Density + seed */}
          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 9, color: "#444" }}>DENSITY</span>
              <input type="range" min={5} max={80} value={Math.round(density * 100)}
                onChange={e => { const d = +e.target.value/100; setDensity(d); reset(ruleStr, d, seed); }}
                style={{ width: 70, accentColor: "#3498db" }} />
              <span style={{ fontSize: 9, color: "#666" }}>{Math.round(density * 100)}%</span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
              <span style={{ fontSize: 9, color: "#444" }}>SEED</span>
              <input type="number" value={seed} onChange={e => { const s = +e.target.value; setSeed(s); reset(ruleStr, density, s); }}
                style={{ width: 55, background: "#0d0d1e", border: "1px solid #2a2a4a", color: "#aaa", fontSize: 10, borderRadius: 4, padding: "2px 6px", fontFamily: "inherit" }} />
            </div>
          </div>

          {/* Rule note */}
          {!customMode && preset && (
            <div style={{
              fontSize: 10, color: "#556", textAlign: "center",
              maxWidth: 460, lineHeight: 1.6, fontStyle: "italic",
            }}>{preset.note}</div>
          )}
        </div>

        {/* Right: Metrics panel */}
        <div style={{
          width: 210, borderLeft: "1px solid #1a1a2e",
          background: "#080814", padding: 14,
        }}>
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 2, textTransform: "uppercase", marginBottom: 14 }}>
            complexity metrics
          </div>

          {metrics ? (
            <>
              {/* Composite score */}
              <div style={{
                background: "#0d0d20", border: "1px solid #2a2a4a",
                borderRadius: 6, padding: "10px 12px", marginBottom: 14,
                textAlign: "center",
              }}>
                <div style={{ fontSize: 9, color: "#555", marginBottom: 4 }}>COMPOSITE C</div>
                <div style={{
                  fontSize: 28, fontWeight: 700,
                  color: scoreColor(metrics.score),
                  transition: "color 0.4s",
                }}>
                  {metrics.score.toFixed(4)}
                </div>
                <div style={{ fontSize: 9, color: "#555", marginTop: 4 }}>
                  {metrics.score > 0.5 ? "★ high complexity" :
                   metrics.score > 0.1 ? "◈ moderate" :
                   metrics.score > 0.01 ? "○ low" : "· trivial"}
                </div>
              </div>

              <Gauge label="w_H (entropy)" value={metrics.wH} max={2} color="#f39c12" />
              <Gauge label="w_OP (opacity↑)" value={metrics.wOP} max={1} color="#9b59b6" />
              <Gauge label="w_T (tcomp)" value={metrics.wT} max={1} color="#3498db" />
              <Gauge label="w_G (gzip proxy)" value={metrics.wG} max={1} color="#1abc9c" />

              <div style={{ height: 1, background: "#1a1a2e", margin: "12px 0" }} />

              <div style={{ fontSize: 9, color: "#444", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>raw metrics</div>
              {[
                ["mean H", metrics.meanH.toFixed(4)],
                ["std H", metrics.stdH.toFixed(4)],
                ["opacity ↑", metrics.opUp.toFixed(4)],
                ["t_comp", metrics.tcomp.toFixed(4)],
                ["gz proxy", metrics.gzipApprox.toFixed(4)],
              ].map(([k, v]) => (
                <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: 10, marginBottom: 4, color: "#888" }}>
                  <span>{k}</span><span style={{ color: "#bbb" }}>{v}</span>
                </div>
              ))}

              <div style={{ height: 1, background: "#1a1a2e", margin: "12px 0" }} />
              <div style={{ fontSize: 9, color: "#444", lineHeight: 1.7 }}>
                Metrics update live. Run for {BURNIN + WINDOW}+ generations for stable readings.
              </div>
              <div style={{ marginTop: 8, fontSize: 9, color: "#333", lineHeight: 1.7 }}>
                ⚠ gzip is approximated in browser (entropy proxy). True values differ from Python framework.
              </div>
            </>
          ) : (
            <div style={{ fontSize: 10, color: "#444", lineHeight: 1.8 }}>
              Run the simulation for {BURNIN + WINDOW}+ generations to compute metrics.
              <br /><br />
              Metrics require a burn-in period to measure the rule's attractor behaviour rather than transient IC effects.
            </div>
          )}

          <div style={{ height: 1, background: "#1a1a2e", margin: "16px 0" }} />
          <div style={{ fontSize: 9, color: "#444", letterSpacing: 2, textTransform: "uppercase", marginBottom: 8 }}>about</div>
          <div style={{ fontSize: 9, color: "#444", lineHeight: 1.7 }}>
            C = w_H × w_OP × w_T × w_G<br /><br />
            Multiplicative: a rule must score well on all dimensions simultaneously.<br /><br />
            Parameters calibrated on 1D ECA Class 4 rules, then held fixed across substrates.
          </div>
        </div>
      </div>
    </div>
  );
}

function btnStyle(bg, color) {
  return {
    background: bg, border: `1px solid ${color}44`,
    color, borderRadius: 5, padding: "6px 14px",
    cursor: "pointer", fontSize: 11, fontFamily: "inherit",
    letterSpacing: 1,
  };
}

function scoreColor(s) {
  if (s > 0.5) return "#2ecc71";
  if (s > 0.1) return "#f39c12";
  if (s > 0.01) return "#e74c3c";
  return "#555";
}
