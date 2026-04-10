import { useState, useEffect, useRef, useCallback } from "react";

function gauss(x, mu, sig) { return Math.exp(-0.5 * ((x - mu) / sig) ** 2); }
function rowEntropy(row) {
  const p1 = row.reduce((s, v) => s + v, 0) / row.length;
  const p0 = 1 - p1;
  if (p1 <= 0 || p0 <= 0) return 0;
  return -(p1 * Math.log2(p1 + 1e-12) + p0 * Math.log2(p0 + 1e-12));
}
function computeMetrics(history, burnin, win) {
  if (history.length < burnin + win) return null;
  const W = history[0].length;
  const Hs = [];
  for (let t = burnin; t < burnin + win; t++) Hs.push(rowEntropy(history[t]));
  const meanH = Hs.reduce((a, b) => a + b, 0) / Hs.length;
  const stdH = Math.sqrt(Hs.map(h => (h - meanH) ** 2).reduce((a, b) => a + b, 0) / Hs.length);
  const joint = {}, marg = {};
  const stride = Math.max(1, Math.floor(W / 120));
  for (let t = burnin; t < burnin + win; t++) {
    const row = history[t];
    const gbin = Math.min(Math.floor((row.reduce((a,b)=>a+b,0)/W) * 8), 7);
    for (let x = 0; x < W; x += stride) {
      const pk = `${row[(x-1+W)%W]}${row[x]}${row[(x+1)%W]}`;
      const jk = `${pk}|${gbin}`;
      joint[jk] = (joint[jk] || 0) + 1;
      marg[pk] = (marg[pk] || 0) + 1;
    }
  }
  const total = Object.values(joint).reduce((a,b)=>a+b,0);
  const lt = Object.values(marg).reduce((a,b)=>a+b,0);
  let hj = 0, hl = 0;
  for (const c of Object.values(joint)) if (c>0) hj -= (c/total)*Math.log2(c/total);
  for (const c of Object.values(marg)) if (c>0) hl -= (c/lt)*Math.log2(c/lt);
  const opUp = Math.max(0, Math.min(1, (hj - hl) / Math.log2(8)));
  let tcSum = 0;
  for (let x = 0; x < W; x++) {
    let flips = 0;
    for (let t = burnin+1; t < burnin+win; t++)
      if (history[t][x] !== history[t-1][x]) flips++;
    tcSum += Math.max(0, 1 - (1 + flips) / win);
  }
  const tcomp = tcSum / W;
  const gzProxy = meanH * 0.16;
  const wH = Math.tanh(50*meanH)*Math.tanh(50*(1-meanH))*(1+gauss(stdH,0.012,0.008));
  const wOP = gauss(opUp,0.14,0.10);
  const wT = Math.max(gauss(tcomp,0.58,0.08),gauss(tcomp,0.73,0.08));
  const wG = gauss(gzProxy,0.10,0.05);
  return { meanH, stdH, opUp, tcomp, gzProxy, wH, wOP, wT, wG, score: wH*wOP*wT*wG };
}
function stepLife(grid, size, birth, survive) {
  const next = new Uint8Array(size*size);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      let nb = 0;
      for (let dy=-1;dy<=1;dy++) for (let dx=-1;dx<=1;dx++) {
        if (!dy&&!dx) continue;
        nb += grid[((y+dy+size)%size)*size+(x+dx+size)%size];
      }
      const v = grid[y*size+x];
      next[y*size+x] = v?(survive.has(nb)?1:0):(birth.has(nb)?1:0);
    }
  return next;
}
function randomGrid(size, density, seed) {
  let s=seed;
  const lcg=()=>{s=(s*1664525+1013904223)&0xffffffff;return(s>>>0)/0xffffffff;};
  const g=new Uint8Array(size*size);
  for(let i=0;i<size*size;i++) g[i]=lcg()<density?1:0;
  return g;
}
function parseRule(str) {
  try {
    const [b,s]=str.split("/");
    return {
      birth:new Set((b||"").replace(/^B/i,"").split("").map(Number).filter(n=>!isNaN(n)&&n>=0&&n<=8)),
      survive:new Set((s||"").replace(/^S/i,"").split("").map(Number).filter(n=>!isNaN(n)&&n>=0&&n<=8))
    };
  } catch { return {birth:new Set([3]),survive:new Set([2,3])}; }
}

const PRESETS = [
  {name:"Conway's Life", rule:"B3/S23",          cls:"C4", note:"Universal computation. Gliders, guns, oscillators."},
  {name:"HighLife",      rule:"B36/S23",          cls:"C4", note:"Like Life, but with a self-replicating pattern."},
  {name:"Morley",        rule:"B368/S245",        cls:"C4", note:"Rich glider dynamics."},
  {name:"Day & Night",   rule:"B3678/S34678",     cls:"C4", note:"Symmetric — alive and dead obey the same rules."},
  {name:"34 Life",       rule:"B34/S34",          cls:"C3", note:"Chaotic explosion. No persistent structures."},
  {name:"Gnarl",         rule:"B1/S1",            cls:"C3", note:"Maximum reactivity."},
  {name:"Maze",          rule:"B3/S12345",        cls:"C2", note:"Grows stable labyrinths — our known edge case."},
  {name:"Coral",         rule:"B3/S45678",        cls:"C2", note:"Coral-like growth, freezes quickly."},
  {name:"Seeds",         rule:"B2/S",             cls:"C2", note:"Born but never survive — explosive."},
  {name:"Survey #19 ★", rule:"B015678/S013467",  cls:"?",  note:"Best candidate from our 1000-rule random survey."},
  {name:"Survey #10",    rule:"B013468/S013457",  cls:"?",  note:"Second-best at full resolution."},
  {name:"Survey #1",     rule:"B01256/S012356",   cls:"?",  note:"Top small-grid score, degrades at full res."},
];
const CC={C4:"#2ecc71",C3:"#e74c3c",C2:"#3498db",C1:"#95a5a6","?":"#f39c12"};

function Gauge({label,value,max=1,color}){
  const pct=Math.min(Math.max(value/max,0),1)*100;
  return(
    <div style={{marginBottom:6}}>
      <div style={{display:"flex",justifyContent:"space-between",fontSize:9,color:"#555",marginBottom:2}}>
        <span>{label}</span><span style={{color:"#999"}}>{value.toFixed(4)}</span>
      </div>
      <div style={{height:3,background:"#0f0f1e",borderRadius:2,overflow:"hidden"}}>
        <div style={{height:"100%",width:`${pct}%`,background:color,transition:"width 0.25s"}}/>
      </div>
    </div>
  );
}

export default function App(){
  const SIZE=68,BURNIN=12,WIN=32;
  const [rule,setRule]=useState("B3/S23");
  const [inp,setInp]=useState("B3/S23");
  const [running,setRunning]=useState(false);
  const [speed,setSpeed]=useState(10);
  const [density,setDensity]=useState(0.35);
  const [seed,setSeed]=useState(42);
  const [gen,setGen]=useState(0);
  const [metrics,setMetrics]=useState(null);
  const [selP,setSelP]=useState(0);
  const [custom,setCustom]=useState(false);
  const st=useRef({grid:null,hist:[],gen:0});
  const anim=useRef(null);
  const cv=useRef(null);

  const reset=useCallback((r=rule,d=density,s=seed)=>{
    const g=randomGrid(SIZE,d,s);
    st.current={grid:g,hist:[Array.from(g)],gen:0};
    setGen(0);setMetrics(null);
  },[rule,density,seed]);

  useEffect(()=>{reset();},[]);

  useEffect(()=>{
    const c=cv.current;if(!c||!st.current.grid)return;
    const ctx=c.getContext("2d");const cs=c.width/SIZE;
    const g=st.current.grid;
    ctx.fillStyle="#07070f";ctx.fillRect(0,0,c.width,c.height);
    for(let y=0;y<SIZE;y++) for(let x=0;x<SIZE;x++){
      if(!g[y*SIZE+x])continue;
      let nb=0;
      for(let dy=-1;dy<=1;dy++)for(let dx=-1;dx<=1;dx++){
        if(!dy&&!dx)continue;
        nb+=g[((y+dy+SIZE)%SIZE)*SIZE+(x+dx+SIZE)%SIZE];
      }
      const t=nb/8;
      ctx.fillStyle=`rgb(${Math.floor(46+t*209)},${Math.floor(204-t*160)},${Math.floor(113+t*30)})`;
      ctx.fillRect(x*cs,y*cs,cs-0.5,cs-0.5);
    }
  });

  useEffect(()=>{
    if(!running){cancelAnimationFrame(anim.current);return;}
    let last=0;const iv=1000/speed;
    const loop=(ts)=>{
      anim.current=requestAnimationFrame(loop);
      if(ts-last<iv)return;last=ts;
      const {birth:b,survive:s}=parseRule(rule);
      const next=stepLife(st.current.grid,SIZE,b,s);
      const hist=st.current.hist;
      hist.push(Array.from(next));
      if(hist.length>BURNIN+WIN+5)hist.shift();
      st.current.grid=next;st.current.gen++;
      setGen(g=>g+1);
      if(hist.length>=BURNIN+WIN)setMetrics(computeMetrics(hist,BURNIN,WIN));
    };
    anim.current=requestAnimationFrame(loop);
    return()=>cancelAnimationFrame(anim.current);
  },[running,speed,rule]);

  const apply=()=>{setRule(inp);reset(inp,density,seed);setRunning(false);setCustom(true);};
  const stepOnce=()=>{
    const {birth:b,survive:s}=parseRule(rule);
    const next=stepLife(st.current.grid,SIZE,b,s);
    const hist=st.current.hist;
    hist.push(Array.from(next));if(hist.length>BURNIN+WIN+5)hist.shift();
    st.current.grid=next;st.current.gen++;setGen(g=>g+1);
    if(hist.length>=BURNIN+WIN)setMetrics(computeMetrics(hist,BURNIN,WIN));
  };
  const selPreset=(i)=>{
    setSelP(i);const r=PRESETS[i].rule;
    setRule(r);setInp(r);reset(r,density,seed);setRunning(false);setCustom(false);
  };
  const toggleBit=(n,type)=>{
    const bs=new Set(rule.split("/")[0].replace(/^B/i,"").split("").map(Number).filter(n=>!isNaN(n)));
    const ss=new Set((rule.split("/")[1]||"").replace(/^S/i,"").split("").map(Number).filter(n=>!isNaN(n)));
    if(type==="b"){bs.has(n)?bs.delete(n):bs.add(n);}else{ss.has(n)?ss.delete(n):ss.add(n);}
    const r=`B${[...bs].sort().join("")}/S${[...ss].sort().join("")}`;
    setRule(r);setInp(r);reset(r,density,seed);setRunning(false);setCustom(true);
  };

  const preset=PRESETS[selP];
  const clr=CC[custom?"?":preset?.cls]||"#888";
  const sc=metrics?.score||0;
  const scClr=sc>0.5?"#2ecc71":sc>0.1?"#f39c12":sc>0.01?"#e74c3c":"#555";
  const bs_=new Set(rule.split("/")[0].replace(/^B/i,"").split("").map(Number).filter(n=>!isNaN(n)));
  const ss_=new Set((rule.split("/")[1]||"").replace(/^S/i,"").split("").map(Number).filter(n=>!isNaN(n)));

  const mono={fontFamily:"'DM Mono','Courier New',monospace"};

  return(
    <div style={{...mono,minHeight:"100vh",background:"#07070f",color:"#ccc",display:"flex",flexDirection:"column",fontSize:11}}>
      <div style={{background:"#080818",borderBottom:"1px solid #141428",padding:"9px 16px",display:"flex",alignItems:"center",gap:14}}>
        <div>
          <div style={{fontSize:8,color:"#3a3a55",letterSpacing:3,textTransform:"uppercase"}}>complexity framework · v5</div>
          <div style={{fontSize:14,color:"#d0d0f0",fontWeight:600}}>2D Life-like Rule Explorer</div>
        </div>
        <div style={{flex:1}}/>
        <div style={{fontSize:8,color:"#2a2a40",textAlign:"right",lineHeight:1.8}}>
          <div>262,144 possible rules</div><div>outer-totalistic B/S</div>
        </div>
      </div>

      <div style={{display:"flex",flex:1,overflow:"hidden"}}>
        {/* presets */}
        <div style={{width:185,background:"#060612",borderRight:"1px solid #141428",padding:9,overflowY:"auto",flexShrink:0}}>
          <div style={{fontSize:8,color:"#2a2a44",letterSpacing:2,textTransform:"uppercase",marginBottom:7}}>preset rules</div>
          {PRESETS.map((p,i)=>(
            <div key={i} onClick={()=>selPreset(i)} style={{
              padding:"6px 7px",marginBottom:3,borderRadius:3,cursor:"pointer",border:"1px solid",
              borderColor:(!custom&&selP===i)?clr:"#111128",
              background:(!custom&&selP===i)?"#0b0b20":"transparent",
            }}>
              <div style={{display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                <span style={{fontSize:10,color:"#bbb"}}>{p.name}</span>
                <span style={{fontSize:8,padding:"1px 4px",borderRadius:2,background:CC[p.cls]+"20",color:CC[p.cls]}}>{p.cls}</span>
              </div>
              <div style={{fontSize:8,color:"#383855",marginTop:1}}>{p.rule}</div>
            </div>
          ))}
        </div>

        {/* main */}
        <div style={{flex:1,display:"flex",flexDirection:"column",alignItems:"center",padding:14,gap:9,overflowY:"auto"}}>
          {/* rule bar */}
          <div style={{display:"flex",gap:7,alignItems:"center",width:"100%",maxWidth:500}}>
            <div style={{flex:1,background:"#0b0b1d",border:"1px solid #222238",borderRadius:4,padding:"5px 10px",display:"flex",alignItems:"center",gap:7}}>
              <span style={{fontSize:8,color:"#383855"}}>RULE</span>
              <input value={inp} onChange={e=>setInp(e.target.value.toUpperCase())} onKeyDown={e=>e.key==="Enter"&&apply()}
                style={{background:"none",border:"none",outline:"none",color:"#d0d0ff",fontSize:13,...mono,flex:1,letterSpacing:1}}/>
              <button onClick={apply} style={{background:"#161630",border:"1px solid #282850",color:"#777",borderRadius:3,padding:"2px 7px",cursor:"pointer",fontSize:8,...mono}}>APPLY</button>
            </div>
            <div style={{padding:"4px 9px",borderRadius:3,background:clr+"18",border:`1px solid ${clr}44`,color:clr,fontSize:10,minWidth:50,textAlign:"center"}}>
              {custom?"CUSTOM":preset?.cls}
            </div>
          </div>

          {/* toggles */}
          <div style={{display:"flex",gap:14,width:"100%",maxWidth:500}}>
            {[["Birth (B)",bs_,"b"],["Survive (S)",ss_,"s"]].map(([lbl,set,type])=>(
              <div key={type} style={{flex:1}}>
                <div style={{fontSize:8,color:"#2e2e48",letterSpacing:2,textTransform:"uppercase",marginBottom:4}}>{lbl}</div>
                <div style={{display:"flex",gap:3}}>
                  {[0,1,2,3,4,5,6,7,8].map(n=>(
                    <div key={n} onClick={()=>toggleBit(n,type)} style={{
                      width:25,height:25,borderRadius:3,display:"flex",alignItems:"center",justifyContent:"center",
                      fontSize:10,cursor:"pointer",border:"1px solid",userSelect:"none",transition:"all 0.1s",
                      borderColor:set.has(n)?clr:"#181830",
                      background:set.has(n)?clr+"25":"#09091a",
                      color:set.has(n)?clr:"#333",
                    }}>{n}</div>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* canvas */}
          <div style={{border:"1px solid #141428",borderRadius:5,overflow:"hidden",position:"relative",flexShrink:0}}>
            <canvas ref={cv} width={408} height={408} style={{display:"block"}}/>
            <div style={{position:"absolute",bottom:5,left:7,fontSize:8,color:"#1e1e30"}}>gen {gen}</div>
          </div>

          {/* controls */}
          <div style={{display:"flex",gap:7,alignItems:"center",flexWrap:"wrap",justifyContent:"center"}}>
            {[
              ["RESET",()=>{reset();setRunning(false);},"#888",false],
              ["STEP",stepOnce,running?"#282840":"#888",running],
              [running?"⏸ PAUSE":"▶ RUN",()=>setRunning(r=>!r),running?"#2ecc71":"#888",false],
            ].map(([lbl,fn,col,dis])=>(
              <button key={lbl} onClick={fn} disabled={dis} style={{
                background:"#0e0e22",border:`1px solid ${col}44`,color:dis?"#2a2a3a":col,
                borderRadius:4,padding:"5px 11px",cursor:dis?"default":"pointer",fontSize:9,...mono,letterSpacing:0.5
              }}>{lbl}</button>
            ))}
            <label style={{fontSize:8,color:"#383855",display:"flex",alignItems:"center",gap:4}}>
              SPEED<input type="range" min={1} max={25} value={speed} onChange={e=>setSpeed(+e.target.value)} style={{width:55,accentColor:"#2ecc71"}}/>
              <span style={{color:"#555",width:25}}>{speed}/s</span>
            </label>
            <label style={{fontSize:8,color:"#383855",display:"flex",alignItems:"center",gap:4}}>
              DENSITY<input type="range" min={5} max={75} value={Math.round(density*100)} onChange={e=>{const d=+e.target.value/100;setDensity(d);reset(rule,d,seed);}} style={{width:55,accentColor:"#3498db"}}/>
              <span style={{color:"#555",width:28}}>{Math.round(density*100)}%</span>
            </label>
            <label style={{fontSize:8,color:"#383855",display:"flex",alignItems:"center",gap:4}}>
              SEED<input type="number" value={seed} onChange={e=>{const s=+e.target.value;setSeed(s);reset(rule,density,s);}} style={{width:50,background:"#0b0b1d",border:"1px solid #222238",color:"#888",fontSize:8,borderRadius:3,padding:"2px 4px",...mono}}/>
            </label>
          </div>
          {!custom&&preset&&<div style={{fontSize:9,color:"#3a3a55",textAlign:"center",maxWidth:420,lineHeight:1.7,fontStyle:"italic"}}>{preset.note}</div>}
        </div>

        {/* metrics */}
        <div style={{width:185,background:"#060612",borderLeft:"1px solid #141428",padding:11,flexShrink:0,overflowY:"auto"}}>
          <div style={{fontSize:8,color:"#2a2a44",letterSpacing:2,textTransform:"uppercase",marginBottom:11}}>complexity metrics</div>
          {metrics?(<>
            <div style={{background:"#0b0b1e",border:"1px solid #1e1e38",borderRadius:4,padding:9,marginBottom:11,textAlign:"center"}}>
              <div style={{fontSize:8,color:"#383855",marginBottom:2}}>COMPOSITE C</div>
              <div style={{fontSize:24,fontWeight:700,color:scClr,transition:"color 0.4s",...mono}}>{sc.toFixed(4)}</div>
              <div style={{fontSize:8,color:"#444",marginTop:2}}>
                {sc>0.5?"★ high complexity":sc>0.1?"◈ moderate":sc>0.01?"○ low":"· trivial"}
              </div>
            </div>
            <Gauge label="w_H (entropy)" value={metrics.wH} max={2} color="#f39c12"/>
            <Gauge label="w_OP (opacity↑)" value={metrics.wOP} max={1} color="#9b59b6"/>
            <Gauge label="w_T (t_comp)" value={metrics.wT} max={1} color="#3498db"/>
            <Gauge label="w_G (gzip~)" value={metrics.wG} max={1} color="#1abc9c"/>
            <div style={{height:1,background:"#141428",margin:"9px 0"}}/>
            <div style={{fontSize:8,color:"#2a2a44",letterSpacing:2,textTransform:"uppercase",marginBottom:5}}>raw values</div>
            {[["mean H",metrics.meanH],["std H",metrics.stdH],["opacity ↑",metrics.opUp],["t_comp",metrics.tcomp]].map(([k,v])=>(
              <div key={k} style={{display:"flex",justifyContent:"space-between",fontSize:9,marginBottom:3,color:"#555"}}>
                <span>{k}</span><span style={{color:"#888"}}>{v.toFixed(4)}</span>
              </div>
            ))}
            <div style={{height:1,background:"#141428",margin:"9px 0"}}/>
            <div style={{fontSize:8,color:"#2a2a44",lineHeight:1.8}}>
              Run {BURNIN+WIN}+ gens for stable readings.<br/><br/>
              ⚠ gzip approximated (entropy proxy) — true values in Python framework.
            </div>
          </>):(
            <div style={{fontSize:9,color:"#2a2a44",lineHeight:1.8}}>
              Press ▶ RUN and wait {BURNIN+WIN}+ generations.<br/><br/>
              Burn-in removes IC transients.
            </div>
          )}
          <div style={{height:1,background:"#141428",margin:"12px 0"}}/>
          <div style={{fontSize:8,color:"#2a2a44",lineHeight:1.8}}>C = w_H × w_OP × w_T × w_G<br/><br/>Multiplicative: all 4 dimensions required.</div>
        </div>
      </div>
    </div>
  );
}
