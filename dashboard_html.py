"""Dashboard + bandwidth HTML for InfiniteModel, extracted from server.py to shrink it.
Imported by server.py via the multi-file self-update; server.py keeps a minimal placeholder
fallback for the brief window before this file lands on a freshly-updated controller."""

BANDWIDTH_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniteModel — Bandwidth</title>
<style>
  body{background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,Consolas,monospace;margin:18px}
  h1{font-size:18px;margin:0 0 4px} a{color:#58a6ff;text-decoration:none} a:hover{text-decoration:underline}
  .sub{color:#8b949e;font-size:12px}
  table{border-collapse:collapse;margin:14px 0;width:100%}
  th,td{padding:5px 10px;border-bottom:1px solid #21262d;text-align:right;white-space:nowrap}
  th:first-child,td:first-child{text-align:left}
  th{color:#8b949e;font-weight:600;border-bottom:1px solid #30363d}
  .dn{color:#58a6ff}.up{color:#3fb950}.nn{color:#d29922}
  .tot{color:#8b949e;font-size:11px}
  .card{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:12px 16px;margin:10px 0}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}
  .ok{background:#3fb950}.off{background:#f85149}
  code{background:#21262d;padding:1px 5px;border-radius:4px}
  .sparkbox{display:inline-block;width:110px;height:30px;vertical-align:middle;line-height:0}
  .sparkbox svg{display:block}
</style></head><body>
<h1>Bandwidth <span class="sub">· <span id="ctl">—</span> · <a href="/">← dashboard</a></span></h1>
<div class="sub">controller↔node is metered on the controller's own sockets (authoritative). node↔node
hidden-state traffic — invisible to the controller during decode — is reported by each worker.
Rates are derived from 2&nbsp;s deltas; totals are cumulative since each process started.</div>

<div class="card">
  <b>Per node</b>
  <table id="nodes"><thead><tr>
    <th>Node</th>
    <th>ctrl→node ⟱</th><th>node→ctrl ⟰</th>
    <th>node↔node in</th><th>node↔node out</th>
    <th>total ↓</th><th>total ↑</th><th>graph ↓/↑</th>
  </tr></thead><tbody><tr><td colspan="8" class="sub">connecting…</td></tr></tbody></table>
</div>

<div class="card">
  <b>Node ↔ node hops</b> <span class="sub">(directed; each worker reports what it SENT to each peer)</span>
  <table id="edges"><thead><tr><th>From</th><th>→ To</th><th>rate</th><th>total</th></tr></thead>
  <tbody><tr><td colspan="4" class="sub">no node-to-node traffic yet (only flows during a multi-node generation)</td></tr></tbody></table>
</div>

<div class="card">
  <b>Totals</b> <span class="sub">(grouped — controller↔all nodes, and all node↔node)</span>
  <table id="totals"><thead><tr><th>Link</th><th>rate</th><th>packets/s</th><th>total</th></tr></thead><tbody>
    <tr><td>controller → nodes</td><td class="dn" id="t-cn-r">—</td><td class="dn" id="t-cn-p">—</td><td class="tot" id="t-cn-b">—</td></tr>
    <tr><td>nodes → controller</td><td class="up" id="t-nc-r">—</td><td class="up" id="t-nc-p">—</td><td class="tot" id="t-nc-b">—</td></tr>
    <tr><td>node ↔ node (all hops)</td><td class="nn" id="t-nn-r">—</td><td class="nn" id="t-nn-p">—</td><td class="tot" id="t-nn-b">—</td></tr>
  </tbody></table>
</div>

<script>
let prev=null, prevT=0, sparkCache={}, lastSparkT=0;
function hb(n){ n=n||0; if(n<1024)return n.toFixed(0)+' B/s'; n/=1024; if(n<1024)return n.toFixed(1)+' KB/s'; n/=1024; if(n<1024)return n.toFixed(1)+' MB/s'; return (n/1024).toFixed(2)+' GB/s'; }
function hbytes(n){ n=n||0; if(n<1024)return n.toFixed(0)+' B'; n/=1024; if(n<1024)return n.toFixed(1)+' KB'; n/=1024; if(n<1024)return n.toFixed(1)+' MB'; return (n/1024).toFixed(2)+' GB'; }
function pps(n){ n=n||0; if(n<1000)return n.toFixed(n<10?1:0)+' p/s'; return (n/1000).toFixed(1)+'k p/s'; }
function rate(cur,key,id,dt){ if(!prev||!prev[id])return 0; const p=prev[id][key]; return dt>0?Math.max(0,(cur-p)/dt):0; }
async function tick(){
  let d; try{ d=await (await fetch('/bandwidthdata',{cache:'no-store'})).json(); }
  catch(e){ document.getElementById('ctl').textContent='controller unreachable'; return; }
  document.getElementById('ctl').textContent=d.controller;
  const now=Date.now()/1000, dt=prevT?now-prevT:0;
  const cur={};
  const nrows=(d.nodes||[]).map(n=>{
    const id='n:'+n.node_id;
    cur[id]={cti:n.ctrl_to_node,ntc:n.node_to_ctrl,nni:n.nn_in,nno:n.nn_out,
             ctip:n.ctrl_to_node_pkts,ntcp:n.node_to_ctrl_pkts,nnip:n.nn_in_pkts,nnop:n.nn_out_pkts};
    const cti=rate(n.ctrl_to_node,'cti',id,dt), ntc=rate(n.node_to_ctrl,'ntc',id,dt);
    const nni=rate(n.nn_in,'nni',id,dt), nno=rate(n.nn_out,'nno',id,dt);
    const ctip=rate(n.ctrl_to_node_pkts,'ctip',id,dt), ntcp=rate(n.node_to_ctrl_pkts,'ntcp',id,dt);
    const nnip=rate(n.nn_in_pkts,'nnip',id,dt), nnop=rate(n.nn_out_pkts,'nnop',id,dt);
    return `<tr><td><span class="dot ${n.alive?'ok':'off'}"></span>${n.hostname}</td>`
      +`<td class="dn">${hb(cti)} <span class="sub">${pps(ctip)}</span><br><span class="tot">${hbytes(n.ctrl_to_node)}</span></td>`
      +`<td class="up">${hb(ntc)} <span class="sub">${pps(ntcp)}</span><br><span class="tot">${hbytes(n.node_to_ctrl)}</span></td>`
      +`<td class="nn">${hb(nni)} <span class="sub">${pps(nnip)}</span><br><span class="tot">${hbytes(n.nn_in)}</span></td>`
      +`<td class="nn">${hb(nno)} <span class="sub">${pps(nnop)}</span><br><span class="tot">${hbytes(n.nn_out)}</span></td>`
      +`<td class="dn">${hb(cti+nni)}</td><td class="up">${hb(ntc+nno)}</td>`
      +`<td><span class="sparkbox" data-host="${n.hostname}" title="recent ↓/↑ — click to expand"></span></td></tr>`;
  }).join('');
  document.querySelector('#nodes tbody').innerHTML=nrows||'<tr><td colspan="8" class="sub">no nodes</td></tr>';
  // Flash-free sparklines: this table is rebuilt every tick (2s), which recreates the sparkbox spans
  // EMPTY. Refill them INSTANTLY from cache (no empty-frame flash), and re-fetch the server-rendered
  // SVGs only every ~10s (a sparkline needs no 2s cadence) — so they stop flickering on every poll.
  document.querySelectorAll('#nodes span.sparkbox').forEach(box=>{ const s=sparkCache[box.dataset.host]; if(s) box.innerHTML=s; });
  if(now-lastSparkT>10){
    lastSparkT=now;
    fetch('/status?graphs=1',{cache:'no-store'}).then(r=>r.json()).then(sd=>{
      (sd.nodes||[]).forEach(n=>{ if(n.spark_bw) sparkCache[n.hostname]=n.spark_bw; });
      document.querySelectorAll('#nodes span.sparkbox').forEach(box=>{ const s=sparkCache[box.dataset.host]; if(s) box.innerHTML=s; });
    }).catch(()=>{});
  }
  const erows=(d.edges||[]).map((e,i)=>{
    const id='e:'+e.src+'>'+e.dst;
    cur[id]={b:e.bytes,p:e.pkts};
    const r=rate(e.bytes,'b',id,dt), rp=rate(e.pkts,'p',id,dt);
    return `<tr><td>${e.src}</td><td>→ ${e.dst}</td><td class="nn">${hb(r)} <span class="sub">${pps(rp)}</span></td><td class="tot">${hbytes(e.bytes)}</td></tr>`;
  }).join('');
  document.querySelector('#edges tbody').innerHTML=erows||'<tr><td colspan="4" class="sub">no node-to-node traffic yet (only flows during a multi-node generation)</td></tr>';
  // grouped totals: controller<->all nodes + all node<->node (rate from the aggregate delta)
  const cnT=(d.nodes||[]).reduce((a,n)=>a+n.ctrl_to_node,0);
  const ncT=(d.nodes||[]).reduce((a,n)=>a+n.node_to_ctrl,0);
  const nnT=(d.edges||[]).reduce((a,e)=>a+e.bytes,0);
  const cnP=(d.nodes||[]).reduce((a,n)=>a+(n.ctrl_to_node_pkts||0),0);
  const ncP=(d.nodes||[]).reduce((a,n)=>a+(n.node_to_ctrl_pkts||0),0);
  const nnP=(d.edges||[]).reduce((a,e)=>a+(e.pkts||0),0);
  cur['totals']={cn:cnT,nc:ncT,nn:nnT,cnp:cnP,ncp:ncP,nnp:nnP};
  document.getElementById('t-cn-r').textContent=hb(rate(cnT,'cn','totals',dt));
  document.getElementById('t-cn-p').textContent=pps(rate(cnP,'cnp','totals',dt));
  document.getElementById('t-cn-b').textContent=hbytes(cnT);
  document.getElementById('t-nc-r').textContent=hb(rate(ncT,'nc','totals',dt));
  document.getElementById('t-nc-p').textContent=pps(rate(ncP,'ncp','totals',dt));
  document.getElementById('t-nc-b').textContent=hbytes(ncT);
  document.getElementById('t-nn-r').textContent=hb(rate(nnT,'nn','totals',dt));
  document.getElementById('t-nn-p').textContent=pps(rate(nnP,'nnp','totals',dt));
  document.getElementById('t-nn-b').textContent=hbytes(nnT);
  prev=cur; prevT=now;
}
tick(); setInterval(tick,2000);
</script>
</body></html>"""

DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniteModel</title>
<style>
  :root { color-scheme: dark; } * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
         background:#0d1117; color:#c9d1d9; }
  header { padding:16px 22px; border-bottom:1px solid #21262d; display:flex;
           align-items:baseline; gap:14px; flex-wrap:wrap; }
  h1 { font-size:18px; margin:0; color:#58a6ff; letter-spacing:.5px; }
  .sub { color:#8b949e; font-size:12px; }
  .wrap { padding:22px 28px; max-width:none; width:100%; }
  .cards { display:flex; gap:14px; flex-wrap:wrap; margin-bottom:22px; }
  .card { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:14px 18px; min-width:150px; }
  /* model/loading cards: cap to ~half the row so TWO sit side by side (drops to 1-up on a narrow
     screen via min-width), and cap height to half the viewport so a card with many warnings can't
     run very long — it scrolls instead. */
  #model-cards > .card { flex:1 1 calc(50% - 7px); max-width:calc(50% - 7px); min-width:320px;
                         box-sizing:border-box; max-height:50vh; overflow:auto; }
  .card .k { color:#8b949e; font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
  .card .v { font-size:24px; color:#e6edf3; margin-top:4px; } .card .v small { font-size:13px; color:#8b949e; }
  .bar { height:8px; background:#21262d; border-radius:4px; overflow:hidden; margin-top:10px; }
  .bar > span { display:block; height:100%; background:linear-gradient(90deg,#1f6feb,#58a6ff); }
  table { width:100%; border-collapse:collapse; font-size:13px; }
  th,td { text-align:left; padding:9px 10px; border-bottom:1px solid #21262d; }
  th { color:#8b949e; font-weight:600; font-size:11px; text-transform:uppercase; letter-spacing:.5px; }
  td.num { text-align:right; font-variant-numeric:tabular-nums; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:7px; }
  .up { background:#3fb950; } .down { background:#f85149; }
  .pill { padding:1px 8px; border-radius:10px; font-size:11px; background:#21262d; color:#8b949e; }
  .pill.act { background:#132e1a; color:#3fb950; }
  .pill.load { background:#2d1416; color:#f85149; animation:pulse 1.1s ease-in-out infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.4} }
  .empty { color:#8b949e; padding:28px 10px; text-align:center; }
  .state-idle{color:#3fb950;} .state-dirty{color:#d29922;} .state-loaded{color:#58a6ff;}
  .panel { background:#161b22; border:1px solid #21262d; border-radius:8px; padding:16px 18px; margin-bottom:22px; }
  select,input,button { font:inherit; background:#0d1117; color:#c9d1d9; border:1px solid #30363d;
         border-radius:6px; padding:6px 10px; }
  button { background:#1f6feb; border-color:#1f6feb; color:#fff; cursor:pointer; }
  button.sec { background:#21262d; border-color:#30363d; color:#c9d1d9; }
  #out { white-space:pre-wrap; background:#0d1117; border:1px solid #21262d; border-radius:6px;
         padding:12px; margin-top:10px; min-height:40px; color:#e6edf3; }
  footer { color:#6e7681; font-size:11px; padding:6px 22px 22px; }
  canvas.spark { display:block; background:#0d1117; border:1px solid #21262d; border-radius:4px; cursor:pointer; }
  canvas.spark:hover { border-color:#30363d; }
  #nettip { position:fixed; z-index:60; pointer-events:none; display:none; background:#161b22;
            border:1px solid #30363d; border-radius:6px; padding:6px 9px; font-size:11px; line-height:1.5;
            color:#e6edf3; white-space:nowrap; box-shadow:0 6px 20px rgba(0,0,0,.55); }
  #netmodal { position:fixed; inset:0; z-index:50; display:none; align-items:center; justify-content:center;
              background:rgba(1,4,9,.72); }
  #netmodal .box { background:#161b22; border:1px solid #30363d; border-radius:10px; padding:18px 20px;
                   width:min(840px,92vw); box-shadow:0 14px 50px rgba(0,0,0,.6); }
  #netmodal h2 { margin:0; font-size:16px; color:#58a6ff; }
  #netmodal .x { float:right; cursor:pointer; color:#8b949e; font-size:20px; line-height:1; padding:0 2px; }
  #netmodal .x:hover { color:#e6edf3; }
  .legend { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:5px; vertical-align:middle; }
  .nstat { display:flex; gap:22px; flex-wrap:wrap; margin-top:14px; font-size:12px; color:#8b949e; }
  .nstat b { color:#e6edf3; font-variant-numeric:tabular-nums; font-weight:600; }
  /* #model-detail: click-a-card modal */
  #mdlov { position:fixed; inset:0; background:rgba(1,4,9,.7); display:none; z-index:1000;
           align-items:flex-start; justify-content:center; overflow:auto; padding:40px 16px; }
  #mdlov.show { display:flex; }
  #mdlbox { background:#0d1117; border:1px solid #30363d; border-radius:10px; max-width:780px;
            width:100%; padding:20px 26px; box-shadow:0 14px 50px rgba(0,0,0,.6); }
  #mdlbox h2 { margin:0; font-size:18px; color:#e6edf3; }
  #mdlbox h3 { font-size:11px; color:#8b949e; margin:18px 0 4px 0; text-transform:uppercase; letter-spacing:.06em; }
  #mdlbox .mdlclose { float:right; cursor:pointer; color:#8b949e; font-size:22px; line-height:1; border:none; background:none; }
  #mdlbox .mdlclose:hover { color:#e6edf3; }
  #mdlbox .tag { display:inline-block; font-size:11px; padding:1px 8px; border-radius:10px;
                 background:#21262d; color:#c9d1d9; margin:3px 5px 0 0; border:1px solid #30363d; }
  #mdlbox .mgrid { display:grid; grid-template-columns:1fr 1fr; gap:0 28px; }
  #mdlbox .mrow { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid #21262d; padding:4px 0; font-size:13px; }
  #mdlbox .mrow b { color:#8b949e; font-weight:normal; }
  #mdlbox .mrow span { color:#e6edf3; font-variant-numeric:tabular-nums; text-align:right; }
  #mdlbox table { width:100%; border-collapse:collapse; font-size:12px; margin-top:4px; }
  #mdlbox th, #mdlbox td { text-align:left; padding:3px 8px; border-bottom:1px solid #21262d; }
  #mdlbox th { color:#8b949e; font-weight:normal; }
  /* #ctx-history: nested context-in/out popup (sits above the model modal) */
  #ctxov { position:fixed; inset:0; background:rgba(1,4,9,.78); display:none; z-index:1100;
           align-items:flex-start; justify-content:center; overflow:auto; padding:40px 16px; }
  #ctxov.show { display:flex; }
  #ctxbox { background:#0d1117; border:1px solid #30363d; border-radius:10px; max-width:980px;
            width:100%; padding:20px 26px; box-shadow:0 14px 50px rgba(0,0,0,.6); }
  #ctxbox h2 { margin:0; font-size:17px; color:#e6edf3; }
  #ctxbox .mdlclose { float:right; cursor:pointer; color:#8b949e; font-size:22px; line-height:1; border:none; background:none; }
  #ctxbox .mdlclose:hover { color:#e6edf3; }
  #ctxbox .sub { color:#8b949e; font-size:12px; }
  .ctxlink { color:#58a6ff; cursor:pointer; text-decoration:underline dotted; }
  .ctxent { margin:10px 0; border:1px solid #21262d; border-radius:6px; }
  .ctxhdr { background:#161b22; color:#8b949e; font-size:11px; padding:4px 10px; border-bottom:1px solid #21262d; }
  .ctxpre { margin:0; padding:10px; max-height:340px; overflow:auto; white-space:pre-wrap; word-break:break-word;
            font-family:ui-monospace,Consolas,monospace; font-size:12px; color:#c9d1d9; }
</style></head><body>
<div id="mdlov" onclick="if(event.target===this)closeModelModal()"><div id="mdlbox"></div></div>
<div id="ctxov" onclick="if(event.target===this)closeCtxHistory()"><div id="ctxbox"></div></div>
<header><h1>∞ InfiniteModel</h1>
  <span class="sub" id="ctl">connecting…</span>
  <span class="sub" style="margin-left:auto"><a href="/bandwidth" style="color:#58a6ff;text-decoration:none" title="full controller↔node + node↔node traffic">Bandwidth →</a></span>
  <span class="sub" id="uptime" title="engine (controller) uptime"></span>
  <span class="sub" id="clock"></span></header>
<div class="wrap">
  <div class="cards">
    <div class="card"><div class="k">Nodes</div><div class="v" id="c-nodes">–</div></div>
    <div class="card"><div class="k">Pool · memory</div><div class="v" id="c-usable">–<small> GB</small></div>
      <div class="bar" id="c-poolbar" onmousemove="poolHover(event)" onmouseleave="hideTip()" style="cursor:help; display:flex"><span id="c-bar" style="width:0%"></span><span id="c-bar-eng" style="width:0%; background:#f85149" title="engine memory"></span></div>
      <div class="sub" id="c-total" style="margin-top:6px"></div></div>
    <div class="card"><div class="k">Cluster</div><div class="v" id="c-state">–</div></div>
    <div class="card"><div class="k">Loaded models</div><div class="v" id="c-model">none</div>
      <div class="sub" id="c-modelsub"></div></div>
    <div class="card"><div class="k">Throughput · 10s</div>
      <div class="v" id="c-tps">0<small> tok/s</small></div>
      <div class="sub" id="c-apinet">API ↓0 ↑0</div>
      <div class="sub" id="c-ctrlnet" title="bytes the controller measured on its own sockets">wire ↓0 ↑0</div>
      <div class="sub" id="c-disk"></div></div>
    <div class="card"><div class="k" title="how busy the fleet's processors are, weighted by capacity (logical cores + GPUs) — i.e. processing used out of what is possible">System load</div>
      <div class="v" id="c-load">0<small>%</small></div>
      <div class="bar" id="c-loadbar" style="display:flex"><span id="c-load-fill" style="width:0%; background:#3fb950"></span></div>
      <div class="sub" id="c-loadsub" style="margin-top:6px"></div></div>
  </div>
  <!-- one live card per RESIDENT model (+ the in-progress load) — a new load adds a card, never replaces -->
  <div class="cards" id="model-cards" style="margin-top:12px"></div>

  <div class="panel">
    <b>Controller activity</b> <span class="sub">newest first</span>
    <div id="activity" style="margin-top:8px; max-height:104px; overflow-y:auto; font-family:ui-monospace,Consolas,monospace; font-size:12px; line-height:1.55"></div>
  </div>

  <div class="panel">
    <b>Why a model unloaded</b> <span class="sub" title="every model departure — manual unload, reload, auto-evict to make room, or a node dying/OOMing mid-serve">newest first · last 12</span>
    <div id="unloads" style="margin-top:8px; max-height:128px; overflow-y:auto; font-size:12px; line-height:1.6"></div>
  </div>

  <div class="panel">
    <b>Slots &amp; queue</b> <span class="sub" id="slotsub">1 slot per model</span>
    <table style="margin-top:10px"><thead><tr>
      <th style="width:90px">State</th><th>Client IP</th><th>Model</th><th class="num">Elapsed</th>
    </tr></thead><tbody id="slotrows"></tbody></table>
  </div>

  <div class="panel">
    <b>Configuration</b>
    <div style="margin-top:10px; display:flex; gap:16px; flex-wrap:wrap; align-items:center">
      <label class="sub" title="max models kept resident at once">max models loaded
        <input id="cfg-max" type="number" min="1" style="width:64px"></label>
      <label class="sub" title="requests allowed WAITING per model beyond the one running in its slot; an arrival past this is rejected (503)">queue depth
        <input id="cfg-queue" type="number" min="0" style="width:64px"></label>
      <label class="sub" title="OFF (default): loaded models stay resident FOREVER — a request never unloads a model, and a new load that doesn't fit simply fails (unload one first). ON: a model idle (no requests) for 60 min is auto-unloaded, and an idle model (never one actively serving) may be evicted LRU-first to make room for a new load.">
        <input id="cfg-auto" type="checkbox"> auto-unload idle models — after 60 min idle, or to make room</label>
      <label class="sub" title="A request for a KNOWN but not-resident model auto-loads it (GPU-first placement, using the Auto-load defaults below) instead of failing.">
        <input id="cfg-autoload" type="checkbox"> auto-load on request</label>
      <span class="sub" style="width:100%;color:#8b949e;border-top:1px solid #21262d;padding-top:8px">Auto-load defaults — used by each model's <b>Load</b> button AND by auto-load when a request hits a non-resident model:</span>
      <label class="sub" title="Quant the Load button + auto-load use. int4 (default) = smallest: ~1/4 the bf16 memory, fits more nodes, serves pre-packed when a shard cache exists. Falls back to bf16 if int4/int8 can't quantize a given model.">quant
        <select id="cfg-aq" style="width:80px"><option value="int4">int4</option><option value="int8">int8</option><option value="none">bf16</option></select></label>
      <label class="sub" title="Default context length the Load button + auto-load use. 8192 (8k) is a sane working window that keeps KV modest. 0 = the model's native training context.">ctx
        <input id="cfg-ctx" type="number" min="0" step="1024" style="width:76px"></label>
      <label class="sub" title="Default placement mode the Load button + auto-load use. auto = GPU-first, fewest nodes (best latency); gpu-spread/distribute/proportional spread across more nodes; single = collapse to one box if it fits.">mode
        <select id="cfg-mode" style="width:118px"><option value="auto">auto</option><option value="single">single</option><option value="gpu-spread">GPU-spread</option><option value="all-gpu">all-GPU</option><option value="distribute">distribute</option><option value="spread">spread</option><option value="proportional">proportional</option></select></label>
      <label class="sub" title="ON (default): pack a new model's WEIGHTS into physically-free VRAM, using resident models' reserved-but-unused KV headroom — so a model lands on GPU when VRAM is free instead of spilling weights to CPU. Each model still reserves its own KV. OFF: conservative — reserve every resident model's full-context KV on GPU (weights spill to CPU before a resident model's KV is touched).">
        <input id="cfg-wf" type="checkbox"> weights-first VRAM</label>
      <button class="sec" onclick="saveConfig()">Save</button>
      <span class="sub" id="cfgmsg"></span>
      <button class="sec" onclick="gcCache()" title="delete HF-cache copies of models already migrated to models/ (pure duplicates ~2x disk); cache-only (never-loaded) models are kept" style="margin-left:auto">Reclaim HF cache</button>
      <span class="sub" id="gcmsg"></span>
    </div>
  </div>

  <div class="panel">
    <b>Load &amp; run</b>
    <div style="margin-top:10px; display:flex; gap:8px; flex-wrap:wrap; align-items:center">
      <select id="m"></select>
      <label class="sub" title="blank = the model's native training context (config max_position_embeddings)">ctx <input id="ctx" type="number" placeholder="auto" style="width:90px"></label>
      <label class="sub" title="how the model is placed across the fleet">run
        <select id="mode" style="width:auto">
          <option value="auto" selected>auto — GPU-first, fewest nodes</option>
          <option value="single">single box — fewest nodes (one box if it fits)</option>
          <option value="gpu-spread">all GPUs — fill VRAM, spill to CPU</option>
          <option value="all-gpu">all GPUs — every GPU, no CPU spill</option>
          <option value="distribute">distribute — split across whole fleet</option>
          <option value="spread">spread — a stage on EVERY node (incl. tiny ones)</option>
          <option value="proportional">proportional — every node, share ∝ capacity (big int4 MoE)</option>
          <option value="tp2">tensor-parallel ×2 (GPU mesh)</option>
          <option value="tp4">tensor-parallel ×4 (GPU mesh)</option>
        </select></label>
      <label class="sub" title="weight quantization for the manual Load/Preview below (the per-model row buttons pick their own)">quant
        <select id="q" style="width:auto">
          <option value="none" selected>bf16 (full)</option>
          <option value="int8">int8 (~½)</option>
          <option value="int4">int4 (~¼)</option>
        </select></label>
      <button class="sec" onclick="doPreview()" title="#60: show WHERE this model would land + a pre-load sanity check (VRAM/RAM split, KV fit, est tok/s tier) WITHOUT loading">Preview</button>
      <button onclick="doLoad()">Load</button>
      <button class="sec" onclick="doUnloadAll()" title="unload EVERY model from every node — drops all shards fleet-wide, frees their RAM/VRAM, and clears the controller's draft state. Reversible (reload from the list).">Unload all</button>
      <button class="sec" onclick="doRestart()" title="RESTART the whole fleet: signal every worker to restart, then restart the controller. ABORTS any in-flight load (use when a load is wedged with no other way out) and relaunches every process clean (supervisor) on the current code.">Restart fleet</button>
      <button class="sec" onclick="doUpdate()" title="UPDATE + restart NOW (forced — does NOT wait for idle): unloads all models, tells every worker to free its RAM, pulls the latest code from GitHub, swaps it in, and relaunches. Auto-load is blocked during the swap so a client request can't reload a model into the box being torn down. Use this to deploy.">Update + restart</button>
      <span class="sub" id="loadmsg"></span>
    </div>
    <div id="previewbox" class="sub" style="margin-top:8px"></div>
    <div style="margin-top:12px; display:flex; gap:8px; align-items:center; flex-wrap:wrap">
      <span class="sub">add a model:</span>
      <input id="addhf" placeholder="Hugging Face id, e.g. deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
             style="flex:1; min-width:320px" onkeydown="if(event.key==='Enter')addModel()">
      <input id="addgguf" placeholder="optional: file.gguf (GGUF-only repo)"
             title="Only for a repo that ships weights as a single llama.cpp .gguf instead of safetensors. Enter the exact .gguf filename (one quant). It is dequantized to safetensors once, then runs like any model."
             style="width:230px" onkeydown="if(event.key==='Enter')addModel()">
      <button class="sec" onclick="addModel()" title="register + download any Hugging Face model id; it then appears in the list to load">Add &amp; download</button>
      <span class="sub" id="addmsg"></span>
    </div>
    <table style="margin-top:14px"><thead><tr>
      <th id="th-model-name" onclick="setSort('model','name')" style="cursor:pointer" title="sort by name">Model</th><th id="th-model-size" class="num" onclick="setSort('model','size')" style="cursor:pointer" title="sort by size">Size</th><th id="th-model-status" onclick="setSort('model','status')" style="cursor:pointer" title="sort by status">Status</th><th>Actions</th>
    </tr></thead><tbody id="modelrows"></tbody></table>
    <div style="margin-top:12px; display:flex; gap:8px;">
      <input id="prompt" placeholder="prompt…" value="The capital of France is" style="flex:1">
      <label class="sub">max <input id="maxtok" type="number" value="32" style="width:80px"></label>
      <button onclick="doGen()">Generate</button>
    </div>
    <div id="out"></div>
  </div>

  <table><thead><tr>
    <th id="th-node-node" onclick="setSort('node','node')" style="cursor:pointer" title="sort by node">Node</th><th id="th-node-host" onclick="setSort('node','host')" style="cursor:pointer" title="sort by host">Host</th><th id="th-node-os" onclick="setSort('node','os')" style="cursor:pointer" title="sort by OS">OS</th><th id="th-node-device" onclick="setSort('node','device')" style="cursor:pointer" title="sort by device">Device</th>
    <th class="num" title="memory FREE / total per tier (green = free), RAM and VRAM">Mem free/total</th><th class="num">Disk</th>
    <th id="th-node-cpu" class="num" onclick="setSort('node','cpu')" style="cursor:pointer" title="sort by CPU%">CPU%</th><th id="th-node-gpu" class="num" onclick="setSort('node','gpu')" style="cursor:pointer" title="sort by GPU% (GPU nodes only)">GPU%</th><th class="num">HB</th>
    <th class="num" title="controller-measured: bytes the controller sent to this node">Net ↓</th>
    <th class="num" title="controller-measured: bytes the controller received from this node">Net ↑</th>
    <th title="recent download (↓ controller→node) and upload (↑ node→controller); hover a point for speed+time, click to expand">Traffic</th>
    <th id="th-node-role" onclick="setSort('node','role')" style="cursor:pointer" title="sort by role">Role</th>
    <th title="enable/disable this node's CPU/RAM or GPU/VRAM contribution (persisted)">Tiers
      <div style="font-weight:normal;font-size:10px;margin-top:2px;white-space:nowrap">
        <label title="enable/disable ALL nodes' CPU/RAM"><input type="checkbox" id="tier-all-cpu" onchange="setAllTiers('ram',this.checked)"> all CPU</label>
        <label style="color:#3fb950" title="enable/disable ALL GPU nodes' VRAM"><input type="checkbox" id="tier-all-gpu" onchange="setAllTiers('vram',this.checked)"> all GPU</label>
      </div>
    </th>
  </tr></thead>
  <tbody id="rows"><tr><td colspan="13" class="empty">no nodes — start a client</td></tr></tbody></table>
</div>
<div id="nettip"></div>
<div id="netmodal" onclick="if(event.target.id==='netmodal')closeNetModal()">
  <div class="box">
    <span class="x" onclick="closeNetModal()" title="close">&#10005;</span>
    <h2 id="nm-title">node traffic</h2>
    <div class="sub" id="nm-sub" style="margin-top:4px"></div>
    <canvas id="nm-canvas" width="800" height="280"
            style="margin-top:12px;width:100%;background:#0d1117;border:1px solid #21262d;border-radius:6px"></canvas>
    <div class="nstat" id="nm-stats"></div>
  </div>
</div>
<footer>auto-refresh 1.5s · Ollama API on this port · net = controller-measured per node
  (mid-pipeline stages exchange hidden states node-to-node, off the controller, during decode)
  · models: <span id="models"></span></footer>
<script>
function gb(x){return x==null?'–':Number(x).toFixed(1);}
function ctxFmt(n){ n=Number(n)||0; if(n>=1048576) return (n/1048576)+'M'; if(n>=1024) return Math.round(n/1024)+'K'; return ''+n; }
function humanBps(b){ b=Number(b)||0; if(b<1024) return b.toFixed(0)+' B/s';
  if(b<1048576) return (b/1024).toFixed(1)+' KB/s'; return (b/1048576).toFixed(2)+' MB/s'; }
function fmtDur(s){ s=Number(s)||0; if(s<60) return s.toFixed(1)+'s';
  const m=Math.floor(s/60); return m+'m'+String(Math.floor(s%60)).padStart(2,'0')+'s'; }
function fmtUptime(s){ s=Math.floor(Number(s)||0); const d=Math.floor(s/86400), h=Math.floor(s%86400/3600),
  m=Math.floor(s%3600/60), sec=s%60;
  if(d) return `${d}d ${h}h ${m}m`; if(h) return `${h}h ${m}m`; if(m) return `${m}m ${sec}s`; return `${sec}s`; }
function esc(x){ return String(x).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function loadColor(v){v=+v||0;const c=v>85?'#f85149':v>60?'#d29922':'#3fb950';return `<span style="color:${c}">${v.toFixed(0)}`;}
let rowModes={};            // per-model run type chosen on its row (name -> auto|single|gpu-spread|distribute)
// per-node traffic history. The SERVER stores + persists this (disk-backed, keyed
// by hostname); the dashboard pulls it incrementally and keeps only a bounded
// window so a long-open tab never piles up unbounded JS memory. Points are
// {t:ms, d:download_bps, u:upload_bps}. Mini sparkline shows the recent tail; the
// detail modal shows the full window.
const NET_HIST={}, NET_CAP=1800, NET_MINI_TAIL=120;   // cap matches server ring (~1 h)
let netSince=0;             // ms watermark for incremental /nethistory fetches
let netModalHost=null;
let LAST=null;              // most recent /status, for hover tooltips
const C_DOWN='#58a6ff', C_UP='#3fb950';
// --- sortable tables (#104/#105): models + nodes lists, click a header to sort, click again to flip
let modelSort={key:'',asc:true}, nodeSort={key:'',asc:true};
const MODEL_GETTERS={ name:m=>(m.name||''), size:m=>(m.size_gb==null?-1:m.size_gb), status:m=>(m.status||'') };
const NODE_GETTERS={ node:n=>(n.node_id||''), host:n=>(n.hostname||''), os:n=>(n.os||''),
  device:n=>(n.device||''), cpu:n=>(n.cpu_percent==null?-1:n.cpu_percent),
  gpu:n=>(n.gpu_util==null?-1:n.gpu_util),
  role:n=>(n.stage!=null?('stage'+String(n.stage).padStart(3,'0')):(n.load_state||'idle')) };
function _cmp(a,b){ if(typeof a==='number'&&typeof b==='number') return a-b;
  return String(a).localeCompare(String(b),undefined,{numeric:true}); }
function sortRows(arr, st, getters){
  if(!st.key||!getters[st.key]) return arr;
  const g=getters[st.key]; arr.sort((x,y)=>{ const r=_cmp(g(x),g(y)); return st.asc?r:-r; }); return arr;
}
function setSort(which,key){
  const st=(which==='model')?modelSort:nodeSort;
  if(st.key===key) st.asc=!st.asc; else { st.key=key; st.asc=true; }
  if(LAST) tick();   // re-render immediately with the new sort (tick is safe to call manually)
}
function _updSortArrows(){
  const M={'th-model-name':[modelSort,'name','Model'],'th-model-size':[modelSort,'size','Size'],
           'th-model-status':[modelSort,'status','Status'],
           'th-node-node':[nodeSort,'node','Node'],'th-node-host':[nodeSort,'host','Host'],
           'th-node-os':[nodeSort,'os','OS'],'th-node-device':[nodeSort,'device','Device'],
           'th-node-cpu':[nodeSort,'cpu','CPU%'],'th-node-gpu':[nodeSort,'gpu','GPU%'],
           'th-node-role':[nodeSort,'role','Role']};
  for(const id in M){ const [st,k,lbl]=M[id]; const el=document.getElementById(id);
    if(el) el.textContent=lbl+(st.key===k?(st.asc?' ▲':' ▼'):''); }
}
function poolHover(e){
  const s=LAST; if(!s) return;
  const p=s.pool;
  const osg=(p.os_gb!=null?p.os_gb:p.used_gb), eng=(p.engine_gb!=null?p.engine_gb:0);
  let h=`<b>Pool memory</b> <span style="color:#d29922">${gb(p.used_gb)} used</span> / ${gb(p.total_gb)} GB`+
    `<br><span style="color:#58a6ff">${gb(osg)} OS / other</span>`+
    `<br><span style="color:#f85149">${gb(eng)} engine</span> <span style="color:#8b949e">(controller ${gb(p.ctrlr_gb)} + workers + GPU shards)</span>`+
    `<br><span style="color:#3fb950">${gb(p.free_gb)} free</span>`+
    (p.ram_free_gb!=null?` <span style="color:#8b949e">(${gb(p.ram_free_gb)} RAM / ${gb(p.vram_free_gb)} VRAM available)</span>`:'')+
    `<br><span style="color:#8b949e">planner budget ${gb(p.usable_gb)} GB free (${gb(p.ram_gb)} RAM + ${gb(p.vram_gb)} VRAM)</span>`;
  const lms=s.cluster.loaded_models||[];
  for(const lm of lms){
    const u=(lm.stages||[]).reduce((a,st)=>a+(st.est_gb||0),0);
    const kv=lm.kv_pos||0, cx=lm.ctx||0;
    h+=`<br><br><b>${lm.display_name||lm.friendly}</b> — ${gb(u)} GB across ${lm.stages.length} stage(s)`+
       ` · ctx ${kv.toLocaleString()}/${cx.toLocaleString()}<br>`+
       lm.stages.map(st=>`&nbsp;&nbsp;${st.hostname}: ${gb(st.est_gb)} GB (${st.num_layers}L)`).join('<br>');
  }
  if(!lms.length) h+=`<br><br><span style="color:#8b949e">idle — nothing loaded</span>`;
  const tip=document.getElementById('nettip');
  tip.innerHTML=h; tip.style.display='block';
  tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px';
}
async function syncNetHistory(){
  let r; try{ r=await (await fetch('/nethistory'+(netSince?('?since='+netSince):''),{cache:'no-store'})).json(); }
  catch(e){ return; }
  const hosts=r.hosts||{}; let maxT=netSince;
  for(const host in hosts){
    const arr=hosts[host]; if(!arr||!arr.length) continue;
    const h=NET_HIST[host]||(NET_HIST[host]=[]);
    for(const a of arr){ h.push({t:a[0],d:a[1],u:a[2]}); if(a[0]>maxT)maxT=a[0]; }
    if(h.length>NET_CAP) h.splice(0, h.length-NET_CAP);
  }
  netSince=maxT;
}
async function tick(){
  let s; try { s=await (await fetch('/status?graphs=1',{cache:'no-store'})).json(); }
  catch(e){ document.getElementById('ctl').textContent='controller unreachable'; return; }
  const c=s.controller,p=s.pool; LAST=s;
  // Auto-reload an OPEN tab when the controller VERSION changes (a deploy self-updated the fleet)
  // so the dashboard never shows STALE UI after an update — the cause of "I still see one card"
  // when new dashboard code shipped but the tab kept running the old JS. First tick records the
  // version; a later mismatch -> reload to pull the fresh HTML/JS.
  if(c.version){ if(window.__cv && window.__cv!==c.version){ location.reload(); return; } window.__cv=c.version; }
  document.getElementById('ctl').textContent=
    `${c.hostname} · ${c.os} · v${c.version} · http :${c.http_port} · control :${c.control_port} · data :${c.data_port}`;
  document.getElementById('uptime').textContent=c.uptime_s!=null?`up ${fmtUptime(c.uptime_s)}`:'';
  const cm=document.getElementById('cfg-max'), ca=document.getElementById('cfg-auto'), cq=document.getElementById('cfg-queue'), cal=document.getElementById('cfg-autoload'), caq=document.getElementById('cfg-aq'), cwf=document.getElementById('cfg-wf'), cctx=document.getElementById('cfg-ctx'), cmode=document.getElementById('cfg-mode');  // don't clobber while editing
  if(cm&&document.activeElement!==cm) cm.value=c.max_loaded;
  if(ca&&document.activeElement!==ca) ca.checked=!!c.auto_unload;
  if(cal&&document.activeElement!==cal&&c.auto_load!=null) cal.checked=!!c.auto_load;
  if(caq&&document.activeElement!==caq&&c.autoload_quant!=null) caq.value=c.autoload_quant;
  if(cctx&&document.activeElement!==cctx&&c.autoload_ctx!=null) cctx.value=c.autoload_ctx;
  if(cmode&&document.activeElement!==cmode&&c.autoload_mode!=null) cmode.value=c.autoload_mode;
  if(cwf&&document.activeElement!==cwf&&c.vram_weights_first!=null) cwf.checked=!!c.vram_weights_first;
  if(cq&&document.activeElement!==cq&&c.queue_depth!=null) cq.value=c.queue_depth;
  document.getElementById('clock').textContent=new Date().toLocaleTimeString();
  document.getElementById('c-nodes').textContent=p.nodes;
  const comp=s.compute;
  if(comp){
    const o=comp.overall_pct||0;
    const cel=document.getElementById('c-load'); if(cel) cel.innerHTML=o.toFixed(0)+'<small>%</small>';
    const f=document.getElementById('c-load-fill');
    if(f){f.style.width=Math.min(100,o)+'%'; f.style.background=o>85?'#f85149':o>60?'#d29922':'#3fb950';}
    const sub=document.getElementById('c-loadsub');
    if(sub) sub.innerHTML=`CPU ${(comp.cpu_pct||0).toFixed(0)}% · ${(comp.cpu_busy_cores||0).toFixed(0)}/${comp.cpu_cores||0} cores`
      +(comp.gpus?` · GPU ${(comp.gpu_pct||0).toFixed(0)}% · ${comp.gpus} GPU`:``);
  }
  document.getElementById('c-usable').innerHTML=gb(p.total_gb)+'<small> GB</small>';
  const osg=(p.os_gb!=null?p.os_gb:p.used_gb), eng=(p.engine_gb!=null?p.engine_gb:0);
  document.getElementById('c-total').innerHTML=
    `<span style="color:#58a6ff" title="OS + other programs">${gb(osg)} OS</span> · `
    +`<span style="color:#f85149" title="controller + worker pythons + their GPU shards">${gb(eng)} engine</span> · `
    +`<span style="color:#3fb950" title="live free memory, by form">${gb(p.free_gb)} free</span>`
    +(p.ram_free_gb!=null?` <span class="sub" style="font-size:11px;color:#3fb950">(${gb(p.ram_free_gb)} RAM / ${gb(p.vram_free_gb)} VRAM)</span>`:'');
  const tot=p.total_gb>0?p.total_gb:1;
  document.getElementById('c-bar').style.width=(Math.round(100*osg/tot))+'%';      // OS (blue)
  document.getElementById('c-bar-eng').style.width=(Math.round(100*eng/tot))+'%';  // engine (red)
  const se=document.getElementById('c-state'); se.textContent=s.cluster.state; se.className='v state-'+s.cluster.state;
  // parallel loads/compiles: cluster.loadings + cluster.compiling are LISTS of cards (one each).
  const lms=(s.cluster.loaded_models||[]), lds=(s.cluster.loadings||[]), cmps=(s.cluster.compiling||[]);
  const nLoad=lds.length+cmps.length;
  const cmEl=document.getElementById('c-model'), csEl=document.getElementById('c-modelsub');
  // Summary status card: COUNT of resident models + COMBINED throughput across all of them.
  const combTps=lms.reduce((a,m)=>a+(m.tok_s||0),0);
  if(lms.length){
    cmEl.innerHTML=`${lms.length} <small style="color:#8b949e">resident</small>`;
    csEl.innerHTML=`combined <b style="color:${combTps>0?'#3fb950':'inherit'}">${combTps.toFixed(1)}</b> tok/s`+(nLoad?` · <span style="color:#d29922">+${nLoad} loading…</span>`:``);
  } else if(nLoad){ cmEl.innerHTML=`<span style="color:#d29922">loading…</span>`; csEl.textContent=''; }
  else { cmEl.textContent='none'; csEl.textContent=''; }
  // One LIVE card per resident model (+ a card for the in-progress load) — a new load ADDS a card,
  // it never replaces the existing one. All fields come from /status cluster.loaded_models[i].
  const _mcard=(lm)=>{
    const kv=lm.kv_pos||0, cx=lm.ctx||0, kvpct=cx?Math.round(100*kv/cx):0;
    const name=lm.display_name||lm.friendly||lm.base||'model';
    // #88 reconfigure control: current layout preselected (pipeline, or TP×N CPU). Offers the common
    // widths; the controller validates geometry and returns a clear error for an invalid tp.
    const _rc=lm.base||lm.friendly;
    const _cur=lm.is_tp?(String(lm.tp_size)+'c'):'1';
    const _rcOpts=[['1','pipeline'],['2c','TP×2 (CPU)'],['4c','TP×4 (CPU)'],['8c','TP×8 (CPU)']]
      .map(o=>`<option value="${o[0]}"${o[0]===_cur?' selected':''}>${o[1]}</option>`).join('');
    return `<div class="card" style="min-width:250px;cursor:pointer" onclick="openModelModal('${_rc}')" title="click for full details">`
      +`<div class="k" title="${esc(lm.target||'')}">${esc(name)}${lm.quant&&lm.quant!=='none'?` <small style="color:#8b949e">${esc(lm.quant)}</small>`:``}${(lm.cpu_frac||0)>=0.3?` <small style="color:#f85149;font-weight:600" title="${Math.round((lm.cpu_frac||0)*100)}% of this model's weights are on CPU because the GPU pool was full at load — it is CPU-bound and will decode SLOWLY (often <1 tok/s). Not hung. Unload another model or use a smaller quant to get it on GPU.">⚠ CPU ${Math.round((lm.cpu_frac||0)*100)}%</small>`:``} <small style="color:#8b949e;float:right">&#9432;</small></div>`
      +((lm.aliases&&lm.aliases.length)?`<div class="sub" style="font-size:11px;color:#8b949e;margin-top:-2px" title="other names that resolve to this model">alias: ${lm.aliases.map(esc).join(', ')}</div>`:``)
      +`<div class="v" style="font-size:15px">${gb(lm.size_gb)} GB <small style="color:#8b949e">${lm.params||''} · ${(lm.stages||[]).length} stg</small></div>`
      +`<div class="sub">`
      +`<span title="KV-cache depth of the current/last generation">ctx <b>${kv.toLocaleString()}</b>/${cx.toLocaleString()} (${kvpct}%)</span>`
      +`<br><span title="weights on-GPU (measured) + the rest in RAM; KV used / reserved">w <span style="color:#3fb950">${gb(lm.vram_used_gb)}</span>+<span style="color:#58a6ff">${gb(lm.ram_used_gb)}</span> GB · KV ${gb(lm.kv_used_gb)}/${gb(lm.kv_reserved_gb)}</span>`
      +`<br><span title="decode throughput (smoothed) + this model's live saturation">▶ <b style="color:${(lm.tok_s||0)>0?'#3fb950':'inherit'}">${(lm.tok_s||0).toFixed(1)}</b> tok/s <small style="color:#8b949e">(avg ${(lm.ema_tok_s||0).toFixed(1)})</small> · ${(lm.active||0)>0?`<span style="color:#3fb950">busy ${lm.active}</span>`:`<span style="color:#8b949e">idle</span>`}${(lm.queued||0)>0?` <span style="color:#d29922">${lm.queued}q</span>`:``}</span>`
      +(lm.plan_basis?`<br><span class="sub" style="font-size:11px" title="how the placement was chosen">${esc(lm.plan_basis)}</span>`:``)
      +((lm.warnings||[]).length?`<br><span style="color:#d29922;font-size:11px" title="pre-load guardrail (#76)">⚠ ${lm.warnings.map(esc).join('<br>⚠ ')}</span>`:``)
      +`</div>`
      +`<div style="margin-top:8px"><button class="sec" style="font-size:11px;padding:2px 8px" `
      +`onclick="event.stopPropagation();doUnloadModel('${lm.base||lm.friendly}')" `
      +`title="unload THIS model only — frees its RAM/VRAM fleet-wide and keeps the other models running (no restart)">Unload</button></div>`
      +`<div style="margin-top:6px;display:flex;gap:4px;align-items:center">`
      +`<select id="rc-${_rc}" class="sec" style="font-size:11px;padding:2px 4px" onclick="event.stopPropagation()">${_rcOpts}</select>`
      +`<button class="sec" style="font-size:11px;padding:2px 8px" onclick="event.stopPropagation();doReconfigure('${_rc}')" `
      +`title="switch this model to/from tensor-parallel (managed reload: briefly unavailable, rolls back to pipeline on failure; refused while busy)">Reconfigure</button></div>`
      +`</div>`;
  };
  // one card per in-flight LOAD (amber) and per in-flight COMPILE (purple) — parallel-safe.
  const _loadcard=(ld,compile)=>{
    const pct=ld.total?Math.round(100*(ld.ready||0)/ld.total):0;
    const col=compile?'#a371f7':'#d29922', verb=compile?'compiling':'loading', unit=compile?'units':'shards';
    return `<div class="card" style="min-width:250px;border-color:${col}">`
      +`<div class="k"><span style="color:${col}">${verb} ${esc(ld.display_model||ld.model)}</span></div>`
      +`<div class="v" style="font-size:15px">${pct}%</div>`
      +`<div class="sub">${ld.ready||0}/${ld.total} ${unit}${ld.stages_total?` · ${ld.stages_ready||0}/${ld.stages_total} nodes`:``}`
      +(ld.started?`<br><span class="sub" style="font-size:11px" title="start time · elapsed · estimated time remaining (from progress)">&#9201; ${new Date(ld.started*1000).toLocaleTimeString()} · ${fmtUptime(ld.elapsed_s||0)} elapsed${ld.eta_s!=null?` · ~${fmtUptime(ld.eta_s)} left`:` · ETA…`}</span>`:``)
      +(ld.basis?`<br><span class="sub" style="font-size:11px">${esc(ld.basis)}</span>`:``)
      +((ld.warnings||[]).length?`<br><span style="color:#d29922;font-size:11px">⚠ ${ld.warnings.map(esc).join('<br>⚠ ')}</span>`:``)
      +`</div>`
      +(compile?``:`<div style="margin-top:8px"><button class="sec" style="font-size:11px;padding:2px 8px;border-color:#f85149;color:#f85149" `
        +`onclick="doCancelLoad('${esc(ld.model||ld.display_model||'')}')" `
        +`title="cancel this load — kill it and free any partial shards (use if it is stuck at 0%); then re-Load to restart">✕ cancel load</button></div>`)
      +`</div>`;
  };
  const rc=s.cluster.reconfiguring;   // #88: keep a card visible during a managed reload (to/from TP)
  const rcName=rc?(rc.model||''):null;
  // #reconfigure-progress: a managed reload sets BOTH the reconfiguring marker AND a normal in-flight
  // load card (which carries ready/total + stages + ETA). Render ONE card — the reconfigure card with
  // that progress folded in — and SKIP the model's duplicate amber load card, so the operator sees a
  // single "reconfiguring X: a→b · NN% · r/t shards" instead of a progress-less purple card next to it.
  let _cards=lms.map(_mcard).join('')
    + lds.filter(l=>!rcName||(l.display_model||l.model)!==rcName).map(x=>_loadcard(x,false)).join('')
    + cmps.map(x=>_loadcard(x,true)).join('');
  if(rc){
    const rl=(lds||[]).find(l=>(l.display_model||l.model)===rcName)||{};   // its in-flight load card
    const pct=rl.total?Math.round(100*(rl.ready||0)/rl.total):0;
    _cards+=`<div class="card" style="min-width:250px;border-color:#a371f7">`
      +`<div class="k"><span style="color:#a371f7">reconfiguring ${esc(rc.model||'')}</span></div>`
      +`<div class="v" style="font-size:15px">${esc(rc.from||'')} → ${esc(rc.to||'')}${rl.total?` · ${pct}%`:``}</div>`
      +`<div class="sub">${rl.total?`${rl.ready||0}/${rl.total} shards${rl.stages_total?` · ${rl.stages_ready||0}/${rl.stages_total} nodes`:``}`:`managed reload — preparing…`}`
      +(rl.started?`<br><span class="sub" style="font-size:11px" title="elapsed · estimated time remaining">&#9201; ${fmtUptime(rl.elapsed_s||0)} elapsed${rl.eta_s!=null?` · ~${fmtUptime(rl.eta_s)} left`:` · ETA…`}</span>`:``)
      +`</div></div>`;
  }
  // don't clobber a card's reconfigure <select> mid-choice: the poll re-renders the cards and would
  // reset the dropdown to the model's CURRENT layout (pipeline) before the user can hit Reconfigure.
  // Skip the cards re-render while an rc- select is focused/open (mirrors the rowmode- guard below).
  const _cae=document.activeElement;
  if(!(_cae&&_cae.tagName==='SELECT'&&_cae.id&&_cae.id.indexOf('rc-')===0))
    document.getElementById('model-cards').innerHTML=_cards;
    if(window.__mdl) renderModelModal();   // #model-detail: keep an open detail modal live
  document.getElementById('models').textContent=s.models.filter(m=>m.ready).map(m=>m.name).join(', ')||'none ready';
  document.getElementById('activity').innerHTML=(s.activity||[]).map(a=>{
    const t=new Date(a.t*1000).toLocaleTimeString();
    const msg=String(a.msg).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
    return `<div><span style="color:#8b949e">${t}</span>  ${msg}</div>`;
  }).join('')||'<div class="sub">idle — no recent activity</div>';
  // why-a-model-unloaded: persistent, color-coded by kind (a node OOM/drop shows red so the
  // operator sees an unexpected departure at a glance vs an intentional unload/evict)
  const UKIND={'node-loss':{c:'#f85149',label:'node lost'},'evict':{c:'#d29922',label:'evicted'},
               'reload':{c:'#58a6ff',label:'reload'},'manual':{c:'#8b949e',label:'manual'}};
  document.getElementById('unloads').innerHTML=(s.unloads||[]).map(u=>{
    const k=UKIND[u.kind]||UKIND.manual;
    const t=new Date(u.t*1000).toLocaleTimeString();
    const model=esc(u.model), reason=esc(u.reason);
    const hosts=(u.hosts||[]).map(esc).join(', ');
    return `<div style="padding:2px 0; border-left:3px solid ${k.c}; padding-left:8px; margin-bottom:3px">`
      +`<span class="pill" style="color:${k.c}">${k.label}</span> `
      +`<b>${model}</b> <span style="color:#8b949e">${t}</span><br>`
      +`<span style="color:#c9d1d9">${reason}</span>`
      +(hosts?` <span class="sub">· freed ${hosts}</span>`:'')+`</div>`;
  }).join('')||'<div class="sub">no models have unloaded this session</div>';
  // slots (1 running per model) + queue (waiters): client IP, model wanted, elapsed
  const cl=s.cluster||{}, sl=cl.slots||[], qu=cl.queue||[];
  document.getElementById('slotsub').textContent=
    `1 slot per model · queue depth ${cl.queue_depth!=null?cl.queue_depth:'?'} · `
    +`${sl.length} running, ${qu.length} queued`;
  const srows=sl.map(r=>`<tr><td><span class="pill act">slot</span></td><td>${esc(r.ip)}</td>`
      +`<td>${esc(r.model)}</td><td class="num" title="time in the slot">${fmtDur(r.running_s)}</td>`
      +`<td><button class="sec" title="disconnect this request" onclick="doCancel(${r.id})">✕</button></td></tr>`)
    .concat(qu.map(r=>`<tr><td><span class="pill load">queued</span></td><td>${esc(r.ip)}</td>`
      +`<td>${esc(r.model)}</td><td class="num" title="time waiting in queue">${fmtDur(r.waiting_s)}</td>`
      +`<td><button class="sec" title="cancel this queued request" onclick="doCancel(${r.id})">✕</button></td></tr>`)).join('');
  document.getElementById('slotrows').innerHTML=srows||'<tr><td colspan="5" class="sub">no active requests</td></tr>';
  { // keep the model <select> IN SYNC with the live list (not one-shot): a model registered
    // after the page loaded — e.g. a custom /add_model like qwen3-4b, or one re-registered after
    // a controller restart — must appear as an option, else doLoadModel() sets select.value to a
    // missing option which silently becomes '' and /load fails with "unknown model ''". Only
    // rebuild when the option set actually changed, and preserve the current selection.
    const sel=document.getElementById('m'); const want=s.models.map(m=>`<option>${esc(m.name)}</option>`).join('');
    if(sel.dataset.opts!==want){ const cur=sel.value; sel.innerHTML=want; sel.dataset.opts=want;
      if(s.models.some(m=>m.name===cur)) sel.value=cur; } }
  // Names display in Ollama 'family:size' form (m.name = 'qwen3:4b'); the dash-form registry
  // key rides along as m.internal_name. Join the per-model dicts on the internal key so the
  // colon display doesn't break the lookups; op buttons send m.name (resolve accepts it).
  const qmap={}; ((s.disk&&s.disk.models)||[]).forEach(x=>{qmap[x.internal_name||x.name]=x;});
  // #72: which quant each LOADED model loaded with + its weight memory split (base/friendly -> …)
  const loadedQuant={}, loadedInfo={}; ((s.cluster&&s.cluster.loaded_models)||[]).forEach(c=>{const k=c.base||c.friendly; loadedQuant[k]=c.quant||'none'; loadedInfo[k]={vram:c.vram_used_gb,ram:c.ram_used_gb};});
  const gMode=(document.getElementById('mode')||{}).value||'auto';   // default for each row's run type
  const _mrows=sortRows(s.models.slice(), modelSort, MODEL_GETTERS).map(m=>{
    const DLING=(m.status==='downloading'||m.status==='pausing'||m.status==='stopping');
    const HALT=(m.status==='paused'||m.status==='stopped');
    let badge = m.status==='ready' ? '<span class="pill act">ready</span>'
      : DLING ? `<span class="pill" style="color:#d29922">${m.status==='pausing'?'pausing…':m.status==='stopping'?'stopping…':(m.dl_pct!=null?'downloading '+m.dl_pct+'%':'downloading…')}</span>`
      : HALT ? `<span class="pill" style="color:#8b949e">${m.status}${m.dl_pct!=null?' '+m.dl_pct+'%':''}</span>`
      : '<span class="pill">not downloaded</span>';
    if(m.loaded){
      badge+=' <span class="pill act">loaded</span>';
      if(m.active||m.queued) badge+=` <span class="pill" style="color:#58a6ff" title="generating + queued requests">▶ ${m.active||0}${m.queued?(' · '+m.queued+' queued'):''}</span>`;
    }
    if(m.cached){ for(const qq in m.cached){ const c=m.cached[qq];   // #shard-cache: pre-compiled quants (click to verify)
      badge+=` <span class="pill" style="cursor:pointer;color:${c.ok?'#a371f7':'#d29922'}" title="pre-compiled shard cache (${qq})${c.ok?': '+c.files+' files, '+c.size_gb+' GB — faster loads. Click to verify.':' — INCOMPLETE, recompile'}" onclick="doVerifyShards('${m.name}','${qq}')">cache:${qq}${c.ok?' '+c.size_gb+'G':' ⚠'}</span>`; } }
    // per-quant load buttons with estimated weight footprint + fit hint (#49)
    const ikey=m.internal_name||m.name;   // dash-form registry key for cross-dict joins
    const q=qmap[ikey]||{}, qg=q.quant_gb||{}, qf=q.quant_fits||{};
    // quant=none loads the model's NATIVE dtype, so label that button accurately (fp32 stays fp32,
    // not "bf16"); unknown (not yet measured) falls back to bf16.
    const ndl=q.src_dtype==='F32'?'fp32':(q.src_dtype==='F16'?'fp16':'bf16');
    const qbtn=(lbl,key)=>{ const v=qg[key]; if(v==null) return '';
      const fit=qf[key]; const style=fit?'':'opacity:.45';
      return `<button class="sec" style="${style}" title="load ${lbl} — ~${v} GB weights, ${fit?'fits the pool':'may NOT fit the pool'}" onclick="doLoadModel('${m.name}','${key}')">${lbl} ~${v}G</button> `; };
    // compile-cache buttons (kept in BOTH loaded + unloaded states): hide a quant whose cache is already ok.
    const compileBtns=(cbtn=>cbtn('int4','~1/4 size, fastest future int4 loads')+cbtn('int8','~1/2 size, higher quality than int4'))(
      (q,desc)=> (m.cached&&m.cached[q]&&m.cached[q].ok) ? ''
        : `<button class="sec" title="pre-compile the ${q} shard cache on the controller (beast) — ${desc}, no per-worker re-quantize" onclick="doCompileShards('${m.name}','${q}')">Compile ${q}</button> `);
    let actions;
    if(m.status==='ready'){
      if(m.loaded){
        // #auto-defaults: ONE row state when loaded — green "loaded <quant>" + its weight memory split
        // (#72 highlight) + compile buttons + Unload. Change quant by Unload then Load.
        const uq=loadedQuant[ikey]||'none', li=loadedInfo[ikey]||{};
        const mem=(li.vram!=null||li.ram!=null)
          ? `<span class="sub" title="weights on GPU VRAM + spilled to CPU RAM">${gb(li.vram||0)}G vram${(li.ram||0)>0?(' + '+gb(li.ram)+'G ram'):''}</span> ` : '';
        actions=`<span class="pill" style="border-color:#3fb950;color:#3fb950;font-weight:600" title="loaded with ${uq}">loaded ${uq}</span> `
          +mem+compileBtns
          +`<button class="sec" onclick="doUnloadModel('${m.name}')" title="unload this model fleet-wide (frees its shards)">Unload</button>`;
      }
      else {
        // #auto-defaults: ONE Load button that uses the Auto-load defaults section (quant/ctx/mode);
        // label it with the chosen quant + that quant's estimated weight footprint.
        const aq=(document.getElementById('cfg-aq')||{}).value||'int4';
        const av=qg[aq], fit=qf[aq];
        actions=`<button class="sec" style="${fit===false?'opacity:.6':''}" onclick="doAutoLoad('${m.name}')" `
            +`title="load with the Auto-load defaults above (quant ${aq}, plus ctx + mode)${av!=null?' — ~'+av+'G weights'+(fit===false?', may NOT fit the pool':''):''}">Load${av!=null?' '+aq+' ~'+av+'G':' '+aq}</button> `
          +compileBtns
          +`<button class="sec" onclick="doDelete('${m.name}')">Delete</button>`;
      }
    }
    else if(DLING){
      const pct = m.dl_pct!=null? m.dl_pct : 0;
      const txt = m.dl_total_gb!=null ? `${m.dl_done_gb} / ${m.dl_total_gb} GB` : `${m.dl_done_gb!=null?m.dl_done_gb+' GB':'pulling…'}`;
      const ctl = m.status==='downloading'
        ? `<button class="sec" title="pause after the current file (cache kept, resumable)" onclick="doDlCtl('${m.name}','pause')">Pause</button> `+
          `<button class="sec" title="stop after the current file (cache kept, resumable)" onclick="doDlCtl('${m.name}','stop')">Stop</button>`
        : `<span class="sub">${m.status}…</span>`;
      actions = `<div style="min-width:150px"><div style="background:#21262d;border-radius:3px;height:6px;overflow:hidden">`+
                `<div style="width:${pct}%;height:100%;background:#d29922;transition:width .4s"></div></div>`+
                `<span class="sub">${txt}</span><div style="margin-top:4px">${ctl}</div></div>`;
    }
    else if(HALT){
      const pct = m.dl_pct!=null? m.dl_pct : 0;
      const txt = m.dl_total_gb!=null ? `${m.dl_done_gb} / ${m.dl_total_gb} GB` : (m.dl_done_gb!=null?m.dl_done_gb+' GB':'');
      actions = `<div style="min-width:150px"><div style="background:#21262d;border-radius:3px;height:6px;overflow:hidden">`+
                `<div style="width:${pct}%;height:100%;background:#8b949e;transition:width .4s"></div></div>`+
                `<span class="sub">${m.status} · ${txt}</span><div style="margin-top:4px">`+
                `<button class="sec" onclick="doResume('${m.name}')">Resume</button> `+
                `<button class="sec" title="delete cached + partial files (start over)" onclick="doClear('${m.name}')">Clear</button></div></div>`;
    }
    else actions=`<button class="sec" onclick="doDownload('${m.name}')">Download</button>`+
      (m.dl_error?` <button class="sec" title="delete cached + partial files" onclick="doClear('${m.name}')">Clear</button>`+
        `<div class="sub" style="color:#f85149;max-width:340px;white-space:normal">${m.dl_error.replace(/</g,'&lt;')}</div>`:'');
    const cx=q.default_ctx?` · <span class="sub" title="native/default context (loads at this when ctx=0)">${ctxFmt(q.default_ctx)} ctx</span>`:'';
    const aliasLine=(m.aliases&&m.aliases.length)?`<div class="sub" style="font-size:11px;color:#8b949e" title="other names that resolve to this model">alias: ${m.aliases.map(esc).join(', ')}</div>`:'';
    return `<tr><td>${m.name}${aliasLine}</td><td class="num">${m.size_gb!=null?m.size_gb+' GB':'–'}${cx}</td><td>${badge}</td><td>${actions}</td></tr>`;
  }).join('');
  // don't clobber a row's run-type <select> mid-choice (the 1.5s refresh would close it)
  const _ae=document.activeElement;
  if(!(_ae&&_ae.tagName==='SELECT'&&_ae.id&&_ae.id.indexOf('rowmode-')===0))
    document.getElementById('modelrows').innerHTML=_mrows;
    _updSortArrows();
  const mt=s.metrics||{};
  document.getElementById('c-tps').innerHTML=`<span style="color:${(Number(mt.tokens_per_s)||0)>0?'#3fb950':'inherit'}">${(Number(mt.tokens_per_s)||0).toFixed(1)}</span><small> tok/s</small>`;
  document.getElementById('c-apinet').textContent=`API ↓${humanBps(mt.api_in_bps)} ↑${humanBps(mt.api_out_bps)}`;
  document.getElementById('c-ctrlnet').textContent=`wire ↓${humanBps(mt.ctrl_in_bps)} ↑${humanBps(mt.ctrl_out_bps)}`;
  const dk=s.disk||{};
  document.getElementById('c-disk').textContent=
    `ctrl disk ${gb(dk.controller_free_gb)} GB free (RAM-bound)`;
  const rows=document.getElementById('rows');
  // reflect the 'all CPU' / 'all GPU' master checkboxes from the per-node tiers: checked when
  // every (GPU-bearing, for GPU) node has it on, indeterminate when only some do.
  (function(){
    const ns=s.nodes||[], gpus=ns.filter(n=>n.has_gpu);
    function tri(id,list,key){ const el=document.getElementById(id); if(!el) return;
      const on=list.filter(n=>n[key]).length, tot=list.length;
      el.checked=tot>0&&on===tot; el.indeterminate=on>0&&on<tot; }
    tri('tier-all-cpu', ns, 'ram_enabled');
    tri('tier-all-gpu', gpus, 'vram_enabled');
  })();
  if(!s.nodes.length){ rows.innerHTML='<tr><td colspan="13" class="empty">no nodes — start a client</td></tr>'; return; }
  const byHost={}; s.nodes.forEach(n=>{ byHost[n.hostname]=n; });   // for the server-SVG mini-graphs below
  rows.innerHTML=sortRows(s.nodes.slice(), nodeSort, NODE_GETTERS).map(n=>{
    let role='<span class="pill">idle</span>';
    if(n.stage!=null){
      const lbl=`stage ${n.stage} · L${n.layer_start}-${n.layer_end}`;
      role = n.load_state==='loading'
        ? `<span class="pill load" title="streaming weights + building this shard">${lbl} · loading…</span>`
        : `<span class="pill act">${lbl}</span>`;
    }
    if(n.can_infer===false) role=`<span class="pill" style="color:#f85149" title="${n.incapable_reason||''}">no-torch · excluded</span>`;
    let tiers=`<label style="font-size:11px;margin-right:6px" title="use this node's CPU/RAM"><input type="checkbox" ${n.ram_enabled?'checked':''} onchange="setTier('${n.hostname}','ram',this.checked)"> CPU</label>`;
    if(n.has_gpu) tiers+=`<label style="font-size:11px;color:#3fb950" title="use this node's GPU/VRAM"><input type="checkbox" ${n.vram_enabled?'checked':''} onchange="setTier('${n.hostname}','vram',this.checked)"> GPU</label>`;
    return `<tr><td><span class="dot ${n.alive?'up':'down'}"></span>${n.node_id}</td>
      <td>${n.hostname}</td>
      <td>${n.os}${n.client_version?`<br><span class="sub" style="font-size:10px">client v${n.client_version}</span>`:''}</td>
      <td>${n.device}${n.device_name?` <span class="sub">${n.device_name}</span>`:''}</td>
      <td class="num"><span style="color:#3fb950" title="RAM free">${gb(n.free_mem_gb)}</span> free / ${gb(n.total_mem_gb)} RAM${n.ram?`<br><span class="sub" style="font-size:10px">${n.ram}</span>`:''}${n.vram_total_gb>0?`<br><span style="color:#3fb950" title="VRAM free">${gb(Math.max(0,n.vram_total_gb-n.vram_used_gb))}</span> free / ${gb(n.vram_total_gb)} <span class="sub" style="font-size:10px">VRAM</span><br><span class="sparkbox" data-host="${n.hostname}" data-kind="vram" title="GPU VRAM used over time — click to expand"></span>`:''}<br><span class="sparkbox" data-host="${n.hostname}" data-kind="ram" title="free RAM over time — click to expand"></span></td>
      <td class="num">${gb(n.free_disk_gb)}</td>
      <td class="num">${loadColor(n.cpu_percent)+'%</span>'}</td>
      <td class="num">${n.has_gpu ? loadColor(n.gpu_util)+'%</span>' : '<span style="color:#484f58">–</span>'}</td>
      <td class="num">${n.age_s}s</td>
      <td class="num">${humanBps(n.net_in_bps)}</td><td class="num">${humanBps(n.net_out_bps)}</td>
      <td><span class="sparkbox" data-host="${n.hostname}" data-kind="bw" title="recent ↓/↑ — click to expand"></span></td>
      <td>${role}</td><td>${tiers}</td></tr>`;}).join('');
  // Server-rendered mini-graphs: the controller built the SVG (data + markup live
  // server-side); we only DISPLAY it by dropping the string into the placeholder.
  // Tooltips are native SVG <title> elements and click-to-expand is the SVG's own
  // <a href="/graph/...">, so no client-side graph JS is needed here.
  document.querySelectorAll('span.sparkbox').forEach(box=>{
    const n=byHost[box.dataset.host];
    if(!n) return;
    const svg=box.dataset.kind==='ram' ? n.spark_ram
            : box.dataset.kind==='vram' ? n.spark_vram   // CPU-only nodes won't have it
            : n.spark_bw;
    if(svg) box.innerHTML=svg;
  });
}
// ---- per-node traffic sparkline + detail modal ----
function drawTraffic(cv, hist, opts){
  const ctx=cv.getContext('2d'), W=cv.width, H=cv.height, mini=!!opts.mini;
  cv._data=hist; cv._mini=mini;   // stash what was drawn so hover stays in sync
  ctx.clearRect(0,0,W,H);
  const x0=mini?2:46, x1=W-(mini?2:10), y0=mini?3:12, y1=H-(mini?3:22);
  if(hist.length<2){ ctx.fillStyle='#6e7681'; ctx.font='10px monospace';
    if(!mini) ctx.fillText('collecting samples…', x0+4, (y0+y1)/2); cv._tx=null; return; }
  let mx=0; for(const p of hist){ if(p.d>mx)mx=p.d; if(p.u>mx)mx=p.u; }
  if(mx<=0) mx=1;
  const n=hist.length, X=i=>x0+(x1-x0)*(i/(n-1)), Y=v=>y1-(y1-y0)*(v/mx);
  if(!mini){
    ctx.strokeStyle='#21262d'; ctx.fillStyle='#6e7681'; ctx.font='10px monospace'; ctx.lineWidth=1;
    for(let g=0; g<=4; g++){ const yy=y0+(y1-y0)*g/4;
      ctx.beginPath(); ctx.moveTo(x0,yy); ctx.lineTo(x1,yy); ctx.stroke();
      ctx.fillText(humanBps(mx*(1-g/4)), 2, yy+3); }
    const span=(hist[n-1].t-hist[0].t)/1000;
    ctx.fillText(span.toFixed(0)+'s ago', x0, H-6);
    ctx.fillText('now', x1-22, H-6);
  }
  const line=(key,color)=>{ ctx.strokeStyle=color; ctx.lineWidth=mini?1:1.7; ctx.beginPath();
    hist.forEach((p,i)=>{ const xx=X(i), yy=Y(p[key]); i?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy); }); ctx.stroke();
    if(!mini){ ctx.lineTo(X(n-1),y1); ctx.lineTo(X(0),y1); ctx.closePath();
      ctx.globalAlpha=.09; ctx.fillStyle=color; ctx.fill(); ctx.globalAlpha=1; } };
  line('d', C_DOWN); line('u', C_UP);
  if(opts.markIdx!=null && opts.markIdx>=0 && opts.markIdx<n){
    const i=opts.markIdx, xx=X(i);
    ctx.strokeStyle='#8b949e'; ctx.lineWidth=1; ctx.beginPath(); ctx.moveTo(xx,y0); ctx.lineTo(xx,y1); ctx.stroke();
    ctx.fillStyle=C_DOWN; ctx.beginPath(); ctx.arc(xx,Y(hist[i].d),mini?2:3.2,0,7); ctx.fill();
    ctx.fillStyle=C_UP;   ctx.beginPath(); ctx.arc(xx,Y(hist[i].u),mini?2:3.2,0,7); ctx.fill();
  }
  cv._tx={x0,x1,n};
}
function trafficHover(e, cv){
  const m=cv._tx, hist=cv._data||[]; if(!m || hist.length<2){ hideTip(); return; }
  const rect=cv.getBoundingClientRect(), sx=cv.width/rect.width;
  const px=(e.clientX-rect.left)*sx;
  let i=Math.round((px-m.x0)/(m.x1-m.x0)*(m.n-1)); i=Math.max(0, Math.min(m.n-1, i));
  const p=hist[i], tip=document.getElementById('nettip');
  tip.innerHTML=`<b>${new Date(p.t).toLocaleTimeString()}</b><br>`+
    `<span style="color:${C_DOWN}">&#8595; ${humanBps(p.d)}</span> &nbsp; `+
    `<span style="color:${C_UP}">&#8593; ${humanBps(p.u)}</span>`;
  tip.style.display='block'; tip.style.left=(e.clientX+14)+'px'; tip.style.top=(e.clientY+14)+'px';
  drawTraffic(cv, hist, {mini:cv._mini, markIdx:i});
}
function hideTip(){ document.getElementById('nettip').style.display='none'; }
function openNetModal(host){ netModalHost=host; document.getElementById('netmodal').style.display='flex'; refreshNetModal(); }
function closeNetModal(){ netModalHost=null; document.getElementById('netmodal').style.display='none'; hideTip(); }
function refreshNetModal(){
  const host=netModalHost; if(!host) return;
  const hist=NET_HIST[host]||[], n=hist.length;
  document.getElementById('nm-title').textContent=host+' · traffic';
  const cv=document.getElementById('nm-canvas');
  drawTraffic(cv, hist, {mini:false});
  cv.onmousemove=e=>trafficHover(e, cv);
  cv.onmouseleave=()=>{ hideTip(); drawTraffic(cv, cv._data||[], {mini:false}); };
  let pd=0,pu=0,sd=0,su=0,totD=0,totU=0;
  hist.forEach((p,i)=>{ if(p.d>pd)pd=p.d; if(p.u>pu)pu=p.u; sd+=p.d; su+=p.u;
    if(i){ const dt=(p.t-hist[i-1].t)/1000; totD+=p.d*dt; totU+=p.u*dt; } });
  const cur=hist[n-1]||{d:0,u:0}, span=n>1?((hist[n-1].t-hist[0].t)/1000):0;
  const fmtB=b=>{ b=Number(b)||0; const u=['B','KB','MB','GB','TB']; let i=0;
    while(b>=1024 && i<u.length-1){ b/=1024; i++; } return (b<10&&i>0?b.toFixed(2):b.toFixed(0))+' '+u[i]; };
  document.getElementById('nm-sub').innerHTML=
    `<span class="legend" style="background:${C_DOWN}"></span>download (controller&#8594;node) &nbsp;&nbsp;`+
    `<span class="legend" style="background:${C_UP}"></span>upload (node&#8594;controller) &nbsp;·&nbsp; `+
    `${n} samples over ${span.toFixed(0)}s · hover for point detail`;
  document.getElementById('nm-stats').innerHTML=
    `<div>now &nbsp;<b style="color:${C_DOWN}">&#8595; ${humanBps(cur.d)}</b> &nbsp;<b style="color:${C_UP}">&#8593; ${humanBps(cur.u)}</b></div>`+
    `<div>peak &nbsp;<b>&#8595; ${humanBps(pd)}</b> &nbsp;<b>&#8593; ${humanBps(pu)}</b></div>`+
    `<div>avg &nbsp;<b>&#8595; ${humanBps(n?sd/n:0)}</b> &nbsp;<b>&#8593; ${humanBps(n?su/n:0)}</b></div>`+
    `<div>total &nbsp;<b>&#8595; ${fmtB(totD)}</b> &nbsp;<b>&#8593; ${fmtB(totU)}</b></div>`;
}
document.addEventListener('keydown',e=>{ if(e.key==='Escape') closeNetModal(); });
async function setTier(host,tier,on){
  try{ await fetch(`/nodeconfig?host=${encodeURIComponent(host)}&${tier}=${on}`,{method:'POST'});
    document.getElementById('loadmsg').textContent=`${host}: ${tier.toUpperCase()} ${on?'enabled':'disabled'} (re-load model to apply)`;
  }catch(e){ document.getElementById('loadmsg').textContent='tier change failed: '+e; }
  tick();
}
async function setAllTiers(tier,on){   // master 'all CPU' / 'all GPU' checkbox -> every node
  try{ await fetch(`/nodeconfig_all?tier=${tier}&enabled=${on}`,{method:'POST'});
    document.getElementById('loadmsg').textContent=`ALL nodes: ${tier==='ram'?'CPU':'GPU'} ${on?'enabled':'disabled'} (re-load model to apply)`;
  }catch(e){ document.getElementById('loadmsg').textContent='bulk tier change failed: '+e; }
  tick();
}
async function gcCache(){
  const el=document.getElementById('gcmsg'); el.textContent='reclaiming…';
  try{
    const r=await (await fetch('/gc_cache',{method:'POST'})).json();
    if(r.ok) el.textContent = r.removed.length
      ? `freed ${r.freed_gb} GB (${r.removed.length} duplicate${r.removed.length>1?'s':''} removed)`
      : 'nothing to reclaim — no duplicates';
    else el.textContent='error: '+(r.error||'failed');
  }catch(e){ el.textContent='error: '+e; }
}
async function saveConfig(){
  const mx=document.getElementById('cfg-max').value, au=document.getElementById('cfg-auto').checked;
  const qd=document.getElementById('cfg-queue').value, al=document.getElementById('cfg-autoload').checked;
  const aq=document.getElementById('cfg-aq').value, wf=document.getElementById('cfg-wf').checked;
  const cx=document.getElementById('cfg-ctx').value, md=document.getElementById('cfg-mode').value;
  document.getElementById('cfgmsg').textContent='saving…';
  try{
    const r=await (await fetch(`/config?max_loaded=${encodeURIComponent(mx)}&auto_unload=${au}&queue_depth=${encodeURIComponent(qd)}&auto_load=${al}&autoload_quant=${encodeURIComponent(aq)}&autoload_ctx=${encodeURIComponent(cx)}&autoload_mode=${encodeURIComponent(md)}&vram_weights_first=${wf}`,{method:'POST'})).json();
    document.getElementById('cfgmsg').textContent=r.ok?`saved · max ${r.config.max_loaded} · auto-unload ${r.config.auto_unload?'on':'off'} · auto-load ${r.config.auto_load?'on':'off'} · ${r.config.autoload_quant}/ctx ${r.config.autoload_ctx}/${r.config.autoload_mode} · weights-first ${r.config.vram_weights_first?'on':'off'} · queue ${r.config.queue_depth}`:'error';
  }catch(e){ document.getElementById('cfgmsg').textContent='error: '+e; }
}
async function doPreview(){   // #60: GET /plan (no load) -> show placement + the #76 assessment
  const m=document.getElementById('m').value, ctx=document.getElementById('ctx').value||0;
  const mode=document.getElementById('mode').value, q=document.getElementById('q').value||'none';
  const box=document.getElementById('previewbox');
  if(!m){ box.textContent='pick a model first'; return; }
  const _lm=document.getElementById('loadmsg'); if(_lm) _lm.textContent='';  // clear stale load/error msg above the preview
  box.textContent='previewing…';
  try{
    const r=await (await fetch(`/plan?model=${encodeURIComponent(m)}&ctx=${ctx}&quant=${q}&mode=${encodeURIComponent(mode)}`)).json();
    if(!r.ok){ box.innerHTML=`<span style="color:#f85149">✗ can't place: ${esc(r.error||'')}</span>`; return; }
    const a=r.assess||{}, tp=mode.indexOf('tp')===0;
    const rows=(r.stages||[]).map(s=>`${esc(s.hostname)} <span style="color:#8b949e">L${s.layer_start}-${s.layer_end} · ${s.num_layers}L · est ${s.est_gb}GB</span>`).join(' → ');
    const warn=(a.warnings||[]).length
      ? `<br>${a.warnings.map(w=>`<span style="color:#d29922">⚠ ${esc(w)}</span>`).join('<br>')}`
      : `<br><span style="color:#3fb950">✓ fits — no warnings</span>`;
    box.innerHTML=`<b>preview</b> @ ctx ${(r.ctx_len||0).toLocaleString()}${q!=='none'?' / '+esc(q):''} · `
      +`needs ${r.required_gb} GB / ${r.pool_usable_gb} GB pool`
      +(a.speed_tier?` <span style="color:#8b949e">[${esc(a.speed_tier)}]</span>`:'')
      +(r.basis?`<br><span class="sub">${esc(r.basis)}</span>`:'')
      +`<br>${rows}`+warn
      +(tp?`<br><span style="color:#8b949e">(tensor-parallel preview is approximate — TP frees the whole fleet and re-plans at load)</span>`:'');
  }catch(e){ box.textContent='preview failed: '+e; }
}
async function doLoad(quant,mode){
  const _n=Date.now(); if(window.__lastLoadClick && _n-window.__lastLoadClick<1500){ return; } window.__lastLoadClick=_n;  // debounce double-clicks
  const m=document.getElementById('m').value, ctx=document.getElementById('ctx').value||0;
  mode = mode || document.getElementById('mode').value;
  const q=quant||document.getElementById('q').value||'none';
  let tp=1;
  if(mode && mode.indexOf('tp')===0){ tp=parseInt(mode.slice(2))||2; mode='auto'; }   // tpN dropdown -> &tp=N
  // #78 guardrail: for a CONSOLIDATING mode (auto/single), pre-check the plan; if it would pile a
  // heavy shard onto the controller box (OOM-drop risk), offer to switch to proportional BEFORE
  // the load goes out. Best-effort — any error here just falls through to the load.
  if(tp<=1 && (mode==='auto'||mode==='single')){
    try{
      const pv=await (await fetch(`/plan?model=${encodeURIComponent(m)}&ctx=${ctx}&mode=${mode}&quant=${q}`)).json();
      if(pv && pv.overload){ const o=pv.overload;
        // #103: two overload reasons -> phrase each. gpu_spill = auto/single oversubscribe one box's
        // VRAM and spill to CPU while other GPUs sit idle; controller_ram = heavy shard on the box
        // that also serves the stream (OOM-drop). Both offer to switch to proportional pre-load.
        let msg;
        if(o.reason==='gpu_spill'){
          msg=`Heads-up: "${o.mode}" mode keeps only ~${o.auto_gpu_gb} GB of this ${o.model_gb} GB model on GPU — about ${o.on_cpu_gb} GB would spill to CPU (slow decode), even though the fleet has ~${o.fleet_gpu_gb} GB of GPU free.\n\nSwitch to "${o.suggest}" mode? It spreads the model across every GPU in the fleet.\n\nOK = use ${o.suggest}    ·    Cancel = keep ${o.mode}`;
        } else {
          msg=`Heads-up: "${o.mode}" mode would put ~${o.stage_gb} GB on ${o.node} — the controller box (${o.node_ram_gb} GB RAM) — which also has to serve the whole model stream, so it may run out of memory and drop mid-load.\n\nSwitch to "${o.suggest}" mode? It spreads the layers across the whole fleet.\n\nOK = use ${o.suggest}    ·    Cancel = keep ${o.mode}`;
        }
        if(confirm(msg)){ mode=o.suggest; }
      }
    }catch(e){ /* plan pre-check is best-effort */ }
  }
  document.getElementById('loadmsg').textContent='loading'+(q!=='none'?(' '+q):'')+'…';
  try{ const r=await (await fetch(`/load?model=${encodeURIComponent(m)}&ctx=${ctx}&mode=${mode}&quant=${q}`+(tp>1?`&tp=${tp}`:''),{method:'POST'})).json();
    if(r.ok){ const warn=(r.warnings||[]).length?`<br><span style="color:#d29922" title="pre-load guardrail (#76)">⚠ ${r.warnings.map(esc).join('<br>⚠ ')}</span>`:'';
      document.getElementById('loadmsg').innerHTML=`loaded (${esc(r.mode)}${r.quant&&r.quant!=='none'?'/'+esc(r.quant):''}) across ${(r.stages||[]).length} stage(s) @ ctx ${r.ctx}: ${(r.stages||[]).map(s=>esc(s.hostname)).join(' → ')}`+warn;
    } else { document.getElementById('loadmsg').textContent=`error: ${r.error||'failed'}`; }
  }catch(e){ document.getElementById('loadmsg').textContent='error: '+e; }
}
async function doRestart(){
  if(!confirm('RESTART THE WHOLE FLEET?\\n\\nSignals every worker to restart, then restarts the controller. Any in-flight load is ABORTED. All processes relaunch clean (supervisor) on the current code.')) return;
  const el=document.getElementById('loadmsg'); el.textContent='restarting fleet…';
  try{ const r=await (await fetch('/restart',{method:'POST'})).json();
    el.textContent=r.ok?`restart: signaled ${r.worker_count} worker(s); controller relaunching…`:`error: ${r.error||'failed'}`;
  }catch(e){ el.textContent='restart sent — controller dropped the connection (relaunching), reload the page in a few seconds'; }
}
async function doUpdate(){
  if(!confirm('UPDATE + RESTART NOW?\\n\\nForced (does NOT wait for idle): unloads ALL models, tells every worker to free its RAM, pulls the latest code from GitHub, swaps it in, and relaunches. Auto-load is blocked during the swap so a client request cannot reload a model mid-update. Any in-flight request is aborted.')) return;
  const el=document.getElementById('loadmsg'); el.textContent='updating: unloading + freeing worker RAM + pulling code…';
  try{ const r=await (await fetch('/update',{method:'POST'})).json();
    el.textContent=r.ok?`update: unloaded ${(r.unloaded||[]).length} model(s), freed ${r.workers_freed} worker(s); controller relaunching on new code…`:`error: ${r.error||'failed'}`;
  }catch(e){ el.textContent='update sent — controller dropped the connection (relaunching on new code), reload the page in ~20s'; }
}
async function doUnloadAll(){
  const lms=(LAST&&LAST.cluster&&LAST.cluster.loaded_models)||[];
  if(!lms.length){ document.getElementById('loadmsg').textContent='nothing loaded'; return; }
  if(!confirm('Unload ALL '+lms.length+' model(s) from every node?\\n\\n'+lms.map(m=>'- '+(m.display_name||m.friendly)).join('\\n')+'\\n\\nThis frees their RAM/VRAM fleet-wide. Reversible (reload from the list).')) return;
  const el=document.getElementById('loadmsg'); el.textContent='unloading all…';
  try{
    const r=await (await fetch('/unload',{method:'POST'})).json();
    const n=(r.unloaded||[]).length;
    el.textContent = n? `unloaded ${n} model(s): ${r.unloaded.join(', ')}` : 'unloaded (nothing was loaded)';
  }catch(e){ el.textContent='error: '+e; }
}
async function doAutoLoad(name){   // #auto-defaults: load using the Auto-load defaults (quant/ctx/mode)
  const aq=(document.getElementById('cfg-aq')||{}).value||'int4';
  const cx=(document.getElementById('cfg-ctx')||{}).value||0;
  const md=(document.getElementById('cfg-mode')||{}).value||'auto';
  const el=document.getElementById('loadmsg'); el.textContent=`loading ${name} (${aq}, ctx ${cx||'auto'}, ${md})…`;
  try{
    const r=await (await fetch(`/load?model=${encodeURIComponent(name)}&quant=${encodeURIComponent(aq)}&ctx=${encodeURIComponent(cx)}&mode=${encodeURIComponent(md)}`,{method:'POST'})).json();
    el.textContent=r.ok?`loaded ${r.model||name} @ ${r.quant||aq} · ctx ${r.ctx||cx} · ${r.mode||md}`:`error: ${r.error||''}`;
  }catch(e){ el.textContent='load failed: '+e; }
}
async function doLoadModel(name,quant){ const sel=document.getElementById('m');
  // robust against a momentarily-stale <select>: inject the option if absent so .value always
  // takes (a missing option would leave value='' -> /load fails "unknown model ''").
  if(![...sel.options].some(o=>o.value===name)) sel.add(new Option(name,name));
  sel.value=name; doLoad(quant, rowModes[name]||undefined); }
async function doUnloadModel(name){   // #72: per-model unload (frees its shards fleet-wide; then a new quant can load)
  if(!confirm('Unload '+name+'?\\n\\nFrees its shards fleet-wide. Reversible (reload from the list).')) return;
  const el=document.getElementById('loadmsg'); el.textContent='unloading '+name+'…';
  try{ const r=await (await fetch('/unload?model='+encodeURIComponent(name),{method:'POST'})).json();
    el.textContent=r.ok?('unloaded '+((r.unloaded||[]).join(', ')||name)):('error: '+(r.error||'')); }
  catch(e){ el.textContent='unload failed: '+e; }
}
async function doCancelLoad(name){   // #stuck-load-override: kill a wedged in-flight load (0%-forever)
  if(!confirm('Cancel the in-flight load of '+name+'?\\n\\nKills it and frees any partial shards. Use this if the load is stuck at 0%. Re-Load it afterward to restart.')) return;
  const el=document.getElementById('loadmsg'); el.textContent='cancelling load '+name+'…';
  try{ const r=await (await fetch('/cancel_load?model='+encodeURIComponent(name),{method:'POST'})).json();
    el.textContent=r.ok?('cancelled load: '+((r.cancelled||[]).join(', ')||name)+' — re-Load to restart'):('error: '+(r.error||'')); }
  catch(e){ el.textContent='cancel failed: '+e; }
}
async function doCompileShards(name, quant){   // #shard-cache: pre-quantize on the controller (beast)
  const el=document.getElementById('loadmsg');
  el.textContent='compiling '+quant+' shard cache for '+name+'… (progress on the loading card; can take minutes for big models)';
  try{ const r=await (await fetch(`/compile_shards?model=${encodeURIComponent(name)}&quant=${encodeURIComponent(quant)}`,{method:'POST'})).json();
    el.textContent=r.ok?`compiled ${quant} shard cache for ${name}: ${r.files} files, ${r.size_gb} GB`:`compile error: ${r.error||'failed'}`;
  }catch(e){ el.textContent='compile error: '+e; }
}
async function doVerifyShards(name, quant){   // #shard-cache: full sha256 integrity check + fix/fail popup
  try{ const r=await (await fetch(`/verify_shards?model=${encodeURIComponent(name)}&quant=${encodeURIComponent(quant)}`,{method:'POST'})).json();
    if(r.ok){ alert(`${name} — ${quant} shard cache is intact (sha256 verified).`); return; }
    if(confirm(`${name} — ${quant} shard cache is BROKEN/incomplete:\\n\\n`+(r.problems||[]).join('\\n')+`\\n\\nRecompile it now? (Cancel = leave as-is; loads fall back to the normal bf16 path.)`)){
      doCompileShards(name, quant);
    }
  }catch(e){ alert('verify error: '+e); }
}
async function doReconfigure(name){   // #88: switch a resident model to/from tensor-parallel (managed reload)
  const sel=document.getElementById('rc-'+name);
  const v=sel?sel.value:'1';
  const tp=parseInt(v,10)||1, cpu=v.indexOf('c')>=0;
  const desc=tp>1?('tensor-parallel ×'+tp+(cpu?' (CPU)':'')):'pipeline';
  if(!confirm('Reconfigure '+name+' to '+desc+'?\\n\\nThe model is briefly UNAVAILABLE while it re-streams (managed reload). On failure it rolls back to a pipeline copy. Refused while the model is busy serving.')) return;
  const el=document.getElementById('loadmsg'); el.textContent='reconfiguring '+name+' → '+desc+'…';
  try{ const r=await (await fetch('/reconfigure?model='+encodeURIComponent(name)+'&tp='+tp+'&cpu_only='+(cpu?'true':'false'),{method:'POST'})).json();
    el.textContent=r.ok?('reconfigured '+name+' → '+((r.to&&r.to.mode)||'')):('reconfigure error: '+(r.error||'')); }
  catch(e){ el.textContent='reconfigure failed: '+e; }
}
async function doCancel(id){
  try{ const r=await (await fetch('/cancel?id='+id,{method:'POST'})).json();
    document.getElementById('slotsub').textContent=r.ok?('cancelled request '+id):('cancel error: '+(r.error||''));
  }catch(e){ document.getElementById('slotsub').textContent='cancel error: '+e; }
}
async function doDownload(name){
  document.getElementById('loadmsg').textContent='downloading '+name+'…';
  await fetch(`/download?model=${encodeURIComponent(name)}`,{method:'POST'}); }
async function doDlCtl(name,action){            // action = 'pause' | 'stop'
  document.getElementById('loadmsg').textContent=action+'… '+name+' (after the current file)';
  try{ const r=await (await fetch(`/download/${action}?model=${encodeURIComponent(name)}`,{method:'POST'})).json();
    if(!r.ok) document.getElementById('loadmsg').textContent=action+' failed: '+(r.error||'?');
  }catch(e){ document.getElementById('loadmsg').textContent=action+' error: '+e; } }
async function doResume(name){
  document.getElementById('loadmsg').textContent='resuming '+name+'…';
  await fetch(`/download/resume?model=${encodeURIComponent(name)}`,{method:'POST'}); }
async function doClear(name){
  if(!confirm('Clear cached + partial files for '+name+'? This deletes the download (you can re-download later).')) return;
  document.getElementById('loadmsg').textContent='clearing '+name+'…';
  try{ const r=await (await fetch(`/download/clear?model=${encodeURIComponent(name)}`,{method:'POST'})).json();
    document.getElementById('loadmsg').textContent = r.ok ? ('cleared '+name+(r.freed_gb?(' — freed '+r.freed_gb+' GB'):''))
      : ('clear failed: '+(r.error||'?'));
  }catch(e){ document.getElementById('loadmsg').textContent='clear error: '+e; } }
async function addModel(){
  const el=document.getElementById('addmsg');
  const hf=document.getElementById('addhf').value.trim();
  const gg=document.getElementById('addgguf').value.trim();   // optional .gguf filename (GGUF-only repo)
  if(!hf || hf.indexOf('/')<0){ el.textContent='enter a HF id like org/name'; return; }
  if(gg && !gg.toLowerCase().endsWith('.gguf')){ el.textContent='gguf file must end in .gguf'; return; }
  el.textContent=gg?'adding + converting GGUF…':'adding + downloading…';
  try{
    let url='/add_model?model='+encodeURIComponent(hf);
    if(gg) url+='&gguf_file='+encodeURIComponent(gg);
    const r=await (await fetch(url,{method:'POST'})).json();
    if(r.ok){ el.textContent='added '+r.friendly+' — '+(r.status||'downloading')+(gg?' (GGUF→safetensors)':'')+' (appears in the list below)';
      document.getElementById('addhf').value=''; document.getElementById('addgguf').value=''; }
    else el.textContent='error: '+(r.error||'failed');
  }catch(e){ el.textContent='error: '+e; }
}
async function doDelete(name){
  if(!confirm('Delete '+name+' COMPLETELY?\\n\\nRemoves its weights + shard cache (and the HF-cache copy) AND unregisters it — including any alias names registered against the same repo. To bring it back you must re-add it (org/name) and re-download.')) return;
  const r=await (await fetch(`/delete?model=${encodeURIComponent(name)}`,{method:'POST'})).json();
  document.getElementById('loadmsg').textContent=r.ok?('deleted '+name):('delete failed: '+r.error); }
async function doGen(){
  const out=document.getElementById('out'); out.textContent='';
  const body={model:document.getElementById('m').value, prompt:document.getElementById('prompt').value,
    stream:true, options:{num_predict:Number(document.getElementById('maxtok').value), temperature:0}};
  const resp=await fetch('/api/generate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!resp.ok){ out.textContent='error: '+(await resp.text()); return; }
  const rd=resp.body.getReader(); const dec=new TextDecoder(); let buf='';
  while(true){ const {done,value}=await rd.read(); if(done) break; buf+=dec.decode(value,{stream:true});
    let nl; while((nl=buf.indexOf('\\n'))>=0){ const ln=buf.slice(0,nl); buf=buf.slice(nl+1);
      if(!ln.trim()) continue; try{ const o=JSON.parse(ln); if(o.response) out.textContent+=o.response; }catch(e){} } }
}
// ---- #model-detail: click-a-card → expanded model details modal ----
function openModelModal(key){ window.__mdl=key; renderModelModal();
  document.getElementById('mdlov').classList.add('show'); }
function closeModelModal(){ window.__mdl=null;
  document.getElementById('mdlov').classList.remove('show'); }
document.addEventListener('keydown',e=>{ if(e.key!=='Escape')return;
  if(document.getElementById('ctxov').classList.contains('show')) closeCtxHistory(); else closeModelModal(); });
// #ctx-history: click a Tokens in/out row -> scrollable popup of the ACTUAL context for that direction.
// Lives only while the model is loaded (controller clears it on unload); decoded on demand by /history.
function closeCtxHistory(){ document.getElementById('ctxov').classList.remove('show'); }
async function openCtxHistory(key,dir){
  const ov=document.getElementById('ctxov'), box=document.getElementById('ctxbox');
  const close='<button class="mdlclose" onclick="closeCtxHistory()">&times;</button>';
  box.innerHTML=close+'<h2>loading…</h2>'; ov.classList.add('show');
  let d;
  try{ d=await (await fetch('/history?model='+encodeURIComponent(key)+'&dir='+dir)).json(); }
  catch(e){ box.innerHTML=close+'<h2>error</h2><div class="sub">'+esc(String(e))+'</div>'; return; }
  const ents=d.entries||[];
  const title=(dir==='in'?'Context sent → ':'Context received ← ')+key;
  let html=close+'<h2>'+esc(title)+'</h2>'
    +'<div class="sub" style="margin-bottom:8px">'+ents.length+' of '+(d.count||ents.length)
    +' kept request(s), newest first — cleared when the model unloads</div>';
  if(!ents.length) html+='<div class="sub">no requests captured yet</div>';
  ents.forEach((e,i)=>{
    const txt=(dir==='in'?e.input:e.output)||'';
    const tk=(dir==='in'?e.tok_in:e.tok_out)||0;
    const when=e.ts?new Date(e.ts).toLocaleString():'';
    html+='<div class="ctxent"><div class="ctxhdr">#'+(ents.length-i)+' · '+esc(when)+' · '
      +tk.toLocaleString()+' tok</div><pre class="ctxpre">'+esc(txt)+'</pre></div>';
  });
  box.innerHTML=html;
}
function renderModelModal(){
  const key=window.__mdl; if(!key) return;
  const lms=(LAST&&LAST.cluster&&LAST.cluster.loaded_models)||[];
  const lm=lms.find(m=>(m.base||m.friendly)===key)||lms.find(m=>m.friendly===key);
  const box=document.getElementById('mdlbox');
  if(!lm){ box.innerHTML='<button class="mdlclose" onclick="closeModelModal()">&times;</button>'
    +'<h2>model not loaded</h2><div class="sub" style="margin-top:8px">It was unloaded since you opened this.</div>'; return; }
  const now=Date.now()/1000;
  const up=lm.loaded_at_ts?fmtUptime(now-lm.loaded_at_ts):'—';
  const idle=lm.last_used_ts?fmtUptime(now-lm.last_used_ts):'—';
  const tags=[];
  tags.push((lm.quant&&lm.quant!=='none')?lm.quant:'bf16');
  if(lm.arch) tags.push(lm.arch);
  tags.push(lm.is_moe?'MoE':'dense');
  tags.push(lm.is_tp?('tensor-parallel ×'+lm.tp_size):'pipeline');
  if(lm.is_embedding) tags.push('embedding');
  if((lm.stages||[]).length>1) tags.push(lm.stages.length+' nodes');
  const hosts=[...new Set((lm.stages||[]).map(s=>s.hostname))].join(', ');
  const r=(k,v)=>`<div class="mrow"><b>${k}</b><span>${v}</span></div>`;
  const st=(lm.stages||[]).map(s=>`<tr><td>${esc(s.hostname)}</td><td>${s.layer_start}–${s.layer_end}</td>`
    +`<td>${s.num_layers}</td><td>${(s.has_embed?'embed ':'')+(s.has_head?'head':'')||'—'}</td>`
    +`<td>${gb(s.est_gb)}</td><td>${gb(s.gpu_gb)}</td></tr>`).join('');
  box.innerHTML='<button class="mdlclose" onclick="closeModelModal()">&times;</button>'
   +`<h2>${esc(lm.display_name||lm.friendly)}</h2>`
   +((lm.aliases&&lm.aliases.length)?`<div class="sub" style="font-size:12px;color:#8b949e;margin-top:-4px" title="other names that resolve to this model">alias: ${lm.aliases.map(esc).join(', ')}</div>`:``)
   +(lm.target?`<div class="sub" style="font-size:12px">${esc(lm.target)}</div>`:``)
   +`<div style="margin:8px 0 2px 0">${tags.map(t=>`<span class="tag">${esc(String(t))}</span>`).join('')}</div>`
   +`<h3>Status</h3><div class="mgrid">`
     +r('Loaded for',up)
     +r('Last used',idle+' ago')
     +r('Now',(lm.active||0)>0?('serving ×'+lm.active):'idle')
     +r('Queued',(lm.queued||0))
     +r('Requests served',(lm.req_total||0).toLocaleString())
     +r('Load took',lm.load_seconds?fmtUptime(lm.load_seconds):'—')
   +`</div>`
   +`<h3>Configuration</h3><div class="mgrid">`
     +r('Quantization',(lm.quant&&lm.quant!=='none')?lm.quant:'none (bf16)')
     +r('Architecture',esc(lm.arch||'?'))
     +r('Type',lm.is_moe?'Mixture-of-Experts':'dense')
     +r('Parameters',esc(lm.params||'?'))
     +r('Layers',(lm.num_layers||0))
     +r('Context (used/max)',(lm.kv_pos||0).toLocaleString()+' / '+(lm.ctx||0).toLocaleString())
     +r('Layout',lm.is_tp?('tensor-parallel ×'+lm.tp_size):('pipeline ('+(lm.stages||[]).length+' stage)'))
     +r('Placed on',esc(hosts||'—'))
   +`</div>`
   +(lm.plan_basis?`<div class="sub" style="font-size:12px;margin-top:4px">${esc(lm.plan_basis)}</div>`:``)
   +`<h3>Memory</h3><div class="mgrid">`
     +r('Weights (total)',gb(lm.size_gb)+' GB')
     +r('On GPU (VRAM)','<span style="color:#3fb950">'+gb(lm.vram_used_gb)+' GB</span>')
     +r('On CPU (RAM)','<span style="color:#58a6ff">'+gb(lm.ram_used_gb)+' GB</span>')
     +r('KV (used/reserved)',gb(lm.kv_used_gb)+' / '+gb(lm.kv_reserved_gb)+' GB')
   +`</div>`
   +`<h3>Throughput &amp; tokens</h3><div class="mgrid">`
     +r('Decode tok/s (live)',(lm.tok_s||0).toFixed(1))
     +r('Decode tok/s (avg)',(lm.ema_tok_s||0).toFixed(1))
     +r('Decode tok/s (peak)',(lm.max_tok_s||0).toFixed(1))
     +r('Tokens in (prompt)',`<span class="ctxlink" onclick="openCtxHistory('${key}','in')">${(lm.tok_in_total||0).toLocaleString()} &#9656;&nbsp;view</span>`)
     +r('Tokens out (gen)',`<span class="ctxlink" onclick="openCtxHistory('${key}','out')">${(lm.tok_out_total||0).toLocaleString()} &#9656;&nbsp;view</span>`)
   +`</div>`
   +`<h3>Stages (${(lm.stages||[]).length})</h3>`
   +`<table><tr><th>host</th><th>layers</th><th>#</th><th>role</th><th>est GB</th><th>GPU GB</th></tr>${st}</table>`
   +((lm.warnings||[]).length?`<h3>Warnings</h3><div style="color:#d29922;font-size:12px">⚠ ${lm.warnings.map(esc).join('<br>⚠ ')}</div>`:``);
}
tick(); setInterval(tick,1500);
</script></body></html>
"""
