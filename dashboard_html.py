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

DASHBOARD_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniteModel</title>
<style>
  :root{
    --bg:#0d1117; --surface:#161b22; --surface2:#1c2230; --border:#2a3038; --border2:#3a424d;
    --text:#e6edf3; --muted:#9aa7b4; --dim:#6e7b89;
    --accent:#4f8cff; --good:#2ea043; --warn:#d29922; --hot:#e0833a; --bad:#da3633;
    --radius:10px; --mono:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;
    --sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5;}
  a{color:var(--accent);text-decoration:none}
  .wrap{max-width:none;margin:0 auto;padding:18px 24px 60px;}
  /* header */
  header{display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap}
  .brand{font-size:20px;font-weight:600;letter-spacing:.2px}
  .ctl{font-size:12px;color:var(--dim);font-family:var(--mono)}
  nav{display:flex;gap:4px;margin-left:8px}
  nav a{font-size:13px;color:var(--muted);padding:5px 11px;border-radius:8px;border:1px solid transparent}
  nav a.on{color:var(--text);background:var(--surface);border-color:var(--border)}
  nav a:hover{background:var(--surface)}
  .grow{flex:1}
  .btn{background:var(--surface);border:1px solid var(--border2);color:var(--text);border-radius:8px;
       padding:6px 12px;font-size:13px;cursor:pointer;display:inline-flex;align-items:center;gap:6px}
  .btn:hover{border-color:var(--accent)} .btn:active{transform:scale(.98)}
  .btn.pri{border-color:var(--accent);color:#cfe0ff}
  .btn.sm{padding:4px 9px;font-size:12px}
  .btn.ghost{background:transparent;border-color:var(--border);color:var(--muted)}
  /* fleet bar */
  .fleet{display:grid;grid-template-columns:repeat(5,1fr);gap:12px;margin-bottom:8px}
  .tile{background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:10px 13px}
  .tile .k{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.4px}
  .tile .v{font-size:22px;font-weight:600;margin-top:2px}
  .tile .v small{font-size:12px;color:var(--dim);font-weight:400}
  .bar{height:5px;background:#0a0e14;border:1px solid var(--border);border-radius:4px;margin-top:7px;overflow:hidden}
  .bar > i{display:block;height:100%;background:var(--accent)}
  .bar.warn > i{background:var(--warn)} .bar.hot > i{background:var(--hot)}
  /* #pool-split: GPU/RAM pool bars show iM's own usage (green) vs other processes (blue) */
  .poolbar{display:flex}
  .poolbar > i{display:block;height:100%}
  .poolbar > i.prog{background:var(--good)} .poolbar > i.oth{background:var(--accent)}
  .poolbar.warn{border-color:var(--warn)} .poolbar.hot{border-color:var(--hot)}
  /* section */
  .sec{display:flex;align-items:baseline;gap:10px;margin:22px 0 9px}
  .sec h2{font-size:16px;font-weight:600;margin:0}
  .sec .hint{font-size:12px;color:var(--dim)}
  input.f{background:var(--surface);border:1px solid var(--border);color:var(--text);border-radius:8px;
          padding:5px 10px;font-size:13px;width:170px}
  input.f::placeholder{color:var(--dim)}
  /* legend */
  .legend{display:flex;gap:15px;font-size:12px;color:var(--muted);margin-bottom:9px;flex-wrap:wrap}
  .dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;vertical-align:1px}
  /* connections rows (#connections — same shape as the Config page's node rows) */
  .nrow{display:flex;align-items:center;gap:14px;padding:9px 15px;border-bottom:1px solid var(--border);font-size:13px}
  .nrow:last-child{border-bottom:none}
  .nrow .nn{min-width:220px;font-weight:600}
  .nrow .nn small{font-weight:400;color:var(--dim);font-size:11px}
  .nrow .em{color:var(--dim)}
  /* model list */
  .list{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .row{display:flex;align-items:center;gap:12px;padding:11px 15px;border-bottom:1px solid var(--border)}
  .row:last-child{border-bottom:none}
  .row:hover{background:var(--surface2)}
  .row .nm{min-width:240px;cursor:pointer}
  .row .nm b{font-weight:600}
  .chip{font-size:10.5px;color:var(--muted);border:1px solid var(--border2);border-radius:9px;padding:1px 7px;margin-left:6px}
  .chip.al{color:var(--dim);border-style:dashed}
  .chip.q4{color:var(--accent);border-color:var(--accent);cursor:pointer;white-space:nowrap}
  .chip.q4:hover{background:var(--accent);color:#0a0e14}
  .row .meta{flex:1;font-size:12.5px;color:var(--muted);min-width:0}
  .row .meta .em{color:var(--text)}
  .row .acts{display:flex;align-items:center;gap:7px}
  .miniprog{height:4px;background:#0a0e14;border-radius:3px;margin-top:5px;max-width:280px;overflow:hidden}
  .miniprog > i{display:block;height:100%;background:var(--warn)}
  .grp{font-size:11px;color:var(--dim);text-transform:uppercase;letter-spacing:.5px;padding:8px 15px 4px;background:var(--bg)}
  /* nodes */
  .node{display:grid;grid-template-columns:210px 1fr 1fr 96px;align-items:center;gap:14px;padding:8px 15px;border-bottom:1px solid var(--border);font-size:13px}
  .node:last-child{border-bottom:none}
  .node .nn{font-weight:600;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
  .node .nn small{font-weight:400;color:var(--dim);font-size:11px}
  .node .mb{display:flex;align-items:center;gap:8px;min-width:0}
  .node .mb .lab{font-size:11px;color:var(--muted);width:34px;flex:none}
  .node .mb .bar{flex:1}
  .node .mb .num{font-size:11px;color:var(--muted);white-space:nowrap;flex:none;width:82px;text-align:right}
  .node .util{font-size:11px;color:var(--dim);text-align:right}
  .node .ver{font-size:10px;color:var(--dim);font-family:var(--mono);margin-top:2px}
  .node .ver.stale{color:var(--warn)}
  /* overlay/modal */
  .ov{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-start;justify-content:center;z-index:50}
  .ov.show{display:flex}
  /* #modal-scroll: while a popup is open the PAGE must not scroll — lock body (also covers
     touch/keyboard), and contain wheel chaining when the modal's own scroll hits an end. */
  body:has(.ov.show){overflow:hidden}
  .modal{background:var(--surface);border:1px solid var(--border2);border-radius:12px;max-width:640px;width:92%;
         margin:60px 0;padding:20px 22px;max-height:80vh;overflow:auto;overscroll-behavior:contain}
  .modal h3{margin:0 0 4px;font-size:17px}
  .modal .x{float:right;cursor:pointer;color:var(--muted);font-size:20px;line-height:1}
  .modal label{display:block;font-size:12px;color:var(--muted);margin:12px 0 4px}
  .modal input,.modal select{width:100%;background:var(--bg);border:1px solid var(--border2);color:var(--text);
         border-radius:8px;padding:7px 10px;font-size:13px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
  .note{font-size:12px;color:var(--warn);margin-top:8px}
  table.kv{width:100%;font-size:12.5px;border-collapse:collapse;margin-top:6px}
  table.kv td{padding:3px 0;color:var(--muted)} table.kv td.v{color:var(--text);text-align:right}
  .empty{padding:26px;text-align:center;color:var(--dim);font-size:13px}
  .err{color:var(--bad);font-size:12px}
</style></head>
<body><div class="wrap">

<header>
  <span class="brand">∞ InfiniteModel</span>
  <span class="ctl" id="ctl">connecting…</span>
  <nav>
    <a class="on" href="/">Models</a>
    <a href="/chat">Chat</a>
    <a href="/config">Config</a>
    <a href="/logs-page">Logs</a>
    <a href="/bandwidth">Bandwidth</a>
  </nav>
  <span class="grow"></span>
  <button class="btn pri" onclick="openAdd()">+ Add model</button>
</header>

<!-- FLEET BAR -->
<div class="fleet" id="fleet"></div>

<!-- MODELS -->
<div class="sec">
  <h2>Models</h2><span class="hint" id="mcount"></span>
  <span class="grow"></span>
  <input class="f" id="filter" placeholder="filter models…" oninput="render()">
</div>
<div class="legend">
  <span><span class="dot" style="background:var(--good)"></span>Loaded</span>
  <span><span class="dot" style="background:var(--warn)"></span>Loading / Compiling</span>
  <span><span class="dot" style="background:var(--dim)"></span>Registered (on disk)</span>
  <span><span class="dot" style="background:var(--accent)"></span>Downloading</span>
  <span><span class="dot" style="background:var(--bad)"></span>Won't fit / not downloaded</span>
</div>
<div class="list" id="models"><div class="empty">loading…</div></div>

<!-- NODES -->
<div class="sec"><h2>Nodes</h2><span class="hint" id="ncount"></span></div>
<div class="list" id="nodes"></div>

<!-- CONNECTIONS (#connections) -->
<div class="sec"><h2>Connections</h2><span class="hint" id="ccount"></span></div>
<div class="list" id="conns"><div class="empty">no clients yet</div></div>

</div>

<div class="ov" id="ov" onclick="if(event.target===this)closeOv()"><div class="modal" id="modal"></div></div>

<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const _up=s=>{s=Math.max(0,Math.floor(s||0));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?d+'d '+h+'h':(h?h+'h '+m+'m':m+'m')};
const gb=v=>(v==null?'—':(Math.round(v*10)/10)+' GB');
const pc=(a,b)=>b>0?Math.min(100,Math.round(100*a/b)):0;
let LAST=null;
let DETAIL_OPEN=null;   // #model-detail: model name whose detail modal is open (live-refreshed each poll)

async function api(path,opts){
  try{const r=await fetch(path,opts);const t=await r.text();let j;try{j=JSON.parse(t)}catch(e){j={ok:r.ok,raw:t}}
      if(!r.ok&&j&&j.error)throw new Error(j.error); if(!r.ok)throw new Error('HTTP '+r.status); return j;}
  catch(e){ throw e; }
}
function toast(msg,bad){ const c=$('#ctl'); const o=c.textContent; c.innerHTML=(bad?'<span class="err">':'<span style="color:var(--good)">')+esc(msg)+'</span>';
  window._toastUntil=Date.now()+(bad?6000:3500); setTimeout(()=>{c.textContent=o;},bad?6000:3500); }   // errors linger longer + survive status polls

async function tick(){
  let d; try{ d=await (await fetch('/status')).json(); }catch(e){ $('#ctl').innerHTML='<span class="err">controller unreachable</span>'; return; }
  LAST=d; render();
}
function render(){
  const d=LAST; if(!d)return;
  const c=d.controller||{}, p=d.pool||{}, comp=d.compute||{}, cl=d.cluster||{};
  if(!(window._toastUntil>Date.now())) $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?')+(c.code_date?' ('+c.code_date+')':'')+(c.uptime_s!=null?' · up '+_up(c.uptime_s):'');  // don't clobber an active toast
  // fleet tiles
  const loaded=(d.models||[]).filter(m=>m.loaded).length, reg=(d.models||[]).length;
  // pool bars: PHYSICAL used (total - free) against PHYSICAL total, one base (#pool-base).
  const vT=p.vram_total_gb||0, rT=p.ram_total_gb||0;
  const vU=Math.max(0,vT-(p.vram_free_gb||0)), rU=Math.max(0,rT-(p.ram_free_gb||0));
  // #pool-split: GREEN = what InfiniteModel itself uses (sum of loaded models' MEASURED VRAM/RAM),
  // BLUE = the rest of the physical usage = OTHER processes on the nodes. So any blue beyond the
  // green tells you something else is holding pool memory. Clamp iM's share to physical-used.
  const _lm=(d.models||[]).filter(m=>m.loaded);
  const vProg=Math.min(vU,_lm.reduce((s,m)=>s+(m.vram_used_gb||0),0));
  const rProg=Math.min(rU,_lm.reduce((s,m)=>s+(m.ram_used_gb||0),0));
  $('#fleet').innerHTML=[
    tile('Nodes', (p.nodes||0)+' <small>· '+(comp.gpus||0)+' GPU</small>'),
    tile('Loaded', loaded+' <small>/ '+reg+' registered</small>'),
    tile('GPU pool', fmt(vU)+'<small> / '+fmt(vT)+' GB</small>', poolbar(vProg,vU-vProg,vT,'InfiniteModel '+fmt(vProg)+' GB (green) · other processes '+fmt(vU-vProg)+' GB (blue) · '+fmt(vT-vU)+' GB free')),
    tile('RAM pool', fmt(rU)+'<small> / '+fmt(rT)+' GB</small>', poolbar(rProg,rU-rProg,rT,'InfiniteModel '+fmt(rProg)+' GB (green) · other processes '+fmt(rU-rProg)+' GB (blue) · '+fmt(rT-rU)+' GB free')),
    tile('Throughput', ((d.metrics||{}).tokens_per_s||0).toFixed(1)+' <small>tok/s · '+Math.round(comp.overall_pct||0)+'% busy</small>'),
  ].join('');
  renderModels(d,cl);
  renderNodes(d);
  renderConns(d);          // #connections: per-client accounting panel (bottom of the page)
  refreshDetailIfOpen();   // #model-detail: live-update the open detail modal's operational section
}
function fmtB(b){ if(b==null)return '—'; if(b<1024)return b+' B'; if(b<1048576)return (b/1024).toFixed(1)+' KB';
  if(b<1073741824)return (b/1048576).toFixed(1)+' MB'; return (b/1073741824).toFixed(2)+' GB'; }
// #connections: one row per client IP — connected-for, idle-for, real bytes both ways (ASGI
// counter, so an active stream's bytes grow live), token totals, what it is using/loading
// RIGHT NOW (INFLIGHT join + loading-card requested_by), and a Terminate button that kills
// every in-flight request from that client (POST /terminate).
function renderConns(d){
  const cs=(d.clients||[]);
  $('#ccount').textContent=cs.length+' client'+(cs.length==1?'':'s');
  if(!cs.length){ $('#conns').innerHTML='<div class="empty">no connections yet</div>'; return; }
  // both name forms: loading cards carry the Ollama display name (colons) AND the friendly
  // key (dashes) — active[].model is the FRIENDLY key, so match against both (review-caught)
  const lnames=((d.cluster||{}).loadings||[]).flatMap(l=>[l.model,l.display_model]).filter(Boolean);
  $('#conns').innerHTML=cs.map(c=>{
    const acts=(c.active||[]).map(a=>{
      const lbl=a.state==='running'?(lnames.includes(a.model)?'loading':'generating'):'queued';
      const col=a.state==='running'?'var(--good)':'var(--warn)';
      return '<span class="chip" style="color:'+col+'">'+esc(a.model)+' · '+lbl+' '+dur(a.s)+'</span>';
    });
    (c.loading||[]).forEach(m=>acts.push('<span class="chip" style="color:var(--warn)">'+esc(m)+' · loading</span>'));
    const act=acts.length?acts.join(' '):(c.last_model?('<span class="em">last: '+esc(c.last_model)+'</span>'):'<span class="em">—</span>');
    const kind=c.api?'':' <span class="chip">dashboard</span>';
    const idle=(c.active||[]).length?'<span style="color:var(--good)">active</span>':('idle '+dur(c.idle_s));
    const term=(c.active||[]).length
      ?' <button class="btn sm ghost" onclick="termClient(\''+esc(c.ip)+'\')">Terminate</button>':'';
    return '<div class="nrow"><div class="nn">'+esc(c.ip)+kind
      +' <small>· connected '+dur(c.connected_s)+' · '+idle+'</small></div>'
      +'<div style="font-size:12px;color:var(--muted)">'
      +fmtB(c.bytes_in)+' in / '+fmtB(c.bytes_out)+' out · '
      +(c.tok_in||0).toLocaleString()+' / '+(c.tok_out||0).toLocaleString()+' tok · '
      +(c.reqs||0)+' req</div>'
      +'<div style="flex:1;text-align:right">'+act+term+'</div></div>';
  }).join('');
}
async function termClient(ip){
  if(!confirm('Terminate every in-flight request from '+ip+'?'))return;
  try{ const r=await api('/terminate?ip='+encodeURIComponent(ip),{method:'POST'});
       toast('terminated '+ip+' · '+r.cancelled+' request(s) cancelled'); tick(); }
  catch(e){ toast(String(e.message||e),1); }
}
function fmt(v){return v==null?'—':(Math.round(v*10)/10)}
function tile(k,v,extra){return '<div class="tile"><div class="k">'+k+'</div><div class="v">'+v+'</div>'+(extra||'')+'</div>';}
function bar(a,b){const r=pc(a,b);const cls=r>=90?'hot':(r>=70?'warn':'');return '<div class="bar '+cls+'"><i style="width:'+r+'%"></i></div>';}
// #pool-split: two-segment bar — prog (green, iM's own usage) then other (blue, everything else),
// both against total. Fullness (green+blue) tints the BORDER warn/hot so the segment colors stay
// meaningful. tip = hover breakdown.
function poolbar(prog,other,total,tip){const pp=pc(prog,total),po=pc(Math.max(0,other),total);const cls=(pp+po)>=90?'hot':((pp+po)>=70?'warn':'');return '<div class="bar poolbar '+cls+'" title="'+tip+'"><i class="prog" style="width:'+pp+'%"></i><i class="oth" style="width:'+po+'%"></i></div>';}

// ---- model state derivation: the one source of truth ----
function mstate(m,cl){
  const id=m.internal_name||m.name;
  const loadings=(cl.loadings||[]), compiling=(cl.compiling||[]);
  const ld=loadings.find(x=>x.model===id||x.display_model===m.name);
  if(ld) return {k:'loading',c:'var(--warn)',rank:1,ld};
  const cp=compiling.find(x=>x.model===id||x.display_model===m.name);
  if(cp) return {k:'compiling',c:'var(--warn)',rank:1,ld:cp};
  const st=m.status||'';
  if(st==='downloading'||st==='pausing'||st==='stopping') return {k:'downloading',c:'var(--accent)',rank:2};
  if(st==='paused'||st==='stopped') return {k:'dlhalt',c:'var(--warn)',rank:2};
  if(m.loaded) return {k:'loaded',c:'var(--good)',rank:0};
  if(m.ready) return {k:'registered',c:'var(--dim)',rank:3};
  return {k:'notdl',c:'var(--bad)',rank:4};
}
function renderModels(d,cl){
  const f=($('#filter').value||'').toLowerCase().trim();
  let ms=(d.models||[]).map(m=>({m,s:mstate(m,cl)}));
  if(f) ms=ms.filter(x=>(x.m.name+' '+(x.m.target||'')+' '+(x.m.aliases||[]).join(' ')).toLowerCase().includes(f));
  ms.sort((a,b)=>a.s.rank-b.s.rank || a.m.name.localeCompare(b.m.name));
  $('#mcount').textContent=ms.length+' model'+(ms.length==1?'':'s');
  if(!ms.length){ $('#models').innerHTML='<div class="empty">no models'+(f?' match "'+esc(f)+'"':'')+'</div>'; return; }
  let html='', grp='';
  const GN={loaded:'Loaded',loading:'Loading',compiling:'Compiling',downloading:'Downloading',dlhalt:'Download paused',registered:'Registered · on disk',notdl:'Not downloaded'};
  for(const {m,s} of ms){
    if(s.k!==grp){ grp=s.k; html+='<div class="grp">'+GN[grp]+'</div>'; }
    html+=modelRow(m,s);
  }
  $('#models').innerHTML=html;
}
function modelRow(m,s){
  const t2i=(m.capabilities||[]).includes('t2i');
  // #cap-badges: every MODALITY capability gets a chip beside the model (config-inferred,
  // see status._model_caps). 'tools' stays out of the row (most chat models have it — noise);
  // it still shows in the detail modal's capabilities line.
  const CAPB={t2i:['🖼 t2i','Text-to-image (diffusers) checkpoint — loads onto a controller-co-located GPU and serves POST /v1/images/generations + the Generate panel.'],
    image:['👁 vision','Accepts image input (vision encoder in the checkpoint) — image→text on all three chat APIs.'],
    video:['🎞 video','Accepts video input (video token support in the checkpoint).'],
    stt:['🎤 stt','Accepts audio input (speech understanding — ask it what was said).'],
    tts:['🔊 tts','Speech output — synthesize WAV via /v1/audio/speech or the Text-to-speech panel in the model detail.'],
    ocr:['🔤 ocr','OCR-specialist checkpoint — built for document/text extraction from images (beyond generic vision).'],
    embedding:['🧮 embed','Embedding encoder — /api/embed + /v1/embeddings, not a chat model.']};
  const capChips=(m.capabilities||[]).filter(c=>CAPB[c]).map(c=>'<span class="chip" title="'+CAPB[c][1]+'">'+CAPB[c][0]+'</span>').join('');
  const arch=archChip(m)+capChips;
  const al=(m.aliases||[]).map(a=>'<span class="chip al">'+esc(a)+'</span>').join('');
  let meta='', acts='';
  if(s.k==='loaded'&&m.t2i){
    // #t2i-serve: image model card — no ctx/tok_s; live per-step render progress instead.
    const parts=[];
    if(m.quant)parts.push(esc(m.quant));
    if(m.vram_used_gb)parts.push('<span class="em">'+gb(m.vram_used_gb)+' VRAM</span>');
    if(m.active>0)parts.push('<span style="color:var(--good)">● rendering'
      +(m.t2i_step?(' step '+m.t2i_step+'/'+(m.t2i_total||'?')):'')+'</span>');
    meta=parts.join(' · ');
    acts='<button class="btn sm" onclick="unload(\''+esc(m.name)+'\')">Unload</button>';
  } else if(s.k==='loaded'){
    const parts=[];
    if(m.quant)parts.push(esc(m.quant));
    if(m.kv_quant&&m.kv_quant!=='none')parts.push('<span class="em" title="TurboQuant KV-cache quantization ('+esc(m.kv_quant)+'): keys/values stored at ~3–4 bits instead of bf16 (data-free rotation + Lloyd-Max, un-rotated on read so attention is unchanged) → ~2× smaller KV cache, so longer context and more co-resident models on the same VRAM. turbo4 ≈ near-lossless, turbo3 more aggressive; best on large models.">KV:'+esc(m.kv_quant)+'</span>');
    if(m.kv_offload)parts.push('<span class="em" title="KV cache offloaded to system RAM (OffloadedCache, per-layer prefetch): the VRAM the KV would reserve goes to model layers instead. Slower decode; useful for long context on small cards.">KV:RAM</span>');
    if(m.def_temperature!=null)parts.push('<span class="em" title="Default sampling temperature for this model — applied when a request does not send its own (explicit request values always win).">t='+esc(String(m.def_temperature))+'</span>');
    if(m.def_min_p!=null)parts.push('<span class="em" title="Default min-p sampling floor — drops tokens below this fraction of the top token\'s probability; confidence-adaptive, pairs with high temperature. Applied when a request sends no min_p.">mp='+esc(String(m.def_min_p))+'</span>');
    parts.push('ctx '+(m.ctx||'?'));
    if(m.vram_used_gb)parts.push('<span class="em">'+gb(m.vram_used_gb)+' VRAM</span>');
    if(m.ram_used_gb)parts.push(gb(m.ram_used_gb)+' RAM');
    // honest tok/s: LIVE (● green) while generating; when idle show the FROZEN last-run rate
    // (last_tok_s is never recomputed while idle, so it stays an honest measurement — not a
    // decaying/averaged value presented as "current").
    if(m.active>0){ parts.push('<span style="color:var(--good)">● '+(m.tok_s||0).toFixed(1)+' tok/s</span>'); }
    else { const _lt=m.last_tok_s||m.ema_tok_s||0; if(_lt)parts.push('<span class="em" title="last run — not live">'+_lt.toFixed(1)+' tok/s</span>'); }
    if(m.cpu_frac>=0.5)parts.push('<span style="color:var(--hot)">'+Math.round(m.cpu_frac*100)+'% CPU</span>');
    if(m.active)parts.push(m.active+' active');
    const b4=int4Badge(m); if(b4)parts.push(b4);   // #int4-badge: loaded but no int4 cache yet
    meta=parts.join(' · ');
    acts='<button class="btn sm" onclick="unload(\''+esc(m.name)+'\')">Unload</button>';
  } else if(s.k==='loading'){
    const ld=s.ld||{}; const r=pc(ld.ready||0,ld.total||1);
    meta='compiling/loading · '+(ld.ready||0)+'/'+(ld.total||'?')+' · '+Math.round(ld.elapsed_s||0)+'s'+(ld.eta_s?(' · eta '+Math.round(ld.eta_s)+'s'):'')
        +'<div class="miniprog"><i style="width:'+r+'%"></i></div>';
    acts='<button class="btn sm ghost" onclick="cancelLoad(\''+esc(m.name)+'\')">Cancel</button>';
  } else if(s.k==='compiling'){
    const cp=s.ld||{}; const r=pc(cp.ready||0,cp.total||1);
    meta='compiling shard cache · '+(cp.ready||0)+'/'+(cp.total||'?')
        +(cp.elapsed_s!=null?' · '+Math.round(cp.elapsed_s)+'s':'')+(cp.eta_s?' · eta '+Math.round(cp.eta_s)+'s':'')
        +'<div class="miniprog"><i style="width:'+r+'%"></i></div>';
    acts='<button class="btn sm ghost" disabled>…</button>';
  } else if(s.k==='downloading'){
    // live download progress — the /status entry carries dl_done_gb/dl_total_gb/dl_pct plus a
    // rolling rate + ETA; render them like the load/compile cards instead of a static line
    // (a static "downloading weights…" reads as a crashed pull on a multi-hour download).
    const p=(m.dl_pct!=null)?m.dl_pct:null;
    meta='downloading · '+gb(m.dl_done_gb||0)+(m.dl_total_gb?' / '+gb(m.dl_total_gb):'')
        +(p!=null?' · '+p.toFixed(1)+'%':'')
        +(m.dl_rate_mbps?' · '+m.dl_rate_mbps.toFixed(1)+' MiB/s':'')
        +(m.dl_eta_s&&m.dl_rate_mbps?' · eta '+dur(m.dl_eta_s):'')
        +((m.status||'')!=='downloading'?' · <span class="em">'+esc(m.status)+'…</span>':'')
        +'<div class="miniprog"><i style="width:'+(p||0)+'%"></i></div>';
    acts='<button class="btn sm ghost" onclick="dl(\''+esc(m.name)+'\',\'pause\')" title="Pause after the current file — kept resumable">Pause</button>'
        +'<button class="btn sm ghost" onclick="dl(\''+esc(m.name)+'\',\'stop\')" title="Stop after the current file — downloaded files are kept, Resume continues from here">Stop</button>';
  } else if(s.k==='dlhalt'){
    const p=(m.dl_pct!=null)?m.dl_pct:null;
    meta='<span class="em">'+esc(m.status)+'</span> at '+gb(m.dl_done_gb||0)
        +(m.dl_total_gb?' / '+gb(m.dl_total_gb):'')+(p!=null?' · '+p.toFixed(1)+'%':'')
        +'<div class="miniprog"><i style="width:'+(p||0)+'%"></i></div>';
    acts='<button class="btn sm pri" onclick="dl(\''+esc(m.name)+'\',\'resume\')">Resume</button>'
        +'<button class="btn sm ghost" onclick="dl(\''+esc(m.name)+'\',\'clear\')" title="Discard the partially downloaded files">Clear</button>';
  } else if(s.k==='registered'){
    meta=fitMeta(m);
    acts=t2i?'<button class="btn sm pri" title="Load the image pipeline onto a controller-co-located GPU: DiT mixed-edge int4 (first+last blocks bf16 — the gate-tested near-bf16 recipe), text encoder on CPU, tiled VAE. Takes a few minutes (quantize + move)." onclick="t2iLoadDlg(\''+esc(m.name)+'\')">Load 🖼</button>'
            :'<button class="btn sm pri" onclick="openLoad(\''+esc(m.name)+'\')">Load ▾</button>';
  } else { // notdl
    meta='<span class="err">not downloaded</span> · '+gb(m.size_gb); acts='<button class="btn sm" onclick="dl(\''+esc(m.name)+'\',\'start\')">Download</button>';
  }
  return '<div class="row"><span class="dot" style="background:'+s.c+'"></span>'
    +'<div class="nm" onclick="openDetail(\''+esc(m.name)+'\')"><b>'+esc(m.name)+'</b>'+arch+al+'</div>'
    +'<div class="meta">'+meta+'</div><div class="acts">'+acts
    +'<span class="btn sm ghost" title="details" onclick="openDetail(\''+esc(m.name)+'\')">⋯</span></div></div>';
}
function archChip(m){
  const t=(m.target||'').toLowerCase(); let a='';
  if(t.includes('moe')||(m.cached&&0))a='';
  return m.arch?('<span class="chip">'+esc(m.arch)+'</span>'):'';
}
function fitMeta(m){
  const cz=m.cached||{};
  // int4 cached -> serve-from-cache ready; else honest "weights on disk" + the compile badge
  // (the old text claimed "cache ready" for uncached models — misleading).
  const q=cz.int4&&cz.int4.ok?('int4 '+gb(cz.int4.size_gb)+' cache ready')
        :((m.size_gb?gb(m.size_gb):'on disk')+int4Badge(m));
  let warn='';
  // surface the devstral-style giant-ctx trap
  const dc=m.default_ctx||0;
  if(dc>=131072) warn=' · <span style="color:var(--hot)">⚠ native ctx '+(dc>=1000?Math.round(dc/1024)+'k':dc)+' — set ctx on load</span>';
  return q+warn;
}
// #int4-badge: one-click int4 shard-cache compile for any on-disk model without one. Tooltip =
// the cost (est. cache size on disk — disk.models' bf16-normalized int4 estimate — and the
// controller's free disk) + the payoff (int4 loads serve from cache instantly). Click fires the
// same compileShards() as the detail modal's Precache button; the row flips to Compiling with a
// progress bar on the next poll. m.ready gate: can't compile weights that aren't downloaded.
function int4Badge(m){
  const cz=m.cached||{};
  // embedding encoders load whole-model float32 (_load_embedding_locked) and NEVER read the
  // shard cache — a compile would fail (decoder-shaped packer) and the payoff is false. No badge.
  if(!m.ready||(cz.int4&&cz.int4.ok)||(m.capabilities||[]).includes('embedding')||(m.capabilities||[]).includes('t2i'))return '';
  const dk=(((LAST||{}).disk||{}).models||[]).find(x=>x.name===m.name||x.internal_name===m.internal_name)||{};
  const est=(dk.quant_gb||{}).int4, free=((LAST||{}).disk||{}).controller_free_gb;
  const tip='Compile the int4 shard cache for '+m.name
    +'\n• est. size on disk: '+(est!=null?'~'+gb(est):'unknown')+(dk.src_dtype?' (packed from '+dk.src_dtype+' weights, group 128)':'')
    +'\n• writes to the controller’s _shards/int4 cache · free disk '+(free!=null?gb(free):'—')
    +((est!=null&&free!=null&&est>free)?' ⚠ may not fit':'')
    +'\n• runs as a background subprocess (below-normal priority) — serving is unaffected'
    +'\n• once cached: int4 loads serve from cache instantly, no on-the-fly quantize'
    +'\n\nClick to compile now.';
  return ' <span class="chip q4" title="'+esc(tip)+'" onclick="compileShards(\''+esc(m.name)+'\',\'int4\')">⚡ int4</span>';
}

function renderNodes(d){
  const ns=(d.nodes||[]).slice().sort((a,b)=>(b.has_gpu?1:0)-(a.has_gpu?1:0)||a.hostname.localeCompare(b.hostname));
  $('#ncount').textContent=ns.length+' nodes';
  // fleet-consensus version = the most common client_version among nodes; a node that differs from it
  // is flagged (warn) so a worker that missed a restart/update stands out. When all match, nothing flags.
  const _vc={}; ns.forEach(n=>{ if(n.client_version) _vc[n.client_version]=(_vc[n.client_version]||0)+1; });
  const _consensus=Object.keys(_vc).sort((a,b)=>_vc[b]-_vc[a])[0]||null;
  // #node-split: per node, GREEN = what InfiniteModel itself uses on THAT node — RAM = the worker
  // process RSS (proc_rss_gb), VRAM = the sum of loaded models' per-stage GPU bytes placed here —
  // vs BLUE = the rest of the physical usage (OS / other processes), so it stands out from everything
  // else and mirrors the GPU/RAM pool tiles. Sum iM's per-node VRAM from the loaded models' stages.
  const imVram={};
  (d.models||[]).filter(m=>m.loaded).forEach(m=>(m.stages||[]).forEach(s=>{
    imVram[s.hostname]=(imVram[s.hostname]||0)+(s.gpu_gb||0);
  }));
  $('#nodes').innerHTML=ns.map(n=>{
    const gpu=n.has_gpu;
    const util=gpu?('GPU '+Math.round(n.gpu_util||0)+'%'):('CPU '+Math.round(n.cpu_percent||0)+'%');
    const dev=(gpu?(n.device_name||'GPU'):((n.cores||'')+'c CPU')).replace(/^NVIDIA GeForce /,'');
    const off=(!n.alive)?' <span class="err">offline</span>':'';
    // GREEN = iM's own usage on this node, BLUE = other/OS usage, then free (poolbar, same as the
    // pool tiles). `im` is clamped to physical-used so a stale estimate can't overflow the bar.
    const memRow=(lab,im,used,tot)=>{
      const g=Math.max(0,Math.min(im,used)),o=Math.max(0,used-g),f=Math.max(0,tot-used);
      return '<div class="mb"><span class="lab">'+lab+'</span>'
        +poolbar(g,o,tot,'InfiniteModel '+fmt(g)+' GB (green) · other '+fmt(o)+' GB (blue) · '+fmt(f)+' GB free')
        +'<span class="num">'+fmt(used)+' / '+fmt(tot)+'</span></div>';
    };
    // Fixed grid columns: name · VRAM · RAM · util. CPU-only nodes leave the VRAM
    // cell empty so their RAM bar still aligns with the GPU nodes' RAM column.
    const ramUsed=Math.max(0,(n.total_mem_gb||0)-(n.free_mem_gb||0));
    const vram=gpu?memRow('VRAM',(imVram[n.hostname]||0),(n.vram_used_gb||0),(n.vram_total_gb||0)):'<div class="mb"></div>';
    const cv=n.client_version||''; const _stale=(_consensus&&cv&&cv!==_consensus)?' stale':'';
    const verCell='<div class="ver'+_stale+'"'+(_stale?' title="version differs from fleet consensus '+esc(_consensus)+'"':'')+'>'+(cv?'v'+esc(cv):'—')+'</div>';
    return '<div class="node"><div class="nn">'+esc(n.hostname)+' <small>'+esc(dev)+'</small>'+off+'</div>'
      +vram+memRow('RAM',(n.proc_rss_gb||0),ramUsed,(n.total_mem_gb||0))
      +'<div class="util">'+util+verCell+'</div></div>';
  }).join('');
}

// ---------- actions ----------
function closeOv(){ $('#ov').classList.remove('show'); DETAIL_OPEN=null; }
// #modal-scroll: wheel over the dark backdrop (outside the popup box) used to scroll the PAGE
// behind the popup. Forward it to the modal instead so the wheel scrolls the popup wherever
// the mouse sits. deltaMode 1 = line-based deltas (Firefox) — scale to ~pixels.
$('#ov').addEventListener('wheel',e=>{ if(e.target===e.currentTarget){ $('#modal').scrollTop+=e.deltaY*(e.deltaMode===1?16:1); e.preventDefault(); } },{passive:false});
function openAdd(){
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Add a model</h3>'
   +'<div style="font-size:12px;color:var(--muted);margin-top:4px">Register any Hugging Face id. It downloads in the background.</div>'
   +'<label>Hugging Face id</label><input id="a-hf" placeholder="org/Name-Instruct">'
   +'<div class="grid2"><div><label>Name (optional)</label><input id="a-nm" placeholder="auto from id"></div>'
   +'<div><label>GGUF file (optional)</label><input id="a-gg" placeholder="*.gguf"></div></div>'
   +'<div style="margin-top:16px;text-align:right"><button class="btn pri" onclick="doAdd()">Add + download</button></div>'
   +'<div id="a-err" class="err" style="margin-top:8px"></div>';
  $('#ov').classList.add('show'); setTimeout(()=>$('#a-hf').focus(),50);
}
async function doAdd(){
  const hf=$('#a-hf').value.trim(), nm=$('#a-nm').value.trim(), gg=$('#a-gg').value.trim();
  if(!hf.includes('/')){ $('#a-err').textContent='enter an org/name Hugging Face id'; return; }
  const q=new URLSearchParams({model:hf}); if(nm)q.set('name',nm); if(gg)q.set('gguf_file',gg);
  try{ await api('/add_model?'+q.toString(),{method:'POST'}); closeOv(); toast('added '+hf); tick(); }
  catch(e){ $('#a-err').textContent=String(e.message||e); }
}
function _devOpts(){
  const ns=(LAST&&LAST.nodes||[]).slice().sort((a,b)=>(b.has_gpu?1:0)-(a.has_gpu?1:0)||a.hostname.localeCompare(b.hostname));
  const g=ns.filter(n=>n.has_gpu&&n.vram_enabled!==false&&n.alive).map(n=>'<option value="g:'+esc(n.hostname)+'">'+esc(n.hostname)+' — GPU ('+fmt(n.vram_total_gb)+' GB)</option>').join('');
  const c=ns.filter(n=>n.alive&&n.ram_enabled!==false).map(n=>'<option value="c:'+esc(n.hostname)+'">'+esc(n.hostname)+' — CPU/RAM ('+fmt(n.total_mem_gb)+' GB)</option>').join('');
  return (g?'<optgroup label="Pin to a GPU device">'+g+'</optgroup>':'')+(c?'<optgroup label="Pin to a CPU node">'+c+'</optgroup>':'');
}
function openLoad(name){
  const m=(LAST.models||[]).find(x=>x.name===name)||{};
  const cz=m.cached||{}; const dc=m.default_ctx||0;
  const ctxDefault = dc>=131072 ? 16384 : (dc||32767);
  const qopt=q=>'<option value="'+q+'">'+q+(cz[q]&&cz[q].ok?' · '+gb(cz[q].size_gb)+' cached':'')+'</option>';
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Load '+esc(name)+'</h3>'
   +'<div class="grid2"><div><label>Quant</label><select id="l-q" onchange="previewSoon(\''+esc(name)+'\')">'+qopt('int4')+qopt('int2')+qopt('int8')+'<option value="none">none (bf16)'+(cz.none&&cz.none.ok?' · '+gb(cz.none.size_gb)+' cached':(m.size_gb?' · '+gb(m.size_gb):''))+'</option></select></div>'
   +'<div><label>Context length</label><input id="l-ctx" type="number" value="'+ctxDefault+'" oninput="previewSoon(\''+esc(name)+'\')"></div></div>'
   +'<label>Placement</label><select id="l-place" onchange="_placeChg();previewSoon(\''+esc(name)+'\')">'
   +'<optgroup label="Auto / distribute"><option value="m:auto">auto · GPU-first</option><option value="m:all-gpu">all-GPU</option>'
   +'<option value="m:distribute">distribute (CPU+GPU)</option><option value="m:proportional">proportional</option></optgroup>'
   +_devOpts()
   +'<optgroup label="Tensor-parallel"><option value="tp:gpu">GPU tensor-parallel</option><option value="tp:cpu">CPU tensor-parallel (RAM)</option></optgroup></select>'
   +'<div id="l-tpwrap" style="display:none;margin-top:8px"><label>TP width (number of nodes)</label><input id="l-tpn" type="number" min="2" value="2" oninput="previewSoon(\''+esc(name)+'\')"></div>'
   +'<div class="grid2" style="margin-top:8px">'
   +'<div><label title="Where the KV cache (the conversation scratchpad — grows with context) lives. GPU = fastest. System RAM (OffloadedCache, per-layer prefetch) frees that VRAM for model layers — useful when pushing context past what the card holds; decode is slower.">KV cache</label>'
   +'<select id="l-kvo"><option value="">GPU (fastest)</option><option value="1">System RAM (frees VRAM)</option></select></div>'
   +'<div><label title="Default sampling temperature for this model: used when a request does not send its own (explicit request values always win). Empty = greedy (0). ~0.7 leans creative, 0 = deterministic.">Default temperature</label>'
   +'<input id="l-temp" type="number" min="0" max="2" step="0.1" placeholder="0 (greedy)"></div></div>'
   +'<div class="grid2" style="margin-top:8px">'
   +'<div><label title="Min-p sampling floor: drops any token whose probability is below this fraction of the top token\'s. Confidence-adaptive — strict when the model is sure, looser when it isn\'t — so it pairs well with HIGH temperature (temp flattens the distribution and lets weird tokens in; min-p cuts them first). At temperature >= 1.0 use 0.05-0.1: lower barely filters, higher eats the variety you raised temperature for. Used when a request sends no min_p of its own.">Default min-p</label>'
   +'<input id="l-minp" type="number" min="0" max="1" step="0.01" placeholder="0 (off)"></div>'
   +'<div></div></div>'
   +(dc>=131072?'<div class="note">⚠ native ctx is '+(Math.round(dc/1024))+'k — a huge KV cache. Keep ctx modest (8–16k) unless you need more.</div>':'')
   +'<div style="margin-top:16px;text-align:right"><button class="btn ghost" onclick="preview(\''+esc(name)+'\')">Preview fit</button> '
   +'<button class="btn pri" onclick="doLoad(\''+esc(name)+'\')">Load</button></div>'
   +'<div id="l-out" style="font-size:12px;color:var(--muted);margin-top:10px"></div>';
  $('#ov').classList.add('show');
  previewSoon(name);   // #mem-preview: show the est VRAM/RAM + KV-at-ctx footprint immediately
}
function _placeChg(){ $('#l-tpwrap').style.display=$('#l-place').value.indexOf('tp:')===0?'block':'none'; }
function placeParams(name){
  const v=$('#l-place').value, p={model:name, quant:$('#l-q').value, ctx:$('#l-ctx').value};
  if(v.indexOf('m:')===0) p.mode=v.slice(2);
  else if(v.indexOf('g:')===0){ p.node=v.slice(2); }
  else if(v.indexOf('c:')===0){ p.node=v.slice(2); p.cpu_only='true'; }
  else if(v==='tp:gpu'){ p.tp=$('#l-tpn').value||2; }
  else if(v==='tp:cpu'){ p.tp=$('#l-tpn').value||2; p.cpu_only='true'; }
  const kvo=$('#l-kvo'); if(kvo&&kvo.value)p.kv_offload='true';      // #kv-offload: KV in system RAM
  const tmp=$('#l-temp'); if(tmp&&tmp.value!=='')p.temperature=tmp.value;  // #load-temp: default temp
  const mp=$('#l-minp'); if(mp&&mp.value!=='')p.min_p=mp.value;            // #min-p: default floor
  return p;
}
let _pvTimer=null;
// #mem-preview: debounce the /plan estimate so typing a ctx / flipping quant re-estimates without a
// request per keystroke. Called on dialog open + on every quant/ctx/placement change.
function previewSoon(name){ clearTimeout(_pvTimer); const el=$('#l-out'); if(el)el.textContent='estimating…'; _pvTimer=setTimeout(()=>preview(name),350); }
async function preview(name){
  const p=placeParams(name); const q=new URLSearchParams({model:p.model,quant:p.quant,ctx:p.ctx});
  if(p.mode)q.set('mode',p.mode); if(p.node)q.set('node',p.node); if(p.cpu_only)q.set('cpu_only',p.cpu_only);
  const out=$('#l-out'); if(!out)return; out.textContent='estimating…';
  let r; try{ r=await (await fetch('/plan?'+q.toString())).json(); }
  catch(e){ out.innerHTML='<span class="err">'+esc(String(e.message||e))+'</span>'; return; }
  const mem=r.mem||{}, a=r.assess||{};
  let html='';
  // Estimated footprint: weights + KV-at-ctx (+ per-1k scaling) + the est VRAM/RAM split.
  if(mem.weights_gb!=null){
    const split=(mem.est_vram_gb!=null)
      ? ('<span style="color:var(--good)">~'+gb(mem.est_vram_gb)+' VRAM</span> · <span class="em">~'+gb(mem.est_ram_gb)+' RAM</span>')
      : '<span class="em">— (does not fit; see below)</span>';
    html+='<div style="margin:2px 0"><b>Estimated footprint</b> · '+esc(mem.quant)+' · ctx '+(mem.ctx||0).toLocaleString()+'</div>'
      +'<table class="kv">'
      +'<tr><td>weights</td><td class="v">'+gb(mem.weights_gb)+'</td></tr>'
      +'<tr><td>KV cache @ '+(mem.ctx||0).toLocaleString()+'</td><td class="v">'+gb(mem.kv_gb)+' <span class="em">(~'+gb(mem.kv_per_1k_gb)+' / 1k tok)</span></td></tr>'
      +'<tr><td>total</td><td class="v"><b>'+gb(mem.total_gb)+'</b></td></tr>'
      +'<tr><td>placement</td><td class="v">'+split+'</td></tr>'
      +'</table>';
  }
  if(!r.ok){ html+='<div class="err" style="margin-top:6px">⚠ '+esc(r.error||'does not fit the fleet at these settings')+'</div>'; out.innerHTML=html; return; }
  const st=(r.stages||[]).map(s=>esc(s.hostname)+' L'+s.layer_start+'-'+s.layer_end).join(', ');
  html+='<div style="margin-top:6px"><b>plan:</b> '+esc(r.basis||'')+'<br>'+st+'</div>';
  (a.warnings||[]).forEach(w=>html+='<div class="note">⚠ '+esc(w)+'</div>');
  if(a.suggested_ctx&&(a.kv_ram_gb||0)>0.1) html+='<div class="note">tip: ctx ≤ '+a.suggested_ctx.toLocaleString()+' keeps the KV cache in VRAM</div>';
  if(p.tp)html+='<div class="note">TP preview is approximate (planned as pipeline)</div>';
  if(r.overload)html+='<div class="note">⚠ '+esc(r.overload.reason)+' — suggest '+esc(r.overload.suggest||'proportional')+'</div>';
  out.innerHTML=html;
}
async function doLoad(name){
  const q=new URLSearchParams(placeParams(name));
  // /load blocks until the load finishes — so close the popup NOW (the loading card appears on the
  // models page via the status poll) instead of holding the dialog open for the whole load.
  closeOv(); toast('loading '+name+'…'); tick();
  try{ await api('/load?'+q.toString(),{method:'POST'}); toast(name+' loaded'); }
  catch(e){ toast(name+' load failed: '+String(e.message||e),1); }
  tick();
}
// #persist / #no-unload: toggle a per-model lifecycle pin. persist -> autoload-on-restart;
// no_unload -> the absolute do-not-auto-unload veto. Unchecking sends the OFF variant.
async function setPin(name,kind,on){
  const p=kind==='persist'?(on?'persist':'unpersist'):(on?'no_unload':'no_unload_off');
  try{ await api('/config?'+p+'='+encodeURIComponent(name),{method:'POST'});
       toast((kind==='persist'?'autoload-on-restart ':'do-not-auto-unload ')+(on?'on':'off')+' · '+name); }
  catch(e){ toast(String(e.message||e),1); }
  tick();
}
async function unload(name){ closeOv(); try{ await api('/unload?model='+encodeURIComponent(name),{method:'POST'}); toast('unloaded '+name); }catch(e){ toast(String(e.message||e),1);} tick(); }
async function cancelLoad(name){ try{ await api('/cancel_load?model='+encodeURIComponent(name),{method:'POST'}); toast('cancelled load'); tick(); }catch(e){ toast(String(e.message||e),1);} }
async function dl(name,action){ try{ await api('/download'+(action==='start'?'':'/'+action)+'?model='+encodeURIComponent(name),{method:'POST'}); toast(action+' '+name); tick(); }catch(e){ toast(String(e.message||e),1);} }

function dur(sec){ sec=Math.max(0,Math.round(sec||0)); if(sec<60)return sec+'s';
  const m=Math.floor(sec/60); if(m<60)return m+'m'+(sec%60?' '+(sec%60)+'s':'');
  const h=Math.floor(m/60); return h+'h'+(m%60?' '+(m%60)+'m':''); }
// #model-detail: the LIVE section (rebuilt from the fresh /status card every poll while the modal is
// open). Loaded models get a full operational readout; not-loaded ones get the card summary (the deep
// architecture/config info is fetched once from /api/show into a separate static section).
function detailLive(name){
  const m=(LAST.models||[]).find(x=>x.name===name); if(!m)return '<div class="empty">model gone</div>';
  const cz=m.cached||{}; const now=Date.now()/1000;
  let rows=''; const add=(k,v)=>{ if(v!=null&&v!=='') rows+='<tr><td>'+k+'</td><td class="v">'+v+'</td></tr>'; };
  add('HF id',esc(m.target)); add('arch',esc(m.arch||'')); add('status',esc(m.status));
  add('aliases',(m.aliases||[]).map(esc).join(', '));
  add('weights size', m.size_gb!=null?gb(m.size_gb):((m.media&&m.media.size_gb!=null)?gb(m.media.size_gb):''));
  add('cached quants',Object.keys(cz).filter(q=>cz[q]&&cz[q].ok).map(q=>q+' '+gb(cz[q].size_gb)).join(', '));
  let out='<table class="kv">'+rows+'</table>';
  // #media-detail: a media model (tts/t2i/t2a) gets a media-appropriate operational block — the
  // LLM section below (tok/s, context, KV, tokens, layers) is all zeros/meaningless for it.
  if(m.loaded && m.media){
    const md=m.media; let o=''; const oadd=(k,v)=>{ if(v!=null&&v!=='') o+='<tr><td>'+k+'</td><td class="v">'+v+'</td></tr>'; };
    const gen=(m.active||0)>0;
    const KIND={tts:'Text-to-speech',t2i:'Text-to-image',t2a:'Text-to-audio (music)'};
    oadd('type',esc(KIND[md.kind]||md.kind||'media')+(md.engine?(' · '+esc(md.engine)):''));
    oadd('state', gen?('<span style="color:var(--good)">● generating'+((m.active||0)>1?(' ×'+m.active):'')+'</span>'):'<span class="em">idle</span>');
    if((m.queued||0)>0) oadd('queue','<span style="color:var(--warn)">'+m.queued+' waiting</span>');
    oadd('device', md.device==='GPU'?'<span style="color:var(--good)">GPU</span>':'<span class="em">CPU</span>');
    const _pf=n=>(n>=1e9?(n/1e9).toFixed(1)+'B':n>=1e6?(n/1e6).toFixed(0)+'M':n>=1e3?(n/1e3).toFixed(0)+'K':String(n));
    oadd('parameters', md.params?_pf(md.params):esc(m.params||''));
    oadd('weights',gb(md.size_gb));
    oadd('VRAM',gb(m.vram_used_gb));
    if((m.ram_used_gb||0)>0.01) oadd('RAM',gb(m.ram_used_gb));
    if(md.sample_rate) oadd('sample rate',(md.sample_rate/1000)+' kHz');
    if(md.n_voices){
      const vlist=(md.voices||[]);
      oadd('voices', vlist.length
        ? '<details><summary style="cursor:pointer;color:var(--accent)">'+md.n_voices+' available</summary><div style="font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:5px;line-height:1.8">'+vlist.map(esc).join(' · ')+'</div></details>'
        : md.n_voices);
    }
    if(md.default_voice) oadd('default voice','<span style="font-family:var(--mono)">'+esc(md.default_voice)+'</span>');
    if(md.last_render_s!=null){
      const spd=(md.last_audio_s>0&&md.last_render_s>0)?(md.last_audio_s/md.last_render_s):null;
      oadd('last render',(md.last_audio_s!=null?(md.last_audio_s+'s audio '):'')+'in '+md.last_render_s+'s'+(spd!=null?(' · '+spd.toFixed(1)+'× real time'):''));
    }
    oadd('requests served', m.req_total!=null?m.req_total:'');
    if(m.loaded_at_ts) oadd('uptime',dur(now-m.loaded_at_ts)+(m.load_seconds?(' · load took '+m.load_seconds+'s'):''));
    if(m.last_used_ts) oadd('last request', gen?'now':(dur(now-m.last_used_ts)+' ago'));
    oadd('placement basis',esc(m.plan_basis||''));
    out+='<h3 style="font-size:13px;margin-top:14px">'+esc(KIND[md.kind]||'Media')+(gen?' · <span style="color:var(--good)">running</span>':'')+'</h3><table class="kv">'+o+'</table>';
  } else if(m.loaded){
    let o=''; const oadd=(k,v)=>{ if(v!=null&&v!=='') o+='<tr><td>'+k+'</td><td class="v">'+v+'</td></tr>'; };
    const gen=(m.active||0)>0;
    oadd('state', gen?('<span style="color:var(--good)">● generating'+((m.active||0)>1?(' ×'+m.active):'')+'</span>'):'<span class="em">idle</span>');
    if((m.queued||0)>0) oadd('queue','<span style="color:var(--warn)">'+m.queued+' waiting</span>');
    // honest tok/s: live while generating, else the FROZEN last-run rate (never recomputed at idle)
    if(gen) oadd('tok/s (live)','<span style="color:var(--good)">● '+(m.tok_s||0).toFixed(1)+'</span>');
    else if(m.last_tok_s) oadd('tok/s (last run)','<span class="em">'+(m.last_tok_s||0).toFixed(1)+' · frozen while idle</span>');
    oadd('tok/s avg · peak','<span class="em">'+(m.ema_tok_s||0).toFixed(1)+' · '+(m.max_tok_s||0).toFixed(1)+' (lifetime)</span>');
    oadd('quant',esc(m.quant)+(m.tp_size>1?(' · TP'+m.tp_size):' · pipeline'));
    if(m.kv_quant&&m.kv_quant!=='none'){
      const _kvb={turbo2:'~2-bit',turbo3:'~3-bit',turbo4:'~4-bit'}[m.kv_quant]||'';
      oadd('<span title="TurboQuant KV-cache quantization: the KV cache (per-token memory the model accumulates) is stored at ~3–4 bits — a data-free random rotation makes the coordinates uniform, then a Lloyd-Max codebook quantizes each; on read it is un-rotated back to normal so the model\'s attention runs UNCHANGED. Effect: ~2× smaller KV cache → longer context or more co-resident models on the same VRAM. turbo4 ≈ near-lossless, turbo3 more aggressive; best on large models.">KV quant&nbsp;ⓘ</span>', esc(m.kv_quant)+(_kvb?(' · '+_kvb+' KV'):''));
    }
    const used=m.kv_pos||0, ctx=m.ctx||0, pctc=ctx?Math.min(100,Math.round(100*used/ctx)):0;
    oadd('context','<b>'+used.toLocaleString()+'</b> / '+ctx.toLocaleString()+' tok'+(ctx?(' · '+pctc+'%'):'')+'<div class="miniprog"><i style="width:'+pctc+'%"></i></div>');
    oadd('VRAM',gb(m.vram_used_gb));
    oadd('RAM',gb(m.ram_used_gb)+(m.cpu_frac>=0.5?(' <span style="color:var(--hot)">('+Math.round(m.cpu_frac*100)+'% on CPU — slow)</span>'):(m.cpu_frac>0?(' ('+Math.round(m.cpu_frac*100)+'% CPU)'):'')));
    oadd('KV used / reserved',gb(m.kv_used_gb)+' / '+gb(m.kv_reserved_gb));
    oadd('layers · params',(m.num_layers||'?')+' · '+esc(m.params||'?'));
    oadd('requests served',m.req_total!=null?m.req_total:'');
    oadd('tokens in / out',m.tok_in_total!=null?((m.tok_in_total||0).toLocaleString()+' / '+(m.tok_out_total||0).toLocaleString()):'');
    if(m.loaded_at_ts) oadd('uptime',dur(now-m.loaded_at_ts)+(m.load_seconds?(' · load took '+m.load_seconds+'s'):''));
    if(m.last_used_ts) oadd('last request', gen?'now':(dur(now-m.last_used_ts)+' ago'));
    oadd('placement basis',esc(m.plan_basis||'')+(m.speed_tier?(' · '+esc(m.speed_tier)):''));
    // #runtime-config: live view of the sampling defaults (edited via the static form below —
    // this row just reflects the current values each poll, so an Apply shows up immediately)
    {const sd=m.sampling_defaults||{}, sdp=[];
     if(m.def_temperature!=null)sdp.push('t='+esc(String(m.def_temperature)));
     if(m.def_min_p!=null)sdp.push('min-p='+esc(String(m.def_min_p)));
     for(const k in sd)if(sd[k]!=null)sdp.push(esc(k)+'='+esc(String(sd[k])));
     if(sdp.length)oadd('sampling defaults',sdp.join(' · '));}
    out+='<h3 style="font-size:13px;margin-top:14px">Operational'+(gen?' · <span style="color:var(--good)">running</span>':'')+'</h3><table class="kv">'+o+'</table>';
  }
  if(m.stages&&m.stages.length){
    // #real-stats: "on GPU x of NODE-VRAM-total" — gpu_gb is the VRAM this stage MEASURED-placed
    // (the old bare "GPU 16 GB" read as the card's capacity); the node's real total is pulled
    // live from the nodes list so nothing about the hardware is assumed. NOTE est includes the
    // KV reserve, so est-minus-gpu is NOT the CPU-resident weight — don't render that.
    const _nodes=(LAST.nodes||[]);
    out+='<h3 style="font-size:13px;margin-top:14px">Placement · '+m.stages.length+' stage'+(m.stages.length>1?'s':'')+'</h3><table class="kv">'
      +m.stages.map(s=>{ const _n=_nodes.find(n=>n.hostname===s.hostname);
        const cap=(_n&&_n.vram_total_gb)?(' of '+gb(_n.vram_total_gb)+' VRAM'):'';
        const dev=(s.gpu_gb>0)?('<span style="color:var(--good)">on GPU '+gb(s.gpu_gb)+cap+'</span>')
                              :'<span class="em">CPU</span>';
        const role=[]; if(s.has_embed)role.push('embed'); if(s.has_head)role.push('head');
        return '<tr><td>'+esc(s.hostname)+'</td><td class="v">L'+s.layer_start+'–'+s.layer_end+' · '+(s.num_layers||0)+'L · '+dev+(role.length?(' · '+role.join('+')):'')+' · est '+gb(s.est_gb)+'</td></tr>';
      }).join('')+'</table>';
  }
  if((m.warnings||[]).length) out+='<div class="note">⚠ '+m.warnings.map(esc).join('<br>⚠ ')+'</div>';
  return out;
}
function refreshDetailIfOpen(){
  if(!DETAIL_OPEN)return;
  const ov=$('#ov'), el=document.getElementById('dlive');
  if(ov&&ov.classList.contains('show')&&el) el.innerHTML=detailLive(DETAIL_OPEN);
}
function _kvtbl(obj){ const ks=Object.keys(obj); if(!ks.length)return '';
  return '<table class="kv">'+ks.map(k=>{ let v=obj[k]; if(v&&typeof v==='object')v=JSON.stringify(v);
    if(typeof v==='number'&&Math.abs(v)>=1000)v=v.toLocaleString();
    return '<tr><td>'+esc(k)+'</td><td class="v" style="word-break:break-word">'+esc(String(v))+'</td></tr>'; }).join('')+'</table>'; }
// #model-detail: the STATIC deep section from /api/show — architecture, capabilities, and the FULL raw
// config.json / generation_config.json (everything about the model, whether or not it's loaded).
function renderShow(sh){
  if(!sh)return '<div class="note">no model info</div>';
  const mi=sh.model_info||{}, det=sh.details||{}, im=sh.infinitemodel||{}, caps=sh.capabilities||[];
  let out='';
  const skip={'tokenizer.ggml.model':1,'general.file_type':1};
  let ar=''; for(const k in mi){ if(skip[k])continue; let v=mi[k]; if(typeof v==='number'&&Math.abs(v)>=1000)v=v.toLocaleString();
    const lbl=k.replace(/^general\./,'').replace(/\./g,' ').replace(/_/g,' '); ar+='<tr><td>'+esc(lbl)+'</td><td class="v">'+esc(String(v))+'</td></tr>'; }
  if(ar) out+='<h3 style="font-size:13px;margin-top:14px">Architecture</h3><table class="kv">'+ar+'</table>';
  let dr=''; const dadd=(k,v)=>{ if(v!=null&&v!=='') dr+='<tr><td>'+k+'</td><td class="v">'+esc(String(v))+'</td></tr>'; };
  dadd('family',det.family); dadd('parameters',det.parameter_size); dadd('format',det.format);
  dadd('on-disk dtype',im.src_dtype); dadd('native ctx',im.default_ctx?im.default_ctx.toLocaleString():'');
  dadd('MoE',im.is_moe?'yes':''); dadd('embedding model',im.is_embedding?'yes':'');
  if(dr) out+='<h3 style="font-size:13px;margin-top:14px">Details</h3><table class="kv">'+dr+'</table>';
  if(caps.length) out+='<div style="margin-top:8px">capabilities: '+caps.map(c=>'<span class="chip">'+esc(c)+'</span>').join(' ')+'</div>';
  const cfg=im.config||{}; if(Object.keys(cfg).length) out+='<details style="margin-top:12px"><summary style="cursor:pointer;font-size:13px;color:var(--accent)">raw config.json · '+Object.keys(cfg).length+' keys</summary>'+_kvtbl(cfg)+'</details>';
  const gc=im.generation_config||{}; if(Object.keys(gc).length) out+='<details style="margin-top:8px"><summary style="cursor:pointer;font-size:13px;color:var(--accent)">generation_config.json · '+Object.keys(gc).length+' keys</summary>'+_kvtbl(gc)+'</details>';
  if(im.engine) out+='<div style="margin-top:8px;font-size:11px;color:var(--dim)">engine '+esc(im.engine)+'</div>';
  return out;
}
async function openDetail(name){
  const m=(LAST.models||[]).find(x=>x.name===name); if(!m)return;
  DETAIL_OPEN=name;
  const cz=m.cached||{};
  // precache (shard cache): compile int4/int8/int2 so future loads serve from cache instantly.
  // int2 is explicit-build ONLY (never on first load): RTN-int2 quality is collapsed until the
  // calibrated packer lands, so the button carries the caveat. t2i checkpoints have no shard
  // cache (diffusers layout — the LLM compiler can't parse them).
  let pre='';
  if(m.ready&&!(m.capabilities||[]).includes('t2i')){ let chips='';
    for(const q of ['int4','int8','int2']){
      if(cz[q]&&cz[q].ok) chips+='<span class="chip al">'+q+' cached '+gb(cz[q].size_gb)+'</span> ';
      else chips+='<button class="btn sm ghost" '+(q==='int2'?'title="~2.5 bits/weight capacity tier, half the int4 cache size. CAVEAT: the current round-to-nearest int2 packing collapses generation quality on dense LLMs — build for experiments only until the calibrated (GPTQ-class) packer lands. Never auto-built on first load; an existing cache does serve int2 loads." ':'')+'onclick="compileShards(\''+esc(name)+'\',\''+q+'\')">Compile '+q+'</button> ';
    }
    pre='<h3 style="font-size:13px;margin-top:14px">Precache (shard cache)</h3><div>'+chips+'</div>';
  }
  // #runtime-config: editable RUNTIME settings for a loaded model — everything changeable without
  // a reload lives here. STATIC section (outside #dlive) so the per-poll live refresh can't clobber
  // the inputs mid-typing. Values apply instantly via POST /model_config; the next request uses them.
  let rt='';
  if(m.loaded){
    const sd=m.sampling_defaults||{};
    const _v=x=>(x!=null)?String(x):'';
    // #runtime-knobs: [input id, label, current value, placeholder-when-unset, suggested values
    // (datalist dropdown — free text still allowed), tooltip]. Every knob a request can send has
    // a runtime default here; empty input = unset (requests fall back to the built-in behavior).
    const RTF=[
      ['rt-temp','Temperature',_v(m.def_temperature),'unset (greedy)',['0','0.3','0.7','1','1.2','1.5'],'Default sampling temperature (0-2), used when a request does not send its own — explicit request values (including 0) always win. Empty = unset (greedy).'],
      ['rt-minp','Min-p',_v(m.def_min_p),'unset (off)',['0.03','0.05','0.08','0.1'],'Min-p floor (0-1): drops tokens below this fraction of the top token\'s probability — confidence-adaptive; pairs with high temperature (useful band 0.05-0.1 at temp >= 1). Empty = off.'],
      ['rt-topp','Top-p',_v(sd.top_p),'unset (1 = off)',['0.8','0.9','0.95','1'],'Nucleus sampling (0-1): keep the smallest set of top tokens whose probabilities sum to top-p, drop the tail. 1 = off.'],
      ['rt-topk','Top-k',_v(sd.top_k),'unset (off)',['1','20','40','100'],'Keep only the k most-probable tokens (0-1000). 1 = always the single top token (deterministic at any temperature). 0 or empty = off.'],
      ['rt-rp','Repeat penalty',_v(sd.repeat_penalty),'unset (1 = off)',['1.05','1.1','1.15','1.3'],'Penalize tokens already present in the recent window (0.5-2, llama.cpp convention: >1 discourages repetition, <1 encourages it). 1 = off.'],
      ['rt-rln','Repeat window',_v(sd.repeat_last_n),'64 tokens',['64','256','1024','-1'],'How many recent tokens (prompt+output) the repeat penalty scans. -1 = the whole context, 0 = disable the window (penalty off). Default 64.'],
      ['rt-pp','Presence penalty',_v(sd.presence_penalty),'unset (0 = off)',['0.5','1','1.5'],'Flat penalty on any token that has already appeared in the OUTPUT (-2 to 2, OpenAI convention; negative values encourage reuse). 0 = off.'],
      ['rt-fp','Frequency penalty',_v(sd.frequency_penalty),'unset (0 = off)',['0.5','1','1.5'],'Penalty scaled by how OFTEN a token has appeared in the output so far (-2 to 2, OpenAI convention). 0 = off.'],
      ['rt-seed','Seed',_v(sd.seed),'unset (random)',[],'Fix the sampling RNG: same prompt + same seed + same settings = same output every time. Empty = random per request.'],
      ['rt-np','Max tokens',_v(sd.num_predict),'unset (256)',['256','512','1024','4096','8192'],'Default response-length cap used when a request sends no num_predict / max_tokens of its own. Explicit request values always win.']];
    let rtf='', rtd='';
    RTF.forEach((f,i)=>{
      const dl='rtdl'+i;
      if(f[4].length)rtd+='<datalist id="'+dl+'">'+f[4].map(o=>'<option value="'+o+'"></option>').join('')+'</datalist>';
      rtf+='<div><label title="'+esc(f[5])+'">'+esc(f[1])+'</label>'
        +'<input id="'+f[0]+'" type="number" step="any"'+(f[4].length?(' list="'+dl+'"'):'')
        +' value="'+esc(f[2])+'" placeholder="'+esc(f[3])+'"></div>';
    });
    rt='<h3 style="font-size:13px;margin-top:14px">Runtime settings <span class="em" style="font-weight:normal">· sampling defaults — apply instantly, no reload; requests that send their own values always win</span></h3>'
      +'<div class="grid2">'+rtf+'</div>'+rtd
      +'<div style="margin-top:8px"><button class="btn sm pri" onclick="applyRt(\''+esc(name)+'\')">Apply</button> '
      +'<span class="em" style="font-size:11px">empty = unset · each box\'s dropdown lists common values (free typing works too) · quant / ctx / KV placement still need a reload</span></div>';
  }
  // #tts-panel: a loaded speech-out model (capability 'tts' — Qwen2.5-Omni) gets an inline
  // synthesize box: type text, pick a voice, POST /v1/audio/speech, play + download the WAV.
  // STATIC (outside #dlive) so the per-poll live refresh can't wipe the textarea mid-typing.
  let tts='';
  if(m.loaded && (m.capabilities||[]).includes('tts')){
    const _ist='font:inherit;padding:6px 7px;border-radius:8px;border:1px solid var(--border2);background:var(--bg);color:var(--text)';
    tts='<h3 style="font-size:13px;margin-top:14px">Text-to-speech <span class="em" style="font-weight:normal">· synthesize a WAV from text via the Omni speech pipeline</span></h3>'
      +'<textarea id="tts-text" rows="3" placeholder="Type text to speak…" style="width:100%;box-sizing:border-box;resize:vertical;'+_ist+'">Hello! This is a test of the InfiniteModel speech engine.</textarea>'
      +'<div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
      +'<label style="font-size:12px;color:var(--muted)">Voice</label>'
      +'<select id="tts-voice" style="'+_ist+'"><option>Chelsie</option><option>Ethan</option></select>'
      +'<button class="btn sm pri" onclick="ttsSpeak(\''+esc(name)+'\')">Speak ▶</button>'
      +'<span id="tts-status" class="em" style="font-size:11px"></span></div>'
      +'<audio id="tts-audio" controls style="width:100%;margin-top:8px;display:none"></audio>'
      +'<div><a id="tts-dl" style="display:none;font-size:11px;color:var(--accent)" download>⬇ download wav</a></div>';
  }
  // #t2i-serve: image-generation box — prompt + knobs, POST /v1/images/generations, show the
  // PNG inline + download. STATIC (outside #dlive) so the poll can't wipe the prompt mid-typing.
  let t2ig='';
  if(m.loaded && (m.capabilities||[]).includes('t2i')){
    const _ig='font:inherit;padding:6px 7px;border-radius:8px;border:1px solid var(--border2);background:var(--bg);color:var(--text)';
    t2ig='<h3 style="font-size:13px;margin-top:14px">Generate image <span class="em" style="font-weight:normal">· flow-matching render on the model\'s GPU worker</span></h3>'
      +'<textarea id="t2i-prompt" rows="3" placeholder="Describe the image…" style="width:100%;box-sizing:border-box;resize:vertical;'+_ig+'"></textarea>'
      +'<div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
      +'<label style="font-size:12px;color:var(--muted)">Size</label>'
      +'<select id="t2i-size" style="'+_ig+'"><option>1024x1024</option><option>1328x1328</option><option>1664x928</option><option>928x1664</option><option>768x768</option></select>'
      +'<label style="font-size:12px;color:var(--muted)" title="Denoising steps: 20 is the tested quality/speed default; fewer = faster, softer">Steps</label>'
      +'<input id="t2i-steps" type="number" value="20" min="1" max="100" style="width:64px;'+_ig+'">'
      +'<label style="font-size:12px;color:var(--muted)" title="Blank = random. Same seed + same prompt + same settings = same image">Seed</label>'
      +'<input id="t2i-seed" type="number" placeholder="rnd" style="width:90px;'+_ig+'">'
      +'<button class="btn sm pri" id="t2i-go" onclick="t2iGen(\''+esc(name)+'\')">Generate ▶</button>'
      +'<span id="t2i-status" class="em" style="font-size:11px"></span></div>'
      +'<div id="t2i-out" style="margin-top:8px"></div>';
  }
  // #t2a-serve: music-generation box — style/genre tags + optional lyrics + all render knobs,
  // POST /v1/audio/music, play the WAV inline + download. STATIC (outside #dlive) so the per-poll
  // live refresh can\'t wipe the fields mid-typing. Served from the controller (not sandboxed) so
  // the download link works here — unlike a shared artifact.
  let t2ag='';
  if(m.loaded && (m.capabilities||[]).includes('t2a')){
    const _ig='font:inherit;padding:6px 7px;border-radius:8px;border:1px solid var(--border2);background:var(--bg);color:var(--text)';
    t2ag='<h3 style="font-size:13px;margin-top:14px">Generate music <span class="em" style="font-weight:normal">· ACE-Step flow-matching render on the model\'s GPU worker</span></h3>'
      +'<textarea id="t2a-prompt" rows="2" placeholder="Style / genre tags — e.g. melodic techno, warm analog pads, driving sub-bass, 124 bpm, instrumental" style="width:100%;box-sizing:border-box;resize:vertical;'+_ig+'"></textarea>'
      +'<textarea id="t2a-lyrics" rows="2" placeholder="Lyrics (optional — blank = instrumental). [verse] / [chorus] tags give it structure." style="width:100%;box-sizing:border-box;resize:vertical;margin-top:6px;'+_ig+'"></textarea>'
      +'<div style="margin-top:8px;display:flex;gap:8px;align-items:center;flex-wrap:wrap">'
      +'<label style="font-size:12px;color:var(--muted)" title="Clip length in seconds (3–240; ACE-Step caps at ~4 min). Longer clips develop more; short ones can loop.">Duration</label>'
      +'<input id="t2a-dur" type="number" value="30" min="3" max="240" style="width:66px;'+_ig+'"><span class="em" style="font-size:11px;margin-left:-4px">s</span>'
      +'<label style="font-size:12px;color:var(--muted)" title="Diffusion steps: 60 default; more = higher quality + slower, fewer = rougher">Steps</label>'
      +'<input id="t2a-steps" type="number" value="60" min="1" max="200" style="width:60px;'+_ig+'">'
      +'<label style="font-size:12px;color:var(--muted)" title="Guidance scale: how literally it follows the prompt (higher = more on-prompt but less varied). 12–15 typical.">Guidance</label>'
      +'<input id="t2a-guid" type="number" value="12" step="0.5" min="0" max="30" style="width:60px;'+_ig+'">'
      +'<label style="font-size:12px;color:var(--muted)" title="Blank = random. Same seed + prompt + settings = the same track every time.">Seed</label>'
      +'<input id="t2a-seed" type="number" placeholder="rnd" style="width:84px;'+_ig+'">'
      +'<button class="btn sm pri" id="t2a-go" onclick="t2aGen(\''+esc(name)+'\')">Generate ▶</button>'
      +'<span id="t2a-status" class="em" style="font-size:11px"></span></div>'
      +'<audio id="t2a-audio" controls style="width:100%;margin-top:8px;display:none"></audio>'
      +'<div><a id="t2a-dl" style="display:none;font-size:11px;color:var(--accent)" download>⬇ download wav</a></div>';
  }
  // #persist / #no-unload: per-model lifecycle pins (work for loaded AND not-loaded models — the
  // controller reports current state via LAST.controller.{persist_models,no_unload_models}).
  const _pc=(LAST&&LAST.controller)||{}, _pk=m.internal_name||m.friendly||name;
  const _isP=!!m.persist||(_pc.persist_models||[]).includes(_pk);
  const _isNU=!!m.no_unload||(_pc.no_unload_models||[]).includes(_pk);
  const pers='<h3 style="font-size:13px;margin-top:14px">Persistence</h3>'
    +'<label style="display:flex;align-items:center;gap:8px;font-size:13px;margin:6px 0;color:var(--text);cursor:pointer" title="Auto-reload this model when the controller restarts or redeploys — it re-streams to the workers after the fleet settles. Independent of the unload pin below.">'
    +'<input type="checkbox" style="width:auto"'+(_isP?' checked':'')+' onchange="setPin(\''+esc(name)+'\',\'persist\',this.checked)"> Autoload on restart</label>'
    +'<label style="display:flex;align-items:center;gap:8px;font-size:13px;margin:6px 0;color:var(--text);cursor:pointer" title="NEVER auto-unload this model — overrides every automatic REMOVAL: idle-unload and LRU eviction to free room for a new load (that new load FAILS instead of evicting this one). The juggler may still RELOAD it to promote it to a faster VRAM-only placement — a promotion, not a removal, so it is never left unloaded. This is the flag that wins.">'
    +'<input type="checkbox" style="width:auto"'+(_isNU?' checked':'')+' onchange="setPin(\''+esc(name)+'\',\'no_unload\',this.checked)"> Do not auto-unload</label>';
  let acts='';
  const _isT2i=(m.capabilities||[]).includes('t2i');
  const _isT2a=(m.capabilities||[]).includes('t2a');
  if(m.loaded&&(_isT2i||_isT2a)) acts='<button class="btn sm" onclick="unload(\''+esc(name)+'\')">Unload</button>';
  else if(m.loaded) acts='<button class="btn sm pri" onclick="location.href=\'/chat?model='+encodeURIComponent(name)+'\'">Chat ↗</button> '
    +'<button class="btn sm" onclick="unload(\''+esc(name)+'\')">Unload</button> '
    +'<button class="btn sm ghost" onclick="openHistory(\''+esc(name)+'\')">View context ▾</button> '
    +'<button class="btn sm ghost" onclick="reconf(\''+esc(name)+'\')">Reconfigure…</button>';
  else if(_isT2i) acts='<button class="btn sm pri" onclick="t2iLoadDlg(\''+esc(name)+'\')">Load 🖼</button> '
    +'<button class="btn sm ghost" onclick="forget(\''+esc(name)+'\')">Forget</button> '
    +'<button class="btn sm ghost" onclick="del(\''+esc(name)+'\')">Delete</button>';
  else if(_isT2a) acts='<button class="btn sm pri" onclick="t2aLoadDlg(\''+esc(name)+'\')">Load 🎵</button> '
    +'<button class="btn sm ghost" onclick="forget(\''+esc(name)+'\')">Forget</button> '
    +'<button class="btn sm ghost" onclick="del(\''+esc(name)+'\')">Delete</button>';
  else acts='<button class="btn sm pri" onclick="closeOv();openLoad(\''+esc(name)+'\')">Load…</button> '
    +'<button class="btn sm ghost" onclick="forget(\''+esc(name)+'\')">Forget</button> '
    +'<button class="btn sm ghost" onclick="del(\''+esc(name)+'\')">Delete</button>';
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>'+esc(name)+'</h3>'
    +'<div id="dlive">'+detailLive(name)+'</div>'+rt+t2ig+t2ag+tts+pre+pers
    +'<div id="mi_more"><div class="empty">loading full model info…</div></div>'
    +'<div style="margin-top:16px">'+acts+'</div>';
  $('#ov').classList.add('show');
  try{ const sh=await api('/api/show',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model:name})});
       const el=document.getElementById('mi_more'); if(el&&DETAIL_OPEN===name) el.innerHTML='<h3 style="font-size:13px;margin-top:14px">Model info</h3>'+renderShow(sh);
  }catch(e){ const el=document.getElementById('mi_more'); if(el) el.innerHTML='<div class="note">full model info unavailable: '+esc(String(e.message||e))+'</div>'; }
}
// #runtime-config: push the detail modal's runtime-settings inputs to the controller. The UI always
// sends EVERY field; an EMPTY input clears that default back to unset. Applies to all replicas.
async function applyRt(name){
  const F={temperature:'rt-temp',min_p:'rt-minp',top_p:'rt-topp',top_k:'rt-topk',
           repeat_penalty:'rt-rp',repeat_last_n:'rt-rln',presence_penalty:'rt-pp',
           frequency_penalty:'rt-fp',seed:'rt-seed',num_predict:'rt-np'};
  const q=new URLSearchParams({model:name});
  for(const k in F){ const el=$('#'+F[k]); q.set(k, el?el.value:''); }
  try{ const r=await api('/model_config?'+q.toString(),{method:'POST'});
       const d=r.defaults||{}, ks=Object.keys(d);
       toast('runtime settings applied · '+(ks.length?ks.map(k=>k+'='+d[k]).join(' · '):'all unset'));
       tick(); }
  catch(e){ toast(String(e.message||e),1); }
}
// #t2i-serve: load an image model (no dialog — the load path picks the gate-tested recipe
// itself). /load blocks for the whole multi-minute build, so fire it WITHOUT awaiting: the
// loading card appears via the status poll, exactly like the LLM load dialog's pattern.
// #t2i-load-dlg: the Load 🖼 CONFIRM dialog — a t2i load is heavyweight (needs ~14-17 GB on the
// controller-co-located GPU and auto-evicts idle residents to get it), so it deserves the same
// deliberate step other loads get, and a FAILURE must be a dialog the user actually sees (the
// old fire-and-forget toast was missed live: the load failed "nothing evictable" invisibly).
// #t2i-load-dlg v2: a REAL form (placement + precision), not two opaque buttons. Every
// option maps to a supported /load param — quant (int4 mixed-edge | none=bf16) and
// t2i_offload — so nothing is "magic". t2i v1 runs the WHOLE pipeline on one
// controller-co-located GPU (no fleet split), so there is no distributed/TP option here;
// the note says so explicitly rather than leaving it a mysterious omission.
function t2iLoadDlg(name){
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Load 🖼 '+esc(name)+'</h3>'
    +'<div class="note" style="margin-top:8px">The image pipeline (v1) runs <b>whole on one GPU that shares the controller box</b> — it is not split across the fleet like an LLM, so there is no distributed / tensor-parallel option. Text encoder runs on that box\'s CPU, VAE decodes tiled. Choose how the DiT weights sit on the GPU:</div>'
    +'<label style="margin-top:10px">Placement</label>'
    +'<select id="t-place" onchange="_t2iPlaceChg()">'
    +'<option value="auto">Auto — GPU int4, spill to CPU if it won\'t fit (recommended)</option>'
    +'<option value="gpu">GPU-resident — weights stay on the GPU</option>'
    +'<option value="offload">CPU offload — weights in RAM, streamed to the GPU</option>'
    +'</select>'
    +'<div id="t-precwrap" style="display:none;margin-top:8px"><label>Precision</label>'
    +'<select id="t-prec">'
    +'<option value="int4">int4 mixed-edge (~13.5 GB — first+last blocks bf16 ≈ bf16 quality)</option>'
    +'<option value="none">bf16 full (~41 GB — needs a big-VRAM / unified-memory card)</option>'
    +'</select></div>'
    +'<div id="t-note" class="note" style="margin-top:10px"></div>'
    +'<div style="margin-top:14px"><button class="btn sm pri" onclick="loadT2i(\''+esc(name)+'\')">Load 🖼</button> '
    +'<button class="btn sm ghost" onclick="closeOv()">Cancel</button></div>';
  $('#ov').classList.add('show');
  _t2iPlaceChg();
}
// Show the precision picker only for a GPU-resident load, and explain the current choice.
function _t2iPlaceChg(){
  var v=$('#t-place')?$('#t-place').value:'auto';
  var pw=$('#t-precwrap'); if(pw) pw.style.display=(v==='gpu')?'block':'none';
  var n=$('#t-note'); if(!n) return;
  if(v==='auto') n.innerHTML='Tries <b>int4 on the GPU</b> (~14–17 GB free VRAM; idle models auto-evict, active ones never do). If it can\'t fit there, it <b>automatically spills to CPU offload</b> — DiT bf16 in RAM (~47 GB), only ~4 GB VRAM, nothing evicted. Fast when there is room, always loads.';
  else if(v==='gpu') n.innerHTML='Weights live on the GPU — fastest renders. <b>int4</b> needs ~14–17 GB free VRAM; <b>bf16</b> needs ~41 GB (om3nbox-class unified memory). Idle residents auto-evict to make room; a model actively serving is never evicted — if it holds the card the load fails with the reason shown.';
  else n.innerHTML='DiT rests <b>bf16 in system RAM</b> (~47 GB) and blocks stream to the GPU each step — needs only <b>~4 GB VRAM</b>, <b>NEVER evicts</b> anything, ~2–3× slower per step. The safe choice on a tight or busy card (e.g. beast\'s 16 GB).';
}
// #t2a-serve: the Load 🎵 dialog. Offload is RECOMMENDED — the DiT rests in RAM and streams to the
// GPU per render, so a multi-minute clip's activations don't OOM a tight card beside other residents
// (GPU-resident holds ~8 GB and long clips can OOM on a 16 GB card).
function t2aLoadDlg(name){
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Load 🎵 '+esc(name)+'</h3>'
    +'<div class="note" style="margin-top:8px">Loads the ACE-Step music pipeline (DiT + VAE + text encoder, ~8 GB bf16) onto the GPU sharing the controller box. Then open its card to generate music.</div>'
    +'<div class="note" style="margin-top:8px"><b>Offloaded (recommended)</b>: components rest in system RAM and the DiT streams to the GPU per render — ~0 GB resident, leaves headroom for long-clip activations, never evicts residents.</div>'
    +'<div class="note" style="margin-top:8px"><b>GPU-resident</b>: the whole pipeline stays on the GPU (~8 GB) — quicker to start each render, but long (multi-minute) clips can OOM on a 16 GB card beside other residents.</div>'
    +'<div style="margin-top:14px"><button class="btn sm pri" onclick="closeOv();loadT2a(\''+esc(name)+'\',1)">Load 🎵 offloaded</button> '
    +'<button class="btn sm" onclick="closeOv();loadT2a(\''+esc(name)+'\')">Load 🎵 GPU-resident</button> '
    +'<button class="btn sm ghost" onclick="closeOv()">Cancel</button></div>';
  $('#ov').classList.add('show');
}
function loadT2a(name,off){
  toast('loading music pipeline for '+name+(off?' (offloaded — components in RAM)':'')+' — up to a minute…');
  api('/load?model='+encodeURIComponent(name)+(off?'&t2i_offload=1':''),{method:'POST'})
    .then(()=>{toast(name+' ready — open its card to generate');tick();})
    .catch(e=>errDlg('music pipeline load failed',String(e.message||e)));
  tick();
}
function errDlg(title,msg){
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>'+esc(title)+'</h3>'
    +'<div class="note" style="margin-top:8px">'+esc(msg)+'</div>'
    +'<div style="margin-top:12px"><button class="btn sm" onclick="closeOv()">OK</button></div>';
  $('#ov').classList.add('show');
}
// Fire one t2i load with an explicit query string, surfacing failures in a dialog the user sees.
function _t2iFire(name,qs,label){
  toast('loading image pipeline for '+name+(label||'')+' — a few minutes…');
  api('/load?model='+encodeURIComponent(name)+qs,{method:'POST'})
    .then(()=>{toast(name+' ready — open its card to generate');tick();})
    .catch(e=>errDlg('image pipeline load failed',String(e.message||e)));
  tick();
}
// Read the load dialog and dispatch. Auto = GPU int4, and on a capacity/eviction failure
// (nothing evictable / not enough co-located VRAM) AUTO-SPILL to CPU offload — the
// "GPU by default, spill to CPU as needed" behavior, done transparently with a toast.
function loadT2i(name){
  var place=$('#t-place')?$('#t-place').value:'auto';
  var prec=$('#t-prec')?$('#t-prec').value:'int4';
  closeOv();
  if(place==='offload'){ _t2iFire(name,'&t2i_offload=1',' (CPU offload — DiT in RAM)'); return; }
  if(place==='gpu'){ _t2iFire(name,'&quant='+encodeURIComponent(prec),
      prec==='none'?' (GPU-resident bf16)':' (GPU-resident int4)'); return; }
  toast('loading '+name+' — GPU int4, will spill to CPU offload if it won\'t fit…');
  api('/load?model='+encodeURIComponent(name)+'&quant=int4',{method:'POST'})
    .then(()=>{toast(name+' ready on GPU (int4)');tick();})
    .catch(e=>{ var msg=String(e.message||e);
      if(/vram|evict|co-located gpu|no room|nothing/i.test(msg.toLowerCase())){
        toast('GPU full — spilling to CPU offload…');
        api('/load?model='+encodeURIComponent(name)+'&t2i_offload=1',{method:'POST'})
          .then(()=>{toast(name+' ready (CPU offload — spilled)');tick();})
          .catch(e2=>errDlg('image pipeline load failed',String(e2.message||e2)));
      } else { errDlg('image pipeline load failed',msg); }
      tick();
    });
  tick();
}
// #t2i-serve: render from the detail modal. Long await (a render is minutes) with a live
// elapsed timer; while it runs the card's status poll shows the worker's step i/n too.
async function t2iGen(name){
  const p=($('#t2i-prompt').value||'').trim(); if(!p){toast('type a prompt',1);return;}
  const sz=($('#t2i-size').value||'1024x1024');
  const steps=parseInt($('#t2i-steps').value||'20')||20;
  const seed=($('#t2i-seed').value||'').trim();
  const btn=$('#t2i-go'), st=$('#t2i-status'), out=$('#t2i-out');
  btn.disabled=true; const t0=Date.now();
  const tick_=setInterval(()=>{ if(st){ const m=(LAST.models||[]).find(x=>x.name===name)||{};
    st.textContent='rendering… '+Math.round((Date.now()-t0)/1000)+'s'
      +(m.t2i_step?(' · step '+m.t2i_step+'/'+(m.t2i_total||'?')):''); } },1000);
  try{
    const body={model:name,prompt:p,size:sz,steps:steps};
    if(seed!=='')body.seed=parseInt(seed);
    const r=await fetch('/v1/images/generations',{method:'POST',
      headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const j=await r.json();
    if(!r.ok) throw new Error((j.error&&j.error.message)||('HTTP '+r.status));
    const b64=j.data&&j.data[0]&&j.data[0].b64_json; if(!b64) throw new Error('empty response');
    const src='data:image/png;base64,'+b64;
    out.innerHTML='<img src="'+src+'" style="max-width:100%;border-radius:10px;border:1px solid var(--border)">'
      +'<div><a href="'+src+'" download="'+esc(name)+'_'+Date.now()+'.png" '
      +'style="font-size:11px;color:var(--accent)">⬇ download png</a>'
      +'<span class="em" style="font-size:11px;margin-left:8px">'
      +Math.round((Date.now()-t0)/1000)+'s · '+esc(sz)+' · '+steps+' steps</span></div>';
    st.textContent='';
  }catch(e){ st.textContent=''; toast(String(e.message||e),1); }
  finally{ clearInterval(tick_); btn.disabled=false; }
}
// #t2a-serve: render music from the detail modal. Raw fetch (NOT api()) — /v1/audio/music returns
// binary WAV; read it as a blob, play it, and offer a working download (object URL revoked on the
// next run so repeats don\'t leak). Errors (503 load/capacity, 400 no prompt) arrive as a JSON body
// even on this binary endpoint, so decode + surface them. A live step i/n rides the status poll.
async function t2aGen(name){
  const p=($('#t2a-prompt').value||'').trim(); if(!p){ toast('enter style / genre tags',1); return; }
  const lyrics=($('#t2a-lyrics').value||'').trim();
  const durS=parseInt($('#t2a-dur').value||'30')||30;
  const steps=parseInt($('#t2a-steps').value||'60')||60;
  const guid=parseFloat($('#t2a-guid').value||'12'); const seed=($('#t2a-seed').value||'').trim();
  const btn=$('#t2a-go'), st=$('#t2a-status'), au=$('#t2a-audio'), dl=$('#t2a-dl');
  btn.disabled=true; const t0=Date.now();
  const tick_=setInterval(()=>{ if(st){ const m=(LAST.models||[]).find(x=>x.name===name)||{};
    st.textContent='rendering '+durS+'s… '+Math.round((Date.now()-t0)/1000)+'s'
      +(m.t2a_step?(' · step '+m.t2a_step+'/'+(m.t2a_total||'?')):''); } },1000);
  try{
    const body={model:name,prompt:p,duration:durS,steps:steps,guidance:guid,response_format:'wav'};
    if(lyrics!=='')body.lyrics=lyrics;
    if(seed!=='')body.seed=parseInt(seed);
    const r=await fetch('/v1/audio/music',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    if(!r.ok){ let msg='HTTP '+r.status; try{ const j=await r.json(); msg=(j.error&&j.error.message)||j.error||msg; }catch(e){} throw new Error(msg); }
    const blob=await r.blob();
    if(au._url)URL.revokeObjectURL(au._url);
    const url=URL.createObjectURL(blob); au._url=url;
    au.src=url; au.style.display='block'; au.play().catch(()=>{});
    dl.href=url; dl.download=name.replace(/[^a-z0-9]+/gi,'_')+'_'+durS+'s_'+Date.now()+'.wav'; dl.style.display='inline';
    st.textContent='done · '+durS+'s · '+(blob.size/1e6).toFixed(1)+' MB · rendered in '+((Date.now()-t0)/1000).toFixed(1)+'s';
  }catch(e){ st.textContent=''; toast(String(e.message||e),1); }
  finally{ clearInterval(tick_); btn.disabled=false; }
}
// #tts-panel: synthesize speech from the detail modal. Raw fetch (NOT api(), which parses JSON) —
// /v1/audio/speech returns binary audio; read it as a blob, play it, and offer a download. Object
// URLs are revoked on the next synth so repeated runs don't leak. Errors (503 load/capacity, 400
// empty input) come back as a JSON body even on this binary endpoint, so decode + surface them.
async function ttsSpeak(name){
  const t=$('#tts-text'), st=$('#tts-status'), au=$('#tts-audio'), dl=$('#tts-dl');
  const text=(t&&t.value||'').trim(), voice=($('#tts-voice')||{}).value||'Chelsie';
  if(!text){ if(st)st.textContent='enter some text first'; return; }
  if(st)st.textContent='synthesizing…'; const t0=Date.now();
  try{
    const r=await fetch('/v1/audio/speech',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:name,input:text,voice:voice,response_format:'wav'})});
    if(!r.ok){ let msg='HTTP '+r.status; try{ const j=await r.json(); msg=(j.error&&j.error.message)||j.error||msg; }catch(e){} throw new Error(msg); }
    const blob=await r.blob();
    if(au._url)URL.revokeObjectURL(au._url);
    const url=URL.createObjectURL(blob); au._url=url;
    au.src=url; au.style.display='block'; au.play().catch(()=>{});
    dl.href=url; dl.download=name.replace(/[^a-z0-9]+/gi,'_')+'.wav'; dl.style.display='inline';
    if(st)st.textContent='done · '+Math.round(blob.size/1024)+' KB · '+((Date.now()-t0)/1000).toFixed(1)+'s';
  }catch(e){ if(st)st.textContent='error: '+(e.message||e); }
}
// Context viewer: one scrollable popup showing the IN/OUT flow of recent requests (newest first).
// History is kept only while the model is resident (cleared on unload).
async function openHistory(name){
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Context · '+esc(name)+'</h3><div class="empty">loading…</div>';
  $('#ov').classList.add('show');
  let d; try{ d=await api('/history?model='+encodeURIComponent(name)+'&dir=both'); }
  catch(e){ $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Context · '+esc(name)+'</h3><div class="note">'+esc(String(e.message||e))+'</div>'; return; }
  const es=d.entries||[];
  const head='<span class="x" onclick="closeOv()">×</span><h3>Context · '+esc(name)+'</h3>'
    +'<div style="font-size:12px;color:var(--dim);margin-bottom:8px">'+es.length+' of '+(d.count||0)+' captured · lifetime '+(d.tok_in_total||0)+' in / '+(d.tok_out_total||0)+' out tokens · newest first '
    +'<button class="btn sm ghost" style="float:right" onclick="openDetail(\''+esc(name)+'\')">← back</button></div>';
  if(!es.length){ $('#modal').innerHTML=head+'<div class="empty">no requests captured yet — generate something, then reopen</div>'; return; }
  const blk=es.map(e=>{
    const t=e.ts?new Date(e.ts*1000).toLocaleTimeString():'';
    return '<div style="border:1px solid var(--border);border-radius:8px;padding:8px;margin-bottom:8px">'
      +'<div style="font-size:11px;color:var(--dim);margin-bottom:4px">'+t+' · '+(e.tok_in||0)+' in → '+(e.tok_out||0)+' out</div>'
      +'<div style="font-size:11px;color:var(--accent)">IN</div><pre style="white-space:pre-wrap;word-break:break-word;margin:2px 0;padding:6px;background:var(--bg);border-radius:6px;font-size:11px;max-height:220px;overflow:auto">'+esc(e.input||'')+'</pre>'
      +'<div style="font-size:11px;color:var(--good);margin-top:4px">OUT</div><pre style="white-space:pre-wrap;word-break:break-word;margin:2px 0;padding:6px;background:var(--bg);border-radius:6px;font-size:11px;max-height:220px;overflow:auto">'+esc(e.output||'')+'</pre></div>';
  }).join('');
  $('#modal').innerHTML=head+'<div style="max-height:60vh;overflow:auto">'+blk+'</div>';
}
async function compileShards(name,quant){ closeOv(); toast('compiling '+quant+' cache for '+name+'…'); tick(); try{ await api('/compile_shards?model='+encodeURIComponent(name)+'&quant='+quant,{method:'POST'}); toast(quant+' cache for '+name+' done'); }catch(e){ toast(String(e.message||e),1);} tick(); }
async function forget(name){ if(!confirm('Forget '+name+'? (keeps weight files)'))return; closeOv(); try{ await api('/forget?model='+encodeURIComponent(name),{method:'POST'}); toast('forgot '+name); }catch(e){ toast(String(e.message||e),1);} tick(); }
async function del(name){ if(!confirm('DELETE '+name+' and its weight files?'))return; closeOv(); toast('deleting '+name+'…'); try{ await api('/delete?model='+encodeURIComponent(name),{method:'POST'}); toast('deleted '+name); }catch(e){ toast(String(e.message||e),1);} tick(); }
async function reconf(name){ const tp=prompt('Reconfigure '+name+' — tp size (1=pipeline):','1'); if(tp==null)return;
  closeOv(); toast('reconfiguring '+name+'…'); try{ await api('/reconfigure?model='+encodeURIComponent(name)+'&tp='+encodeURIComponent(tp),{method:'POST'}); toast('reconfigured '+name); }catch(e){ toast(String(e.message||e),1);} tick(); }

tick(); setInterval(tick,2000);
</script>
</body></html>
"""


CONFIG_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniteModel — Config</title>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--surface2:#1c2230;--border:#2a3038;--border2:#3a424d;
    --text:#e6edf3;--muted:#9aa7b4;--dim:#6e7b89;--accent:#4f8cff;--good:#2ea043;--warn:#d29922;--bad:#da3633;
    --radius:10px;--mono:ui-monospace,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  .wrap{max-width:none;margin:0 auto;padding:18px 24px 60px}
  header{display:flex;align-items:center;gap:14px;margin-bottom:18px;flex-wrap:wrap}
  .brand{font-size:20px;font-weight:600} .ctl{font-size:12px;color:var(--dim);font-family:var(--mono)}
  nav{display:flex;gap:4px;margin-left:8px} nav a{font-size:13px;color:var(--muted);padding:5px 11px;border-radius:8px;border:1px solid transparent}
  nav a.on{color:var(--text);background:var(--surface);border-color:var(--border)} nav a:hover{background:var(--surface)}
  .grow{flex:1}
  .btn{background:var(--surface);border:1px solid var(--border2);color:var(--text);border-radius:8px;padding:7px 13px;font-size:13px;cursor:pointer}
  .btn:hover{border-color:var(--accent)} .btn:active{transform:scale(.98)} .btn.pri{border-color:var(--accent);color:#cfe0ff}
  .btn.sm{padding:4px 9px;font-size:12px} .btn.danger{border-color:#5a2a2a;color:#ff9a9a} .btn.danger:hover{border-color:var(--bad)}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:16px 18px;margin-bottom:16px}
  .card h2{font-size:15px;margin:0 0 4px} .card .sub{font-size:12px;color:var(--dim);margin-bottom:12px}
  .frm{display:grid;grid-template-columns:1fr 1fr;gap:13px 22px}
  .fld{display:flex;flex-direction:column;gap:4px} .fld label{font-size:12px;color:var(--muted)}
  .fld[title]{cursor:help} .fld[title]>label{width:max-content;max-width:100%;border-bottom:1px dotted var(--border2)}
  .fld input,.fld select{background:var(--bg);border:1px solid var(--border2);color:var(--text);border-radius:8px;padding:7px 10px;font-size:13px}
  .tog{display:flex;align-items:center;gap:8px;font-size:13px} .tog input{width:16px;height:16px}
  .nrow{display:flex;align-items:center;gap:14px;padding:8px 0;border-bottom:1px solid var(--border);font-size:13px}
  .nrow:last-child{border-bottom:none} .nrow .nn{min-width:150px;font-weight:600}
  .nrow .nn small{font-weight:400;color:var(--dim);font-size:11px} .nrow .grow{flex:1}
  .saved{color:var(--good);font-size:12px;margin-left:10px} .err{color:var(--bad);font-size:12px;margin-left:10px}
  .actrow{display:flex;gap:10px;flex-wrap:wrap;align-items:center}
</style></head>
<body><div class="wrap">
<header>
  <span class="brand">∞ InfiniteModel</span><span class="ctl" id="ctl">…</span>
  <nav><a href="/">Models</a><a href="/chat">Chat</a><a class="on" href="/config">Config</a><a href="/logs-page">Logs</a><a href="/bandwidth">Bandwidth</a></nav>
</header>

<div class="card">
  <h2>Engine settings</h2><div class="sub">Runtime config, persisted across restarts.</div>
  <div class="frm">
    <div class="fld" title="Cap on how many models can be resident at once. Loading one more first evicts the least-recently-used IDLE model (if LRU auto-unload is on); with no evictable victim the load fails. Pinned (do-not-auto-unload) models are never evicted."><label>Max concurrent loaded models</label><input id="max_loaded" type="number" min="1"></div>
    <div class="fld" title="How many requests may wait in line per model — generations for one model run one at a time. When the queue is full, further requests get a retryable 429/503 (honest backpressure) instead of piling up behind a long generation."><label>Per-model queue depth</label><input id="queue_depth" type="number" min="1"></div>
    <div class="fld" title="Quantization tier used when a request or dashboard click auto-loads a model. int4 is the recommended default; if a quantized auto-load fails, one bf16 retry is attempted. Hover each option in the list for details."><label>Auto-load default quant</label><select id="autoload_quant"><option title="Fused 4-bit: ~4x smaller than bf16, near-lossless, fastest fused GPU kernels — the recommended default">int4</option><option title="Experimental 2-bit (~2.5 bits/weight, dense models only — a MoE auto-downgrades to int4). Current round-to-nearest packing collapses output quality; awaits a calibrated (GPTQ-class) packer">int2</option><option title="8-bit: 2x smaller than bf16, effectively lossless. No 8-bit MoE-expert packer — a MoE auto-downgrades to int4">int8</option><option value="none" title="Full bf16 weights — exact but largest (~2 bytes per parameter); needs the most VRAM/RAM">none (bf16)</option></select></div>
    <div class="fld" title="Context window (tokens) sized into auto/click loads. The per-layer KV-cache memory reserve grows with ctx, so bigger windows make placement fatter. 0 = the model's native maximum (can be huge — 128k+ on many models)."><label>Auto-load default ctx (0 = native)</label><input id="autoload_ctx" type="number" min="0"></div>
    <div class="fld" title="Placement mode for auto-loads. auto = GPU-VRAM-first on the fewest nodes — best latency, the recommended default. If models seem to randomly load slow into system RAM, check this is not set to single. Hover each option in the list for details."><label>Auto-load placement mode</label><select id="autoload_mode"><option title="GPU-VRAM-first, fewest nodes — best latency (recommended)">auto</option><option title="A stage on EVERY enabled GPU, nothing on CPU — fails if the fleet's total VRAM cannot hold the model">all-gpu</option><option title="Spread stages across the whole fleet, CPUs and GPUs alike">distribute</option><option title="Layers across every capable node proportional to its capacity — for huge MoE models too big for the GPU-first subset">proportional</option><option title="Fewest nodes counting RAM+VRAM together — collapses to one box if it fits; RAM-first, usually the slow choice for big models">single</option></select></div>
    <div class="fld" title="Prefill stall-watchdog: reclaim a generation that has produced NO tokens and reports no per-layer forward progress for this many seconds — its slot, queue and per-model lock reset so the next request re-flows the pipeline. Slow-but-ADVANCING prefills are never reclaimed (workers report progress over their heartbeat). Repeated reclaims of one model trip the wedge quarantine, which forces a fresh automatic re-place. 0 = watchdog off."><label>Prefill stall-watchdog (s, 0=off)</label><input id="gen_stall_s" type="number" min="0"></div>
    <div class="fld" title="Mid-decode stall-watchdog: once at least one token has been produced, reclaim if no NEW token arrives for this many seconds. Tighter than the prefill threshold because a healthy decode ticks every second or two. 0 = off."><label>Decode stall-watchdog (s, 0=off)</label><input id="gen_stall_decode_s" type="number" min="0"></div>
    <div class="fld" title="Unload any model that has served no requests for this many minutes. 0 or -1 = keep every model loaded forever (the default; -1 is the Ollama-style spelling and saves as -1). Pinned (📌) models and models with an active or queued request are never idle-unloaded."><label>Idle unload (min, 0/-1 = keep forever)</label><input id="idle_unload_m" type="number" min="-1" step="any" list="dl-idleu"><datalist id="dl-idleu"><option value="-1"></option><option value="0"></option><option value="5"></option><option value="15"></option><option value="60"></option><option value="240"></option><option value="1440"></option></datalist></div>
    <div class="fld" title="On controller startup, wait at least this many seconds — AFTER the worker fleet has settled — before auto-reloading persisted (autoload-on-restart) models, so API clients have time to (re)connect before the box gets busy streaming weights. 0 = no extra wait."><label>Autostart delay (s, client-connect grace)</label><input id="autostart_delay_s" type="number" min="0" step="any"></div>
    <div class="fld tog" title="When a new load needs room (or the max-loaded cap is hit), automatically evict the least-recently-used IDLE model. Busy models and pinned (do-not-auto-unload) models are never victims — if no evictable victim exists, the load fails instead. Off = nothing is ever evicted to make space."><input type="checkbox" id="auto_unload"><label for="auto_unload">LRU auto-unload</label></div>
    <div class="fld tog" title="A request naming a model that is not resident triggers an automatic load using the auto-load defaults above, then serves the request (Ollama-style). Off = requests for non-resident models fail immediately."><input type="checkbox" id="auto_load"><label for="auto_load">Auto-load on request</label></div>
    <div class="fld tog" title="Budget a new model's weights against each GPU's LIVE physically-free VRAM (committed-aware) rather than its nominal capacity — protects resident models' full-context KV reserves from being overcommitted. Leave on unless debugging placement."><input type="checkbox" id="vram_weights_first"><label for="vram_weights_first">Budget weights vs physical-free VRAM</label></div>
    <div class="fld tog" title="Checks every ~60s (and right after an idle-unload frees VRAM) whether the hottest resident model running split across GPU+RAM would now fit entirely in VRAM, and if so promotes it. Only while that model is momentarily idle, it engages a barrier (new requests wait) and re-places it VRAM-only, then resumes — the client connection just pauses briefly, no reconnect. A busy model is skipped (caught at a gap by a later sweep) rather than stalled; embeddings and models that can't fit GPU are skipped. Lets models auto-load hybrid under memory pressure and migrate to full-GPU speed as the busy one out-competes quieter models for VRAM."><input type="checkbox" id="juggler"><label for="juggler">Juggler (promote hybrid→VRAM on free)</label></div>
  </div>
  <div style="margin-top:14px"><button class="btn pri" onclick="save()" title="Apply and persist every engine setting above — takes effect immediately and survives controller restarts.">Save settings</button><span id="cfg-msg"></span></div>
</div>

<div class="card">
  <h2>Nodes — compute tiers</h2><div class="sub">Enable/disable each node's CPU/RAM and GPU/VRAM in the placement pool. Changes re-plan affected models.</div>
  <div class="actrow" style="margin-bottom:8px">
    <span style="font-size:12px;color:var(--muted)">Bulk:</span>
    <button class="btn sm" onclick="bulk('ram',true)" title="Offer every node's CPU + system RAM to the placement pool">All RAM on</button><button class="btn sm" onclick="bulk('ram',false)" title="Remove every node's CPU/RAM from the placement pool — models placed there re-plan onto what remains">All RAM off</button>
    <button class="btn sm" onclick="bulk('vram',true)" title="Offer every GPU's VRAM to the placement pool">All VRAM on</button><button class="btn sm" onclick="bulk('vram',false)" title="Remove every GPU from the placement pool — models placed there re-plan onto what remains">All VRAM off</button>
    <span id="node-msg"></span>
  </div>
  <div id="nodes"></div>
</div>

<div class="card">
  <h2>Controller</h2><div class="sub">Fleet-level operations. Use with care.</div>
  <div class="actrow">
    <button class="btn" onclick="ctlAct('/restart?workers=0','Restart the CONTROLLER only?')" title="Restart the controller process only. Workers stay up but drop their shards when the control link drops — persisted (autoload-on-restart) models re-stream after startup + the autostart delay; everything else re-auto-loads on demand.">Restart controller</button>
    <button class="btn" onclick="ctlAct('/restart?workers=1','Restart the WHOLE fleet (controller + all workers)?')" title="Restart the controller AND every worker process — the full reset that clears stale worker state, wedged loads and allocator-held VRAM. Required after a code deploy that touches worker files (a plain update does NOT restart worker processes).">Restart fleet</button>
    <button class="btn" onclick="ctlAct('/update?workers=1','Pull latest code from GitHub and restart the fleet?')" title="Pull the latest code from GitHub onto every node, then restart the whole fleet on it. Raw-CDN propagation can lag a push by minutes per file — if a just-pushed change is missing, wait a little and update again.">Update + deploy</button>
    <button class="btn" onclick="ctlAct('/gc_cache','Reclaim disk by removing redundant HF-cache copies?')" title="Delete the HuggingFace-cache copy of every model that is also complete in models/ (pure duplicates). Models existing only in the cache are kept — this never deletes the only copy.">GC disk cache</button>
    <span id="ctl-msg"></span>
  </div>
</div>
</div>

<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const _up=s=>{s=Math.max(0,Math.floor(s||0));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?d+'d '+h+'h':(h?h+'h '+m+'m':m+'m')};
const FIELDS=['max_loaded','queue_depth','autoload_quant','autoload_ctx','autoload_mode','gen_stall_s','gen_stall_decode_s','idle_unload_m','autostart_delay_s'];
const TOGS=['auto_unload','auto_load','vram_weights_first','juggler'];
// #cfg-dirty: fields the user touched since the last successful save. The 5s /status poll
// used to overwrite EVERY field wholesale, silently reverting an in-progress edit before
// the user could click Save. A dirty field is never overwritten by the poll; Save clears
// the set so server values show through again.
const DIRTY=new Set();
let NODES=[];
async function load(){
  let d; try{ d=await (await fetch('/status')).json(); }catch(e){ $('#ctl').innerHTML='<span style="color:var(--bad)">controller unreachable</span>'; return; }
  const c=d.controller||{}; $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?')+(c.code_date?' ('+c.code_date+')':'')+(c.uptime_s!=null?' · up '+_up(c.uptime_s):'');
  FIELDS.forEach(k=>{ const el=$('#'+k); if(el&&c[k]!=null&&!DIRTY.has(k)&&el!==document.activeElement)el.value=c[k]; });
  TOGS.forEach(k=>{ const el=$('#'+k); if(el&&!DIRTY.has(k))el.checked=!!c[k]; });
  NODES=d.nodes||[]; renderNodes();
}
function renderNodes(){
  $('#nodes').innerHTML=NODES.slice().sort((a,b)=>(b.has_gpu?1:0)-(a.has_gpu?1:0)||a.hostname.localeCompare(b.hostname)).map(n=>{
    const gpu=n.has_gpu;
    return '<div class="nrow"><div class="nn">'+esc(n.hostname)+' <small>'+(gpu?esc(n.device_name||'GPU'):((n.cores||'')+'c CPU'))+(n.alive?'':' · offline')+'</small></div>'
      +'<label class="tog" title="Include the CPU + system RAM of this node in the placement pool — toggling re-plans any model placed on it"><input type="checkbox" '+(n.ram_enabled?'checked':'')+' onchange="setTier(\''+esc(n.hostname)+'\',\'ram\',this.checked)">RAM</label>'
      +(gpu?'<label class="tog" title="Include the GPU VRAM of this node in the placement pool — toggling re-plans any model placed on it"><input type="checkbox" '+(n.vram_enabled?'checked':'')+' onchange="setTier(\''+esc(n.hostname)+'\',\'vram\',this.checked)">VRAM</label>':'<span style="color:var(--dim);font-size:12px" title="This node has no usable GPU — it contributes CPU/RAM only">no GPU</span>')
      +'<span class="grow"></span><span style="font-size:11px;color:var(--dim)" title="Live used / total memory of the primary tier on this node">'+(gpu?fmt(n.vram_used_gb)+'/'+fmt(n.vram_total_gb)+' GB VRAM':fmt((n.total_mem_gb||0)-(n.free_mem_gb||0))+'/'+fmt(n.total_mem_gb)+' GB RAM')+'</span></div>';
  }).join('')||'<div style="color:var(--dim);font-size:13px">no nodes</div>';
}
function fmt(v){return v==null?'—':Math.round(v*10)/10}
async function post(path){ const r=await fetch(path,{method:'POST'}); const t=await r.text(); let j;try{j=JSON.parse(t)}catch(e){j={ok:r.ok}} if(!r.ok||j.ok===false)throw new Error(j.error||('HTTP '+r.status)); return j; }
function msg(id,txt,bad){ $(id).innerHTML=(bad?'<span class="err">':'<span class="saved">')+esc(txt)+'</span>'; setTimeout(()=>{$(id).innerHTML='';},3500); }
async function save(){
  const q=new URLSearchParams();
  FIELDS.forEach(k=>{ const v=$('#'+k).value; if(v!=='')q.set(k,v); });
  TOGS.forEach(k=>{ q.set(k,$('#'+k).checked?'true':'false'); });
  try{ await post('/config?'+q.toString()); DIRTY.clear(); msg('#cfg-msg','saved'); load(); }catch(e){ msg('#cfg-msg',String(e.message||e),1); }
}
async function setTier(host,tier,on){ try{ await post('/nodeconfig?host='+encodeURIComponent(host)+'&'+tier+'='+(on?'true':'false')); msg('#node-msg',host+' '+tier+'='+on); }catch(e){ msg('#node-msg',String(e.message||e),1);} }
async function bulk(tier,on){ try{ await post('/nodeconfig_all?tier='+tier+'&enabled='+(on?'true':'false')); msg('#node-msg','all '+tier+'='+on); load(); }catch(e){ msg('#node-msg',String(e.message||e),1);} }
async function ctlAct(path,confirmMsg){ if(!confirm(confirmMsg))return; try{ await post(path); msg('#ctl-msg','sent — controller restarting if applicable'); }catch(e){ msg('#ctl-msg',String(e.message||e),1);} }
FIELDS.concat(TOGS).forEach(k=>{ const el=$('#'+k); if(el){ el.addEventListener('input',()=>DIRTY.add(k)); el.addEventListener('change',()=>DIRTY.add(k)); } });
load(); setInterval(load,5000);
</script>
</body></html>
"""

CHAT_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniteModel — Chat</title>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--surface2:#1c2230;--border:#2a3038;--border2:#3a424d;
    --text:#e6edf3;--muted:#9aa7b4;--dim:#6e7b89;--accent:#4f8cff;--good:#2ea043;--warn:#d29922;--bad:#da3633;
    --radius:10px;--mono:ui-monospace,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  .wrap{max-width:none;margin:0 auto;padding:18px 24px 30px}
  header{display:flex;align-items:center;gap:14px;margin-bottom:14px;flex-wrap:wrap}
  .brand{font-size:20px;font-weight:600} .ctl{font-size:12px;color:var(--dim);font-family:var(--mono)}
  nav{display:flex;gap:4px;margin-left:8px} nav a{font-size:13px;color:var(--muted);padding:5px 11px;border-radius:8px;border:1px solid transparent}
  nav a.on{color:var(--text);background:var(--surface);border-color:var(--border)} nav a:hover{background:var(--surface)}
  .bar{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
  select,textarea{background:var(--bg);border:1px solid var(--border2);color:var(--text);border-radius:8px;padding:8px 10px;font-size:14px;font-family:var(--sans)}
  .btn{background:var(--surface);border:1px solid var(--border2);color:var(--text);border-radius:8px;padding:8px 14px;font-size:14px;cursor:pointer}
  .btn:hover{border-color:var(--accent)} .btn.pri{border-color:var(--accent);color:#cfe0ff} .btn.sm{padding:4px 9px;font-size:12px}
  .btn:disabled{opacity:.5;cursor:default} .grow{flex:1}
  #log{border:1px solid var(--border);border-radius:12px;background:var(--surface);padding:12px;height:58vh;overflow:auto;margin-bottom:10px}
  .msg{margin-bottom:10px} .who{font-size:11px;margin-bottom:2px} .who.u{color:var(--accent)} .who.a{color:var(--good)}
  .bub{white-space:pre-wrap;word-break:break-word;background:var(--bg);border-radius:8px;padding:8px 10px;font-size:13px}
  .empty{color:var(--dim);text-align:center;padding:40px 0}
  .inrow{display:flex;gap:8px} .inrow textarea{flex:1;resize:vertical;min-height:46px}
  .hint{font-size:12px;color:var(--dim)}
</style></head>
<body><div class="wrap">
<header>
  <span class="brand">∞ InfiniteModel</span><span class="ctl" id="ctl">…</span>
  <nav><a href="/">Models</a><a class="on" href="/chat">Chat</a><a href="/config">Config</a><a href="/logs-page">Logs</a><a href="/bandwidth">Bandwidth</a></nav>
</header>
<div class="bar">
  <label class="hint">Model</label>
  <select id="model" onchange="switchModel()"></select>
  <span class="hint" id="mhint"></span>
  <span class="grow"></span>
  <button class="btn sm" onclick="clearChat()">Clear</button>
</div>
<div id="log"></div>
<div class="inrow">
  <textarea id="in" rows="2" placeholder="type a prompt — Enter to send, Shift+Enter for newline" onkeydown="key(event)"></textarea>
  <button class="btn pri" id="send" onclick="send()">Send</button>
</div>
<div class="hint" style="margin-top:6px">Throwaway test chat · streams live · leaving this tab or switching models ends the generation.</div>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const _up=s=>{s=Math.max(0,Math.floor(s||0));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?d+'d '+h+'h':(h?h+'h '+m+'m':m+'m')};
const qp=new URLSearchParams(location.search);
let MODELS=[], cur='', msgs=[], live='', busy=false, abort=null;
function aborter(){ if(abort){ try{abort.abort();}catch(e){} abort=null; } }
async function loadModels(){
  let d; try{ d=await (await fetch('/status',{cache:'no-store'})).json(); }
  catch(e){ $('#ctl').innerHTML='<span style="color:var(--bad)">controller unreachable</span>'; return; }
  const c=d.controller||{}; $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?')+(c.code_date?' ('+c.code_date+')':'')+(c.uptime_s!=null?' · up '+_up(c.uptime_s):'');
  MODELS=(d.models||[]).filter(m=>m.loaded).map(m=>m.name);
  const sel=$('#model'); const want=cur||qp.get('model')||MODELS[0]||'';
  // #cfg-dirty sibling: don't rebuild the option list while the dropdown is focused/open
  // (the 5s poll would snap the native popup shut mid-selection), and skip no-op rebuilds
  // when the loaded-model set hasn't changed.
  const mkey=MODELS.join('\u0001');
  if(sel!==document.activeElement&&sel.dataset.mkey!==mkey){
    sel.dataset.mkey=mkey;
    sel.innerHTML=MODELS.length
      ? MODELS.map(n=>'<option value="'+esc(n)+'"'+(n===want?' selected':'')+'>'+esc(n)+'</option>').join('')
      : '<option value="">(no models loaded)</option>';
  }
  if(MODELS.length){ cur=sel.value; $('#mhint').textContent=''; $('#send').disabled=false; $('#in').disabled=false; }
  else { cur=''; $('#mhint').innerHTML='no models loaded — <a href="/">load one on the Models tab</a>'; $('#send').disabled=true; $('#in').disabled=true; }
}
function switchModel(){ aborter(); cur=$('#model').value; msgs=[]; live=''; busy=false; render(); $('#in').focus(); }
function clearChat(){ aborter(); msgs=[]; live=''; busy=false; render(); }
function key(e){ if(e.key==='Enter'&&!e.shiftKey){ e.preventDefault(); send(); } }
function bub(role,text){ const u=role==='user';
  return '<div class="msg"><div class="who '+(u?'u':'a')+'">'+(u?'you':esc(cur))+'</div><div class="bub">'+esc(text||'')+'</div></div>'; }
function render(){ const l=$('#log'); if(!l)return;
  let h=msgs.map(m=>bub(m.role,m.content)).join(''); if(busy)h+=bub('assistant',live||'…');
  l.innerHTML=h||'<div class="empty">select a loaded model and send a prompt</div>'; l.scrollTop=l.scrollHeight; }
async function send(){
  if(busy||!cur)return; const ta=$('#in'); const t=ta.value.trim(); if(!t)return;
  ta.value=''; msgs.push({role:'user',content:t}); live=''; busy=true; $('#send').disabled=true; render();
  const ctrl=new AbortController(); abort=ctrl;
  try{
    const r=await fetch('/api/chat',{method:'POST',signal:ctrl.signal,headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:cur,messages:msgs,stream:true,options:{temperature:0.7,num_predict:512}})});
    if(!r.ok){ const tx=await r.text(); throw new Error(tx||('HTTP '+r.status)); }
    const rd=r.body.getReader(), dec=new TextDecoder(); let buf='';
    while(true){ const x=await rd.read(); if(x.done)break; buf+=dec.decode(x.value,{stream:true}); let nl;
      while((nl=buf.indexOf('\n'))>=0){ const ln=buf.slice(0,nl).trim(); buf=buf.slice(nl+1); if(!ln)continue;
        let j; try{j=JSON.parse(ln);}catch(e){continue;} if(j.error)live+='\n[error] '+j.error;
        const p=(j.message&&j.message.content)||j.response||''; if(p){ live+=p; render(); } } }
    msgs.push({role:'assistant',content:live||'(no output)'});
  }catch(e){ if(e.name!=='AbortError') msgs.push({role:'assistant',content:'[error] '+String(e.message||e)}); }
  finally{ busy=false; live=''; abort=null; $('#send').disabled=(!cur); render(); const i=$('#in'); if(i)i.focus(); }
}
window.addEventListener('beforeunload',aborter);
loadModels(); render(); setInterval(loadModels,5000);
</script>
</body></html>
"""

LOGS_HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>InfiniteModel — Logs</title>
<style>
  :root{--bg:#0d1117;--surface:#161b22;--surface2:#1c2230;--border:#2a3038;--border2:#3a424d;
    --text:#e6edf3;--muted:#9aa7b4;--dim:#6e7b89;--accent:#4f8cff;--good:#2ea043;--warn:#d29922;--bad:#da3633;
    --radius:10px;--mono:ui-monospace,Menlo,Consolas,monospace;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;}
  *{box-sizing:border-box} body{margin:0;background:var(--bg);color:var(--text);font-family:var(--sans);font-size:14px;line-height:1.5}
  a{color:var(--accent);text-decoration:none}
  .wrap{max-width:none;margin:0 auto;padding:18px 24px 40px}
  header{display:flex;align-items:center;gap:14px;margin-bottom:16px;flex-wrap:wrap}
  .brand{font-size:20px;font-weight:600} .ctl{font-size:12px;color:var(--dim);font-family:var(--mono)}
  nav{display:flex;gap:4px;margin-left:8px} nav a{font-size:13px;color:var(--muted);padding:5px 11px;border-radius:8px;border:1px solid transparent}
  nav a.on{color:var(--text);background:var(--surface);border-color:var(--border)} nav a:hover{background:var(--surface)}
  .grow{flex:1}
  .btn{background:var(--surface);border:1px solid var(--border2);color:var(--text);border-radius:8px;padding:6px 12px;font-size:13px;cursor:pointer}
  .btn:hover{border-color:var(--accent)} .btn.on{border-color:var(--accent);color:#cfe0ff}
  select.f{background:var(--surface);border:1px solid var(--border2);color:var(--text);border-radius:8px;padding:6px 10px;font-size:13px}
  .tog{display:inline-flex;align-items:center;gap:6px;font-size:13px;color:var(--muted)}
  .panes{display:grid;grid-template-columns:2.4fr 1fr;gap:14px;align-items:start}
  @media(max-width:860px){.panes{grid-template-columns:1fr}}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .card h2{font-size:14px;margin:0;padding:10px 14px;border-bottom:1px solid var(--border);color:var(--muted)}
  pre#log{margin:0;padding:12px 14px;font-family:var(--mono);font-size:12px;line-height:1.5;color:var(--text);
    white-space:pre-wrap;word-break:break-word;height:72vh;overflow:auto;background:#0a0e14}
  .act{padding:4px 0} .act div{padding:6px 14px;border-bottom:1px solid var(--border);font-size:12px;color:var(--muted)}
  .act div:last-child{border-bottom:none} .act .t{color:var(--dim);font-size:11px;margin-right:6px}
  .empty{padding:20px;color:var(--dim);font-size:13px;text-align:center}
</style></head>
<body><div class="wrap">
<header>
  <span class="brand">∞ InfiniteModel</span><span class="ctl" id="ctl">…</span>
  <nav><a href="/">Models</a><a href="/chat">Chat</a><a href="/config">Config</a><a class="on" href="/logs-page">Logs</a><a href="/bandwidth">Bandwidth</a></nav>
  <span class="grow"></span>
  <span class="tog">source <select class="f" id="src" onchange="refresh()"></select></span>
  <button class="btn on" id="auto" onclick="toggleAuto()">auto ⟳</button>
  <button class="btn" onclick="refresh()">refresh</button>
</header>
<div class="panes">
  <div class="card"><h2 id="logtitle">controller log</h2><pre id="log">loading…</pre></div>
  <div class="card"><h2>activity</h2><div class="act" id="act"></div></div>
</div>
<div class="card" style="margin-top:14px"><h2>HTTP errors · 4xx / 5xx returned to clients &amp; nodes</h2><div class="act" id="errs"></div></div>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const _up=s=>{s=Math.max(0,Math.floor(s||0));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?d+'d '+h+'h':(h?h+'h '+m+'m':m+'m')};
let AUTO=true, SRCS=false;
async function srcList(){
  try{ const d=await (await fetch('/status')).json();
    const c=d.controller||{}; $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?')+(c.code_date?' ('+c.code_date+')':'')+(c.uptime_s!=null?' · up '+_up(c.uptime_s):'');
    if(!SRCS){ const sel=$('#src'); const cur=sel.value;
      sel.innerHTML='<option value="">controller</option>'+(d.nodes||[]).map(n=>'<option value="'+esc(n.hostname)+'">'+esc(n.hostname)+(n.has_gpu?' (GPU)':'')+'</option>').join('');
      if(cur)sel.value=cur; SRCS=true; }
    renderAct(d.activity||[]);
    renderErrors(d.errors||[]);
  }catch(e){ $('#ctl').innerHTML='<span style="color:var(--bad)">controller unreachable</span>'; }
}
function renderErrors(es){
  if(!es.length){ $('#errs').innerHTML='<div class="empty">no HTTP errors recorded</div>'; return; }
  $('#errs').innerHTML=es.slice(0,80).map(e=>{
    const ts=e.t?new Date(e.t*1000).toLocaleTimeString():'';
    const col=(e.status>=500)?'var(--bad)':'var(--warn)';
    return '<div><span class="t">'+esc(ts)+'</span><b style="color:'+col+'">'+esc(e.status)+'</b> '
      +esc(e.method||'')+' <span style="font-family:var(--mono)">'+esc(e.path||'')+'</span>'
      +' <span style="color:var(--dim)">· '+esc(e.ip||'')+'</span>'
      +(e.detail?('<div style="color:var(--dim);font-size:11px;margin-top:2px">'+esc(e.detail)+'</div>'):'')+'</div>';
  }).join('');
}
function renderAct(a){
  if(!a.length){ $('#act').innerHTML='<div class="empty">no recent activity</div>'; return; }
  $('#act').innerHTML=a.slice(-60).reverse().map(e=>{
    const txt=typeof e==='string'?e:(e.msg||e.text||JSON.stringify(e)); const t=(e&&e.ts)?('<span class="t">'+esc(String(e.ts).slice(11,19))+'</span>'):'';
    return '<div>'+t+esc(txt)+'</div>';
  }).join('');
}
async function refresh(){
  const host=$('#src').value;
  $('#logtitle').textContent=host?('worker log · '+host):'controller log';
  try{
    const r=await fetch('/logs'+(host?('?node='+encodeURIComponent(host)):''));
    const t=await r.text();
    const el=$('#log'); const atBottom=el.scrollHeight-el.scrollTop-el.clientHeight<40;
    el.textContent=t||'(empty)';
    if(atBottom) el.scrollTop=el.scrollHeight;
  }catch(e){ $('#log').textContent='error fetching log: '+(e.message||e); }
}
function toggleAuto(){ AUTO=!AUTO; $('#auto').classList.toggle('on',AUTO); $('#auto').textContent=AUTO?'auto ⟳':'auto off'; }
async function tick(){ await srcList(); if(AUTO) refresh(); }
refresh(); tick(); setInterval(tick,3000);
</script>
</body></html>
"""
