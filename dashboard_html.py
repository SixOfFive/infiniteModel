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
  .wrap{max-width:1180px;margin:0 auto;padding:18px 20px 60px;}
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
  .node{display:flex;align-items:center;gap:12px;padding:9px 15px;border-bottom:1px solid var(--border);font-size:13px}
  .node:last-child{border-bottom:none}
  .node .nn{min-width:150px;font-weight:600}
  .node .nn small{font-weight:400;color:var(--dim);font-size:11px}
  .node .mb{flex:1;display:flex;align-items:center;gap:8px;max-width:420px}
  .node .mb .lab{font-size:11px;color:var(--muted);width:46px}
  .node .util{font-size:11px;color:var(--dim);width:120px;text-align:right}
  /* overlay/modal */
  .ov{position:fixed;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-start;justify-content:center;z-index:50}
  .ov.show{display:flex}
  .modal{background:var(--surface);border:1px solid var(--border2);border-radius:12px;max-width:640px;width:92%;
         margin:60px 0;padding:20px 22px;max-height:80vh;overflow:auto}
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
  $('#fleet').innerHTML=[
    tile('Nodes', (p.nodes||0)+' <small>· '+(comp.gpus||0)+' GPU</small>'),
    tile('Loaded', loaded+' <small>/ '+reg+' registered</small>'),
    tile('GPU pool', fmt(vU)+'<small> / '+fmt(vT)+' GB</small>', bar(vU,vT)),
    tile('RAM pool', fmt(rU)+'<small> / '+fmt(rT)+' GB</small>', bar(rU,rT)),
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

// ---- model state derivation: the one source of truth ----
function mstate(m,cl){
  const id=m.internal_name||m.name;
  const loadings=(cl.loadings||[]), compiling=(cl.compiling||[]);
  const ld=loadings.find(x=>x.model===id||x.display_model===m.name);
  if(ld) return {k:'loading',c:'var(--warn)',rank:1,ld};
  const cp=compiling.find(x=>x.model===id||x.display_model===m.name);
  if(cp) return {k:'compiling',c:'var(--warn)',rank:1,ld:cp};
  if((m.status||'')==='downloading') return {k:'downloading',c:'var(--accent)',rank:2};
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
  const GN={loaded:'Loaded',loading:'Loading',compiling:'Compiling',downloading:'Downloading',registered:'Registered · on disk',notdl:'Not downloaded'};
  for(const {m,s} of ms){
    if(s.k!==grp){ grp=s.k; html+='<div class="grp">'+GN[grp]+'</div>'; }
    html+=modelRow(m,s);
  }
  $('#models').innerHTML=html;
}
function modelRow(m,s){
  const arch=archChip(m), al=(m.aliases||[]).map(a=>'<span class="chip al">'+esc(a)+'</span>').join('');
  let meta='', acts='';
  if(s.k==='loaded'){
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
    meta='downloading weights…'; acts='<button class="btn sm ghost" onclick="dl(\''+esc(m.name)+'\',\'stop\')">Stop</button>';
  } else if(s.k==='registered'){
    meta=fitMeta(m); acts='<button class="btn sm pri" onclick="openLoad(\''+esc(m.name)+'\')">Load ▾</button>';
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
  if(!m.ready||(cz.int4&&cz.int4.ok)||(m.capabilities||[]).includes('embedding'))return '';
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
  $('#nodes').innerHTML=ns.map(n=>{
    const gpu=n.has_gpu;
    const used=gpu?(n.vram_used_gb||0):((n.total_mem_gb||0)-(n.free_mem_gb||0));
    const tot=gpu?(n.vram_total_gb||0):(n.total_mem_gb||0);
    const util=gpu?('GPU '+Math.round(n.gpu_util||0)+'%'):('CPU '+Math.round(n.cpu_percent||0)+'%');
    const dev=gpu?(n.device_name||'GPU'):((n.cores||'')+'c CPU');
    const off=(!n.alive)?' <span class="err">offline</span>':'';
    return '<div class="node"><div class="nn">'+esc(n.hostname)+' <small>'+esc(dev)+'</small>'+off+'</div>'
      +'<div class="mb"><span class="lab">'+(gpu?'VRAM':'RAM')+'</span>'+bar(used,tot)
      +'<span style="font-size:11px;color:var(--muted);white-space:nowrap">'+fmt(used)+' / '+fmt(tot)+'</span></div>'
      +'<div class="util">'+util+'</div></div>';
  }).join('');
}

// ---------- actions ----------
function closeOv(){ $('#ov').classList.remove('show'); DETAIL_OPEN=null; }
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
  const ctxDefault = dc>=131072 ? 16384 : (dc||8192);
  const qopt=q=>'<option value="'+q+'">'+q+(cz[q]&&cz[q].ok?' · '+gb(cz[q].size_gb)+' cached':'')+'</option>';
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>Load '+esc(name)+'</h3>'
   +'<div class="grid2"><div><label>Quant</label><select id="l-q" onchange="previewSoon(\''+esc(name)+'\')">'+qopt('int4')+qopt('int8')+'<option value="none">none (bf16)'+(cz.none&&cz.none.ok?' · '+gb(cz.none.size_gb)+' cached':(m.size_gb?' · '+gb(m.size_gb):''))+'</option></select></div>'
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
  add('aliases',(m.aliases||[]).map(esc).join(', ')); add('weights size',gb(m.size_gb));
  add('cached quants',Object.keys(cz).filter(q=>cz[q]&&cz[q].ok).map(q=>q+' '+gb(cz[q].size_gb)).join(', '));
  let out='<table class="kv">'+rows+'</table>';
  if(m.loaded){
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
  // precache (shard cache): compile int4/int8 so future loads serve from cache instantly.
  let pre='';
  if(m.ready){ let chips='';
    for(const q of ['int4','int8']){
      if(cz[q]&&cz[q].ok) chips+='<span class="chip al">'+q+' cached '+gb(cz[q].size_gb)+'</span> ';
      else chips+='<button class="btn sm ghost" onclick="compileShards(\''+esc(name)+'\',\''+q+'\')">Compile '+q+'</button> ';
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
  let acts='';
  if(m.loaded) acts='<button class="btn sm pri" onclick="location.href=\'/chat?model='+encodeURIComponent(name)+'\'">Chat ↗</button> '
    +'<button class="btn sm" onclick="unload(\''+esc(name)+'\')">Unload</button> '
    +'<button class="btn sm ghost" onclick="openHistory(\''+esc(name)+'\')">View context ▾</button> '
    +'<button class="btn sm ghost" onclick="reconf(\''+esc(name)+'\')">Reconfigure…</button>';
  else acts='<button class="btn sm pri" onclick="closeOv();openLoad(\''+esc(name)+'\')">Load…</button> '
    +'<button class="btn sm ghost" onclick="forget(\''+esc(name)+'\')">Forget</button> '
    +'<button class="btn sm ghost" onclick="del(\''+esc(name)+'\')">Delete</button>';
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>'+esc(name)+'</h3>'
    +'<div id="dlive">'+detailLive(name)+'</div>'+rt+pre
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
  .wrap{max-width:980px;margin:0 auto;padding:18px 20px 60px}
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
    <div class="fld"><label>Max concurrent loaded models</label><input id="max_loaded" type="number" min="1"></div>
    <div class="fld"><label>Per-model queue depth</label><input id="queue_depth" type="number" min="1"></div>
    <div class="fld"><label>Auto-load default quant</label><select id="autoload_quant"><option>int4</option><option>int8</option><option value="none">none (bf16)</option></select></div>
    <div class="fld"><label>Auto-load default ctx (0 = native)</label><input id="autoload_ctx" type="number" min="0"></div>
    <div class="fld"><label>Auto-load placement mode</label><select id="autoload_mode"><option>auto</option><option>all-gpu</option><option>distribute</option><option>proportional</option><option>single</option></select></div>
    <div class="fld"><label>Prefill stall-watchdog (s, 0=off)</label><input id="gen_stall_s" type="number" min="0"></div>
    <div class="fld"><label>Decode stall-watchdog (s, 0=off)</label><input id="gen_stall_decode_s" type="number" min="0"></div>
    <div class="fld"><label title="Unload any model that has served no requests for this many minutes. 0 or -1 = keep every model loaded forever (the default; -1 is the Ollama-style spelling and saves as -1). Pinned (📌) models and models with an active or queued request are never idle-unloaded.">Idle unload (min, 0/-1 = keep forever)</label><input id="idle_unload_m" type="number" min="-1" step="any" list="dl-idleu"><datalist id="dl-idleu"><option value="-1"></option><option value="0"></option><option value="5"></option><option value="15"></option><option value="60"></option><option value="240"></option><option value="1440"></option></datalist></div>
    <div class="fld tog"><input type="checkbox" id="auto_unload"><label for="auto_unload">LRU auto-unload</label></div>
    <div class="fld tog"><input type="checkbox" id="auto_load"><label for="auto_load">Auto-load on request</label></div>
    <div class="fld tog"><input type="checkbox" id="vram_weights_first"><label for="vram_weights_first">Budget weights vs physical-free VRAM</label></div>
  </div>
  <div style="margin-top:14px"><button class="btn pri" onclick="save()">Save settings</button><span id="cfg-msg"></span></div>
</div>

<div class="card">
  <h2>Nodes — compute tiers</h2><div class="sub">Enable/disable each node's CPU/RAM and GPU/VRAM in the placement pool. Changes re-plan affected models.</div>
  <div class="actrow" style="margin-bottom:8px">
    <span style="font-size:12px;color:var(--muted)">Bulk:</span>
    <button class="btn sm" onclick="bulk('ram',true)">All RAM on</button><button class="btn sm" onclick="bulk('ram',false)">All RAM off</button>
    <button class="btn sm" onclick="bulk('vram',true)">All VRAM on</button><button class="btn sm" onclick="bulk('vram',false)">All VRAM off</button>
    <span id="node-msg"></span>
  </div>
  <div id="nodes"></div>
</div>

<div class="card">
  <h2>Controller</h2><div class="sub">Fleet-level operations. Use with care.</div>
  <div class="actrow">
    <button class="btn" onclick="ctlAct('/restart?workers=0','Restart the CONTROLLER only?')">Restart controller</button>
    <button class="btn" onclick="ctlAct('/restart?workers=1','Restart the WHOLE fleet (controller + all workers)?')">Restart fleet</button>
    <button class="btn" onclick="ctlAct('/update?workers=1','Pull latest code from GitHub and restart the fleet?')">Update + deploy</button>
    <button class="btn" onclick="ctlAct('/gc_cache','Reclaim disk by removing redundant HF-cache copies?')">GC disk cache</button>
    <span id="ctl-msg"></span>
  </div>
</div>
</div>

<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const _up=s=>{s=Math.max(0,Math.floor(s||0));const d=Math.floor(s/86400),h=Math.floor(s%86400/3600),m=Math.floor(s%3600/60);return d?d+'d '+h+'h':(h?h+'h '+m+'m':m+'m')};
const FIELDS=['max_loaded','queue_depth','autoload_quant','autoload_ctx','autoload_mode','gen_stall_s','gen_stall_decode_s','idle_unload_m'];
const TOGS=['auto_unload','auto_load','vram_weights_first'];
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
      +'<label class="tog"><input type="checkbox" '+(n.ram_enabled?'checked':'')+' onchange="setTier(\''+esc(n.hostname)+'\',\'ram\',this.checked)">RAM</label>'
      +(gpu?'<label class="tog"><input type="checkbox" '+(n.vram_enabled?'checked':'')+' onchange="setTier(\''+esc(n.hostname)+'\',\'vram\',this.checked)">VRAM</label>':'<span style="color:var(--dim);font-size:12px">no GPU</span>')
      +'<span class="grow"></span><span style="font-size:11px;color:var(--dim)">'+(gpu?fmt(n.vram_used_gb)+'/'+fmt(n.vram_total_gb)+' GB VRAM':fmt((n.total_mem_gb||0)-(n.free_mem_gb||0))+'/'+fmt(n.total_mem_gb)+' GB RAM')+'</span></div>';
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
  .wrap{max-width:980px;margin:0 auto;padding:18px 20px 30px}
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
  .wrap{max-width:1180px;margin:0 auto;padding:18px 20px 40px}
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
