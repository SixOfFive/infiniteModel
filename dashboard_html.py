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
  /* model list */
  .list{background:var(--surface);border:1px solid var(--border);border-radius:12px;overflow:hidden}
  .row{display:flex;align-items:center;gap:12px;padding:11px 15px;border-bottom:1px solid var(--border)}
  .row:last-child{border-bottom:none}
  .row:hover{background:var(--surface2)}
  .row .nm{min-width:240px;cursor:pointer}
  .row .nm b{font-weight:600}
  .chip{font-size:10.5px;color:var(--muted);border:1px solid var(--border2);border-radius:9px;padding:1px 7px;margin-left:6px}
  .chip.al{color:var(--dim);border-style:dashed}
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
  .ov{position:absolute;inset:0;background:rgba(0,0,0,.55);display:none;align-items:flex-start;justify-content:center;z-index:50}
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

</div>

<div class="ov" id="ov" onclick="if(event.target===this)closeOv()"><div class="modal" id="modal"></div></div>

<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const gb=v=>(v==null?'—':(Math.round(v*10)/10)+' GB');
const pc=(a,b)=>b>0?Math.min(100,Math.round(100*a/b)):0;
let LAST=null;

async function api(path,opts){
  try{const r=await fetch(path,opts);const t=await r.text();let j;try{j=JSON.parse(t)}catch(e){j={ok:r.ok,raw:t}}
      if(!r.ok&&j&&j.error)throw new Error(j.error); if(!r.ok)throw new Error('HTTP '+r.status); return j;}
  catch(e){ throw e; }
}
function toast(msg,bad){ const c=$('#ctl'); const o=c.textContent; c.innerHTML=(bad?'<span class="err">':'<span style="color:var(--good)">')+esc(msg)+'</span>';
  setTimeout(()=>{c.textContent=o;},3500); }

async function tick(){
  let d; try{ d=await (await fetch('/status')).json(); }catch(e){ $('#ctl').innerHTML='<span class="err">controller unreachable</span>'; return; }
  LAST=d; render();
}
function render(){
  const d=LAST; if(!d)return;
  const c=d.controller||{}, p=d.pool||{}, comp=d.compute||{}, cl=d.cluster||{};
  $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?');
  // fleet tiles
  const loaded=(d.models||[]).filter(m=>m.loaded).length, reg=(d.models||[]).length;
  const vU=p.vram_gb-p.vram_free_gb, rU=p.ram_gb-p.ram_free_gb;
  $('#fleet').innerHTML=[
    tile('Nodes', (p.nodes||0)+' <small>· '+(comp.gpus||0)+' GPU</small>'),
    tile('Loaded', loaded+' <small>/ '+reg+' registered</small>'),
    tile('GPU pool', fmt(vU)+'<small> / '+fmt(p.vram_gb)+' GB</small>', bar(vU,p.vram_gb)),
    tile('RAM pool', fmt(rU)+'<small> / '+fmt(p.ram_gb)+' GB</small>', bar(rU,p.ram_gb)),
    tile('Throughput', ((d.metrics||{}).tokens_per_s||0).toFixed(1)+' <small>tok/s · '+Math.round(comp.overall_pct||0)+'% busy</small>'),
  ].join('');
  renderModels(d,cl);
  renderNodes(d);
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
  if(compiling.find(x=>x.model===id||x.display_model===m.name)) return {k:'compiling',c:'var(--warn)',rank:1};
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
    parts.push('ctx '+(m.ctx||'?'));
    if(m.vram_used_gb)parts.push('<span class="em">'+gb(m.vram_used_gb)+' VRAM</span>');
    if(m.ram_used_gb)parts.push(gb(m.ram_used_gb)+' RAM');
    if(m.last_tok_s)parts.push('<span style="color:var(--good)">'+m.last_tok_s.toFixed(1)+' tok/s</span>');
    if(m.active)parts.push(m.active+' active');
    meta=parts.join(' · ');
    acts='<button class="btn sm" onclick="unload(\''+esc(m.name)+'\')">Unload</button>';
  } else if(s.k==='loading'){
    const ld=s.ld||{}; const r=pc(ld.ready||0,ld.total||1);
    meta='compiling/loading · '+(ld.ready||0)+'/'+(ld.total||'?')+' · '+Math.round(ld.elapsed_s||0)+'s'+(ld.eta_s?(' · eta '+Math.round(ld.eta_s)+'s'):'')
        +'<div class="miniprog"><i style="width:'+r+'%"></i></div>';
    acts='<button class="btn sm ghost" onclick="cancelLoad(\''+esc(m.name)+'\')">Cancel</button>';
  } else if(s.k==='compiling'){
    meta='compiling shard cache…'; acts='<button class="btn sm ghost" disabled>…</button>';
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
  const cz=m.cached||{}; const q=cz.int4&&cz.int4.ok?'int4 '+gb(cz.int4.size_gb):(m.size_gb?gb(m.size_gb):'on disk');
  let warn='';
  // surface the devstral-style giant-ctx trap
  const dc=m.default_ctx||0;
  if(dc>=131072) warn=' · <span style="color:var(--hot)">⚠ native ctx '+(dc>=1000?Math.round(dc/1024)+'k':dc)+' — set ctx on load</span>';
  return q+' cache ready'+warn;
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
function closeOv(){ $('#ov').classList.remove('show'); }
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
   +'<div class="grid2"><div><label>Quant</label><select id="l-q">'+qopt('int4')+qopt('int8')+'<option value="none">none (bf16)</option></select></div>'
   +'<div><label>Context length</label><input id="l-ctx" type="number" value="'+ctxDefault+'"></div></div>'
   +'<label>Placement</label><select id="l-place" onchange="_placeChg()">'
   +'<optgroup label="Auto / distribute"><option value="m:auto">auto · GPU-first</option><option value="m:all-gpu">all-GPU</option>'
   +'<option value="m:distribute">distribute (CPU+GPU)</option><option value="m:proportional">proportional</option></optgroup>'
   +_devOpts()
   +'<optgroup label="Tensor-parallel"><option value="tp:gpu">GPU tensor-parallel</option><option value="tp:cpu">CPU tensor-parallel (RAM)</option></optgroup></select>'
   +'<div id="l-tpwrap" style="display:none;margin-top:8px"><label>TP width (number of nodes)</label><input id="l-tpn" type="number" min="2" value="2"></div>'
   +(dc>=131072?'<div class="note">⚠ native ctx is '+(Math.round(dc/1024))+'k — a huge KV cache. Keep ctx modest (8–16k) unless you need more.</div>':'')
   +'<div style="margin-top:16px;text-align:right"><button class="btn ghost" onclick="preview(\''+esc(name)+'\')">Preview fit</button> '
   +'<button class="btn pri" onclick="doLoad(\''+esc(name)+'\')">Load</button></div>'
   +'<div id="l-out" style="font-size:12px;color:var(--muted);margin-top:10px"></div>';
  $('#ov').classList.add('show');
}
function _placeChg(){ $('#l-tpwrap').style.display=$('#l-place').value.indexOf('tp:')===0?'block':'none'; }
function placeParams(name){
  const v=$('#l-place').value, p={model:name, quant:$('#l-q').value, ctx:$('#l-ctx').value};
  if(v.indexOf('m:')===0) p.mode=v.slice(2);
  else if(v.indexOf('g:')===0){ p.node=v.slice(2); }
  else if(v.indexOf('c:')===0){ p.node=v.slice(2); p.cpu_only='true'; }
  else if(v==='tp:gpu'){ p.tp=$('#l-tpn').value||2; }
  else if(v==='tp:cpu'){ p.tp=$('#l-tpn').value||2; p.cpu_only='true'; }
  return p;
}
async function preview(name){
  const p=placeParams(name); const q=new URLSearchParams({model:p.model,quant:p.quant,ctx:p.ctx});
  if(p.mode)q.set('mode',p.mode); if(p.node)q.set('node',p.node);
  $('#l-out').textContent='planning…';
  try{ const r=await (await fetch('/plan?'+q.toString())).json();
    if(!r.ok){ $('#l-out').innerHTML='<span class="err">'+esc(r.error||'cannot place')+'</span>'; return; }
    const st=(r.stages||[]).map(s=>esc(s.hostname)+' L'+s.layer_start+'-'+s.layer_end).join(', ');
    $('#l-out').innerHTML='<b>plan:</b> '+(esc(r.basis||''))+'<br>'+st
      +(p.tp?'<div class="note">TP preview is approximate (planned as pipeline)</div>':'')
      +(r.overload?'<div class="note">⚠ '+esc(r.overload.reason)+' — suggest '+esc(r.overload.suggest||'proportional')+'</div>':''); }
  catch(e){ $('#l-out').innerHTML='<span class="err">'+esc(String(e.message||e))+'</span>'; }
}
async function doLoad(name){
  const q=new URLSearchParams(placeParams(name));
  $('#l-out').textContent='loading…';
  try{ await api('/load?'+q.toString(),{method:'POST'}); closeOv(); toast('loading '+name); tick(); }
  catch(e){ $('#l-out').innerHTML='<span class="err">'+esc(String(e.message||e))+'</span>'; }
}
async function unload(name){ try{ await api('/unload?model='+encodeURIComponent(name),{method:'POST'}); toast('unloaded '+name); tick(); }catch(e){ toast(String(e.message||e),1);} }
async function cancelLoad(name){ try{ await api('/cancel_load?model='+encodeURIComponent(name),{method:'POST'}); toast('cancelled load'); tick(); }catch(e){ toast(String(e.message||e),1);} }
async function dl(name,action){ try{ await api('/download'+(action==='start'?'':'/'+action)+'?model='+encodeURIComponent(name),{method:'POST'}); toast(action+' '+name); tick(); }catch(e){ toast(String(e.message||e),1);} }

function openDetail(name){
  const m=(LAST.models||[]).find(x=>x.name===name); if(!m)return;
  const cz=m.cached||{};
  let rows='';
  const add=(k,v)=>{ if(v!=null&&v!=='') rows+='<tr><td>'+k+'</td><td class="v">'+v+'</td></tr>'; };
  add('HF id',esc(m.target)); add('arch',esc(m.arch||'')); add('status',esc(m.status));
  add('aliases',(m.aliases||[]).map(esc).join(', ')); add('size',gb(m.size_gb));
  add('native ctx',m.default_ctx||''); add('cached quants',Object.keys(cz).filter(q=>cz[q]&&cz[q].ok).join(', '));
  if(m.loaded){ add('loaded ctx',m.ctx); add('quant',esc(m.quant)); add('VRAM',gb(m.vram_used_gb)); add('RAM',gb(m.ram_used_gb)); add('KV reserved',gb(m.kv_reserved_gb)); }
  let stages='';
  if(m.stages&&m.stages.length) stages='<h3 style="font-size:13px;margin-top:14px">Placement</h3>'
    +'<table class="kv">'+m.stages.map(s=>'<tr><td>'+esc(s.hostname)+'</td><td class="v">L'+s.layer_start+'-'+s.layer_end+(s.role?(' · '+esc(s.role)):'')+'</td></tr>').join('')+'</table>';
  let acts='';
  if(m.loaded) acts='<button class="btn sm" onclick="unload(\''+esc(name)+'\')">Unload</button> '
    +'<button class="btn sm ghost" onclick="reconf(\''+esc(name)+'\')">Reconfigure…</button>';
  else acts='<button class="btn sm pri" onclick="closeOv();openLoad(\''+esc(name)+'\')">Load…</button> '
    +'<button class="btn sm ghost" onclick="forget(\''+esc(name)+'\')">Forget</button> '
    +'<button class="btn sm ghost" onclick="del(\''+esc(name)+'\')">Delete</button>';
  $('#modal').innerHTML='<span class="x" onclick="closeOv()">×</span><h3>'+esc(name)+'</h3>'
    +'<table class="kv">'+rows+'</table>'+stages
    +((m.warnings||[]).length?'<div class="note">⚠ '+m.warnings.map(esc).join('<br>⚠ ')+'</div>':'')
    +'<div style="margin-top:16px">'+acts+'</div>';
  $('#ov').classList.add('show');
}
async function forget(name){ if(!confirm('Forget '+name+'? (keeps weight files)'))return; try{ await api('/forget?model='+encodeURIComponent(name),{method:'POST'}); closeOv(); toast('forgot '+name); tick(); }catch(e){ toast(String(e.message||e),1);} }
async function del(name){ if(!confirm('DELETE '+name+' and its weight files?'))return; try{ await api('/delete?model='+encodeURIComponent(name),{method:'POST'}); closeOv(); toast('deleted '+name); tick(); }catch(e){ toast(String(e.message||e),1);} }
async function reconf(name){ const tp=prompt('Reconfigure '+name+' — tp size (1=pipeline):','1'); if(tp==null)return;
  try{ await api('/reconfigure?model='+encodeURIComponent(name)+'&tp='+encodeURIComponent(tp),{method:'POST'}); closeOv(); toast('reconfiguring '+name); tick(); }catch(e){ toast(String(e.message||e),1);} }

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
  <nav><a href="/">Models</a><a class="on" href="/config">Config</a><a href="/logs-page">Logs</a><a href="/bandwidth">Bandwidth</a></nav>
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
    <div class="fld"></div>
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
const FIELDS=['max_loaded','queue_depth','autoload_quant','autoload_ctx','autoload_mode','gen_stall_s','gen_stall_decode_s'];
const TOGS=['auto_unload','auto_load','vram_weights_first'];
let NODES=[];
async function load(){
  let d; try{ d=await (await fetch('/status')).json(); }catch(e){ $('#ctl').innerHTML='<span style="color:var(--bad)">controller unreachable</span>'; return; }
  const c=d.controller||{}; $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?');
  FIELDS.forEach(k=>{ if($('#'+k)&&c[k]!=null)$('#'+k).value=c[k]; });
  TOGS.forEach(k=>{ if($('#'+k))$('#'+k).checked=!!c[k]; });
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
  try{ await post('/config?'+q.toString()); msg('#cfg-msg','saved'); load(); }catch(e){ msg('#cfg-msg',String(e.message||e),1); }
}
async function setTier(host,tier,on){ try{ await post('/nodeconfig?host='+encodeURIComponent(host)+'&'+tier+'='+(on?'true':'false')); msg('#node-msg',host+' '+tier+'='+on); }catch(e){ msg('#node-msg',String(e.message||e),1);} }
async function bulk(tier,on){ try{ await post('/nodeconfig_all?tier='+tier+'&enabled='+(on?'true':'false')); msg('#node-msg','all '+tier+'='+on); load(); }catch(e){ msg('#node-msg',String(e.message||e),1);} }
async function ctlAct(path,confirmMsg){ if(!confirm(confirmMsg))return; try{ await post(path); msg('#ctl-msg','sent — controller restarting if applicable'); }catch(e){ msg('#ctl-msg',String(e.message||e),1);} }
load(); setInterval(load,5000);
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
  <nav><a href="/">Models</a><a href="/config">Config</a><a class="on" href="/logs-page">Logs</a><a href="/bandwidth">Bandwidth</a></nav>
  <span class="grow"></span>
  <span class="tog">source <select class="f" id="src" onchange="refresh()"></select></span>
  <button class="btn on" id="auto" onclick="toggleAuto()">auto ⟳</button>
  <button class="btn" onclick="refresh()">refresh</button>
</header>
<div class="panes">
  <div class="card"><h2 id="logtitle">controller log</h2><pre id="log">loading…</pre></div>
  <div class="card"><h2>activity</h2><div class="act" id="act"></div></div>
</div>
</div>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s==null?'':s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let AUTO=true, SRCS=false;
async function srcList(){
  try{ const d=await (await fetch('/status')).json();
    const c=d.controller||{}; $('#ctl').textContent=(c.hostname||'?')+':'+(c.http_port||'')+' · v'+(c.version||'?');
    if(!SRCS){ const sel=$('#src'); const cur=sel.value;
      sel.innerHTML='<option value="">controller</option>'+(d.nodes||[]).map(n=>'<option value="'+esc(n.hostname)+'">'+esc(n.hostname)+(n.has_gpu?' (GPU)':'')+'</option>').join('');
      if(cur)sel.value=cur; SRCS=true; }
    renderAct(d.activity||[]);
  }catch(e){ $('#ctl').innerHTML='<span style="color:var(--bad)">controller unreachable</span>'; }
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
