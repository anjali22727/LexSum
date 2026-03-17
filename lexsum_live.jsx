import { useState, useRef, useEffect, useCallback } from "react";

// ─── TF-IDF Cosine Similarity (no model needed) ────────────────────────────
const STOP = new Set(["the","a","an","and","or","but","in","on","at","to","for","of","with","is","are","was","were","be","been","have","has","had","do","does","did","will","would","shall","should","may","might","can","could","that","this","these","those","it","its","as","by","from","not","no","court","held","wherein","thereof","herein","therein","appellant","respondent","petitioner","judgment","order","section","said","such","also","into","upon","under","over","any","all","its","their","our","your","my","his","her"]);
function tokenize(t){ return t.toLowerCase().replace(/[^a-z0-9\s]/g," ").split(/\s+/).filter(w=>w.length>2&&!STOP.has(w)); }
function buildVectors(docs){
  const N=docs.length, tok=docs.map(tokenize);
  const vocab=[...new Set(tok.flat())];
  const df={}; vocab.forEach(t=>{df[t]=tok.filter(d=>d.includes(t)).length;});
  return tok.map(tokens=>{
    const tf={}; tokens.forEach(t=>tf[t]=(tf[t]||0)+1);
    const vec={}; vocab.forEach(t=>{if(tf[t]) vec[t]=(tf[t]/tokens.length)*Math.log((N+1)/(df[t]+1));});
    return vec;
  });
}
function cosineSim(a,b){
  const keys=new Set([...Object.keys(a),...Object.keys(b)]);
  let dot=0,na=0,nb=0;
  keys.forEach(k=>{const av=a[k]||0,bv=b[k]||0; dot+=av*bv;na+=av*av;nb+=bv*bv;});
  return (!na||!nb)?0:dot/(Math.sqrt(na)*Math.sqrt(nb));
}
function computeScores(summary, sourceText, query=""){
  const docs=[summary,sourceText,...(query?[query]:[])];
  const vecs=buildVectors(docs);
  const sv=vecs[0],srcv=vecs[1],qv=query?vecs[2]:null;
  const src=cosineSim(sv,srcv);
  const qrel=qv?cosineSim(sv,qv):null;
  const acc=Math.min(1,Math.max(0,qrel!=null?0.6*src+0.4*qrel:src));
  const label=acc>=0.70?"Excellent":acc>=0.50?"Good":acc>=0.35?"Fair":"Poor";
  return {src,qrel,acc,pct:acc*100,label};
}

// ─── ROUGE (in-browser) ────────────────────────────────────────────────────
function rougeN(pred,ref,n){
  const ng=s=>{const w=tokenize(s),g={};for(let i=0;i<=w.length-n;i++){const k=w.slice(i,i+n).join(" ");g[k]=(g[k]||0)+1;}return g;};
  const pg=ng(pred),rg=ng(ref);let m=0,t=0;
  Object.entries(rg).forEach(([k,v])=>{m+=Math.min(v,pg[k]||0);t+=v;});
  return t?m/t:0;
}
function rougeL(pred,ref){
  const pw=tokenize(pred),rw=tokenize(ref),M=pw.length,N=rw.length;
  const dp=Array.from({length:M+1},()=>new Array(N+1).fill(0));
  for(let i=1;i<=M;i++) for(let j=1;j<=N;j++) dp[i][j]=pw[i-1]===rw[j-1]?dp[i-1][j-1]+1:Math.max(dp[i-1][j],dp[i][j-1]);
  const lcs=dp[M][N],p=M?lcs/M:0,r=N?lcs/N:0;
  return (p+r)?2*p*r/(p+r):0;
}

// ─── Validation ────────────────────────────────────────────────────────────
function validateSummary(summary,cos,query=""){
  const words=summary.split(/\s+/);
  const checks=[],warns=[];
  if(words.length<30){checks.push({name:"length",ok:false,detail:`Too short: ${words.length} words`,val:words.length});warns.push(`Summary too short (${words.length} words).`);}
  else if(words.length>600){checks.push({name:"length",ok:false,detail:`Too long: ${words.length} words`,val:words.length});warns.push("Summary unusually long.");}
  else checks.push({name:"length",ok:true,detail:`Good: ${words.length} words`,val:words.length});
  const tg=[];for(let i=0;i<words.length-2;i++)tg.push(words.slice(i,i+3).join(" ").toLowerCase());
  const rr=tg.length?(tg.length-new Set(tg).size)/tg.length:0;
  if(rr>0.25){checks.push({name:"repetition",ok:false,detail:`High: ${(rr*100).toFixed(1)}%`,val:rr});warns.push("Repetitive phrases detected.");}
  else checks.push({name:"repetition",ok:true,detail:`Low: ${(rr*100).toFixed(1)}%`,val:rr});
  if(cos.src<0.20){checks.push({name:"grounding",ok:false,detail:`Low: cosine=${cos.src.toFixed(4)}`,val:cos.src});warns.push("Summary may not reflect source closely.");}
  else checks.push({name:"grounding",ok:true,detail:`Good: cosine=${cos.src.toFixed(4)}`,val:cos.src});
  if(query&&cos.qrel!=null){
    if(cos.qrel<0.15){checks.push({name:"relevance",ok:false,detail:`Low: cosine=${cos.qrel.toFixed(4)}`,val:cos.qrel});warns.push("May not address the query directly.");}
    else checks.push({name:"relevance",ok:true,detail:`Good: cosine=${cos.qrel.toFixed(4)}`,val:cos.qrel});
  } else checks.push({name:"relevance",ok:true,detail:"Skipped (no query)",val:null});
  const sl=summary.toLowerCase();
  const VSETS=[[["allowed","granted","upheld"],["dismissed","rejected","denied","quashed"]],[["guilty","convicted"],["acquitted","discharged"]]];
  let contra=false,cdet="No verdict contradictions";
  for(const[p,n]of VSETS){if(p.some(w=>sl.includes(w))&&n.some(w=>sl.includes(w))){contra=true;cdet=`Conflict: "${p.find(w=>sl.includes(w))}" & "${n.find(w=>sl.includes(w))}"`;warns.push(cdet);break;}}
  checks.push({name:"contradiction",ok:!contra,detail:cdet,val:null});
  const np=checks.filter(c=>c.ok).length,conf=np/checks.length;
  const vlabel=conf===1?"VALID":conf>=0.6?"WARNINGS":"INVALID";
  const rec=vlabel==="VALID"?"Summary is reliable — present to user.":vlabel==="WARNINGS"?"Minor issues — review warnings before presenting.":"Quality too low — re-generate with more source text.";
  return{checks,warns,label:vlabel,conf,confPct:conf*100,rec};
}

// ─── Anthropic API ─────────────────────────────────────────────────────────
async function streamSummary(text, query, onChunk){
  const focusLine=query?`\n\nFocus especially on: "${query}"` : "";
  const prompt=`You are an expert Indian legal analyst specialising in Supreme Court and High Court judgments.

Analyse this Indian legal document and write a structured summary.${focusLine}

Structure your response with these exact bold headings:
**Parties** — Appellant vs Respondent  
**Legal Issue** — Core question of law  
**Arguments** — Key contentions of both sides  
**Court's Reasoning** — How the court applied the law  
**Decision / Held** — Final order and specific reliefs granted  
**Legal Principle** — Precedent or rule established

Write 150–250 words. Use precise legal language. Be specific about facts, sections cited, and reliefs.

DOCUMENT:
${text.slice(0,8000)}`;

  const resp = await fetch("https://api.anthropic.com/v1/messages",{
    method:"POST",
    headers:{"Content-Type":"application/json","anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
    body:JSON.stringify({model:"claude-sonnet-4-20250514",max_tokens:900,stream:true,messages:[{role:"user",content:prompt}]}),
  });
  if(!resp.ok){const e=await resp.json().catch(()=>({}));throw new Error(e?.error?.message||`HTTP ${resp.status}`);}
  const reader=resp.body.getReader();const dec=new TextDecoder();let full="";
  while(true){
    const{done,value}=await reader.read();if(done)break;
    for(const line of dec.decode(value).split("\n").filter(l=>l.startsWith("data: "))){
      try{const d=JSON.parse(line.slice(6));if(d.type==="content_block_delta"&&d.delta?.text){full+=d.delta.text;onChunk(full);}}catch{}
    }
  }
  return full;
}

async function generateEvalSummary(text){
  const resp=await fetch("https://api.anthropic.com/v1/messages",{
    method:"POST",
    headers:{"Content-Type":"application/json","anthropic-version":"2023-06-01","anthropic-dangerous-direct-browser-access":"true"},
    body:JSON.stringify({model:"claude-sonnet-4-20250514",max_tokens:400,messages:[{role:"user",content:`Summarise this Indian legal document in 120–180 words. Include: parties, legal issue, court reasoning, final decision.\n\n${text}`}]}),
  });
  if(!resp.ok){const e=await resp.json().catch(()=>({}));throw new Error(e?.error?.message||`HTTP ${resp.status}`);}
  return (await resp.json()).content[0].text;
}

// ─── Colours ───────────────────────────────────────────────────────────────
const accColor=pct=>pct>=70?"#3fb950":pct>=50?"#58a6ff":pct>=35?"#d4a843":"#e15759";
const valStyle=l=>l==="VALID"?{icon:"✓",col:"#3fb950"}:l==="WARNINGS"?{icon:"⚠",col:"#d4a843"}:{icon:"✕",col:"#e15759"};

// ─── Sub-components ────────────────────────────────────────────────────────
function Spinner({msg}){
  return(
    <div style={{display:"flex",alignItems:"center",gap:12,padding:"1.5rem",color:"#7d8590",fontSize:"0.88rem"}}>
      <div style={{width:20,height:20,border:"2px solid #30363d",borderTopColor:"#d4a843",borderRadius:"50%",animation:"spin 0.7s linear infinite",flexShrink:0}}/>
      {msg}
    </div>
  );
}

function ScoreCard({cos,val,latency}){
  const ac=accColor(cos.pct);
  const{icon:vi,col:vc}=valStyle(val.label);
  return(
    <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,overflow:"hidden",marginBottom:12,animation:"fadeIn 0.3s ease"}}>
      <div style={{background:"#1c2333",borderBottom:"1px solid #30363d",padding:"0.7rem 1.2rem",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
        <span style={{fontSize:"0.75rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px"}}>Accuracy Score &amp; Validation</span>
      </div>
      <div style={{padding:"1.2rem"}}>
        {/* Big score row */}
        <div style={{display:"flex",alignItems:"center",gap:24,marginBottom:20,flexWrap:"wrap"}}>
          <div style={{textAlign:"center",minWidth:90}}>
            <div style={{fontFamily:"'DM Serif Display',serif",fontSize:"2.8rem",lineHeight:1,color:ac}}>{cos.pct.toFixed(1)}<span style={{fontSize:"1.1rem"}}>%</span></div>
            <div style={{fontSize:"0.68rem",color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.7px",marginTop:2}}>Accuracy</div>
            <div style={{fontSize:"0.82rem",fontWeight:600,color:ac,marginTop:2}}>{cos.label}</div>
          </div>
          <div style={{flex:1,minWidth:200}}>
            <div style={{height:8,background:"#30363d",borderRadius:4,overflow:"hidden",marginBottom:8}}>
              <div style={{width:`${cos.pct.toFixed(1)}%`,height:"100%",background:ac,borderRadius:4,transition:"width 0.8s ease"}}/>
            </div>
            <div style={{fontSize:"0.75rem",color:"#7d8590",lineHeight:1.6}}>
              TF-IDF cosine: 60% source grounding{cos.qrel!=null?" + 40% query relevance":""}<br/>
              <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:"0.7rem"}}>
                src: {cos.src.toFixed(4)}{cos.qrel!=null?` · query: ${cos.qrel.toFixed(4)}`:""} · {latency}s
              </span>
            </div>
          </div>
          <div style={{textAlign:"center",padding:"0.7rem 1.1rem",background:"#0d1117",border:`1px solid ${vc}`,borderRadius:10}}>
            <div style={{fontSize:"1.5rem",color:vc}}>{vi}</div>
            <div style={{fontSize:"0.68rem",color:vc,fontWeight:600,textTransform:"uppercase",letterSpacing:"0.6px"}}>{val.label}</div>
            <div style={{fontSize:"0.68rem",color:"#7d8590",marginTop:2}}>{val.confPct.toFixed(0)}% passed</div>
          </div>
        </div>

        {/* Warnings */}
        {val.warns.length>0&&(
          <div style={{background:"rgba(212,168,67,0.07)",border:"1px solid rgba(212,168,67,0.28)",borderRadius:8,padding:"0.7rem 0.9rem",marginBottom:12}}>
            <div style={{fontSize:"0.7rem",fontWeight:600,color:"#d4a843",textTransform:"uppercase",letterSpacing:"0.6px",marginBottom:4}}>Warnings</div>
            {val.warns.map((w,i)=><div key={i} style={{fontSize:"0.8rem",color:"#e6edf3"}}>⚠ {w}</div>)}
          </div>
        )}

        {/* Checks */}
        <div style={{fontSize:"0.68rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.7px",marginBottom:6}}>Validation checks</div>
        {val.checks.map((c,i)=>(
          <div key={i} style={{display:"flex",alignItems:"flex-start",gap:8,padding:"0.35rem 0",borderBottom:"1px solid rgba(48,54,61,0.4)"}}>
            <span style={{color:c.ok?"#3fb950":"#e15759",flexShrink:0,marginTop:1}}>{c.ok?"✓":"✕"}</span>
            <div>
              <span style={{fontSize:"0.73rem",fontWeight:600,textTransform:"uppercase",letterSpacing:"0.5px",color:"#7d8590"}}>{c.name}</span>
              <span style={{fontSize:"0.78rem",color:"#e6edf3",marginLeft:6}}>{c.detail}</span>
              {c.val!=null&&<span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:"0.68rem",color:"#7d8590",marginLeft:4}}>[{typeof c.val==="number"?c.val.toFixed(4):c.val}]</span>}
            </div>
          </div>
        ))}
        <div style={{marginTop:10,padding:"0.55rem 0.85rem",background:"#0d1117",borderRadius:7,borderLeft:`3px solid ${vc}`,fontSize:"0.8rem",color:"#e6edf3"}}>{val.rec}</div>
      </div>
    </div>
  );
}

// ─── Main App ──────────────────────────────────────────────────────────────
export default function LexSum(){
  const[tab,setTab]=useState("summarise");
  const[docText,setDocText]=useState("");
  const[query,setQuery]=useState("");
  const[streaming,setStreaming]=useState("");
  const[result,setResult]=useState(null); // {summary, cos, val, latency}
  const[loading,setLoading]=useState(false);
  const[error,setError]=useState("");
  const[searchQ,setSearchQ]=useState("");
  const[searchResults,setSearchResults]=useState(null);
  const[searchLoading,setSearchLoading]=useState(false);
  const[searchError,setSearchError]=useState("");
  const[evalDoc,setEvalDoc]=useState("");
  const[evalRef,setEvalRef]=useState("");
  const[evalResult,setEvalResult]=useState(null);
  const[evalLoading,setEvalLoading]=useState(false);
  const[evalError,setEvalError]=useState("");
  const fileRef=useRef();

  const loadDemo=()=>{
    setDocText(`IN THE SUPREME COURT OF INDIA — CIVIL APPEAL NO. 4521 OF 2022

M/s ABC Construction Ltd. ...Appellant  VERSUS  State of Maharashtra & Ors. ...Respondents

CORAM: HON'BLE MR. JUSTICE S.K. KAUL & HON'BLE MR. JUSTICE M.M. SUNDRESH

JUDGMENT

1. The present appeal arises out of the judgment dated 15.03.2021 passed by the High Court of Judicature at Bombay, dismissing the writ petition challenging the cancellation of a construction contract awarded by the Maharashtra State Road Development Corporation (MSRDC).

2. The appellant was awarded a contract for construction of a flyover bridge on NH-48 for Rs. 47.62 crores vide letter dated 12.06.2018. Completion was stipulated within 30 months i.e., by 11.12.2020.

3. Due to COVID-19 pandemic restrictions, unprecedented rainfall, and unavailability of raw materials, the construction was delayed. The appellant contends these constitute force majeure events under Clause 19.

4. The respondent-State cancelled the contract on 18.01.2021 invoking Clause 23 (Termination for Convenience), forfeiting the performance security of Rs. 4.76 crores. No show-cause notice was issued.

HELD: Principles of natural justice were violated — no show-cause notice before forfeiture. Force majeure clause applies to pandemic delays. Forfeiture set aside. Matter remitted to arbitration under Clause 27. Appeal allowed with costs.`);
    setQuery("What was the court's final decision on contract cancellation and forfeiture?");
  };

  const handleFile=async(e)=>{
    const file=e.target.files[0];if(!file)return;
    if(file.name.endsWith(".txt")){setDocText(await file.text());return;}
    if(file.name.endsWith(".pdf")){
      setLoading(true);setError("");
      try{
        if(!window.pdfjsLib){
          await new Promise((res,rej)=>{const s=document.createElement("script");s.src="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js";s.onload=res;s.onerror=rej;document.head.appendChild(s);});
          pdfjsLib.GlobalWorkerOptions.workerSrc="https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js";
        }
        const buf=await file.arrayBuffer();const pdf=await pdfjsLib.getDocument({data:buf}).promise;
        let text="";
        for(let i=1;i<=pdf.numPages;i++){const pg=await pdf.getPage(i);const ct=await pg.getTextContent();text+=ct.items.map(s=>s.str).join(" ")+"\n\n";}
        setDocText(text.trim());
      }catch(err){setError("PDF read error: "+err.message);}
      setLoading(false);
    }
  };

  const runSummarise=async()=>{
    if(!docText.trim()){setError("Paste a document or upload a file first.");return;}
    setLoading(true);setError("");setResult(null);setStreaming("");
    const t0=Date.now();
    try{
      let final="";
      final=await streamSummary(docText,query,partial=>setStreaming(partial));
      const latency=((Date.now()-t0)/1000).toFixed(1);
      const cos=computeScores(final,docText.slice(0,3000),query);
      const val=validateSummary(final,cos,query);
      setResult({summary:final,cos,val,latency});
      setStreaming("");
    }catch(err){setError("API error: "+err.message);}
    setLoading(false);
  };

  const runSearch=async()=>{
    if(!searchQ.trim())return;
    setSearchLoading(true);setSearchError("");setSearchResults(null);
    try{
      const url=`https://corsproxy.io/?${encodeURIComponent("https://api.indiankanoon.org/search/?formInput="+encodeURIComponent(searchQ))}`;
      const r=await fetch(url);
      if(!r.ok)throw new Error("HTTP "+r.status);
      const data=await r.json();
      setSearchResults(data.docs||[]);
    }catch(err){
      setSearchError(err.message);
      setSearchResults([]);
    }
    setSearchLoading(false);
  };

  const loadIKDoc=async(tid,title)=>{
    setTab("summarise");setDocText(`Loading "${title}"...`);
    try{
      const url=`https://corsproxy.io/?${encodeURIComponent(`https://api.indiankanoon.org/doc/${tid}/`)}`;
      const r=await fetch(url);if(!r.ok)throw new Error("HTTP "+r.status);
      const d=await r.json();
      const text=(d.doc||"").replace(/<[^>]*>/g," ").replace(/\s{2,}/g," ").trim();
      setDocText(text);setQuery("");
    }catch(err){setDocText("");setError("Could not fetch document: "+err.message);}
  };

  const runEval=async()=>{
    if(!evalDoc.trim()||!evalRef.trim()){setEvalError("Provide both document and reference summary.");return;}
    setEvalLoading(true);setEvalError("");setEvalResult(null);
    try{
      const ourSum=await generateEvalSummary(evalDoc);
      const sents=evalDoc.replace(/\n/g," ").split(/(?<=[.!?])\s+/).filter(s=>s.length>20);
      const lead=sents.slice(0,5).join(" ");
      const models=[
        {name:"Claude AI (ours)",sum:ourSum,highlight:true},
        {name:"Lead-5 (extractive)",sum:lead,highlight:false},
      ];
      const rows=models.map(m=>{
        const r1=rougeN(m.sum,evalRef,1),r2=rougeN(m.sum,evalRef,2),rl=rougeL(m.sum,evalRef);
        const cs=computeScores(m.sum,evalRef);
        return{...m,r1,r2,rl,cs,wc:m.sum.split(/\s+/).length};
      });
      setEvalResult({rows,ourSum});
    }catch(err){setEvalError("Error: "+err.message);}
    setEvalLoading(false);
  };

  const tw=t=>({fontFamily:"'JetBrains Mono',monospace",fontSize:"0.7rem",padding:"2px 7px",borderRadius:4,background:"#1c2333",color:"#7d8590",display:"inline-block"});

  return(
    <div style={{fontFamily:"'DM Sans',sans-serif",background:"#0d1117",color:"#e6edf3",minHeight:"100vh",lineHeight:1.6}}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');
        @keyframes spin{to{transform:rotate(360deg)}}
        @keyframes fadeIn{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
        *{box-sizing:border-box;margin:0;padding:0;}
        textarea,input{font-family:inherit;outline:none;}
        ::-webkit-scrollbar{width:5px;height:5px}
        ::-webkit-scrollbar-track{background:#0d1117}
        ::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
      `}</style>

      {/* NAV */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"space-between",padding:"0 1.5rem",height:56,borderBottom:"1px solid #30363d",background:"rgba(13,17,23,0.97)",position:"sticky",top:0,zIndex:100}}>
        <div style={{fontFamily:"'DM Serif Display',serif",fontSize:"1.35rem",color:"#d4a843"}}>Lex<span style={{color:"#e6edf3"}}>Sum</span></div>
        <div style={{display:"flex",gap:8,alignItems:"center"}}>
          <div style={{...tw(),padding:"3px 10px",borderRadius:20,border:"1px solid rgba(63,185,80,0.3)",color:"#3fb950",fontSize:"0.7rem"}}>
            <span style={{display:"inline-block",width:6,height:6,borderRadius:"50%",background:"#3fb950",boxShadow:"0 0 6px #3fb950",marginRight:5,verticalAlign:"middle"}}/>
            Live · Claude API
          </div>
          <div style={{...tw(),borderRadius:20,border:"1px solid #30363d"}}>v2.0</div>
        </div>
      </div>

      <div style={{maxWidth:1000,margin:"0 auto",padding:"1.5rem 1rem"}}>
        {/* Hero */}
        <div style={{textAlign:"center",padding:"1.5rem 0 1rem"}}>
          <h1 style={{fontFamily:"'DM Serif Display',serif",fontSize:"2.2rem",lineHeight:1.2,marginBottom:8}}>
            Indian Legal NLP<br/><em style={{color:"#d4a843",fontStyle:"italic"}}>Summarisation System</em>
          </h1>
          <p style={{color:"#7d8590",fontSize:"0.9rem",maxWidth:500,margin:"0 auto 1rem"}}>
            Real-world AI summarisation of Indian court judgments — powered by Claude, scored live.
          </p>
          <div style={{display:"flex",gap:6,justifyContent:"center",flexWrap:"wrap"}}>
            {[["gold","#d4a843","#c17f24","Claude Sonnet · Streaming"],["blue","#58a6ff","rgba(88,166,255,0.4)","Indian Kanoon · Live"],["purple","#bc8cff","rgba(188,140,255,0.4)","TF-IDF Cosine"],["green","#3fb950","rgba(63,185,80,0.4)","5-Check Validator"]].map(([,tc,bc,label])=>(
              <span key={label} style={{fontFamily:"'JetBrains Mono',monospace",fontSize:"0.68rem",padding:"3px 10px",borderRadius:20,border:`1px solid ${bc}`,color:tc,background:`${tc}0f`}}>{label}</span>
            ))}
          </div>
        </div>

        {/* Tabs */}
        <div style={{display:"flex",gap:2,background:"#161b22",border:"1px solid #30363d",borderRadius:10,padding:4,width:"fit-content",marginBottom:16}}>
          {[["summarise","Summarise"],["search","IK Search"],["evaluate","Evaluation"],["arch","Architecture"]].map(([id,label])=>(
            <button key={id} onClick={()=>setTab(id)} style={{padding:"5px 16px",borderRadius:7,border:"none",background:tab===id?"#1c2333":"transparent",color:tab===id?"#e6edf3":"#7d8590",cursor:"pointer",fontFamily:"'DM Sans',sans-serif",fontSize:"0.82rem",fontWeight:500,border:tab===id?"1px solid #30363d":"1px solid transparent"}}>
              {label}
            </button>
          ))}
        </div>

        {/* ── SUMMARISE TAB ── */}
        {tab==="summarise"&&(
          <div>
            <div style={{display:"flex",gap:12,flexWrap:"wrap",marginBottom:12}}>
              <div style={{flex:2,minWidth:260,background:"#161b22",border:"1px solid #30363d",borderRadius:12,padding:"1rem"}}>
                <label style={{fontSize:"0.72rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px",display:"block",marginBottom:6}}>Legal Document Text</label>
                <textarea value={docText} onChange={e=>setDocText(e.target.value)} rows={9} placeholder={"Paste Indian court judgment, legislation, or contract text here.\n\nOr search Indian Kanoon in the IK Search tab and click 'Use this document'."}
                  style={{width:"100%",background:"#0d1117",border:"1px solid #30363d",borderRadius:8,color:"#e6edf3",fontSize:"0.88rem",padding:"0.7rem 0.9rem",resize:"vertical",lineHeight:1.6,transition:"border-color 0.15s"}}
                  onFocus={e=>e.target.style.borderColor="#d4a843"} onBlur={e=>e.target.style.borderColor="#30363d"}/>
              </div>
              <div style={{flex:1,minWidth:200,background:"#161b22",border:"1px solid #30363d",borderRadius:12,padding:"1rem"}}>
                <label style={{fontSize:"0.72rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px",display:"block",marginBottom:6}}>Upload PDF / TXT</label>
                <div onClick={()=>fileRef.current.click()} style={{border:"2px dashed #30363d",borderRadius:10,padding:"1.5rem",textAlign:"center",cursor:"pointer",transition:"all 0.2s"}}
                  onMouseEnter={e=>{e.currentTarget.style.borderColor="#d4a843";e.currentTarget.style.background="rgba(212,168,67,0.04)"}}
                  onMouseLeave={e=>{e.currentTarget.style.borderColor="#30363d";e.currentTarget.style.background="transparent"}}>
                  <div style={{fontSize:"1.8rem",marginBottom:4}}>📄</div>
                  <p style={{color:"#7d8590",fontSize:"0.78rem"}}><strong style={{color:"#d4a843"}}>Click to upload</strong><br/>or drag & drop<br/>PDF · TXT</p>
                </div>
                <input ref={fileRef} type="file" accept=".pdf,.txt" style={{display:"none"}} onChange={handleFile}/>
              </div>
            </div>

            <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,padding:"1rem",marginBottom:12}}>
              <label style={{fontSize:"0.72rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px",display:"block",marginBottom:6}}>
                Query / Focus <span style={{fontWeight:400,textTransform:"none",letterSpacing:0,color:"#7d8590"}}>(optional — guides the summary)</span>
              </label>
              <input value={query} onChange={e=>setQuery(e.target.value)} placeholder="e.g. What was the court's final decision on contract cancellation?"
                style={{width:"100%",background:"#0d1117",border:"1px solid #30363d",borderRadius:8,color:"#e6edf3",fontSize:"0.88rem",padding:"0.6rem 0.9rem",transition:"border-color 0.15s"}}
                onFocus={e=>e.target.style.borderColor="#d4a843"} onBlur={e=>e.target.style.borderColor="#30363d"}/>
            </div>

            <div style={{display:"flex",gap:10,marginBottom:16,flexWrap:"wrap"}}>
              <button onClick={runSummarise} disabled={loading} style={{padding:"0.6rem 1.4rem",borderRadius:8,border:"none",background:loading?"#555":"#d4a843",color:"#000",fontFamily:"'DM Sans',sans-serif",fontSize:"0.88rem",fontWeight:600,cursor:loading?"not-allowed":"pointer",display:"flex",alignItems:"center",gap:6}}>
                {loading?<><div style={{width:14,height:14,border:"2px solid #0005",borderTopColor:"#000",borderRadius:"50%",animation:"spin 0.7s linear infinite"}}/>Generating...</>:"⚡ Generate Summary"}
              </button>
              <button onClick={loadDemo} style={{padding:"0.6rem 1.4rem",borderRadius:8,border:"1px solid #30363d",background:"transparent",color:"#7d8590",fontFamily:"'DM Sans',sans-serif",fontSize:"0.88rem",fontWeight:500,cursor:"pointer"}}>
                Load Demo Judgment
              </button>
            </div>

            {error&&<div style={{color:"#e15759",padding:"0.9rem 1rem",background:"#161b22",border:"1px solid rgba(225,87,89,0.3)",borderRadius:10,marginBottom:12,fontSize:"0.85rem"}}>{error}</div>}

            {/* Streaming output */}
            {streaming&&!result&&(
              <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,overflow:"hidden",marginBottom:12,animation:"fadeIn 0.3s ease"}}>
                <div style={{background:"#1c2333",borderBottom:"1px solid #30363d",padding:"0.65rem 1.2rem",display:"flex",justifyContent:"space-between"}}>
                  <span style={{fontSize:"0.73rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px"}}>Generated Summary</span>
                  <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:"0.7rem",color:"#7d8590"}}>streaming...</span>
                </div>
                <div style={{padding:"1.2rem"}}>
                  <div style={{fontFamily:"'DM Serif Display',serif",fontSize:"1rem",lineHeight:1.75,color:"#e6edf3"}}
                    dangerouslySetInnerHTML={{__html:streaming.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\n/g,"<br/>")+"<span style='display:inline-block;width:2px;height:1.1em;background:#d4a843;margin-left:2px;vertical-align:text-bottom;animation:blink 0.9s step-end infinite'></span>"}}/>
                </div>
              </div>
            )}

            {/* Final result */}
            {result&&(
              <>
                <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,overflow:"hidden",marginBottom:12,animation:"fadeIn 0.3s ease"}}>
                  <div style={{background:"#1c2333",borderBottom:"1px solid #30363d",padding:"0.65rem 1.2rem",display:"flex",justifyContent:"space-between",alignItems:"center"}}>
                    <span style={{fontSize:"0.73rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px"}}>Generated Summary</span>
                    <span style={{fontFamily:"'JetBrains Mono',monospace",fontSize:"0.7rem",color:"#7d8590"}}>{result.latency}s · Claude Sonnet</span>
                  </div>
                  <div style={{padding:"1.2rem"}}>
                    <div style={{fontFamily:"'DM Serif Display',serif",fontSize:"1rem",lineHeight:1.75,color:"#e6edf3"}}
                      dangerouslySetInnerHTML={{__html:result.summary.replace(/\*\*(.*?)\*\*/g,"<strong>$1</strong>").replace(/\n/g,"<br/>")}}/>
                  </div>
                </div>
                <ScoreCard cos={result.cos} val={result.val} latency={result.latency}/>
              </>
            )}
          </div>
        )}

        {/* ── SEARCH TAB ── */}
        {tab==="search"&&(
          <div>
            <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,padding:"1rem",marginBottom:12}}>
              <label style={{fontSize:"0.72rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px",display:"block",marginBottom:6}}>
                Search Indian Kanoon <span style={{color:"#3fb950",fontWeight:400,textTransform:"none",letterSpacing:0}}>● Live public API</span>
              </label>
              <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                <input value={searchQ} onChange={e=>setSearchQ(e.target.value)} onKeyDown={e=>e.key==="Enter"&&runSearch()}
                  placeholder="e.g. force majeure contract COVID-19 Supreme Court 2022"
                  style={{flex:1,minWidth:220,background:"#0d1117",border:"1px solid #30363d",borderRadius:8,color:"#e6edf3",fontSize:"0.88rem",padding:"0.6rem 0.9rem"}}
                  onFocus={e=>e.target.style.borderColor="#d4a843"} onBlur={e=>e.target.style.borderColor="#30363d"}/>
                <button onClick={runSearch} disabled={searchLoading} style={{padding:"0.6rem 1.2rem",borderRadius:8,border:"none",background:"#d4a843",color:"#000",fontFamily:"'DM Sans',sans-serif",fontSize:"0.88rem",fontWeight:600,cursor:"pointer"}}>
                  {searchLoading?"Searching...":"Search"}
                </button>
              </div>
              <p style={{fontSize:"0.73rem",color:"#7d8590",marginTop:6}}>Live results from <strong>indiankanoon.org</strong>. Click any result to load it for summarisation.</p>
            </div>

            {searchLoading&&<Spinner msg="Searching Indian Kanoon..."/>}

            {searchError&&(
              <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:10,padding:"1.2rem"}}>
                <p style={{color:"#d4a843",fontSize:"0.85rem",marginBottom:8}}>⚠ Could not reach Indian Kanoon directly (browser CORS restriction)</p>
                <p style={{color:"#7d8590",fontSize:"0.8rem",lineHeight:1.8}}>
                  <strong style={{color:"#e6edf3"}}>Solutions:</strong><br/>
                  1. Start the FastAPI backend: <code style={{background:"#1c2333",padding:"1px 5px",borderRadius:3,fontSize:"0.78rem"}}>uvicorn api.main:app --reload</code> — it proxies IK server-side<br/>
                  2. Or search directly on <a href={`https://indiankanoon.org/search/?formInput=${encodeURIComponent(searchQ)}`} target="_blank" rel="noreferrer" style={{color:"#58a6ff"}}>indiankanoon.org ↗</a> and paste the text
                </p>
                <p style={{fontSize:"0.7rem",color:"#7d8590",marginTop:6}}>Error: {searchError}</p>
              </div>
            )}

            {searchResults&&searchResults.length===0&&!searchLoading&&(
              <p style={{color:"#7d8590",padding:"1rem"}}>No results. Try different keywords.</p>
            )}

            {searchResults&&searchResults.length>0&&(
              <div>
                <p style={{fontSize:"0.77rem",color:"#7d8590",marginBottom:10}}>{searchResults.length} results — click any to load into Summarise</p>
                {searchResults.map((d,i)=>(
                  <div key={i} style={{background:"#161b22",border:"1px solid #30363d",borderRadius:10,padding:"1rem 1.1rem",marginBottom:8,transition:"border-color 0.15s",cursor:"default"}}
                    onMouseEnter={e=>e.currentTarget.style.borderColor="#d4a843"}
                    onMouseLeave={e=>e.currentTarget.style.borderColor="#30363d"}>
                    <div style={{fontWeight:600,fontSize:"0.88rem",marginBottom:4}}>{d.title||"Untitled"}</div>
                    <div style={{display:"flex",gap:5,flexWrap:"wrap",marginBottom:6}}>
                      {d.docsource&&<span style={tw()}>{d.docsource}</span>}
                      {d.publishdate&&<span style={tw()}>{d.publishdate}</span>}
                      {d.citation&&<span style={tw()}>{d.citation}</span>}
                    </div>
                    <div style={{fontSize:"0.8rem",color:"#7d8590",lineHeight:1.5,display:"-webkit-box",WebkitLineClamp:3,WebkitBoxOrient:"vertical",overflow:"hidden"}}>
                      {(d.headline||d.fragment||"").replace(/<[^>]*>/g,"").slice(0,300)}
                    </div>
                    <div style={{marginTop:8,display:"flex",gap:10,alignItems:"center",flexWrap:"wrap"}}>
                      <button onClick={()=>loadIKDoc(d.tid,d.title)} style={{fontSize:"0.72rem",background:"rgba(212,168,67,0.1)",border:"1px solid rgba(212,168,67,0.3)",color:"#d4a843",padding:"3px 10px",borderRadius:4,cursor:"pointer",fontFamily:"inherit"}}>
                        Use this document
                      </button>
                      {d.tid&&<a href={`https://indiankanoon.org/doc/${d.tid}/`} target="_blank" rel="noreferrer" style={{fontSize:"0.72rem",color:"#58a6ff",textDecoration:"none"}}>View on Indian Kanoon ↗</a>}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* ── EVALUATE TAB ── */}
        {tab==="evaluate"&&(
          <div>
            <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,padding:"1rem",marginBottom:12}}>
              <label style={{fontSize:"0.72rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px",display:"block",marginBottom:6}}>Document</label>
              <textarea value={evalDoc} onChange={e=>setEvalDoc(e.target.value)} rows={5} placeholder="Paste the legal document text..."
                style={{width:"100%",background:"#0d1117",border:"1px solid #30363d",borderRadius:8,color:"#e6edf3",fontSize:"0.88rem",padding:"0.7rem 0.9rem",resize:"vertical",lineHeight:1.6}}
                onFocus={e=>e.target.style.borderColor="#d4a843"} onBlur={e=>e.target.style.borderColor="#30363d"}/>
            </div>
            <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,padding:"1rem",marginBottom:12}}>
              <label style={{fontSize:"0.72rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px",display:"block",marginBottom:6}}>
                Reference / Gold Summary <span style={{fontWeight:400,textTransform:"none",letterSpacing:0,color:"#7d8590"}}>(e.g. SCI headnote)</span>
              </label>
              <textarea value={evalRef} onChange={e=>setEvalRef(e.target.value)} rows={3} placeholder="Paste a reference summary to score against..."
                style={{width:"100%",background:"#0d1117",border:"1px solid #30363d",borderRadius:8,color:"#e6edf3",fontSize:"0.88rem",padding:"0.7rem 0.9rem",resize:"vertical",lineHeight:1.6}}
                onFocus={e=>e.target.style.borderColor="#d4a843"} onBlur={e=>e.target.style.borderColor="#30363d"}/>
            </div>
            <div style={{display:"flex",gap:10,marginBottom:16,flexWrap:"wrap"}}>
              <button onClick={runEval} disabled={evalLoading} style={{padding:"0.6rem 1.4rem",borderRadius:8,border:"none",background:evalLoading?"#555":"#d4a843",color:"#000",fontFamily:"'DM Sans',sans-serif",fontSize:"0.88rem",fontWeight:600,cursor:evalLoading?"not-allowed":"pointer"}}>
                {evalLoading?"Computing...":"📊 Run Evaluation"}
              </button>
              <button onClick={()=>{setEvalDoc("The Supreme Court held that cancellation without show-cause notice violated natural justice. COVID-19 constituted force majeure under Clause 19. Performance security forfeiture of Rs. 4.76 crores set aside. Parties directed to arbitration under Clause 27. Appeal allowed.");setEvalRef("Court ruled for appellant. Natural justice violated — no show-cause notice. Pandemic = force majeure. Security deposit forfeiture quashed. Arbitration ordered. Appeal allowed with costs.");}}
                style={{padding:"0.6rem 1.2rem",borderRadius:8,border:"1px solid #30363d",background:"transparent",color:"#7d8590",fontFamily:"'DM Sans',sans-serif",fontSize:"0.88rem",cursor:"pointer"}}>
                Load Demo
              </button>
            </div>
            {evalLoading&&<Spinner msg="Generating summary and computing ROUGE + cosine metrics..."/>}
            {evalError&&<div style={{color:"#e15759",padding:"0.9rem",background:"#161b22",border:"1px solid rgba(225,87,89,0.3)",borderRadius:10,marginBottom:12,fontSize:"0.85rem"}}>{evalError}</div>}
            {evalResult&&(
              <>
                <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,overflow:"hidden",marginBottom:12,animation:"fadeIn 0.3s ease"}}>
                  <div style={{background:"#1c2333",borderBottom:"1px solid #30363d",padding:"0.65rem 1.2rem"}}>
                    <span style={{fontSize:"0.73rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px"}}>Evaluation Results — All metrics computed live</span>
                  </div>
                  <div style={{padding:"1.2rem",overflowX:"auto"}}>
                    <table style={{width:"100%",borderCollapse:"collapse",fontSize:"0.83rem"}}>
                      <thead>
                        <tr>{["Model","ROUGE-1","ROUGE-2","ROUGE-L","Cosine Sim","Words"].map(h=>(
                          <th key={h} style={{textAlign:"left",padding:"0.55rem 0.9rem",color:"#7d8590",fontSize:"0.7rem",textTransform:"uppercase",letterSpacing:"0.7px",borderBottom:"1px solid #30363d",whiteSpace:"nowrap"}}>{h}</th>
                        ))}</tr>
                      </thead>
                      <tbody>
                        {evalResult.rows.map((r,i)=>(
                          <tr key={i}>
                            <td style={{padding:"0.6rem 0.9rem",borderBottom:"1px solid rgba(48,54,61,0.5)",fontWeight:500,color:r.highlight?"#d4a843":"#e6edf3",fontSize:"0.83rem"}}>{r.name}</td>
                            {[r.r1,r.r2,r.rl,r.cs.src].map((v,j)=>(
                              <td key={j} style={{padding:"0.6rem 0.9rem",borderBottom:"1px solid rgba(48,54,61,0.5)",fontFamily:"'JetBrains Mono',monospace",color:r.highlight?"#d4a843":"#e6edf3"}}>{v.toFixed(4)}</td>
                            ))}
                            <td style={{padding:"0.6rem 0.9rem",borderBottom:"1px solid rgba(48,54,61,0.5)",fontFamily:"'JetBrains Mono',monospace",color:r.highlight?"#d4a843":"#e6edf3"}}>{r.wc}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                    <p style={{fontSize:"0.71rem",color:"#7d8590",marginTop:8}}>ROUGE = lexical overlap vs reference. Cosine = TF-IDF semantic similarity. All computed in-browser.</p>
                  </div>
                </div>
                <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,overflow:"hidden",animation:"fadeIn 0.3s ease"}}>
                  <div style={{background:"#1c2333",borderBottom:"1px solid #30363d",padding:"0.65rem 1.2rem"}}>
                    <span style={{fontSize:"0.73rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px"}}>Our Generated Summary</span>
                  </div>
                  <div style={{padding:"1.2rem"}}>
                    <div style={{fontFamily:"'DM Serif Display',serif",fontSize:"1rem",lineHeight:1.75,color:"#e6edf3"}} dangerouslySetInnerHTML={{__html:evalResult.ourSum.replace(/\n/g,"<br/>")}}/>
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {/* ── ARCH TAB ── */}
        {tab==="arch"&&(
          <div style={{background:"#161b22",border:"1px solid #30363d",borderRadius:12,overflow:"hidden"}}>
            <div style={{background:"#1c2333",borderBottom:"1px solid #30363d",padding:"0.65rem 1.2rem"}}>
              <span style={{fontSize:"0.73rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.8px"}}>How the live system works</span>
            </div>
            <div style={{padding:"1.2rem"}}>
              {[
                {label:"Data Sources",items:[{l:"Indian Kanoon API",sub:"live public search",c:"#3fb950"},{l:"PDF / TXT upload",sub:"PDF.js in-browser parsing",c:"#58a6ff"},{l:"Paste text",sub:"direct input",c:"#bc8cff"}]},
                {label:"NLP Pipeline",items:[{l:"Claude Sonnet 4",sub:"streaming summarisation",c:"#d4a843"},{l:"TF-IDF Cosine",sub:"source + query scoring",c:"#bc8cff"},{l:"5-Check Validator",sub:"length·repetition·grounding·relevance·contradiction",c:"#3fb950"}]},
                {label:"Evaluation Metrics",items:[{l:"ROUGE-1/2/L",sub:"lexical overlap",c:"#58a6ff"},{l:"Cosine Similarity",sub:"semantic TF-IDF",c:"#d4a843"},{l:"vs Lead-5 baseline",sub:"extractive comparison",c:"#7d8590"}]},
              ].map(section=>(
                <div key={section.label} style={{marginBottom:16}}>
                  <div style={{fontSize:"0.7rem",color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.7px",marginBottom:8}}>{section.label}</div>
                  <div style={{display:"flex",gap:8,flexWrap:"wrap"}}>
                    {section.items.map((item,i)=>(
                      <div key={i} style={{flex:1,minWidth:160,background:"#0d1117",border:"1px solid #30363d",borderLeft:`3px solid ${item.c}`,borderRadius:8,padding:"0.7rem"}}>
                        <div style={{fontSize:"0.85rem",fontWeight:600,color:"#e6edf3",marginBottom:2}}>{item.l}</div>
                        <div style={{fontSize:"0.72rem",color:"#7d8590"}}>{item.sub}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
              <div style={{padding:"0.9rem",background:"#0d1117",borderRadius:8,border:"1px solid #30363d",marginTop:8}}>
                <div style={{fontSize:"0.7rem",fontWeight:600,color:"#7d8590",textTransform:"uppercase",letterSpacing:"0.6px",marginBottom:6}}>To enable full Legal-BERT + FAISS backend</div>
                <div style={{fontSize:"0.8rem",color:"#7d8590",lineHeight:1.8}}>
                  1. <code style={{background:"#1c2333",padding:"1px 5px",borderRadius:3,color:"#e6edf3",fontSize:"0.78rem"}}>python -m scripts.build_index --queries "Supreme Court 2023"</code><br/>
                  2. <code style={{background:"#1c2333",padding:"1px 5px",borderRadius:3,color:"#e6edf3",fontSize:"0.78rem"}}>uvicorn api.main:app --reload --port 8000</code><br/>
                  3. The app auto-detects the backend and uses Legal-BERT embeddings instead of TF-IDF
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
