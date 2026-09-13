"""A local, blinded review page. It never reads workflow predictions."""
from common import ROOT, read, canonical

HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FARM binding reference review</title>
<style>
body{font:16px/1.5 system-ui,sans-serif;background:#f2f5f8;color:#172d40;margin:0}
header{background:#12364e;color:white;padding:24px 5vw}h1{font-size:26px;margin:0}
main{max-width:1200px;margin:24px auto;padding:0 20px}.row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button,select,input{font:inherit;padding:8px;border:1px solid #91a4b4;border-radius:5px}button{cursor:pointer;background:#fff;color:#12364e}
button.primary{background:#006f84;color:white;border:0}.card{background:white;padding:20px;margin:16px 0;border:1px solid #d2dee7;border-radius:8px}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eef3f6;padding:12px;font-size:13px}
textarea{width:100%;box-sizing:border-box;font:14px/1.45 ui-monospace,monospace;padding:10px;border:1px solid #a4b5c2;border-radius:5px}
.muted{color:#506779}.warning{background:#fff4d7;border-left:5px solid #ca8900;padding:12px}.pill{background:#e5eef3;border-radius:12px;padding:3px 10px}
label{display:block;margin:10px 0}h2{font-size:20px}h3{font-size:17px;margin:6px 0}.proposal{background:#f3f7fa;padding:12px;border-radius:5px}
.small{font-size:13px}a{color:#006985}.good{color:#08744c}#notice{min-height:24px}input[type=checkbox]{width:18px;height:18px}
@media(max-width:750px){.grid{grid-template-columns:1fr}}
</style>
<header><h1>FARM semantic binding references</h1><p>150 requests · independent AI proposals · human review before reporting verified accuracy</p></header>
<main>
<div class="warning">This page contains private research requests. It works locally and does not upload your edits. Neither proposal contains a workflow prediction. Agreement between the two AI annotators is not proof of correctness.</div>
<div class="card"><div class="row"><label>Reviewer name <input id="reviewer" placeholder="Your name"></label>
<button onclick="exportReviews()" class="primary">Export review file</button>
<label>Import saved review <input id="importFile" type="file" accept=".json"></label></div>
<div class="row"><button onclick="move(-1)">Previous</button><select id="caseSelect"></select><button onclick="move(1)">Next</button><span id="progress" class="pill"></span></div>
<div id="notice" role="status"></div></div>
<section id="context"></section><section id="fields"></section>
<div class="card"><h2>Coherent alternative binding sets</h2><p class="small">Each inner list is one complete valid alternative. Never merge incompatible alternatives. Leave exhaustive unchecked when open-ended fields prevent enumerating all valid choices.</p>
<textarea id="sets" rows="5"></textarea><label><input id="exhaustive" type="checkbox"> These alternatives exhaust the admissible dynamic binding sets</label>
<label>Cross-field constraints requiring separate judgment (JSON list)<textarea id="constraints" rows="2"></textarea></label></div>
<div class="card"><label><input id="confirm" type="checkbox"> I reviewed every field against the request and available evidence. Unjudgeable fields remain explicitly marked.</label>
<button class="primary" onclick="approve()">Save this case as human reviewed</button><p class="small">Saving records your name and review time. It does not claim a second human review or successful execution.</p></div>
</main>
<script id="packets" type="application/json">__PACKETS__</script>
<script>
'use strict';
const packets=JSON.parse(document.getElementById('packets').textContent);
const storageKey='farm-binding-review-'+packets[0].consensus.input_sha256.slice(0,12);
let stored={};try{stored=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(e){}
let index=0,working;
const $=id=>document.getElementById(id),clone=x=>JSON.parse(JSON.stringify(x));
function el(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(stored))}catch(e){$('notice').textContent='Local save is unavailable. Export your review file before leaving.'}}
packets.forEach((p,i)=>{const o=el('option',String(i+1).padStart(3,'0')+' · '+p.case.query.slice(0,75));o.value=i;$('caseSelect').append(o)});
$('caseSelect').onchange=()=>{index=Number($('caseSelect').value);render()};
function move(step){index=Math.max(0,Math.min(packets.length-1,index+step));render()}
function proposal(p,side,slug){return Array.isArray(p?.reference?.fields)?p.reference.fields.find(f=>f.side===side&&f.field===slug):undefined}
function useProposal(side,slug,label){const f=proposal(packets[index].proposals[label],side,slug);if(!f)return;const pos=working.fields.findIndex(x=>x.side===side&&x.field===slug);working.fields[pos]=clone(f);renderFields()}
function readFieldEditors(){working.fields.forEach((f,i)=>{f.status=$('status'+i).value;f.acceptable=JSON.parse($('accepted'+i).value);f.rationale=$('rationale'+i).value;f.evidence_quotes=JSON.parse($('evidence'+i).value)})}
function renderFields(){
 const target=$('fields');target.replaceChildren();
 working.fields.forEach((f,i)=>{
  const card=el('article',undefined,'card');card.append(el('h2',f.side+' · '+f.field));
  const ep=packets[index].case.endpoints[f.side], meta=ep.fields.find(x=>x.slug===f.field);
  card.append(el('p','Label: '+meta.label+' · Required: '+String(meta.required),'muted'));if(meta.help_text)card.append(el('p',meta.help_text));
  const grid=el('div',undefined,'grid');
  ['A','B'].forEach(label=>{const panel=el('div',undefined,'proposal'),r=proposal(packets[index].proposals[label],f.side,f.field);panel.append(el('h3','AI proposal '+label));
   panel.append(el('pre',r?JSON.stringify(r,null,2):'Missing or malformed proposal'));
   const b=el('button','Use proposal '+label);b.onclick=()=>{try{readFieldEditors();useProposal(f.side,f.field,label)}catch(e){$('notice').textContent='Fix malformed JSON before switching proposals.'}};panel.append(b);grid.append(panel)});card.append(grid);
  const lab=el('label','Reviewed status');const select=el('select');select.id='status'+i;['closed','missing_context','open_ended','unjudgeable'].forEach(s=>{let o=el('option',s);o.value=s;select.append(o)});select.value=f.status;lab.append(select);card.append(lab);
  [['accepted','Acceptable alternatives (JSON list)',JSON.stringify(f.acceptable,null,2),4],['rationale','Reason',f.rationale||'',2],['evidence','Evidence quotations (JSON list of exact source excerpts)',JSON.stringify(f.evidence_quotes||[],null,2),3]].forEach(([id,title,value,rows])=>{const l=el('label',title),ta=el('textarea');ta.id=id+i;ta.rows=rows;ta.value=value;l.append(ta);card.append(l)});
  card.append(el('p','Examples: [{"kind":"trigger_output","ingredient_slug":"Body"}] · [{"kind":"literal","value":"green"}] · [{"kind":"needs_input"}] · [{"kind":"omit"}]','small muted'));
  target.append(card);
 });
}
function render(){
 const p=packets[index],saved=stored[p.case.case_id];working=clone(saved?.reference||p.consensus.reference);
 $('caseSelect').value=index;$('confirm').checked=false;$('notice').textContent=saved?.reference_state==='human_reviewed'?'Previously reviewed; saving again replaces that review.':'';
 $('progress').textContent=Object.values(stored).filter(x=>x.reference_state==='human_reviewed').length+' / 150 reviewed';
 const c=$('context');c.replaceChildren();const card=el('article',undefined,'card');card.append(el('h2','Request '+p.case.case_number));card.append(el('p',p.case.query));
 const endpoints=el('div',undefined,'grid');['trigger','action'].forEach(side=>{let ep=p.case.endpoints[side],box=el('div');box.append(el('h3',side+': '+ep.service+' — '+ep.function));box.append(el('p',ep.description));let a=el('a','Official endpoint documentation');a.href=ep.endpoint_id;a.target='_blank';a.rel='noopener noreferrer';box.append(a);
 if(side==='trigger')box.append(el('pre',JSON.stringify(ep.ingredients,null,2)));
 const docs=el('details');docs.append(el('summary','Archived documentation and reference warnings'));docs.append(el('pre',p.case.official_documentation[side].text||'Documentation unavailable'));
 const missing=p.case.official_documentation[side].frozen_field_slugs_absent_from_page;if(missing.length)docs.append(el('p','Frozen fields absent from current page: '+missing.join(', '),'warning'));box.append(docs);endpoints.append(box)});
 card.append(endpoints);let warning=[];['A','B'].forEach(l=>{const issues=p.proposals[l].validation_errors;if(issues.length)warning.push('Proposal '+l+': '+issues.join('; '))});if(warning.length)card.append(el('pre',warning.join('\n'),'warning small'));c.append(card);
 renderFields();$('sets').value=JSON.stringify(working.binding_sets||[],null,2);$('exhaustive').checked=working.binding_sets_exhaustive===true;$('constraints').value=JSON.stringify(working.cross_field_constraints||[],null,2);
}
function approve(){
 try{
  if(!$('reviewer').value.trim())throw Error('Enter the actual reviewer name.');
  if(!$('confirm').checked)throw Error('Confirm that you reviewed every field.');
  readFieldEditors();working.binding_sets=JSON.parse($('sets').value);working.binding_sets_exhaustive=$('exhaustive').checked;working.cross_field_constraints=JSON.parse($('constraints').value);
  for(const f of working.fields){if(!Array.isArray(f.acceptable)||!Array.isArray(f.evidence_quotes))throw Error('Alternatives and evidence must be JSON lists.');if(['closed','missing_context'].includes(f.status)&&(!f.acceptable.length||!f.evidence_quotes.length))throw Error(f.field+' needs explicit alternatives and evidence.');}
  const base=clone(packets[index].consensus);base.reference=clone(working);base.reference_state='human_reviewed';base.human_review={reviewer_name:$('reviewer').value.trim(),reviewed_at_utc:new Date().toISOString(),confirmed_all_fields:true};stored[base.case_id]=base;persist();render();$('notice').textContent='Review saved locally. Export the review file to retain a portable copy.';
 }catch(e){$('notice').textContent=e.message}
}
function exportReviews(){const items=packets.map(p=>stored[p.case.case_id]||p.consensus);const blob=new Blob([JSON.stringify(items,null,2)],{type:'application/json'});const a=el('a');a.href=URL.createObjectURL(blob);a.download='HUMAN_REVIEWED.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),1000);}
$('importFile').onchange=async()=>{try{const values=JSON.parse(await $('importFile').files[0].text());for(const v of values){const p=packets.find(p=>p.case.case_id===v.case_id);if(!p||p.consensus.input_sha256!==v.input_sha256)throw Error('The review belongs to different frozen inputs.');stored[v.case_id]=v;}persist();render();$('notice').textContent='Review file imported.'}catch(e){$('notice').textContent=e.message}};
render();
</script></html>'''

if __name__ == '__main__':
    packets = read(ROOT / 'private/REVIEW_PACKETS.json')
    assert len(packets) == 150
    raw = canonical(packets).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    out = ROOT / 'private/REVIEW_BINDINGS.html'
    out.write_text(HTML.replace('__PACKETS__', raw))
    out.chmod(0o600)
    print('Created blinded local review page for 150 cases, without workflow outputs.')
