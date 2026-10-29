"""Portable, offline visual review: original media, labels, and raw answers."""
import base64
import hashlib
import json
import mimetypes
from pathlib import Path
from .common import jsonl

def gallery(suite, scores, models):
    suite=Path(suite).resolve();out=Path(scores).resolve();cases=jsonl(suite/'cases.jsonl')
    bymodel={m:{(r['id'],r['condition']):r for r in jsonl(out/m/'scored.jsonl')} for m in models}
    entries=[];assets={}
    for case in cases:
        if not all((case['id'],'image') in bymodel[m] for m in models):continue
        for media in case['media']:
            if media['sha256'] in assets:continue
            path=suite/media['path'];content=path.read_bytes()
            assert hashlib.sha256(content).hexdigest()==media['sha256'],f'Image hash mismatch: {path}'
            mime=mimetypes.guess_type(path.name)[0]
            assert mime and mime.startswith('image/'),f'Unknown image type: {path}'
            assets[media['sha256']]=f'data:{mime};base64,'+base64.b64encode(content).decode('ascii')
        entries.append({**case,'models':{model:{'image':bymodel[model][(case['id'],'image')],
                       'no_image':bymodel[model].get((case['id'],'no_image'))} for model in models}})
    policy_path=out/'review-policy.json'
    policy=json.loads(policy_path.read_text()) if policy_path.exists() else None
    contexts={}
    for model in models:
        summary_path=out/model/'summary.json'
        if summary_path.exists():contexts[model]=json.loads(summary_path.read_text()).get('execution_context',{})
    payload=json.dumps({'models':models,'cases':entries,'assets':assets,'review_policy':policy,'execution_contexts':contexts},ensure_ascii=False).replace('&','\\u0026').replace('<','\\u003c').replace('>','\\u003e')
    template='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>VLM 逐题核查</title><style>
body{font:15px/1.6 system-ui,sans-serif;background:#f5f7fa;color:#182433;margin:0 auto;padding:24px;max-width:1400px}h1{margin-bottom:6px}header{position:sticky;top:0;background:#f5f7faf5;padding:8px 0;z-index:2}input,select{padding:9px;margin-right:8px;border:1px solid #bbc5d1;border-radius:6px}article{background:white;padding:20px;border:1px solid #d9e1eb;border-radius:9px;margin:18px 0}h2{font-size:19px;margin:0}.meta{color:#526177}pre{white-space:pre-wrap;word-break:break-word;font:14px/1.6 system-ui;margin:8px 0}.media{display:flex;flex-wrap:wrap;gap:10px}.media figure{margin:0;max-width:31%;min-width:180px}.media img{max-width:100%;max-height:340px;object-fit:contain;background:#eee}.media figcaption{font-size:12px}.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}.answer{padding:12px;border:1px solid #d6dce5;border-radius:7px}.pass{color:#187346}.fail{color:#b82b32}.gold{padding:8px;background:#edf2fa}small{display:block;color:#526177}a{color:#1765a8}
dialog{max-width:96vw;max-height:95vh;border:1px solid #bbc5d1;border-radius:9px;padding:16px}dialog::backdrop{background:#0009}dialog img{display:block;max-width:90vw;max-height:78vh;object-fit:contain}dialog button{float:right;padding:6px 14px;cursor:pointer}dialog p{margin:8px 0}details pre{max-height:300px;overflow:auto}
</style><h1>VLM 逐题核查</h1><p>同一冻结输入、复核后的参考答案与原始输出。绿色表示内容正确，灰色条目对全部模型统一不计分。视频为统一抽取的 8 帧。全部原图已内嵌，可单独保存本 HTML 离线查看；点击图片放大。每个判定附复核依据；表达差异单独标注，不产生另一套分数。</p><p id="api-note" hidden>API 模型使用所选别名的默认思考设置，已有本地基线关闭思考。API 完整耗时包含代理、网络和远端处理。模型名是代理返回的 ID，底层版本未经独立核实；两类运行的预算与耗时口径不同。</p><details><summary>运行设置</summary><pre id="contexts"></pre></details><details><summary>答案处理规则</summary><pre id="policy"></pre></details>
<header><input id="search" placeholder="搜索 ID、题目、类别"><select id="dataset"><option value="">全部数据集</option></select><select id="filter"><option value="all">全部结果</option><option value="different">模型得分不同</option><option value="allwrong">全部模型未通过</option><option value="truncated">至少一个模型截断</option><option value="excluded">统一不计分</option></select><span id="count"></span></header><main id="results"></main>
<dialog id="image-view"><button id="close-image" type="button">关闭</button><p id="image-caption"></p><img id="original" alt=""></dialog>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent), $=id=>document.getElementById(id);
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
$('policy').textContent=data.review_policy?JSON.stringify(data.review_policy,null,2):'本报告未提供额外答案处理规则。';
$('contexts').textContent=JSON.stringify(data.execution_contexts,null,2);$('api-note').hidden=!Object.values(data.execution_contexts).some(c=>c.backend==='openai_compatible_api');
$('close-image').addEventListener('click',()=>$('image-view').close());
$('image-view').addEventListener('close',()=>$('original').removeAttribute('src'));
for(const name of [...new Set(data.cases.map(c=>c.dataset))]){const o=node('option',name);o.value=name;$('dataset').append(o)}
function render(){const q=$('search').value.toLowerCase(), ds=$('dataset').value, filter=$('filter').value;
const rows=data.cases.filter(c=>{const rs=data.models.map(m=>c.models[m].image);return (!ds||c.dataset===ds)&&(!q||[c.id,c.question,c.category].join(' ').toLowerCase().includes(q))&&(filter==='all'||filter==='different'&&new Set(rs.map(r=>r.correct)).size>1||filter==='allwrong'&&rs.every(r=>r.eligible&&!r.correct)||filter==='truncated'&&rs.some(r=>r.length_limit_reached)||filter==='excluded'&&!rs[0].eligible)});
$('count').textContent=rows.length+' 题';$('results').replaceChildren();for(const c of rows){const a=node('article');a.append(node('h2',c.id),node('div',c.dataset+' · '+c.category,'meta'),node('pre',c.prompt));const media=node('div',undefined,'media');
for(const m of c.media){const f=node('figure'),link=node('a'),img=node('img'),url=data.assets[m.sha256];link.href=url;link.addEventListener('click',event=>{event.preventDefault();$('original').src=url;$('original').alt=c.id;$('image-caption').textContent=c.id+' · '+m.width+' × '+m.height;$('image-view').showModal()});img.loading='lazy';img.src=url;img.alt=c.id;link.append(img);f.append(link,node('figcaption',m.timestamp_seconds!==undefined?m.timestamp_seconds.toFixed(3)+' 秒':m.width+' × '+m.height));media.append(f)}a.append(media);
const gold=c.models[data.models[0]].image.gold;a.append(node('div','复核参考答案：'+JSON.stringify(gold),'gold'));const reviewed=c.models[data.models[0]].image;a.append(node('pre',(reviewed.eligible?'计分':'统一不计分')+' · '+reviewed.case_review_reason));if(reviewed.case_decision==='correct_reference')a.append(node('small','原始标签：'+JSON.stringify(reviewed.original_gold)+'；已依据原图修正。'));const answers=node('div',undefined,'answers');
for(const m of data.models){const r=c.models[m].image,b=node('section',undefined,'answer');b.append(node('strong',m+' · '+(!r.eligible?'不计分':r.correct?'内容正确':'内容错误'),!r.eligible?'meta':r.correct?'pass':'fail'),node('pre',r.answer||r.error||r.status));b.append(node('small','状态 '+r.status+' · 格式 '+(r.format_ok===null?'不单列':r.format_ok?'合规':'不合规')+' · 完整耗时 '+(!Number.isFinite(r.total_seconds)?'—':r.total_seconds.toFixed(3)+' s')+' · 输出 '+(r.generated_tokens??'—')+' tokens'+(r.reasoning_tokens==null?'':' · 思考 '+r.reasoning_tokens+' tokens')+(r.length_limit_reached?' · 达到输出上限':'')));
b.append(node('small','复核解释：'+JSON.stringify(r.parsed)),node('small',r.review_reason));if(r.format_issue)b.append(node('small','表达或格式差异已标注；内容判定见上。'));const no=c.models[m].no_image;if(no)b.append(node('small','无图诊断：'+(no.answer||no.status)+'（'+(no.correct?'正确':'错误')+'）'));answers.append(b)}a.append(answers);a.append(node('small','输入 SHA256：'+c.input_sha256));$('results').append(a)}}
for(const id of ['search','dataset','filter'])$(id).addEventListener('input',render);render();
</script></html>'''
    (out/'review.html').write_text(template.replace('PAYLOAD',payload))

def markdown(summaries,out):
    out=Path(out);models=[s['model'] for s in summaries]
    lines=['# 复核后的固定题库成绩','',
           '[逐题原图、原始回答和复核依据](review.html) · [逐题 CSV](comparison.csv) · [汇总 JSON](comparison.json) · [内容评分规则](review-policy.json)','',
           '七个模型使用相同有效题集。仅保留一套内容成绩；表达或格式问题单独标注。原始图片、问题、提示词、回答及冻结标签未改，错标修正记录绑定原始哈希。','',
           '| 数据集 | '+' | '.join(models)+' |','|---|'+'---|'*len(models)]
    for dataset in summaries[0]['datasets']:
        vals=[s['datasets'][dataset] for s in summaries]
        lines.append('| '+dataset+' | '+' | '.join(f"{v['correct']}/{v['n']} ({v['accuracy']:.1%})" if v['n'] else '0/0' for v in vals)+' |')
    lines+=['','| 模型 | 正确 / 有效题数 | 原始有图请求成功 | 严格格式合规 | 完整耗时中位数 | P90 | 触及输出上限 |',
            '|---|---:|---:|---:|---:|---:|---:|']
    for summary in summaries:
        v=summary['overall'];d=v['total_seconds'];fmt=lambda x:'—' if x is None else f'{x:.3f} s'
        lines.append(f"| {summary['model']} | {v['correct']}/{v['n']} ({v['accuracy']:.1%}) | {v['ok']}/{v['run_n']} | {v['format_correct']}/{v['format_n']} | {fmt(d['median'])} | {fmt(d['p90'])} | {v['length_limit_reached']} |")
    lines+=['','耗时、输出上限和 token 统计仍覆盖原始完整运行，包含不计分条目；预热排除，无图诊断不计入主成绩。格式合规按原始有图选择题及是非题要求统计，不影响内容正确率。运行失败始终留在有效题分母。','',
            '| 统一不计分条目 | 依据 |','|---|---|']
    for case in summaries[0]['excluded_cases']:lines.append('| '+case['id']+' | '+case['reason']+' |')
    lines+=['','其余错标修正及逐回答复核记录见归档 review/ 目录。数值 ChartQA 保留 5% 相对容差；年份列表、字符 OCR 不使用数值容差。','']
    api=[s for s in summaries if s.get('api_token_usage')]
    if api:
        lines+=['Gemini high 启用代理默认思考，本地基线关闭思考；预算不同。API 完整耗时包含图片编码、代理、网络和服务端处理，不能视为同硬件推理速度。模型身份按代理 ID 记录，底层版本未经核实。','',
                '| API 模型 | 有 usage / 正式请求 | Prompt | Completion | Reasoning | Total |','|---|---:|---:|---:|---:|---:|']
        for summary in api:
            u=summary['api_token_usage'];lines.append('| '+summary['model']+' | '+str(u['requests_with_usage'])+'/'+str(u['planned_requests'])+' | '+' | '.join(str(u[k]['sum']) if u[k]['sum'] is not None else '—' for k in ['prompt_tokens','completion_tokens','reasoning_tokens','total_tokens'])+' |')
        lines+=['','Usage 沿用代理原字段，覆盖原始144有图＋6无图正式请求；未测量字段留空，不假定各字段相加关系。','']
    lines+=['这是内部固定子集复核，不代表完整公开榜单或模型总体排名。','']
    (out/'README.md').write_text('\n'.join(lines))
