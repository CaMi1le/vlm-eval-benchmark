"""Offline response gallery template and measured latency statistics."""
import math
import statistics

def quantile(values,p):
    values=sorted(values)
    if not values:return None
    index=(len(values)-1)*p;lower=math.floor(index);upper=math.ceil(index)
    return values[lower]+(values[upper]-values[lower])*(index-lower)

def timing(rows):
    ok=[r for r in rows if r['status']=='ok'];times=[r['total_seconds'] for r in ok]
    first=[r['ttft_seconds'] for r in ok if r.get('ttft_seconds') is not None]
    observed_generation=[r for r in ok if isinstance(r.get('generation_seconds'),(int,float))
                         and r['generation_seconds']>0 and isinstance(r.get('generated_tokens'),(int,float))]
    generation=sum(r['generation_seconds'] for r in observed_generation)
    tokens=[r['generated_tokens'] for r in ok if isinstance(r.get('generated_tokens'),(int,float))]
    return {'planned':len(rows),'ok':len(ok),'failed':len(rows)-len(ok),
            'empty_answers':sum(not isinstance(r.get('answer'),str) or not r['answer'].strip() for r in ok),
            'mean_seconds':statistics.mean(times) if times else None,'median_seconds':quantile(times,.5),
            'p95_seconds':quantile(times,.95),'total_request_seconds':sum(times),
            'ttft_observed':len(first),'ttft_median_seconds':quantile(first,.5),
            'generated_tokens':sum(tokens) if tokens else None,
            'generation_tokens_per_second':sum(r['generated_tokens'] for r in observed_generation)/generation if generation else None,
            'length_limited':sum(bool(r.get('length_limit_reached')) for r in ok),
            'time_limited':sum(bool(r.get('time_limit_reached')) for r in ok)}

TEMPLATE='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>1,440 题四模型原图与回答</title><style>
body{max-width:1500px;margin:24px auto;padding:0 20px;background:#f5f7fa;color:#182433;font:15px/1.6 system-ui}header{position:sticky;top:0;background:#f5f7faf5;padding:12px 0;z-index:1}select,input,button{padding:8px;margin:3px}article{background:white;border:1px solid #d9e1eb;border-radius:8px;padding:18px;margin:20px 0}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui}.media{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;max-width:31%;min-width:140px}img{max-width:100%;max-height:300px;object-fit:contain}figcaption,small{color:#526177;font-size:12px}.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(290px,1fr));gap:10px}.answer{border:1px solid #d9e1eb;padding:12px;border-radius:5px}table{border-collapse:collapse}td,th{padding:8px 14px;border-bottom:1px solid #cdd5df;text-align:left}dialog{max-width:95vw;max-height:95vh}dialog img{max-width:90vw;max-height:80vh}.error{color:#ad2525}
</style><h1>1,440 题四模型原图与回答</h1><p>全部原始回答；参考标签与答案均待本轮内容审查，本页不判对错。完整 1,440 题包含旧 144 题。视频按统一 8 帧输入。每个模型独占一张卡并行运行，预热、独立冒烟和无图诊断不计入主测速度。</p><table id="summary"></table><p>原图随本页保存在 media 文件夹中，移动时请保留整个目录。点击图片查看原始分辨率。</p>
<header><input id="search" placeholder="搜索题号、题目、类别"><select id="dataset"><option value="">全部数据集</option></select><select id="view"><option value="core">全部 1440 题</option><option value="regression">旧 144 题</option><option value="delta">新增 1296 题</option></select><select id="filter"><option value="all">全部输出</option><option value="error">有请求失败</option><option value="length">有回答达到长度上限</option></select><button id="prev">上一页</button><button id="next">下一页</button><span id="count"></span></header><main id="results"></main>
<dialog id="image"><button id="close">关闭</button><p id="caption"></p><img id="original"></dialog>
<script id="data" type="application/json">PAYLOAD</script><script>
const data=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),size=12;let page=0;
const node=(tag,text,cls)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;if(cls)n.className=cls;return n};
const head=node('tr');for(const text of ['模型','部署 / 并发','完成','平均 / 秒','中位数 / 秒','P95 / 秒'])head.append(node('th',text));$('summary').append(head);
for(const model of data.models){const p=data.performance[model],s=p.core,tr=node('tr'),setting=p.config?.backend==='openai_compatible_api'?'API · '+p.config.concurrency+' 路 · 默认思考':(p.host??'—')+' · GPU '+(p.gpus??[]).join(',')+' · 单请求';for(const value of [model,setting,s.ok+'/'+s.planned,...[s.mean_seconds,s.median_seconds,s.p95_seconds].map(v=>v==null?'—':v.toFixed(3))])tr.append(node('td',value));$('summary').append(tr)}
for(const model of data.models){const p=data.performance[model];if(p.config?.backend==='openai_compatible_api'){const wall=p.formal_wall_seconds==null?'未观测':(p.formal_wall_seconds/60).toFixed(1)+' 分钟';$('summary').after(node('p',model+' 正式请求墙钟：'+wall+'；单题延迟与并发吞吐分别记录。completion 与 reasoning 用量不假定可相加。'))}}
for(const name of [...new Set(data.cases.map(c=>c.dataset))]){const o=node('option',name);o.value=name;$('dataset').append(o)}
function render(){const q=$('search').value.toLowerCase(),ds=$('dataset').value,view=new Set(data.views[$('view').value]),filter=$('filter').value;
const rows=data.cases.filter(c=>(!ds||c.dataset===ds)&&view.has(c.id)&&(!q||[c.id,c.prompt,c.category].join(' ').toLowerCase().includes(q))&&(filter==='all'||filter==='error'&&data.models.some(m=>c.models[m].image.status!=='ok')||filter==='length'&&data.models.some(m=>c.models[m].image.length_limit_reached)));
page=Math.max(0,Math.min(page,Math.ceil(rows.length/size)-1));$('count').textContent=rows.length+' 题 · '+(page+1)+' / '+Math.max(1,Math.ceil(rows.length/size))+' 页';$('prev').disabled=page===0;$('next').disabled=(page+1)*size>=rows.length;$('results').replaceChildren();
for(const c of rows.slice(page*size,(page+1)*size)){const a=node('article');a.append(node('h2',c.id),node('small',c.dataset+' · '+c.category+' · '+(data.views.regression.includes(c.id)?'旧题':'新增题')),node('pre',c.prompt));const media=node('div',undefined,'media');
for(const m of c.media){const f=node('figure'),link=node('a'),img=node('img');link.href=m.path;link.target='_blank';img.src=m.path;img.loading='lazy';img.alt=c.id;link.append(img);link.addEventListener('click',e=>{e.preventDefault();$('original').src=m.path;$('caption').textContent=c.id+' · '+m.width+' × '+m.height;$('image').showModal()});f.append(link,node('figcaption',m.timestamp_seconds===undefined?m.width+' × '+m.height:m.timestamp_seconds.toFixed(3)+' 秒'));media.append(f)}a.append(media,node('pre','原始参考标签（待审查）：'+JSON.stringify(c.reference.answers)));const answers=node('div',undefined,'answers');
for(const model of data.models){const r=c.models[model].image,b=node('div',undefined,'answer');b.append(node('strong',model),node('pre',r.answer||r.error||(r.status==='ok'?'（空回答）':r.status),r.status==='ok'?'':'error'),node('small','状态 '+r.status+' · '+(r.total_seconds==null?'—':r.total_seconds.toFixed(3))+' 秒 · '+(r.generated_tokens??'—')+' tokens'+(r.length_limit_reached?' · 达到长度上限':'')));if(r.recovery)b.append(node('small','补测：无客户端时限 · 首轮 '+r.recovery.first_attempt_status+' / '+r.recovery.first_attempt_seconds.toFixed(3)+' 秒；当前耗时为补测单次延迟。'));if(r.api_usage)b.append(node('small',' API reasoning '+(r.reasoning_tokens??'—')+' tokens · response model '+(r.response_model??'未提供')));const no=c.models[model].no_image;if(no){b.append(node('pre','独立无图诊断：'+(no.answer||no.error||(no.status==='ok'?'（空回答）':no.status))));if(no.recovery)b.append(node('small','无图诊断也经无客户端时限补测；首轮 '+no.recovery.first_attempt_seconds.toFixed(3)+' 秒超时，补测 '+(no.total_seconds==null?'—':no.total_seconds.toFixed(3))+' 秒。'));}answers.append(b)}a.append(answers,node('small','输入 SHA256：'+c.input_sha256));$('results').append(a)}}
for(const id of ['search','dataset','view','filter'])$(id).addEventListener('input',()=>{page=0;render()});$('prev').onclick=()=>{page--;render();window.scrollTo(0,0)};$('next').onclick=()=>{page++;render();window.scrollTo(0,0)};$('close').onclick=()=>$('image').close();$('image').addEventListener('close',()=>$('original').removeAttribute('src'));render();
</script></html>'''
