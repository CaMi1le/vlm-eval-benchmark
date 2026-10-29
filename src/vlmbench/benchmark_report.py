"""Scored public-label results with a portable relative-image viewer."""
import json
import os
from pathlib import Path
import shutil

from .benchmark_score import POLICY
from .viewer_base import TEMPLATE, timing
from .common import jsonl, sha


def gallery(suite, root, summaries, rows, deployment, introduction=None):
    suite,root=Path(suite),Path(root)
    cases=jsonl(suite/'cases.jsonl');refs={r['id']:r for r in jsonl(suite/'references.jsonl')}
    views=json.loads((suite/'views.json').read_text());assets=set()
    for case in cases:
        for media in case['media']:
            if media['path'] in assets:continue
            source=suite/media['path'];destination=(root/media['path']).resolve()
            assert destination.is_relative_to(root.resolve())
            destination.parent.mkdir(parents=True,exist_ok=True)
            if not destination.exists():
                try:os.link(source,destination)
                except OSError:shutil.copyfile(source,destination)
            assert sha(destination)==media['sha256']
            assets.add(media['path'])
    bymodel={m:{(r['id'],r['condition']):r for r in v} for m,v in rows.items()}
    entries=[{**c,'reference':refs[c['id']],'models':{m:{'image':bymodel[m][c['id'],'image'],
        'no_image':bymodel[m].get((c['id'],'no_image'))} for m in rows}} for c in cases]
    performance={}
    for summary in summaries:
        name=summary['model'];p=deployment.get(name,{})
        image=[{**r,'total_seconds':r['timing_seconds']} for r in rows[name] if r['condition']=='image']
        performance[name]={**p,'config':summary['execution_context'],'core':timing(image),'grade':summary['overall']}
    payload=json.dumps({'models':list(rows),'cases':entries,'views':views,'performance':performance,
        'scoring_protocol':POLICY['schema']},ensure_ascii=False).replace('&','\u0026').replace('<','\u003c').replace('>','\u003e')
    html=TEMPLATE.replace('旧 144 题', f"回归 {len(views.get('regression', []))} 题").replace('新增 1296 题', f"增量 {len(views.get('delta', []))} 题").replace('四模型',f'{len(rows)} 模型评分与原图').replace('1,440',f'{len(cases):,}').replace('1440',str(len(cases)))
    start=html.index('<p>全部原始回答');end=html.index('</p>',start)+4
    html=html[:start]+(f'<p>按冻结公开标签自动评分，全部 {len(cases):,} 题计分；统一提取最终答案，格式差异单列。'
        '未人工改标或剔题。绿色为本评分规则判对，红色为判错；解析和参考标签均可逐题核查。'
        '视频统一为 8 帧；API 默认思考与本地预算不同，补测题速度按累计请求耗时统计。'
        ' <a href="dashboard/dashboard.html">七维能力图</a> · <a href="scores/comparison.csv">逐题 CSV</a>'
        ' · <a href="scores/scoring-policy.json">评分规则</a></p>')+html[end:]
    html=html.replace("['模型','部署 / 并发','完成','平均 / 秒','中位数 / 秒','P95 / 秒']",
        "['模型','部署 / 并发','正确 / 题数','完成','累计请求平均 / 秒','累计请求中位 / 秒','P95 / 秒']")
    html=html.replace('[model,setting,s.ok+\'/\'+s.planned,',"[model,setting,p.grade.correct+'/'+p.grade.n,s.ok+'/'+s.planned,")
    html=html.replace('<option value="length">有回答达到长度上限</option>',
        '<option value="length">有回答达到长度上限</option><option value="different">模型判分不同</option>'
        '<option value="allwrong">所有模型判错</option><option value="unresolved">有答案未解析</option>'
        '<option value="wrong">所选模型判错</option>')
    html=html.replace('<button id="prev">','<select id="model"><option value="">判定模型：全部</option></select><button id="prev">')
    html=html.replace('function render(){',"for(const m of data.models){const o=node('option',m);o.value=m;$('model').append(o)}\nfunction render(){")
    old="filter==='all'||filter==='error'"
    new="filter==='all'||filter==='different'&&new Set(data.models.map(m=>c.models[m].image.correct)).size>1||filter==='allwrong'&&data.models.every(m=>!c.models[m].image.correct)||filter==='unresolved'&&data.models.some(m=>c.models[m].image.parsed===null)||filter==='wrong'&&($('model').value?[$('model').value]:data.models).some(m=>!c.models[m].image.correct)||filter==='error'"
    assert old in html;html=html.replace(old,new)
    html=html.replace("['search','dataset','view','filter']","['search','dataset','view','filter','model']")
    html=html.replace('原始参考标签（待审查）：','冻结公开标签：')
    html=html.replace("node('strong',model),node('pre'", "node('strong',model+' · '+(r.correct?'正确':'错误'),r.correct?'pass':'error'),node('pre'")
    html=html.replace("if(r.recovery)b.append", "b.append(node('small','提取答案：'+JSON.stringify(r.parsed)+' · '+r.extraction_method+(r.extraction_issue?' · '+r.extraction_issue:'')+' · 严格格式：'+(r.format_ok===null?'不单列':r.format_ok?'合规':'有差异')));if(r.recovery)b.append")
    html=html.replace("r.status==='ok'?'':'error'","r.correct?'pass':'error'")
    html=html.replace('.error{color:#ad2525}', '.error{color:#ad2525}.pass{color:#187346}')
    if introduction is not None:
        start=html.index('<h1>');start=html.index('<p>',start);end=html.index('</p>',start)+4
        html=html[:start]+'<p>'+introduction+'</p>'+html[end:]
    # Keep the actual final response easy to compare; raw token decoding and
    # reasoning remain accessible, with the same grading as the JSONL records.
    html=html.replace("node('pre',r.answer||r.error||", "node('pre',(r.final_answer_text??r.answer)||r.error||")
    marker="if(r.recovery)b.append"
    details="if(r.reasoning_text){const d=node('details');d.append(node('summary','思考过程（不计分）'),node('pre',r.reasoning_text));b.append(d)}if(r.raw_generated_text){const d=node('details');d.append(node('summary','原始生成文本（含特殊标记）'),node('pre',r.raw_generated_text));b.append(d)}"
    html=html.replace(marker,details+marker)
    (root/'review.html').write_text(html.replace('PAYLOAD',payload))
    return len(assets)
