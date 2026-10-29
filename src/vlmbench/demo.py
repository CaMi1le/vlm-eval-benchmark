"""Generate original synthetic inputs and scripted responses; never inference."""
import json
from pathlib import Path
from PIL import Image, ImageDraw
from .common import json_hash, prompt, sha, write
from .public_workflow import build_report


def demo(output):
    root=Path(output);root.mkdir(parents=True,exist_ok=False)
    suite=root/'suite';(suite/'media').mkdir(parents=True)
    cases=[];references=[]
    for dataset in ['MMStar','BLINK','OCRBench','ChartQA','POPE','MVBench']:
        for i in range(24):
            path=suite/'media'/f'{dataset}-{i}.png'
            im=Image.new('RGB',(256,128),'white');draw=ImageDraw.Draw(im)
            draw.rectangle((16,16,112,112),fill='blue');draw.text((128,40),f'DEMO {i}',fill='black');im.save(path)
            media=[{'path':str(path.relative_to(suite)),'sha256':sha(path),'width':256,'height':128}]
            if dataset=='OCRBench':task,question,choices,answer='ocr','Read the text.',[],f'DEMO {i}'
            elif dataset=='ChartQA':task,question,choices,answer='chart','Read the number after DEMO.',[],str(i)
            elif dataset=='POPE':task,question,choices,answer='yesno','Is there a blue square?',[],'yes'
            else:task,question,choices,answer='mcq','What color is the square?',['blue','red'],'A'
            if dataset=='MVBench':media=[{**media[0],'timestamp_seconds':float(t)} for t in range(8)]
            case={'id':f'{dataset}:demo-{i}','dataset':dataset,'category':'synthetic-demo','task':task,
                  'question':question,'choices':choices,'prompt':prompt(question,choices,task),'media':media}
            case['input_sha256']=json_hash({k:case[k] for k in ['question','choices','media','task','prompt']})
            cases.append(case);references.append({'id':case['id'],'answers':[answer]})
    for name,values in [('cases',cases),('references',references)]:
        (suite/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))
    ids=[c['id'] for c in cases];diagnostics=ids[::24]
    write(suite/'views.json',{'core':ids,'smoke':ids[::6],'regression':ids[:24],'delta':ids[24:],'no_image_diagnostic':diagnostics})
    write(suite/'freeze.json',{'suite':'synthetic-demo','case_count':len(cases),'files':[
        {'file':str(p.relative_to(suite)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(suite.rglob('*')) if p.is_file()]})
    runs=[]
    for model in ['demo-scripted-perfect','demo-scripted-noisy']:
        run=root/'runs'/model;run.mkdir(parents=True);runs.append(run);rows=[]
        for condition,selected in [('image',cases),('no_image',cases[::24])]:
            for i,c in enumerate(selected):
                answer=references[cases.index(c)]['answers'][0]
                if model.endswith('noisy') and i%4==0:answer='unresolved synthetic response'
                rows.append({'id':c['id'],'model':model,'condition':condition,'dataset':c['dataset'],
                    'category':c['category'],'input_sha256':c['input_sha256'],'answer':answer,'status':'ok',
                    'total_seconds':0.25 if model.endswith('perfect') else 0.5,'generated_tokens':1})
        write(run/'run.json',{'model':model,'suite_sha256':sha(suite/'freeze.json'),
            'config':{'backend':'synthetic_demo','identity_note':'SCRIPTED DEMO: no model inference; timings are synthetic.'}})
        write(run/'planned.json',[{k:r[k] for k in ['id','condition','input_sha256']} for r in rows])
        (run/'outputs.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=build_report(suite,runs,root/'report')
    (root/'README.md').write_text('# Synthetic demo\n\nAll images, questions, responses and timings are synthetic. No model ran.\n\nOpen report/review.html or report/dashboard/dashboard.html.\n')
    return {**result,'synthetic':True,'model_calls':0}
