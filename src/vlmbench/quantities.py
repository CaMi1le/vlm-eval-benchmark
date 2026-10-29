"""Conservative numeric parsing using input-bound chart units, without gold text search."""
import math
import re

NUMBER=r'[+-]?(?:\d{1,3}(?:[, ]\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?|[+-]?\.\d+'
SCALES={'thousand':1e3,'thousands':1e3,'million':1e6,'millions':1e6,
        'billion':1e9,'billions':1e9,'trillion':1e12,'trillions':1e12,'k':1e3,'m':1e6,'b':1e9}
UNITS={
    'usd':('USD',1),'u.s. dollar':('USD',1),'u.s. dollars':('USD',1),'us dollars':('USD',1),'dollars':('USD',1),
    'euro':('EUR',1),'euros':('EUR',1),'eur':('EUR',1),'gbp':('GBP',1),'british pounds':('GBP',1),
    'pounds':('GBP',1),'yuan':('CNY',1),'cny':('CNY',1),'yen':('JPY',1),'jpy':('JPY',1),
    'saudi riyal':('SAR',1),'saudi riyals':('SAR',1),'sar':('SAR',1),
    'people':('count',1),'persons':('count',1),'users':('count',1),'views':('count',1),'visitors':('count',1),
    'departures':('count',1),'cases':('count',1),'years':('year',1),'year':('year',1),
    'tonnes':('mass_tonne',1),'tonne':('mass_tonne',1),'tons':('mass_tonne',1),'metric tons':('mass_tonne',1),'t':('mass_tonne',1),
    'kg':('mass_tonne',.001),'kilograms':('mass_tonne',.001),'ounces':('mass_ounce',1),
    'hectares':('area_ha',1),'ha':('area_ha',1),'square meters':('area_m2',1),'square metres':('area_m2',1),
    'tonne-kilometers':('tonne_km',1),'tonne-kilometres':('tonne_km',1),
    't/ha':('yield_tonne_ha',1),'tonnes per hectare':('yield_tonne_ha',1),'kg/ha':('yield_tonne_ha',.001),
    'kg per hectare':('yield_tonne_ha',.001)}


def quantity(text, context=None, question=''):
    """Return possible physical values from declared unit conventions; no target input."""
    context=context or {};expected_dimension=context.get('dimension')
    value=str(text).strip().rstrip('.').strip().replace('\u2212','-')
    value=re.sub(r'^(?:approximately|approx\.?|about|around|roughly|~)\s*','',value,flags=re.I)
    if re.search(r'\bratio\b',question,re.I):
        match=re.fullmatch(r'\s*('+NUMBER+r')\s*[:/]\s*('+NUMBER+r')\s*',value)
        if match:
            a=float(match.group(1).replace(',','').replace(' ',''));b=float(match.group(2).replace(',','').replace(' ',''))
            return {'dimension':'ratio','values':[a/b],'unit_explicit':True} if b else None
    alternate=re.search(r'\s*\((?:or\s+|approximately\s+)(.*?)\)\s*$',value,re.I)
    if alternate:
        first=quantity(value[:alternate.start()],context,question)
        second=quantity(alternate.group(1),context,question)
        if first and second and first['dimension']==second['dimension'] and any(
            math.isclose(a,b,rel_tol=.001,abs_tol=1e-12) for a in first['values'] for b in second['values']):return first
        return None
    prefix=None
    if value.startswith(('$','€','£')):
        prefix={'$':'USD','€':'EUR','£':'GBP'}[value[0]];value=value[1:].strip()
    match=re.fullmatch('('+NUMBER+r')\s*(.*?)',value,re.I)
    if not match:return None
    number=float(match.group(1).replace(',','').replace(' ',''));suffix=match.group(2).lower().strip().rstrip('.')
    if not math.isfinite(number):return None
    if not suffix and prefix is None:
        scales=context.get('bare_scales',[context.get('reference_scale',1)])
        return {'dimension':expected_dimension or 'number','values':[number*s for s in scales],'unit_explicit':False}
    if suffix in ('%','percent','percentage points'):
        return {'dimension':'ratio','values':[number*.01],'unit_explicit':True}
    if re.fullmatch(r'per (?:one )?thousand (?:people|persons)',suffix):
        return {'dimension':'ratio','values':[number*.001],'unit_explicit':True}
    rate=re.fullmatch(r'deaths? per ([\d, ]+) live births',suffix)
    if rate:
        denominator=float(rate.group(1).replace(',','').replace(' ',''))
        return {'dimension':'deaths_per_birth','values':[number/denominator],'unit_explicit':True} if denominator else None
    scale=1
    scaled=re.match(r'^(thousands?|millions?|billions?|trillions?|[kmb])\b\s*',suffix)
    if scaled:
        token=scaled.group(1);scale=SCALES[token]
        scale*=context.get('abbreviation_scale_multiplier',{}).get(token,1)
        suffix=suffix[scaled.end():].strip()
    if suffix:
        unit=UNITS.get(suffix)
        if unit is None:return None
        dimension,factor=unit
        if prefix is not None and prefix!=dimension:return None
    else:dimension,factor=prefix or expected_dimension or 'number',1
    if dimension=='mass_tonne' and expected_dimension=='yield_tonne_ha':dimension=expected_dimension
    return {'dimension':dimension,'values':[number*scale*factor],'unit_explicit':True}


def compatible(prediction, target, context, question):
    predicted=quantity(prediction,context,question)
    reference=quantity(target,{'dimension':context['dimension'],'reference_scale':context['reference_scale']},question)
    if predicted is None or reference is None or predicted['dimension']!=reference['dimension']:return False
    exact=context['dimension']=='year'
    return any(abs(a-b)<= (0 if exact or b==0 else abs(b)*.05)+1e-12
               for a in predicted['values'] for b in reference['values'])
