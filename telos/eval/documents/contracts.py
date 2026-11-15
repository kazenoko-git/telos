"""Frozen extraction contract and deterministic executor; gold used only by graders."""
from datetime import datetime
from decimal import Decimal, InvalidOperation
import re

DATE_FORMATS = ('%d %B %Y', '%Y-%m-%d')


def public_plan(plan):
    return {k: v for k, v in plan.items() if k != 'reference_fields'}


def number(text):
    if not re.fullmatch(r'[-+]?\d+(?:\.\d+)?', text.strip()):
        raise ValueError('Operand is not a verbatim numeric literal')
    return Decimal(text.strip())


def date_value(text):
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text.strip(), fmt).date()
        except ValueError:
            pass
    raise ValueError('Unsupported date literal')


def execute(document, plan, fields):
    """Execute public, human-written task plan using model-extracted literals only."""
    try:
        known = {f['name'] for f in plan['fields']}
        if not isinstance(fields, dict) or set(fields) != known:
            raise ValueError('Field set does not match schema')
        for vals in fields.values():
            if not isinstance(vals, list) or any(not isinstance(v,str) or not v or v not in document for v in vals):
                raise ValueError('Extracted values must be nonempty verbatim document spans')
        def one(name):
            vals=fields[name]
            if len(vals)!=1: raise ValueError(f'Expected one {name} operand')
            return vals[0]
        def n(name): return number(one(name))
        op=plan['operation'];kind=plan['output_type']
        if op=='resolve':
            vals=list(dict.fromkeys(fields['values']))
            if not vals:return {'status':'ABSENT','value':None}
            if len(vals)>1:return {'status':'CONFLICT','value':None}
            if kind=='money':
                currencies=list(dict.fromkeys(fields['currency']))
                if len(currencies)!=1:raise ValueError('Missing or conflicting currency')
                value={'amount':str(number(vals[0])),'currency':currencies[0]}
            elif kind=='date':value=date_value(vals[0]).isoformat()
            else:value=str(number(vals[0]))
        elif op=='list':value=fields['values']
        elif op=='sum':value=str(sum((n(f['name']) for f in plan['fields']),Decimal(0)))
        elif op=='subtract':value=str(n('initial')-n('removed'))
        elif op=='add_subtract':value=str(n('initial')+n('added')-n('removed'))
        elif op=='multiply':value=str(n('instances')*n('per_instance'))
        elif op=='increase_pct':value=str((n('after')-n('before'))/n('before')*100)
        elif op=='decrease_pct':value=str((n('before')-n('after'))/n('before')*100)
        elif op=='share_pct':value=str(n('part')/n('total')*100)
        elif op=='elapsed_days':value=str((date_value(one('end'))-date_value(one('start'))).days)
        elif op=='rate':value=str(n('quantity')/n('duration'))
        elif op=='kb_to_mb':value=str(n('kilobytes')/Decimal(1024))
        else:raise ValueError('Unknown public operation')
        return {'status':'ANSWER','value':value}
    except (ValueError,KeyError,TypeError,InvalidOperation,ZeroDivisionError) as err:
        return {'status':'CONTRACT_ERROR','value':None,'error':str(err)}


def grade_extraction(document, plan, fields):
    expected=plan['reference_fields']
    if not isinstance(fields,dict) or any(not isinstance(v,list) or any(not isinstance(x,str) for x in v) for v in fields.values()):return {'exact_fields':False,'grounded':False,'matching_fields':0,'total_fields':len(expected)}
    grounded=all(isinstance(v,list) and all(isinstance(x,str) and bool(x) and x in document for x in v) for v in fields.values())
    count=0
    for name, ref in expected.items():
        actual=fields.get(name)
        if plan['operation']!='list' and isinstance(actual,list):
            match=sorted(set(actual))==sorted(set(ref)) and len(actual)==len(set(actual))
        else:match=actual==ref
        count+=bool(match)
    return {'exact_fields':count==len(expected) and set(fields)==set(expected),'grounded':grounded,'matching_fields':count,'total_fields':len(expected)}
