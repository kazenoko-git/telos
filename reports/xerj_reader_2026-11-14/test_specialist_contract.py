"""Offline boundary tests for extraction grading and deterministic execution."""
import copy
import json
from pathlib import Path
import unittest
from specialist_contract import execute, grade_extraction, public_plan
ROOT=Path(__file__).resolve().parent


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.plan={'operation':'subtract','output_type':'number','fields':[{'name':'initial'},{'name':'removed'}],
                   'reference_fields':{'initial':['9'],'removed':['4']}}
    def test_executor_uses_only_extracted_literals(self):
        public=public_plan(self.plan)
        self.assertNotIn('reference_fields',public)
        self.assertEqual(execute('Initially 18; removed 7.',public,{'initial':['18'],'removed':['7']}),{'status':'ANSWER','value':'11'})
        changed=copy.deepcopy(self.plan);changed['reference_fields']={'initial':['999'],'removed':['0']}
        self.assertEqual(execute('Initially 18; removed 7.',public_plan(changed),{'initial':['18'],'removed':['7']})['value'],'11')
    def test_fabricated_and_missing_operands_rejected(self):
        self.assertEqual(execute('Initially 9; removed 4.',public_plan(self.plan),{'initial':['10'],'removed':['4']})['status'],'CONTRACT_ERROR')
        self.assertEqual(execute('Initially 9; removed 4.',public_plan(self.plan),{'initial':['9']})['status'],'CONTRACT_ERROR')
    def test_extraction_does_not_credit_correct_arithmetic(self):
        g=grade_extraction('Initially 9; removed 4. Result 5.',self.plan,{'initial':['5'],'removed':[]})
        self.assertFalse(g['exact_fields'])
    def test_competing_values_and_absence_use_same_resolver(self):
        plan={'operation':'resolve','output_type':'number','fields':[{'name':'values'}]}
        self.assertEqual(execute('Records 4 and 7.',plan,{'values':['4','7']})['status'],'CONFLICT')
        self.assertEqual(execute('No recorded quantity.',plan,{'values':[]})['status'],'ABSENT')
        self.assertEqual(execute('Recorded 7.',plan,{'values':['7']})['value'],'7')
    def test_ordered_lists_preserve_order(self):
        plan={'operation':'list','fields':[{'name':'values'}],'output_type':'list','reference_fields':{'values':['alpha','beta']}}
        self.assertFalse(grade_extraction('alpha beta',plan,{'values':['beta','alpha']})['exact_fields'])
    def test_all_frozen_references_are_grounded(self):
        cases={c['id']:c for c in json.loads((ROOT/'suite_512_cases.json').read_text())}
        plans=json.loads((ROOT/'pipeline_task_plans.json').read_text())['cases']
        self.assertEqual(len(plans),512)
        for plan in plans:
            result=grade_extraction(cases[plan['case_id']]['document'],plan,plan['reference_fields'])
            self.assertTrue(result['exact_fields'] and result['grounded'],plan['case_id'])

if __name__=='__main__':unittest.main()
