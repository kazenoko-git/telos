"""Optional real-browser regression for comparison tracks and phone layout."""
import os
import pytest


def test_phone_comparison_tracks_and_safe_case_details(tmp_path):
    browser_api=pytest.importorskip('playwright.sync_api')
    from telos.eval.display import write_html_report
    hostile='</script><script>window.injected=true</script>'
    def row(model,track,value):
        return dict(id='one',model=model,suite=track,category='scalar',correct=True,
                    status='ANSWER',value=value,expected=value,question=hostile,
                    confidence=None,latency_ms=None,document='Source text',evidence=[])
    rows=[row('A','direct_qa',1),row('B','direct_qa',1),row('A','extraction',{'value':['1']}),row('A','calculator',1)]
    report=dict(schema='telos.evaluation.display.v1',rows=rows,note='Synthetic tasks',
                suites=[dict(model=r['model'],suite=r['suite'],score=100,total=1) for r in rows],
                presentation=dict(featured_model='A',default_suite='direct_qa'))
    path=write_html_report(report,tmp_path/'report.html')
    with browser_api.sync_playwright() as api:
        options={'headless':True}
        if os.environ.get('TELOS_CHROMIUM_EXECUTABLE'):
            options['executable_path']=os.environ['TELOS_CHROMIUM_EXECUTABLE']
        with api.chromium.launch(**options) as browser:
            page=browser.new_page(viewport={'width':390,'height':844},is_mobile=True,has_touch=True)
            errors=[]
            page.on('pageerror',lambda error:errors.append(str(error)))
            page.goto(path.as_uri())
            page.wait_for_function('window.telosReportReady')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            assert page.locator('.rank-row').count()==2
            assert page.locator('#metrics').is_hidden()
            page.get_by_role('button',name='Source extraction',exact=True).click()
            assert page.locator('.rank-row').count()==1
            assert page.locator('#rows tr').count()==1
            page.get_by_role('button',name='Calculator',exact=True).click()
            assert page.locator('#suite').input_value()=='calculator'
            page.locator('#rows button').click()
            assert page.locator('dialog').is_visible()
            assert page.locator('#detail-title').inner_text()==hostile
            assert page.evaluate('window.injected === undefined')
            assert not errors
