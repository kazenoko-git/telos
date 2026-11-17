import math
import random

from scripts.compare_mlx_switchback import schedule_modes
from scripts.analyze_mlx_switchback import hardware_block,matched_pair,paired_summary


def test_schedule_is_balanced_reproducible_and_does_not_change_global_rng():
    before=random.getstate()
    modes=schedule_modes(24,17)
    assert random.getstate()==before
    assert modes==schedule_modes(24,17)
    assert sum(modes)==24
    assert all(a!=b for a,b in zip(modes[::2],modes[1::2]))
    assert sum(modes[::2])==12
    assert all(a!=b for a,b in zip(modes,[not x for x in modes]))


def sample(start,end,mhz,active,cpu):
    return dict(window_start_epoch_ns=start,window_end_epoch_ns=end,
        interval_seconds=(end-start)/1e9,gpu_active_frequency_mhz=mhz,
        gpu_frequency_mapping_coverage=1,gpu_active_residency=active,
        thermal_label='fair',power_source="Now drawing from 'AC Power'",
        processes=f'1 {cpu} 100 /System/Applications/Screen Sharing\n2 99 100 /tmp/python',
        performance_states={'GPUPH':[('OFF',0),('P1',100)]})


def test_hardware_intervals_must_be_wholly_inside_block_and_use_active_weight():
    rows=[sample(0,10**9,1000,.5,10),sample(10**9,2*10**9,1020,1,20)]
    result=hardware_block(rows,0,2*10**9)
    assert math.isclose(result['frequency_mhz'],(500+1020)/1.5)
    assert result['median_top_other_cpu_pct']==15
    assert result['all_ac']
    clipped=hardware_block(rows,1,2*10**9-1)
    assert clipped['full_intervals']==0
    assert 'frequency_mhz' not in clipped


def test_matching_rejects_clock_thermal_and_background_mismatches():
    policy=dict(min_full_hardware_intervals_per_block=6,max_block_frequency_cv=.02,
        max_pair_frequency_relative_difference=.01,max_pair_other_cpu_pct_difference=15)
    base=dict(full_intervals=10,mapped_intervals=10,frequency_mhz=1000,frequency_cv=.005,
        thermal_states=['fair'],all_ac=True,median_top_other_cpu_pct=80,active_state_distribution={'P1':1})
    sync={'hardware':base}
    assert matched_pair(sync,{'hardware':{**base,'frequency_mhz':1005}},policy)['matched']
    for override,reason in [({'frequency_mhz':1020},'frequency mismatch'),
            ({'thermal_states':['nominal']},'thermal-state mismatch'),
            ({'median_top_other_cpu_pct':96},'background CPU mismatch')]:
        result=matched_pair(sync,{'hardware':{**base,**override}},policy)
        assert not result['matched'] and reason in result['rejections']


def test_paired_bootstrap_uses_pairs_and_reports_positive_constant_effect():
    result=paired_summary([dict(sync_rate=100,async_rate=105)]*8)
    assert math.isclose(result['geometric_gain_pct'],5)
    assert all(math.isclose(x,5) for x in result['bootstrap_95_pct'])
    assert result['pairs']==result['positive_pairs']==8
    assert paired_summary([])=={'pairs':0}


def test_one_pair_does_not_report_a_zero_width_confidence_interval():
    result=paired_summary([dict(sync_rate=100,async_rate=105)])
    assert result['pairs']==1 and result['bootstrap_95_pct'] is None
