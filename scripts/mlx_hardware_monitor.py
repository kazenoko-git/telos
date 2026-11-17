"""Read-only, non-root Apple Silicon telemetry; private IOReport is best effort.

Frequency mapping follows the pmgr/GPUPH layout documented in macmon's sources:
https://github.com/vladkens/macmon/blob/main/src/{sources,metrics}.rs
Energy Model readings are estimates and can freeze on macOS 27. Raw values and
units are retained; a zero reading must not establish zero physical power.
"""
import ctypes as C
import plistlib
import struct
import subprocess
import time


class HardwareMonitor:
    def __init__(self):
        self.cf=C.CDLL('/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation')
        self.ior=C.CDLL('/usr/lib/libIOReport.dylib')
        P=C.c_void_p;I=C.c_int64
        def bind(lib,name,restype,args):
            f=getattr(lib,name);f.restype=restype;f.argtypes=args;return f
        self.string=bind(self.cf,'CFStringCreateWithCString',P,[P,C.c_char_p,C.c_uint32])
        self.decode=bind(self.cf,'CFStringGetCString',C.c_bool,[P,C.c_char_p,I,C.c_uint32])
        self.release=bind(self.cf,'CFRelease',None,[P])
        self.value=bind(self.cf,'CFDictionaryGetValue',P,[P,P])
        self.count=bind(self.cf,'CFArrayGetCount',I,[P])
        self.at=bind(self.cf,'CFArrayGetValueAtIndex',P,[P,I])
        self.copy=bind(self.ior,'IOReportCopyChannelsInGroup',P,[P,P,C.c_uint64,C.c_uint64,C.c_uint64])
        self.merge=bind(self.ior,'IOReportMergeChannels',None,[P,P,P])
        self.subscribe=bind(self.ior,'IOReportCreateSubscription',P,[P,P,C.POINTER(P),C.c_uint64,P])
        self.snapshot=bind(self.ior,'IOReportCreateSamples',P,[P,P,P])
        self.delta=bind(self.ior,'IOReportCreateSamplesDelta',P,[P,P,P])
        self.names={key:bind(self.ior,'IOReportChannelGet'+key,P,[P]) for key in ('Group','ChannelName','UnitLabel')}
        self.states_count=bind(self.ior,'IOReportStateGetCount',C.c_int,[P])
        self.state_name=bind(self.ior,'IOReportStateGetNameForIndex',P,[P,C.c_int])
        self.residency=bind(self.ior,'IOReportStateGetResidency',I,[P,C.c_int])
        self.integer=bind(self.ior,'IOReportSimpleGetIntegerValue',I,[P,C.c_int])
        self.channels=None
        for group in ('GPU Stats','CPU Stats','Energy Model'):
            key=self.s(group);ch=self.copy(key,None,0,0,0);self.release(key)
            if not ch:continue
            if self.channels:self.merge(self.channels,ch,None);self.release(ch)
            else:self.channels=ch
        if not self.channels:raise RuntimeError('No IOReport channels')
        self.subscribed=P()
        self.subscription=self.subscribe(None,self.channels,C.byref(self.subscribed),0,None)
        if not self.subscription:raise RuntimeError('IOReport subscription unavailable')
        self.previous=self.snapshot(self.subscription,self.channels,None)
        if not self.previous:raise RuntimeError('Initial IOReport snapshot unavailable')
        self.previous_time=time.monotonic()
        data=subprocess.check_output(['ioreg','-a','-r','-n','pmgr'],timeout=5)
        def find(obj):
            if isinstance(obj,dict):
                if 'voltage-states9' in obj:return obj['voltage-states9']
                for val in obj.values():
                    got=find(val)
                    if got is not None:return got
            if isinstance(obj,list):
                for val in obj:
                    got=find(val)
                    if got is not None:return got
        table=find(plistlib.loads(data))
        self.gpu_mhz=[f/1e6 for f,v in struct.iter_unpack('<II',table)] if table else []
        C.CDLL('/System/Library/Frameworks/Foundation.framework/Foundation')
        objc=C.CDLL('/usr/lib/libobjc.A.dylib')
        klass=bind(objc,'objc_getClass',P,[C.c_char_p]);sel=bind(objc,'sel_registerName',P,[C.c_char_p])
        self.msg=C.CFUNCTYPE(P,P,P)(('objc_msgSend',objc))
        self.msg_int=C.CFUNCTYPE(C.c_long,P,P)(('objc_msgSend',objc))
        self.process_info=self.msg(klass(b'NSProcessInfo'),sel(b'processInfo'))
        self.thermal_selector=sel(b'thermalState')

    def s(self,text):return self.string(None,text.encode(),0x08000100)

    def text(self,pointer):
        if not pointer:return ''
        buf=C.create_string_buffer(1024)
        return buf.value.decode() if self.decode(pointer,buf,len(buf),0x08000100) else ''

    def sample(self):
        now=time.monotonic();current=self.snapshot(self.subscription,self.channels,None)
        if not current:raise RuntimeError('IOReport snapshot unavailable')
        delta=self.delta(self.previous,current,None)
        if not delta:
            self.release(current)
            raise RuntimeError('IOReport delta unavailable')
        dt=now-self.previous_time
        self.release(self.previous);self.previous=current;self.previous_time=now
        key=self.s('IOReportChannels');array=self.value(delta,key);self.release(key)
        if not array:
            self.release(delta)
            raise RuntimeError('IOReport delta contains no channels')
        states={};energy={}
        for index in range(self.count(array)):
            item=self.at(array,index)
            group=self.text(self.names['Group'](item));name=self.text(self.names['ChannelName'](item))
            if group=='Energy Model' and ('CPU Energy' in name or name=='GPU Energy'):
                unit=self.text(self.names['UnitLabel'](item));value=self.integer(item,0)
                scale={'mJ':1e-3,'uJ':1e-6,'nJ':1e-9}.get(unit)
                energy[name]=dict(raw_delta=value,unit=unit,estimated_watts=value*scale/dt if scale else None)
            if group in ('GPU Stats','CPU Stats'):
                n=self.states_count(item)
                if n>0:states[name]=[(self.text(self.state_name(item,i)),self.residency(item,i)) for i in range(n)]
        self.release(delta)
        gpu=states.get('GPUPH',[])
        active=[v for name,v in gpu if name not in ('OFF','IDLE','DOWN')]
        table=[f for f in self.gpu_mhz if f>0]
        freq=None
        mapped=sum(active[:len(table)])
        coverage=mapped/sum(active) if sum(active)>0 else None
        if coverage==1:
            freq=sum(f*v for f,v in zip(table,active))/sum(active)
        thermal=int(self.msg_int(self.process_info,self.thermal_selector))
        return dict(time=time.time(),interval_seconds=dt,gpu_active_frequency_mhz=freq,
            gpu_frequency_mapping_coverage=coverage,
            gpu_active_pstate_index=sum((i+1)*v for i,v in enumerate(active))/sum(active) if sum(active)>0 else None,
            gpu_frequency_table_mhz=self.gpu_mhz,gpu_active_residency=sum(active)/sum(v for _,v in gpu) if gpu and sum(v for _,v in gpu)>0 else None,
            energy_model=energy,thermal_state=thermal,thermal_label={0:'nominal',1:'fair',2:'serious',3:'critical'}.get(thermal,'unknown'),
            performance_states=states)


if __name__=='__main__':
    import json
    monitor=HardwareMonitor();time.sleep(1)
    print(json.dumps(monitor.sample()))
