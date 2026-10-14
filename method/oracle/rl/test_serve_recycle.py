"""Model-free multiprocessing regression for graceful verifier retirement."""
import itertools,multiprocessing as mp,os,threading,unittest,tempfile,json,queue,types,sys
from unittest.mock import patch,Mock
from pathlib import Path
from concurrent.futures import Future
import serve_scores
import serve_edits

def fake_worker(inbox,outbox,ready,release,calls):
 rid,req=inbox.get();calls.put(rid);outbox.put((rid,[req['task']],None))
 ready.set();release.wait(5)
 outbox.close();outbox.join_thread();os._exit(serve_scores.GRACEFUL_RECYCLE_EXIT)

class Harness(serve_scores.Pool):
 graceful_recycle_exit = serve_scores.GRACEFUL_RECYCLE_EXIT
 def __init__(self):
  self.ctx=mp.get_context('spawn');self.out=self.ctx.Queue();self.inboxes=[self.ctx.Queue()]
  self.ready=self.ctx.Event();self.release=self.ctx.Event();self.calls=self.ctx.Queue()
  self.ids=itertools.count();self.waiting={};self.owner={};self.lock=threading.Lock()
  self.procs=[self.start(self.inboxes[0])]
 def start(self,q):
  p=self.ctx.Process(target=fake_worker,args=(q,self.out,self.ready,self.release,self.calls));p.start();return p
 def collect_one(self):
  rid,scores,err=self.out.get(timeout=5)
  with self.lock:fut=self.waiting.pop(rid);self.owner.pop(rid)
  fut.set_result(scores)
 def close(self):
  for p in self.procs:
   p.join(1)
   if p.is_alive():p.terminate();p.join()
  for q in [self.out,self.calls,*self.inboxes]:q.close();q.join_thread()

class RecycleTests(unittest.TestCase):
 def test_chunk_failure_and_cardinality_preserve_group_error_and_cleanup(self):
  for failure in ('raise','short'):
   with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp:
    path=os.path.join(tmp,'goals');Path(path).write_text(json.dumps({'id':'g','pref':{}})+'\n')
    inbox=queue.Queue();inbox.put((0,{'task':'g','answers':['a','b','c','d','e']}));outbox=Mock()
    calls=[]
    def answers(pref,items,part):
     calls.append(items)
     if len(calls)==2:
      if failure=='raise':raise ValueError('chunk failed')
      return []
     return [{} for _ in items]
    env=types.SimpleNamespace(load=lambda device:(None,types.SimpleNamespace(answers=answers)))
    cleanup=Mock()
    with patch.dict(sys.modules,{'edit_env':env}),patch.object(serve_edits,'RECYCLE',1),patch.object(serve_edits,'CHUNK',4),patch.object(serve_edits,'free_device',cleanup),patch.object(serve_edits.os,'_exit',side_effect=SystemExit):
     with self.assertRaises(SystemExit):serve_edits.work(None,path,inbox,outbox)
    self.assertEqual(cleanup.call_count,2);rid,scores,error=outbox.put.call_args.args[0]
    self.assertEqual(rid,0);self.assertIsNone(scores);self.assertTrue(error)
 def test_retirement_preserves_delayed_reply_and_new_queued_request_once(self):
  h=Harness()
  try:
   first=h.submit({'task':'first'});self.assertTrue(h.ready.wait(5))
   second=h.submit({'task':'second'}) # enqueued during retirement, collector deliberately delayed
   h.release.set();h.procs[0].join(5);self.assertEqual(h.procs[0].exitcode,75)
   h.restart_dead();self.assertFalse(first.done());self.assertFalse(second.done())
   h.collect_one();h.collect_one()
   self.assertEqual(first.result(),['first']);self.assertEqual(second.result(),['second'])
   self.assertEqual([h.calls.get(timeout=5),h.calls.get(timeout=5)],[0,1]);self.assertFalse(h.waiting)
  finally:h.close()
 def test_unplanned_crash_still_fails_inflight(self):
  h=Harness()
  try:
   f=h.submit({'task':'first'});self.assertTrue(h.ready.wait(5));h.procs[0].terminate();h.procs[0].join()
   h.restart_dead()
   with self.assertRaises(RuntimeError):f.result()
  finally:h.close()
 def test_generic_pool_does_not_treat_exit75_as_graceful(self):
  h=Harness();h.graceful_recycle_exit=None
  try:
   f=h.submit({'task':'first'});self.assertTrue(h.ready.wait(5));h.release.set();h.procs[0].join(5)
   self.assertEqual(h.procs[0].exitcode,75);h.restart_dead()
   with self.assertRaises(RuntimeError):f.result()
  finally:h.close()
if __name__=='__main__':unittest.main()
