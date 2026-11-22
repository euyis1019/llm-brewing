"""Temporary MATH-500 pool (same pid convention 'math500:<row>') so calibration can start before pool.jsonl."""
import json, sys
src, dst = sys.argv[1], sys.argv[2]
with open(src) as f, open(dst, 'w') as g:
    for i, l in enumerate(f):
        r = json.loads(l)
        g.write(json.dumps(dict(pid=f'math500:{i}', source='math500', problem=r['problem'], answer=r['answer'],
                                answer_type='math', level=r.get('level'), orig_id=r.get('unique_id'))) + '\n')
