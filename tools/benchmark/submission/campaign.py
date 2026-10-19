"""Atomic benchmark artifacts."""
import json

def save(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data,indent=2)+'\n')
    temp.replace(path)
