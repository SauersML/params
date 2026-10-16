"""Minimal dataset adapter for installed MLX CacheDataset; no training algorithm."""
class ExactDataset:
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):return self.rows[i]
    def process(self,row):
        prefix=list(row['prompt_ids']);completion=list(row['completion_ids'])
        assert prefix and completion and len(prefix)+len(completion)<=6144
        return prefix+completion,len(prefix)
