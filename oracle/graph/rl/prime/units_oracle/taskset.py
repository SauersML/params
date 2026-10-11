"""The unit-circuit oracle (#2951) as a verifiers v1 taskset for prime-rl: one task per question (units.dataset rows
{"prompt": [user message], "info": {"task"}}, or units_sft.py questions output {"task", "question"}), the reward the
units verifier's (rl/serve_units.py over HTTP: the carried share less 0.5 x units / units_all, 0 for an answer without a
line for every listed position), with the carried share, units, validity and necessity as metrics.

  env.taskset.id = "units-oracle", env.taskset.questions = "<JSONL>", env.taskset.task.scorer_url
"""

import json
from pathlib import Path

import httpx

import verifiers.v1 as vf


class UnitsOracleData(vf.TaskData):
    task: str
    """The question's text id: the scorer reads TEXTS/vpd4l/<task>.json."""


class UnitsOracleTaskConfig(vf.TaskConfig):
    scorer_url: str = "http://127.0.0.1:8766/score"
    timeout: float = 3600.0
    """Seconds one scoring request may take."""


class UnitsOracleTask(vf.Task[UnitsOracleData, vf.State, UnitsOracleTaskConfig]):
    @vf.reward(weight=1.0)
    async def verifier(self, trace: vf.Trace) -> float:
        replies = trace.assistant_messages
        reply = (replies[-1].content or "") if replies else ""
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            r = await client.post(self.config.scorer_url, json={"task": self.data.task, "answers": [reply]})
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"serve_units: {body.get('error')}")
        s = body["scores"][0]
        for k in ("carried", "units", "rest_carried", "unknown"):
            trace.record_metric(k, float(s[k]))
        trace.record_metric("valid", float(s["valid"]))
        return s["reward"]


class UnitsOracleEnv(vf.SingleAgentEnv):
    """One answer per episode."""

    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn()


class UnitsOracleConfig(vf.TasksetConfig):
    questions: str = ""
    """units.dataset rows or units_sft.py questions output, one question per line."""
    task: UnitsOracleTaskConfig = UnitsOracleTaskConfig()


class UnitsOracleTaskset(vf.Taskset[UnitsOracleTask, UnitsOracleConfig]):
    def load(self) -> list[UnitsOracleTask]:
        tasks = []
        for line in Path(self.config.questions).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tid, text = (r["task"], r["question"]) if "question" in r else (r["info"]["task"], r["prompt"][0]["content"])
            tasks.append(UnitsOracleTask(UnitsOracleData(id=tid, prompt=text, task=tid), self.config.task))
        return tasks
