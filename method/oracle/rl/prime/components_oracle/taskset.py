"""The component-circuit oracle (#2951) as a verifiers v1 taskset for prime-rl: one task per text, the question from
components.py questions output, the reward components.reward's (rl/serve_components.py over HTTP; an invalid answer
earns -(1 + size weight)), with its terms, size and validity as metrics.

  env.taskset.id = "components-oracle", env.taskset.questions = "<JSONL>", env.taskset.task.scorer_url
"""

import json
from pathlib import Path

import httpx

import verifiers.v1 as vf

class ComponentsOracleData(vf.TaskData):
    task: str
    """The question's text id: the scorer reads TEXTS/vpd4l/<task>.json and its evidence row."""


class ComponentsOracleTaskConfig(vf.TaskConfig):
    scorer_url: str = "http://127.0.0.1:8767/score"
    timeout: float = 3600.0
    """Seconds one scoring request may take."""


class ComponentsOracleTask(vf.Task[ComponentsOracleData, vf.State, ComponentsOracleTaskConfig]):
    @vf.reward(weight=1.0)
    async def verifier(self, trace: vf.Trace) -> float:
        replies = trace.assistant_messages
        reply = (replies[-1].content or "") if replies else ""
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            r = await client.post(self.config.scorer_url, json={"task": self.data.task, "answers": [reply]})
        body = r.json()
        if r.status_code != 200:
            raise RuntimeError(f"serve_components: {body.get('error')}")
        s = body["scores"][0]
        for k in ("F1", "N", "F3", "E1", "E2", "size", "nodes", "edges", "dropped_edges"):
            if s.get(k) is not None:
                trace.record_metric(k, float(s[k]))
        trace.record_metric("valid", float(s["valid"]))
        return s["reward"]


class ComponentsOracleEnv(vf.SingleAgentEnv):
    """One answer per episode."""

    async def run(self, task, agents):
        async with agents.agent.interaction(task) as interaction:
            await interaction.turn()


class ComponentsOracleConfig(vf.TasksetConfig):
    questions: str = ""
    """components.py questions output, one text per line."""
    task: ComponentsOracleTaskConfig = ComponentsOracleTaskConfig()


class ComponentsOracleTaskset(vf.Taskset[ComponentsOracleTask, ComponentsOracleConfig]):
    def load(self) -> list[ComponentsOracleTask]:
        tasks = []
        for line in Path(self.config.questions).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            tasks.append(ComponentsOracleTask(ComponentsOracleData(id=r["task"], prompt=r["question"], task=r["task"]), self.config.task))
        return tasks
