"""Progressive Layered Extraction (Tang et al. 2020) for the credit funnel.

Tasks: p(click), p(apply | click), p(approve | apply) and the ZILN *amount* tower.
Click and approval pull the shared representation in different directions (a flashy
high-APR card is clicky but rarely approved), the classic *seesaw* phenomenon.  PLE
gives every task its own experts, keeps a set of shared experts, and uses a
**Customized Gate Control** (CGC) block per level so task ``k`` only ever mixes
``{its own experts} ∪ {shared experts}``: task-specific parameters are isolated from
other tasks' gradients while shared knowledge still flows.  Stacking CGC blocks
(``num_levels > 1``) is the "progressive" extraction.

Placement: PLE sits on top of the shared HSTU fusion vector (one backbone, four
towers) rather than per-task backbones or a shared-bottom / MMoE; the towers emit raw
logits which Stage 4 calibrates.  Nothing about business value lives here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn


@dataclass(frozen=True)
class PLEConfig:
    input_dim: int
    task_names: tuple[str, ...] = ("click", "apply", "approve", "amount")
    task_output_dims: tuple[int, ...] = (1, 1, 1, 3)  # amount = ZILN [p_logit, mu, sigma_raw]
    num_shared_experts: int = 2
    num_task_experts: int = 1
    expert_hidden: tuple[int, ...] = (64,)
    expert_dim: int = 32
    num_levels: int = 2
    tower_hidden: tuple[int, ...] = (32,)
    dropout: float = 0.0
    amount_mu_bias_init: float = 9.0  # ~ mean log-amount of approved loans / limits

    def __post_init__(self) -> None:
        if len(self.task_names) != len(self.task_output_dims):
            raise ValueError("task_names and task_output_dims must align")
        if self.num_levels < 1:
            raise ValueError("num_levels must be >= 1")

    @property
    def num_tasks(self) -> int:
        return len(self.task_names)


def _mlp(
    dims: tuple[int, ...], dropout: float = 0.0, final_activation: bool = True
) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2 or final_activation:
            layers += [nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden: tuple[int, ...], expert_dim: int, dropout: float):
        super().__init__()
        self.net = _mlp((input_dim, *hidden, expert_dim), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.net(x)
        return out


class CGCBlock(nn.Module):
    """One Customized-Gate-Control level.

    Inputs: one tensor per task plus one shared tensor (all ``(..., input_dim)``).
    Task ``k``'s gate (softmax over ``num_task_experts + num_shared_experts``) is
    computed from *its own* input and mixes only its experts and the shared ones.
    The shared gate mixes every expert and feeds the next level (absent on the last).
    """

    def __init__(self, config: PLEConfig, input_dim: int, is_last: bool) -> None:
        super().__init__()
        c = config
        self.num_tasks = c.num_tasks
        self.is_last = is_last
        self.task_experts = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Expert(input_dim, c.expert_hidden, c.expert_dim, c.dropout)
                        for _ in range(c.num_task_experts)
                    ]
                )
                for _ in range(c.num_tasks)
            ]
        )
        self.shared_experts = nn.ModuleList(
            [
                Expert(input_dim, c.expert_hidden, c.expert_dim, c.dropout)
                for _ in range(c.num_shared_experts)
            ]
        )
        self.task_gates = nn.ModuleList(
            [
                nn.Linear(input_dim, c.num_task_experts + c.num_shared_experts)
                for _ in range(c.num_tasks)
            ]
        )
        total = c.num_tasks * c.num_task_experts + c.num_shared_experts
        self.shared_gate = None if is_last else nn.Linear(input_dim, total)
        self.last_task_gates: list[torch.Tensor] = []

    def forward(
        self, task_inputs: list[torch.Tensor], shared_input: torch.Tensor
    ) -> tuple[list[torch.Tensor], torch.Tensor | None]:
        shared_out = torch.stack([e(shared_input) for e in self.shared_experts], dim=-2)
        task_outs: list[torch.Tensor] = []
        all_task_experts: list[torch.Tensor] = []
        gates: list[torch.Tensor] = []
        for k in range(self.num_tasks):
            experts_k = cast(nn.ModuleList, self.task_experts[k])
            own = torch.stack([e(task_inputs[k]) for e in experts_k], dim=-2)
            all_task_experts.append(own)
            experts = torch.cat([own, shared_out], dim=-2)  # (..., n_own + n_shared, E)
            gate = torch.softmax(self.task_gates[k](task_inputs[k]), dim=-1)  # (..., n)
            gates.append(gate.detach())
            task_outs.append((gate.unsqueeze(-1) * experts).sum(dim=-2))
        self.last_task_gates = gates
        shared_next: torch.Tensor | None = None
        if self.shared_gate is not None:
            everything = torch.cat([*all_task_experts, shared_out], dim=-2)
            g = torch.softmax(self.shared_gate(shared_input), dim=-1)
            shared_next = (g.unsqueeze(-1) * everything).sum(dim=-2)
        return task_outs, shared_next


@dataclass
class PLEOutput:
    logits: dict[str, torch.Tensor]  # task name -> (...,) or (..., out_dim) when out_dim > 1
    task_reprs: list[torch.Tensor]  # per task (..., expert_dim)

    @property
    def click_logit(self) -> torch.Tensor:
        return self.logits["click"]

    @property
    def apply_logit(self) -> torch.Tensor:
        return self.logits["apply"]

    @property
    def approve_logit(self) -> torch.Tensor:
        return self.logits["approve"]

    @property
    def amount_logits(self) -> torch.Tensor | None:
        return self.logits.get("amount")


class PLE(nn.Module):
    def __init__(self, config: PLEConfig) -> None:
        super().__init__()
        self.config = config
        levels = []
        d = config.input_dim
        for level in range(config.num_levels):
            levels.append(CGCBlock(config, d, is_last=level == config.num_levels - 1))
            d = config.expert_dim
        self.levels = nn.ModuleList(levels)
        towers = []
        for name, out_dim in zip(config.task_names, config.task_output_dims, strict=True):
            hidden_dim = config.tower_hidden[-1] if config.tower_hidden else config.expert_dim
            head = nn.Linear(hidden_dim, out_dim)
            if name == "amount" and out_dim == 3:
                with torch.no_grad():
                    head.bias[1] = config.amount_mu_bias_init
            towers.append(
                nn.Sequential(_mlp((config.expert_dim, *config.tower_hidden), config.dropout), head)
            )
        self.towers = nn.ModuleList(towers)

    def cgc_blocks(self) -> list[CGCBlock]:
        return [cast(CGCBlock, b) for b in self.levels]

    def forward(self, x: torch.Tensor) -> PLEOutput:
        task_inputs = [x for _ in self.config.task_names]
        shared: torch.Tensor | None = x
        for block in self.levels:
            assert shared is not None
            task_inputs, shared = block(task_inputs, shared)
        logits: dict[str, torch.Tensor] = {}
        for name, out_dim, tower, rep in zip(
            self.config.task_names, self.config.task_output_dims, self.towers, task_inputs,
            strict=True,
        ):  # fmt: skip
            out: torch.Tensor = tower(rep)
            logits[name] = out.squeeze(-1) if out_dim == 1 else out
        return PLEOutput(logits=logits, task_reprs=task_inputs)
