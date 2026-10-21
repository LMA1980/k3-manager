# k3-manager

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs">
    <img src="https://hook.lma1980.duckdns.org/cards/LMA1980/k3-manager/card.svg" alt="k3-manager Factory Scorecard" width="520"/>
  </a>
</p>

<p align="center">
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-manager/status.svg" alt="CI Status"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-manager/score.svg" alt="Quality Score"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-manager/coverage.svg" alt="Coverage"/></a>
  <a href="https://hook.lma1980.duckdns.org/jobs"><img src="https://hook.lma1980.duckdns.org/badges/LMA1980/k3-manager/benchmark.svg" alt="Benchmark"/></a>
</p>

---

Control plane for the SpacemiT K3 AI platform.

## Overview
`k3-manager` provides the orchestration layer, session management, and external APIs for the K3 platform:
- **IPC Daemon**: Apache Arrow IPC serialization over Unix Domain Sockets (UDS) + shared memory.
- **Session State**: Active in-memory sessions backed by SQLite for reboot persistence.
- **Modular WebAPI**: OpenAI and Anthropic compatible endpoint implementations.
- **Model Subsystem**: Registry, model pull/search, and vendor adapters.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [model-runner](https://github.com/LMA1980/model-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/model-runner/blob/main/PLAN.md)
- Specs: [specs/](https://github.com/LMA1980/model-runner/tree/main/specs)
