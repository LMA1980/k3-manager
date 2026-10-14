# k3-manager

Control plane for the SpacemiT K3 AI platform.

## Overview
`k3-manager` provides the orchestration layer, session management, and external APIs for the K3 platform:
- **IPC Daemon**: Apache Fury serialization over Unix Domain Sockets (UDS) + shared memory.
- **Session State**: Active in-memory sessions backed by SQLite for reboot persistence.
- **Modular WebAPI**: OpenAI and Anthropic compatible endpoint implementations.
- **Model Subsystem**: Registry, model pull/search, and vendor adapters.

## Architecture & Specifications
For master architecture and detailed component specifications, see the parent [model-runner](https://github.com/LMA1980/model-runner) repository:
- Master Plan: [PLAN.md](https://github.com/LMA1980/model-runner/blob/main/PLAN.md)
- Specs: [specs/](https://github.com/LMA1980/model-runner/tree/main/specs)
