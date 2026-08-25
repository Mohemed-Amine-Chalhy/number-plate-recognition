# Two-minute agentic product walkthrough

← [Video package](README.md) · [Recording guide](recording-guide.md)

The cut follows the product from campus-wide context into one bounded agent trajectory, its human
decision boundary, the surrounding access workflow, configuration, and system architecture. Scene
lengths and narration beats are the same timing contract used by `scripts/build_demo_video.py` and
`captions.vtt`.

## Scene contract

| Scene | Time | Source | On-screen callout | Product beat |
| --- | --- | --- | --- | --- |
| S01 | 00:00–00:08 | Generated title card | `Agentic Campus Access` | Establish fixed, intent-selected triage with visible human authority. |
| S02 | 00:08–00:24 | `command-center.png` | `Control every gate from one view` | Build the shared operational context from which an operator selects a gate. |
| S03 | 00:24–00:44 | `agent-operations.png` | `Inspect a fixed, intent-selected trajectory` | Show retained operator context, typed steps, allowlisted tools, evidence, policy results, trace versions, and the approval boundary. |
| S04 | 00:44–00:56 | `gate-workspace.png` | `Keep people in control of consequential work` | Keep perception evidence, operating context, and a staged response together. |
| S05 | 00:56–01:08 | `access-approvals.png` | `Replace inbox searching with structured access` | Show that agentic triage complements—not replaces—the typed access workflow. |
| S06 | 01:08–01:23 | `campus-setup.png` | `Configure authority without creating a product fork` | Present tenant topology, devices, roles, locale, API, and agent scope as deployment configuration. |
| S07 | 01:23–01:35 | `mobile-rtl.png` | `Review the agent handoff in Arabic` | Demonstrate the Agent Operations trajectory, evidence, and decision boundary in a responsive Arabic right-to-left layout. |
| S08 | 01:35–01:53 | Generated architecture card | `Engineer the agent boundary, not only the prompt` | Explain the planner seam, tool registry, policy, human handoff, durable trace, inference, and edge boundaries. |
| S09 | 01:53–02:00 | Generated end card | `A complete agentic platform slice` | Close on working product behavior and the engineering system that evaluates and delivers it. |

During UI scenes, use one quiet source note in the upper safe area:
`Reference scenario · generated operational data`.

## Timed narration

These rows match the WebVTT cues exactly. Keep each sentence inside its assigned scene when replacing
the draft voice track.

| Beat | Time | Narration |
| --- | --- | --- |
| N01 | 00:00–00:04 | Campus Access runs a fixed, typed gate-health trajectory selected by one supported operational intent. |
| N02 | 00:04–00:08 | It coordinates people, gate systems, and AI perception without hiding who controls consequential work. |
| N03 | 00:08–00:16 | The command center brings gate state, queues, arrivals, reviews, incidents, and device health into one context. |
| N04 | 00:16–00:24 | From that shared picture, an operator selects a gate and records a narrow objective as triage context. |
| N05 | 00:24–00:34 | The fixed planner retains that context without interpreting it; the intent selects typed steps and only server-registered, gate-scoped tools. |
| N06 | 00:34–00:44 | Every observation carries evidence, policy results, planner and trace versions, while incident work pauses for a person. |
| N07 | 00:44–00:50 | The gate workspace keeps plate evidence, camera state, matching context, and the proposed response together. |
| N08 | 00:50–00:56 | Approval or rejection records the actor and reason; idempotent execution prevents a transport retry from repeating the effect. |
| N09 | 00:56–01:02 | Hosts still create structured requests with vehicle, site, gate, purpose, and a bounded time window. |
| N10 | 01:02–01:08 | Reviewers work from a prioritized queue, so a decision becomes searchable operational state instead of a lost email. |
| N11 | 01:08–01:16 | Administrators configure tenant identity, topology, devices, roles, locale, time zone, API, and agent scope. |
| N12 | 01:16–01:23 | Those controls remain configuration, keeping the product and its authority model reusable without campus-specific forks. |
| N13 | 01:23–01:29 | The Agent Operations workspace adapts to a mobile booth view without losing the run scope, evidence, or pending handoff. |
| N14 | 01:29–01:35 | In its Arabic right-to-left view, trace evidence and the human decision boundary remain together. |
| N15 | 01:35–01:44 | A typed control plane persists domain state and agent trajectories; the planner interface remains replaceable by design. |
| N16 | 01:44–01:53 | Tool allowlists, external policy, human handoff, durable traces, and scenario evaluations constrain any future model-backed planner. |
| N17 | 01:53–01:57 | The result is a runnable agentic platform slice: |
| N18 | 01:57–02:00 | multi-gate context, bounded triage, AI evidence, tests, evaluations, runbooks, and reproducible delivery. |

## Edit direction

- Favor the product hierarchy over decorative motion: a restrained 3.5% crop drift and short
  dissolves are enough.
- Keep the command-center map readable during S02; do not crop away gate pins, health state, or the
  selected-gate panel.
- Keep the retained operator objective, fixed plan, tool steps, Evidence coverage, trace metadata, and
  decision controls readable together during S03.
- Keep gate-workspace evidence and operational context visible together during S04.
- Use the checked-in light-theme desktop captures for the main flow and the dark Arabic Agent
  Operations capture only for S07.
- Keep captions to two lines, above the lower callout panel, with a solid or semitransparent backing.
- Use clean narration without notification sounds. If music is added, keep it clearly below speech
  and use only licensed material.

## Content boundary

The walkthrough describes product behavior visible in the repository. The current planner is the
deterministic, offline reference provider: `gate_health_triage` selects one fixed typed trajectory,
while the objective is retained only as operator context and is not interpreted or decomposed. A
future model-backed planner is presented only as a replaceable seam constrained by the same tools
and policy. Physical controllers and site-edge hardware remain explicit integration seams.
Interface numbers are generated reference data and are not narrated as measured outcomes.

Before export, verify the scene and cue ranges against the automated timeline test, then watch once
muted to confirm that the agent trajectory and human authority remain understandable without
narration.
