# Data Model

## ResearchProject

Tracks the local research project:

- `project_id`
- `name`
- `user_goal`
- `research_questions`
- `scope`
- `status`
- `created_at`
- `updated_at`

## Paper And Versioning

`ScholarlyWork` is the canonical work. `PaperVersion` records arXiv, conference, journal, local PDF, demo, or unknown versions.

Dedup priority:

1. DOI
2. arXiv ID
3. source internal ID
4. normalized title
5. title similarity

Duplicates are merged as versions where possible. Evidence always references a concrete `version_id`.

## EvidenceUnit

An evidence unit records:

- `evidence_id`
- `work_id`
- `version_id`
- `section`
- `page`
- `paragraph_index`
- `table_figure_id`
- `content`
- `evidence_type`
- `source_type`
- `extraction_confidence`
- `verification_status`
- `content_hash`

`abstract_only` evidence is allowed but cannot pretend to be full-text reading.

## Claim

Claim types:

```text
paper_fact
author_claim
cross_paper_synthesis
agent_inference
research_hypothesis
```

Support states:

```text
verified
partially_supported
contested
insufficient_evidence
rejected
```

`paper_fact` requires at least one supporting evidence ID at schema validation and Quality Gate validation.

## Hypothesis

A hypothesis records:

- motivation
- supporting claims
- counter evidence
- generated queries
- falsification condition
- minimum experiment
- risks
- status

Hypotheses are research candidates. They are not stored as paper facts.

## Task And RunEvent

Tasks record structured role work. Run events provide JSONL progress and audit records:

```text
run.started
task.started
task.completed
search.query
paper.discovered
paper.selected
paper.read
evidence.created
claim.created
hypothesis.created
warning
error
run.completed
```
