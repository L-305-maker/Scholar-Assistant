from __future__ import annotations

from scholar_assistant.schemas.evidence import Claim, ClaimType, EvidenceUnit, Hypothesis
from scholar_assistant.schemas.paper import Paper
from scholar_assistant.schemas.research import QueryPlan

CLAIM_LABELS = {
    ClaimType.PAPER_FACT: "[论文事实]",
    ClaimType.AUTHOR_CLAIM: "[作者主张]",
    ClaimType.CROSS_PAPER_SYNTHESIS: "[跨论文归纳]",
    ClaimType.AGENT_INFERENCE: "[Agent推断]",
    ClaimType.RESEARCH_HYPOTHESIS: "[待验证假设]",
}


def render_report(
    *,
    question: str,
    query_plan: QueryPlan,
    papers: list[Paper],
    evidence_units: list[EvidenceUnit],
    claims: list[Claim],
    hypotheses: list[Hypothesis],
    retrieval_mode: str,
    warnings: list[str],
) -> str:
    evidence_by_id = {evidence.evidence_id: evidence for evidence in evidence_units}
    paper_by_id = {paper.work_id: paper for paper in papers}

    lines = [
        f"# Research Brief: {question}",
        "",
        "## 1. 研究问题与范围",
        f"- 研究问题：{question}",
        "- 范围：本 MVP 聚焦论文搜索、摘要/PDF 证据抽取、证据约束的 Claim 与候选假设。",
        "",
        "## 2. 检索策略",
        f"- 核心查询：`{query_plan.core_query}`",
        f"- 检索模式：`{retrieval_mode}`",
        "- 查询类型：核心查询、同义词查询、方法导向查询、问题导向查询、负面结果查询、验证查询。",
        "",
        "## 3. 文献覆盖情况",
        f"- 核心论文数：{len(papers)}",
        f"- 证据单元数：{len(evidence_units)}",
        f"- Claim 数：{len(claims)}",
        f"- 候选假设数：{len(hypotheses)}",
        "",
        "## 4. 方法分类",
    ]
    roles = {}
    for paper in papers:
        roles.setdefault(paper.paper_role.value, 0)
        roles[paper.paper_role.value] += 1
    if roles:
        lines.extend([f"- {role}: {count}" for role, count in sorted(roles.items())])
    else:
        lines.append("- [证据不足] 未获得可分类论文。")

    lines.extend(["", "## 5. 核心论文"])
    for index, paper in enumerate(papers, start=1):
        year = paper.year if paper.year else "n.d."
        identifier = paper.arxiv_id or paper.doi or paper.work_id
        lines.append(
            f"{index}. {paper.title} ({year}) - `{identifier}` - "
            f"role `{paper.paper_role.value}` - score {paper.relevance_score:.4f}"
        )

    lines.extend(["", "## 6. 跨论文比较"])
    synthesis = [claim for claim in claims if claim.type == ClaimType.CROSS_PAPER_SYNTHESIS]
    if synthesis:
        for claim in synthesis:
            lines.append(f"- {CLAIM_LABELS[claim.type]} {claim.content}")
    else:
        lines.append("- [证据不足] 当前证据不足以形成可靠跨论文比较。")

    lines.extend(["", "## 7. 领域共识"])
    author_claims = [claim for claim in claims if claim.type == ClaimType.AUTHOR_CLAIM]
    if author_claims:
        for claim in author_claims[:6]:
            lines.append(f"- {CLAIM_LABELS[claim.type]} {claim.content}")
    else:
        lines.append("- [证据不足] 当前主要是元数据或少量全文证据，不能声称领域共识。")

    lines.extend(["", "## 8. 冲突与争议"])
    lines.append("- [证据不足] MVP 未发现经过双向证据验证的冲突结论。")

    lines.extend(["", "## 9. 实验设计问题"])
    lines.append(
        "- [Agent推断] 比较长期记忆检索噪声时，需要固定记忆库、检索预算、"
        "上下文长度、任务集和噪声定义。"
    )
    lines.append("- [证据不足] 没有统一实验条件时，系统不会输出直接优劣排名。")

    lines.extend(["", "## 10. 未解决问题"])
    lines.append("- [Agent推断] 需要验证噪声记忆的定义、标注协议和下游任务影响是否可复现。")
    lines.append("- [Agent推断] 需要区分陈旧记忆、无关记忆和矛盾记忆三类噪声。")

    lines.extend(["", "## 11. 候选研究假设"])
    for hypothesis in hypotheses:
        lines.append(f"- [待验证假设] {hypothesis.content}")
        lines.append(f"  - 最小实验：{hypothesis.minimum_experiment}")
        lines.append(f"  - 证伪条件：{hypothesis.falsification_condition}")
    if not hypotheses:
        lines.append("- [证据不足] 当前没有生成候选假设。")

    lines.extend(["", "## 12. 检索和证据限制"])
    if warnings:
        lines.extend([f"- {warning}" for warning in warnings])
    lines.append("- 摘要证据不能替代全文深读；报告会保留 `abstract_only` 标记。")
    lines.append("- 离线 demo 元数据只用于无网络烟测，不应当作为正式文献结论。")

    lines.extend(["", "## 13. 参考文献"])
    for paper in papers:
        identifier = paper.arxiv_id or paper.doi or paper.work_id
        lines.append(
            f"- {paper.title}. {paper.venue or 'unknown venue'}, "
            f"{paper.year or 'n.d.'}. `{identifier}`"
        )

    lines.extend(["", "## Evidence Index"])
    for evidence in evidence_units:
        paper = paper_by_id.get(evidence.work_id)
        paper_title = paper.title if paper else evidence.work_id
        location = f"page {evidence.page}" if evidence.page else evidence.section or "abstract"
        lines.append(
            f"- `{evidence.evidence_id}`: {paper_title}, "
            f"version `{evidence.version_id}`, {location}, `{evidence.source_type.value}`"
        )

    lines.extend(["", "## Claim Evidence Links"])
    for claim in claims:
        label = CLAIM_LABELS[claim.type]
        refs = []
        for evidence_id in claim.supporting_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence:
                location = f"p.{evidence.page}" if evidence.page else evidence.section or "abstract"
                refs.append(f"{evidence_id}/{evidence.version_id}/{location}")
        lines.append(f"- {label} `{claim.claim_id}` {claim.support_status.value}: {claim.content}")
        if refs:
            lines.append(f"  - Evidence: {', '.join(refs)}")
    return "\n".join(lines) + "\n"
