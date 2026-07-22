from __future__ import annotations

import re

from scholar_assistant.schemas.research import QueryPlan

TERM_MAP = {
    "长期记忆": ["long-term memory", "memory management", "persistent memory"],
    "检索噪声": ["retrieval noise", "noisy retrieval", "irrelevant memory retrieval"],
    "智能体": ["LLM agents", "language model agents", "agent memory"],
    "代理": ["LLM agents", "language model agents"],
    "大模型": ["large language models", "LLM"],
    "知识库": ["retrieval augmented generation", "RAG"],
}


class TaskPlanner:
    def plan_search(self, user_question: str) -> QueryPlan:
        english_terms = self._expand_terms(user_question)
        core_query = " ".join(english_terms[:4]) if english_terms else user_question
        if not core_query.strip():
            core_query = user_question
        return QueryPlan(
            user_question=user_question,
            core_query=core_query,
            english_terms=english_terms,
            synonym_queries=[
                "LLM agent memory retrieval noise",
                "long-term memory large language model agents",
                "retrieval augmented generation noisy context memory",
            ],
            method_queries=[
                "agent memory retrieval reranking filtering",
                "memory consolidation retrieval augmented language agents",
                "vector database memory LLM agents irrelevant retrieval",
            ],
            problem_queries=[
                "retrieval noise long context LLM agents",
                "irrelevant memories degrade language model agents",
                "memory retrieval failure cases LLM agents",
            ],
            negative_result_queries=[
                "LLM agent memory limitations retrieval noise",
                "negative results retrieval augmented generation noise",
            ],
            verification_queries=[
                "survey LLM agents memory mechanism retrieval",
                "benchmark long-term memory LLM agents retrieval",
            ],
        )

    def _expand_terms(self, text: str) -> list[str]:
        terms: list[str] = []
        for key, values in TERM_MAP.items():
            if key in text:
                terms.extend(values)
        ascii_terms = re.findall(r"[A-Za-z][A-Za-z0-9 -]{2,}", text)
        terms.extend(term.strip() for term in ascii_terms)
        seen: set[str] = set()
        unique: list[str] = []
        for term in terms:
            key = term.lower()
            if key not in seen:
                seen.add(key)
                unique.append(term)
        return unique
