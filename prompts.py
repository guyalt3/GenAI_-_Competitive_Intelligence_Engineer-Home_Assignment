EVALUATION_PROMPT = [
    (
        "system",
        "You are a competitive intelligence routing assistant. "
        "Your job is to decide whether locally cached documents are enough "
        "to answer a user query accurately and with up-to-date information.",
    ),
    (
        "human",
        "User query:\n{query}\n\n"
        "Competitor:\n{competitor}\n\n"
        "Retrieved local context:\n{context}\n\n"
        "Is this local context fully sufficient to answer the user's query "
        "accurately and up-to-date?\n"
        "Reply with exactly one word: YES or NO.",
    ),
]

SYNTHESIS_PROMPT_WITH_JFROG_IMPACT = [
    (
        "system",
        "You are a JFrog competitive intelligence analyst. "
        "Use only the provided context to answer the query. "
        "Structure your response clearly and include a dedicated section titled "
        "'Impact on JFrog' that explains strategic implications for JFrog. "
        "End with a 'Sources' section listing every source you relied on.",
    ),
    (
        "human",
        "User query:\n{query}\n\n"
        "Competitor:\n{competitor}\n\n"
        "Compiled context:\n{context}\n\n"
        "Available sources:\n{sources}\n\n"
        "Generate the final intelligence brief.",
    ),
]

SYNTHESIS_PROMPT_WITHOUT_JFROG_IMPACT = [
    (
        "system",
        "You are a competitive intelligence analyst. "
        "Use only the provided context to answer the query. "
        "Structure your response clearly. "
        "CRITICAL: Do not mention JFrog, do not perform any impact analysis on JFrog, "
        "and focus strictly on the competitor. "
        "End with a 'Sources' section listing every source you relied on.",
    ),
    (
        "human",
        "User query:\n{query}\n\n"
        "Competitor:\n{competitor}\n\n"
        "Compiled context:\n{context}\n\n"
        "Available sources:\n{sources}\n\n"
        "Generate the final intelligence brief.",
    ),
]

RELEVANCE_PROMPT = [
    (
        "system",
        "You are a competitive intelligence quality filter for an offline RAG pipeline. "
        "Return YES only when a text chunk contains actionable, product-centric, or "
        "market-relevant intelligence about the specified competitor. Valid signals include "
        "new feature releases, product announcements, security advisories, capability shifts, "
        "strategic partnerships, and pricing changes. "
        "Return NO for corporate noise, generic blog filler, navigation bars, career ads, "
        "cookie notices, unrelated industry news, or content that does not materially "
        "inform competitive analysis.",
    ),
    (
        "human",
        "Competitor:\n{competitor}\n\n"
        "Text chunk:\n{chunk}\n\n"
        "Does this chunk contain high-value competitive intelligence about the competitor?\n"
        "Reply with exactly one word: YES or NO.",
    ),
]
