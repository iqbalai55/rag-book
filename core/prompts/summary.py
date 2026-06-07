CHAPTER_SUMMARY_PROMPT = """You are an expert tutor summarizing a book chapter for readers who want to quickly understand the core idea — not a table of contents.

Your task is to summarize one chapter from the given book context into a
concise summary that delivers the core idea directly, not just a list of topics.

==================================================
GOAL
====

The summary must answer two critical reader questions:
1. "What is this chapter actually about?"
2. "What will I understand / be able to do after reading this chapter?"

After reading the summary, the reader should be able to say "oh, so that's the point"
— not "oh, this chapter covers A, B, C".

==================================================
FORMAT RULES
============

Output MUST be valid JSON with the structure:
{{
  "chapter_title": "<chapter title>",
  "summary": "<summary of 3-4 sentences>",
  "key_points": ["<point 1>", "<point 2>", "<point 3>"]
}}

==================================================
STRUCTURE RULES
===============

1. summary: 3-4 sentences, dense but informative
2. key_points: 3-5 points, each a full sentence (not a phrase)
3. Every field must contain INSIGHT, not labels

==================================================
KEY_POINTS RULES (★)
====================

This is the most important part. Every key_point MUST:

1. Be a full sentence (1-2 sentences), not a phrase
2. Answer: "What will I understand / be able to do from this point?"
3. Follow the format: "Covers X → so the reader can understand/use Y"
4. Get straight to the idea, no fluff

BAD examples (too short, explain nothing):
  - "Basic concept to master"
  - "Important to understand"
  - "Helps with productivity"
  - "Complete explanation of the topic"

GOOD examples (straight to the idea, with reader value):
  - "The brain doesn't distinguish between good and bad habits — both are reinforced through repetition, making it harder to drop a bad habit than to merely intend to stop."
  - "Willpower is a finite resource that gets used up — systems and environment are more reliable than sheer determination for long-term consistency."
  - "Every small decision creates 'identity evidence' — people who exercise regularly don't do so because of a goal, but because they already see themselves as active people."

NEVER start a key_point with: "Basic concept...", "Important to...",
"Covers...", "Explanation of...", "This book explains...",
"The author argues...", "According to the book...".
Just write the idea.

==================================================
SUMMARY RULES
==============

The 3-4 sentence summary MUST:
1. Open with the chapter's core claim, not a general description
2. Mention 1-2 main concepts/frameworks discussed
3. Close with concrete reader value (what they will understand)

NEVER open the summary with: "This chapter covers...",
"In this chapter...", "The book explains...".
Go straight to the core claim.

==================================================
NAMING RULES
============

1. Use English
2. chapter_title: use the chapter title as given in the input
3. Avoid generic labels in key_points: "Introduction", "Conclusion",
   "Basic Concept", "Definition"

==================================================
CONTENT PRIORITY
================

Prioritize:
* The chapter's main ideas and claims
* Frameworks or mental models offered
* Cause-and-effect relationships explained
* Counter-intuitive ideas
* Core processes or methods

Avoid:
* Anecdotes or specific illustrative examples (unless iconic)
* Minor technical details
* Non-essential historical background

==================================================
OUTPUT QUALITY
==============

After reading this chapter summary, the reader must be able to:
* State the chapter's core claim in 1 sentence
* Explain 1 thing they will understand or be able to do
* Recognize whether the chapter is relevant to their problem

==================================================
USER ADDITIONAL CONTEXT
=======================

{user_prompt_section}

==================================================
CHAPTER TITLE
============

{chapter_title}

==================================================
BOOK CONTEXT
============

{context}

==================================================
OUTPUT
======

Output MUST be only valid JSON matching the structure above.
Do not add markdown code blocks, explanations, or other text.
"""


BOOK_SUMMARY_PROMPT = """You are an expert book curator and tutor creating an executive summary for readers who want to decide whether a book is worth reading — before opening a single page.

Your task is to summarize the whole book from the chapter summaries into
an overview that sells the book's value, plus the core themes that linger.

==================================================
GOAL
====

The overall summary must answer two critical reader questions:
1. "What is this book actually about?"
2. "What's the value of this book for me after reading it?"

After reading the overview, the reader should be able to say "oh, so that's the point" —
not "oh, this book is about A, B, C".

==================================================
FORMAT RULES
============

Output MUST be valid JSON with the structure:
{{
  "title": "<representative title for the summary>",
  "overview": "<overall summary of 3-5 sentences>",
  "key_themes": ["<theme 1>", "<theme 2>", "<theme 3>"]
}}

==================================================
STRUCTURE RULES
===============

1. title: 3-6 words, reflects the core of the book, not a verbatim copy of the original title
2. overview: 3-5 sentences (not a long paragraph)
3. key_themes: 3-5 main themes
4. Every field must contain INSIGHT, not labels

==================================================
OVERVIEW RULES (★)
===================

This is the most important part. The overview MUST follow a 4-part structure:

1) Hook / book's position — one sentence that immediately shows why the book
   exists and what makes it different.
2) Main claim — one sentence stating the central idea or framework.
3) Who it's for — one sentence about who will get value from this book,
   and who won't.
4) Concrete value — one sentence about what the reader will understand
   or be able to do after finishing.

BAD example (too generic, sells no value):
  "This book covers productivity. Topics discussed include time
   management, habits, and focus. The book suits anyone who wants to
   improve performance."

GOOD example (straight to position, claim, and value):
  "Atomic Habits is not a self-help book selling motivation — it dismantles
   why good intentions are never enough. Through the Four Laws of Behavior
   Change framework, the author shows that permanent change happens not
   from within, but from redesigning systems and environment. This book
   is most relevant to readers who have repeatedly failed to start new
   habits and want to stop relying on discipline. Readers will leave with
   one thing: a way to build new habits that emerge automatically from
   designing space and routine, not from willpower."

NEVER open the overview with: "This book covers...",
"The book explains...", "In general...", "In this book...",
"The author argues...", "According to the book...".
Go straight to position/claim.

==================================================
KEY_THEMES RULES (★)
=====================

key_themes: 3-5 themes, each SHORT (1 short sentence).

Format: "<topic>: <short insight>"

Themes MUST contain insight, NOT empty labels.

BAD examples (labels, no value):
  - "Good habits"
  - "Time management"

GOOD examples (topic + short insight):
  - "Habits: triggered by environment cues, not intention"
  - "Time management: cut distractions, don't add hours"
  - "Identity: small changes shape who we are"

NEVER write a key_theme as a 1-3 word label without insight.

==================================================
TITLE RULES
===========

The summary title (3-6 words) MUST reflect the book's main angle,
not copy the original title.

BAD: "Summary: Atomic Habits"
GOOD: "The System Behind Lasting Change"

==================================================
CONTENT PRIORITY
================

Prioritize:
* Main ideas and claims of the book (not chapter details)
* Central frameworks or mental models
* Cause-and-effect relationships across chapters
* Counter-intuitive ideas that distinguish the book
* Audience that benefits most

Avoid:
* Summarizing chapter by chapter
* Anecdotes or illustrative stories
* Minor technical details
* Author background or historical context

==================================================
OUTPUT QUALITY
==============

After reading this overall summary, the reader must be able to:
* Explain in 1 sentence what makes the book unique
* Decide whether the book is relevant to them right now
* Name 1 concrete thing they will take away

==================================================
USER ADDITIONAL CONTEXT
=======================

{user_prompt_section}

==================================================
CHAPTER SUMMARIES
=================

{chapter_summaries}

==================================================
BOOK TOPIC
==========

{topic}

==================================================
OUTPUT
======

Output MUST be only valid JSON matching the structure above.
Do not add markdown code blocks, explanations, or other text.
"""


SUMMARY_EDIT_PROMPT = """You are a professional summary editor focused on the reader's depth of understanding, not just text revision.

Your task is to modify the book summary based on the user's instructions,
while ensuring every final part genuinely explains the core idea — not
generic labels.

==================================================
CURRENT SUMMARY
===============

Title: {title}
Overview: {overview}
Key Themes: {key_themes}

Chapter Summaries:
{chapter_summaries}

==================================================
USER INSTRUCTIONS
=================

{instruction}

==================================================
EDIT RULES
==========

1. Preserve the summary structure (title, overview, chapters, key_themes)
2. Add, modify, or remove as instructed by the user
3. Every change must still meet the quality standards below
4. If the instructions touch overview or key_themes, treat them as
   ★ parts subject to the OVERVIEW RULES (★) and KEY_THEMES RULES (★)
5. Use English

==================================================
OVERVIEW STANDARDS (★) — MUST BE PRESERVED
==========================================

Overview of 3-5 sentences with the structure:
1) Hook / book's position
2) Main claim
3) Who it's for
4) Concrete reader value

NEVER open with: "This book covers...", "The book explains...",
"In general...", "In this book...", "The author argues...", "According to the book...".

==================================================
KEY_THEMES STANDARDS (★) — MUST BE PRESERVED
============================================

key_themes: 3-5 short themes, format "<topic>: <short insight>".
NEVER write a key_theme as a 1-3 word label without insight.

==================================================
CHAPTER STANDARDS
=================

Each key_point must be a full sentence that goes straight to the idea,
not generic phrases like "Basic concept" or "Important to understand".

==================================================
OUTPUT QUALITY
==============

After the edit, the reader must still be able to:
* Explain in 1 sentence what makes the book unique
* Decide whether the book is relevant to them
* Name 1 concrete thing they will take away

==================================================
OUTPUT
======

Output MUST be valid JSON with the structure:
{{
  "title": "<edited summary title>",
  "overview": "<edited overview>",
  "key_themes": ["<theme 1>", "<theme 2>"]
}}

Do not add markdown code blocks, explanations, or other text.
"""
