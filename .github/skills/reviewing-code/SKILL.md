---
name: reviewing-code
description: Code review dispatch — gather diff, dispatch code-reviewer agent, present findings
allowed-tools:
  - Bash(git *)
  - Read
  - Glob
  - Grep
  - Task
  - AskUserQuestion
---

# Code Review Dispatch

Gather the relevant diff, dispatch a code-reviewer agent, and present structured findings.

## Input

$ARGUMENTS = file list, "staged", or branch name.

If $ARGUMENTS is empty, ask via AskUserQuestion with options:

- "Staged changes" — review what's currently staged
- "Current branch vs develop" — review all changes on this feature branch
- "Specific files" — then ask for file paths
- "Last N commits" — then ask for N

## Step 1: Gather the Diff

Based on input, run the appropriate git command:

- "staged" or "Staged changes":

  ```bash
  git diff --cached
  ```

- Branch name or "Current branch vs develop":

  ```bash
  git diff develop...HEAD
  ```

  (Replace develop with the specified branch if different)

- File list (space-separated paths):

  ```bash
  git diff -- {file1} {file2} ...
  ```

- "Last N commits":

  ```bash
  git diff HEAD~{N}
  ```

Also gather the stat summary:

```bash
git diff {same_args} --stat
```

If the diff is empty, report "No changes found for the specified scope" and stop.

## Step 2: Assess Diff Size

Count total changed lines from the stat summary.

If diff exceeds 500 lines:

- Note the total size
- Group files by directory or subsystem
- The reviewer agent will receive the full diff but be informed of the groupings for structured review

## Step 3: Generate Slug

Derive a slug for the review report file name:

- From branch name: feature/add-auth -> add-auth
- From timestamp if no branch context: {YYYY-MM-DD-HHMMSS}
- Sanitize: lowercase, hyphens only, no special characters

## Step 4: Dispatch Code-Reviewer Agent

Create a Task agent (subagent_type: general-purpose) with code-reviewer instructions.

Provide the agent with:

- The full diff content
- The stat summary (files changed, lines added/removed)
- Project conventions: read CLAUDE.md from project root if it exists, summarize relevant conventions
- Focus areas: if user specified any, pass them through
- Instructions: follow the 6-stage review process (Understand, Architecture, Correctness, Quality, Security, Summary)

Wait for the agent to complete and collect the review report.

## Step 5: Collect Review Results

Parse the agent's report to extract:

- Total findings count
- Findings by severity: critical, major, minor
- Individual findings with file:line references
- Overall verdict: APPROVE, REQUEST CHANGES, or COMMENT

## Step 6: Write Review Report

Create the review report file:

```bash
mkdir -p docs/reviews
```

Write to docs/reviews/{slug}-review.md using the review report template from templates/review-report.md.

Fill in all template fields:

- {SLUG}: the generated slug
- {DATE}: current date in YYYY-MM-DD format
- {BRANCH}: current git branch name
- Scope and intent from the reviewer's understanding
- All findings in the table format
- Severity counts
- Verdict
- Prioritized action items

## Step 7: Present Summary

Output the findings summary:

```text
## Review Summary

Findings: {total}
  Critical: {N}
  Major: {N}
  Minor: {N}

Top Issues:
1. {severity_symbol} {file}:{line} — {description}
2. {severity_symbol} {file}:{line} — {description}
3. {severity_symbol} {file}:{line} — {description}

Full report: docs/reviews/{slug}-review.md
```

Severity symbols: open diamond for minor, filled diamond for major, double filled diamond for critical.

Then offer next steps via AskUserQuestion:

- "Address critical issues now" — if critical findings exist, start fixing them immediately
- "Address all issues" — work through all findings from highest to lowest severity
- "Acknowledge and continue" — user has seen the review and will handle it later
- "Request re-review after fixes" — user will fix issues and re-run the review

## Rules

- Always gather the actual diff — never review from memory or assumptions
- The code-reviewer agent is a separate subagent with its own context
- Every finding in the report must include file:line reference
- Review report is always written to docs/reviews/ for traceability
- If no changes are found, stop early — do not dispatch a reviewer for an empty diff
