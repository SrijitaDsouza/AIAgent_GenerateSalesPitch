---
title: Generate Sales Pitch (AI agent)
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.28.0
app_file: app.py
pinned: false
---

# BrokerDesk – Generate Sales Pitch (AI agent)
This program does the below:
Sales Manager Agent working in insurance company:
    a. Generates 3 sample sales pitch emails using different LLMs (tools and agents)
    b. Selects the best email
    c. Formats the email with Subject, body and content (agent as a tool)
    d. Sends the email to End-Customer (function as a tool)
Guardrails - There is an Agent which is checking if any person's name is used, then trip.
Various AI Agents are created for Sales Email Generation, Email Agent to draft and send, Manager Role to monitor
Tools have been used and given to Agent
Handoff Control to Agent for using Email Tools 
SendGrid Tool used for Sending the email


# Steps to Run
Go to the project folder containing the files
install requirements
uvicorn server:app --reload   

Then open http://localhost:8000 and click "Generate Sales Pitch".

Flow (human in the loop):

1. Dialog -> POST /api/sales-pitch/drafts. The name-check guardrail runs,
   sales_agent1/2/3 write drafts in parallel, and a judge agent (same model as
   careful_sales_manager) picks the best one with a one-line reason.
2. The page shows the three drafts side by side, with the AI pick in green.
   The user accepts it or selects another draft.
3. POST /api/sales-pitch/send -> emailer_agent writes the subject, converts to
   HTML and calls send_html_email. Progress streams back, so the page shows the
   final subject and HTML while SendGrid sends, then confirms it was sent.
