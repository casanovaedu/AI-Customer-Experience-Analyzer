# AI-Powered Customer Experience (CX) Intelligence Platform

This project is an interactive web application designed to transform raw, unstructured customer feedback into a strategic, data-driven executive dashboard. It was developed to solve a real-world business challenge: understanding the root causes behind a significant NPS gap between different customer markets.



## The Business Problem

While managing customer experience, a key challenge was a persistent NPS gap between our Spanish (ES) market and other global markets (Rest of World). The "what" was clear, but the "why" was buried in thousands of free-text reviews in multiple languages. We needed a way to quickly and scientifically diagnose the specific issues driving this dissatisfaction.

## The Solution: A CX Intelligence Platform

I led the development of this internal tool to automate the analysis process. The platform allows any stakeholder, regardless of their technical skill, to upload raw feedback data and generate a high-level executive dashboard in minutes.

### Key Features:
* **AI-Powered Pain Point Analysis:** Leverages a multilingual `transformer` model to read customer comments (in English, Spanish, etc.) and classify them into specific, actionable categories like "Itinerary Pace" or "Hotel Location."
* **Strategic Deep-Dive Dashboard:** Provides a high-level overview comparing two markets or a global summary, with clear metrics, data visualizations, and a "Top 3 Gaps" analysis to immediately highlight problem areas.
* **"Why did NPS fall?" Diagnostic:** A one-click report that automatically analyzes weekly performance, identifies the highest-impact destinations on the global NPS, and pinpoints the market and pain point driving the change.

## Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Data Manipulation:** Pandas
* **AI / NLP:** Hugging Face Transformers (`mDeBERTa-v3-base`)
* **Data Visualization:** Altair

## Note on My Role & Development
As a business stakeholder with a deep interest in data, I conceptualized and guided the development of this project. It was rapidly prototyped by leveraging AI coding assistants as a development partner. My role was to define the business logic, direct the analytical approach, and translate the complex data outputs into a clear, strategic tool for executive decision-making. This project demonstrates my ability to bridge the gap between business needs and technical solutions to drive company-wide improvements.
