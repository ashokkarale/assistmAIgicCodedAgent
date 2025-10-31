### ✨ **AssistmAIgic** – Intelligent Email Agent for Order & Product Support

**AssistmAIgic** is a specialized automation agent designed to streamline customer email handling for product orders and inquiries. Built on LangGraph’s state machine architecture and integrated with UiPath’s Model Context Portal (MCP), it delivers fast, accurate, and context-aware responses while ensuring human oversight for sensitive cases.

---

### Key Capabilities

1. **Multilingual Email Handling**
   - Accepts customer emails in any language.
   - Automatically translates incoming emails to a processing language (e.g., English, Marathi, Hindi).
   - Replies are generated and translated back into the customer's original language for seamless and personalize communication.

2. **Order ID Extraction & Validation**
   - Parses email content to extract an 8-digit order ID.
   - If found, retrieves order details via MCP tools.
   - If missing, sends a polite rejection email requesting the order ID.

3. **Email Categorization & Sentiment Analysis**
   - Classifies emails into categories like:
     - *Product Inquiry*
     - *Order Status*
     - *Complaint*
     - *etc.*
   - Assesses sentiment as *Positive*, *Negative*, or *Very Negative*.

4. **Contextual Response Generation**
   - Uses a grounding retriever to pull relevant product specs, warranty info, and FAQs from the knowledge base.
   - Crafts informative, personalized replies based on the customer's query and order context.

5. **Human-in-the-Loop (HITL) Escalation**
   - If sentiment is *Very Negative*, the agent routes the draft response to a human reviewer for approval or refinement ensuring empathy and precision in sensitive situations.

6. **Automated Response Delivery**
   - Sends the final email response (automated or HITL-approved) to the customer in their original language.

7. **Interaction Logging**
   - Logs the outcome of each email interaction for traceability and analytics.

---

### Benefits

- Reduces manual workload for support teams.
- Ensures fast, multilingual customer service.
- Escalates complex or emotional cases to human agents.
- Maintains high accuracy and contextual relevance in responses.
