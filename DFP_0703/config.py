import textwrap

CATEGORIES = textwrap.dedent(
    """<category>
    <label>Price Changes</label>
    <content>
        Questions regarding recent changes in product or service pricing
        Requests for clarification on the reasons behind price adjustments
        Inquiries about future pricing and cost expectations
    </content>
</category>
<category>
    <label>Promotions</label>
    <content>
        Inquiries about current or upcoming promotional offers
        Requests for information on discounts, deals, and limited-time offers
        Questions regarding eligibility and terms for specific promotions
    </content>
</category>
<category>
    <label>Customer Complaint</label>
    <content>
        Feedback or grievances related to product or service experiences
        Requests for resolution on specific customer issues or incidents
        Questions about escalation processes and complaint handling policies
    </content>
</category>
<category>
    <label>New Product Launches</label>
    <content>
        Questions explicitly about newly introduced products or services
        Inquiries focusing on the launch schedules, availability, or features of products newly added to the catalog
        Requests for detailed specifications of new products
        (Does not include queries about switching to existing products)
    </content>
</category>
<category>
    <label>Out of Stock</label>
    <content>
        Questions about the availability and restocking dates of products
        Inquiries on alternatives for out-of-stock items
        Requests for notifications on restocked products
    </content>
</category>
<category>
    <label>Customer Request</label>
    <content>
        Requests for products or services, questions about products
        Suggestions for new products or features customers would like to see
        Inquiries about the possibility of fulfilling specific customer needs outside the standard catalog
    </content>
</category>
<category>
    <label>Other</label>
    <content>
        General inquiries or requests that do not fit into predefined categories
        Miscellaneous questions or feedback about company policies
        Assistance with issues or topics outside standard support topics
    </content>
</category>"""
)

CITATION_FORMAT_PROMPT = f"""
"""

GLOBAL_CHALLENGES_FORMAT = f"""
Don't add any descriptions. No impact analysis, no recommendations.
Support insights with multiple examples (as many as possible, minimum 5), citing their Row_ID in this format:
[Row_ID:row_id]
Example Citation Format:
- [Row_ID:80]
"""

LOCATION_DETAILS_PROMPT = f"""
**Location Breakdown**:
- Summarize feedback trends and challenges for each region or duty-free station, describe them in two or three bullet points.
- Propose **location-specific recommendations** tailored to local issues.
- Cite several (at least 3) examples to support the findings and insights
{CITATION_FORMAT_PROMPT}
"""

LOCATION_INSTRUCTIONS_PROMPT = """
**Location-Specific Challenges**
- Break down the findings by **region or specific duty-free stations**, Describe them in two or three bullet points.
- Highlight patterns or recurrent issues (e.g., pricing complaints in Region A, staff shortages in Station B).
- Support observations with **specific data examples** (Field Intelligence and Row_ID) and recommend **location-specific actions** to address these challenges.
"""

TRENDS_DETAILS_PROMPT = f"""
**Timing/Seasonal Trends**:
- Detail patterns in feedback volume and sentiment over time, describe them in two or three bullet points.
- Offer recommendations for addressing time-sensitive challenges effectively.
- Cite several (at least 3) examples to support the findings and insights
{CITATION_FORMAT_PROMPT}
"""

TRENDS_INSTRUCTIONS_PROMP = """
**Timing and Seasonal Trends**
- Analyze feedback for patterns tied to **specific times of the year** (e.g., holidays, weekends, or seasonal events), Describe them in two or three bullet points.
- Identify spikes in issues or unique challenges during these periods.
- Provide examples illustrating these trends and suggest proactive measures to prevent recurring problems.
"""
