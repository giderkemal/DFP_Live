import textwrap

CATEGORIES = textwrap.dedent("""<category> 
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
</category>""")
