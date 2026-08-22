"""The evaluation set.

Each case lists what a correct answer must contain, not the exact wording -
the model phrases things differently each run, but the figures must be right.

'route' is the branch the agent should take. Recording it turns the routing
logic into something measurable rather than something assumed.
"""

QUESTIONS = [

    # --- English: prices and package details ---
    {
        "question": "How much does Emerald 430 cost per month?",
        "must_contain": ["430"],
        "route": "generate",
        "label": "Emerald 430 price (en)",
    },
    {
        "question": "What internet quota does Emerald 780 include?",
        "must_contain": ["780"],
        "route": "generate",
        "label": "Emerald 780 quota (en)",
    },
    {
        "question": "What does Hekaya Internet 46 include?",
        "must_contain": ["46", "1,250"],
        "route": "generate",
        "label": "Hekaya Internet 46 (en)",
    },
    {
        "question": "What is the cheapest Mini Hekaya Internet plan?",
        "must_contain": ["10.5"],
        "route": "generate",
        "label": "cheapest mini plan (en)",
    },

    # --- English: rules and services ---
    {
        "question": "Can I use a Data Line SIM in my mobile phone?",
        "must_contain": ["MiFi"],
        "route": "generate",
        "label": "Data Line device restriction (en)",
    },
    {
        "question": "What is the Double the Package offer?",
        "must_contain": ["first recharge", "double"],
        "route": "generate",
        "label": "Double the Package (en)",
    },
    {
        "question": "How many Mixes equal one minute to another network?",
        "must_contain": ["5"],
        "route": "generate",
        "label": "Mix conversion rate (en)",
    },
    {
        "question": "What roaming internet does Emerald 3450 include?",
        "must_contain": ["3450"],
        "route": "generate",
        "label": "Emerald 3450 roaming (en)",
    },

    # --- English: codes ---
    {
        "question": "What code do I use to subscribe to Aqwa Card?",
        "must_contain": ["811"],
        "route": "generate",
        "label": "Aqwa Card code (en)",
    },
    {
        "question": "What is the master code for Hekaya Mixat?",
        "must_contain": ["319"],
        "route": "generate",
        "label": "Hekaya Mixat master code (en)",
    },

    # --- Arabic ---
    {
        "question": "كام سعر باقة اميرالد 430 في الشهر؟",
        "must_contain": ["430"],
        "route": "generate",
        "label": "Emerald 430 price (ar)",
    },
    {
        "question": "ايه هو عرض ضعف الباقة في خط الداتا؟",
        "must_contain": ["ضعف"],
        "route": "generate",
        "label": "Double the Package (ar)",
    },
    {
        "question": "كام مكس يساوي دقيقة لشبكة تانية؟",
        "must_contain": ["5"],
        "route": "generate",
        "label": "Mix conversion rate (ar)",
    },
    {
        "question": "ايه الكود بتاع أقوى كارت؟",
        "must_contain": ["811"],
        "route": "generate",
        "label": "Aqwa Card code (ar)",
    },
    {
        "question": "هل خط الداتا بيشتغل على الموبايل؟",
        "must_contain": ["MiFi", "مايفاي"],
        "route": "generate",
        "label": "Data Line device restriction (ar)",
    },
    {
        "question": "ايه باقات حكاية انترنت الشهرية؟",
        "must_contain": ["46"],
        "route": "generate",
        "label": "Hekaya Internet monthly plans (ar)",
    },

    # --- mixed script: how customers actually type ---
    {
        "question": "عايز اعرف سعر Emerald 780",
        "must_contain": ["780"],
        "route": "generate",
        "label": "mixed script price (ar+en)",
    },
    {
        "question": "ايه الفرق بين Hekaya Mixat و Hekaya Internet؟",
        "must_contain": ["مكس", "ميكس", "Mix"],
        "route": "generate",
        "label": "mixed script comparison (ar+en)",
    },

    # --- small talk: should skip retrieval entirely ---
    {
        "question": "hi",
        "must_contain": [],
        "route": "smalltalk",
        "label": "greeting (en)",
    },
    {
        "question": "السلام عليكم",
        "must_contain": [],
        "route": "smalltalk",
        "label": "greeting (ar)",
    },
    {
        "question": "thanks",
        "must_contain": [],
        "route": "smalltalk",
        "label": "closing (en)",
    },

        # --- off topic: retrieval finds loosely similar chunks, so these route to
    # generate; the system prompt is what refuses them ---
    {
        "question": "What's the weather in Cairo today?",
        "must_contain": [],
        "route": "generate",
        "label": "off topic - weather, refused by prompt (en)",
    },
    {
        "question": "Does e& Egypt offer home fibre internet in Alexandria?",
        "must_contain": [],
        "route": "generate",
        "label": "off topic - not in documents, refused by prompt (en)",
    },
    {
        "question": "ممكن تقولي رصيدي كام؟",
        "must_contain": [],
        "route": "generate",
        "label": "off topic - account balance, refused by prompt (ar)",
    },
]