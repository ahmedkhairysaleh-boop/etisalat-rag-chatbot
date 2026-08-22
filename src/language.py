"""Detect whether text is Arabic or English.

Used in two places: to filter retrieval to chunks in the matching language,
and to tell the model which language to answer in.
"""

# Arabic script occupies U+0600 to U+06FF in Unicode
ARABIC_RANGE_START = "\u0600"
ARABIC_RANGE_END = "\u06FF"


def detect_language(text):
    """Return 'ar' if the text contains any Arabic, otherwise 'en'.

    One Arabic character is enough. Customers mix scripts freely - a question
    like "عايز اعرف سعر Emerald 780" is mostly Arabic with an English product
    name, and the customer expects an Arabic answer.
    """
    for character in text:
        if ARABIC_RANGE_START <= character <= ARABIC_RANGE_END:
            return "ar"
    return "en"


def other_language(lang):
    """The opposite language, for falling back when retrieval finds nothing
    useful in the language the question was asked in."""
    return "en" if lang == "ar" else "ar"