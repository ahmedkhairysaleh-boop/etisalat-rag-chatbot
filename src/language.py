"""Detect whether text is Arabic or English, and normalise Arabic numerals.

Used in three places: to filter retrieval to chunks in the matching language,
to tell the model which language to answer in, and to make a question written
with Arabic numerals searchable against documents written with Western ones.
"""

# Arabic script occupies U+0600 to U+06FF in Unicode
ARABIC_RANGE_START = "؀"
ARABIC_RANGE_END = "ۿ"

# Arabic-Indic digits (٠-٩, U+0660-U+0669) are what an Arabic keyboard produces
# by default in Egypt. Extended Arabic-Indic (۰-۹, U+06F0-U+06F9) come from
# Persian and Urdu layouts and turn up occasionally. Both are mapped to the
# Western digits the knowledge base is written with.
#
# The two separators are included because they travel with the digits: ٫ is the
# Arabic decimal mark and ٬ the thousands mark, so "٧٤٫٢٩" is 74.29.
DIGIT_TRANSLATION = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩" "۰۱۲۳۴۵۶۷۸۹" "٫٬",
    "0123456789" "0123456789" ".,",
)


def detect_language(text):
    """Return 'ar' if the text contains any Arabic, otherwise 'en'.

    One Arabic character is enough. Customers mix scripts freely - a question
    like "عايز اعرف سعر Emerald 780" is mostly Arabic with an English product
    name, and the customer expects an Arabic answer.

    Note that Arabic-Indic digits fall inside the Arabic range, so a question
    like "Emerald ٤٣٠" counts as Arabic even though it contains no Arabic
    letters. That is deliberate: someone typing ٤٣٠ is on an Arabic keyboard
    and most likely wants an Arabic answer. If that turns out to be wrong, the
    fix is to skip U+0660-U+0669 and U+06F0-U+06F9 in the loop below.
    """
    for character in text:
        if ARABIC_RANGE_START <= character <= ARABIC_RANGE_END:
            return "ar"
    return "en"


def other_language(lang):
    """The opposite language, for falling back when retrieval finds nothing
    useful in the language the question was asked in."""
    return "en" if lang == "ar" else "ar"


# Characters that look like ordinary spacing and hyphens but are not.
#
# The query rewriter emits these: the log shows it returning "داتا لاين\u202f25"
# with narrow no-break spaces around the number, and "rolled\u2011over" with a
# non-breaking hyphen. They are invisible on screen and different to the
# embedding model, so a rewritten query can quietly stop matching the chunk it
# was meant to find.
INVISIBLE_TRANSLATION = str.maketrans({
    "\u00a0": " ",   # no-break space
    "\u202f": " ",   # narrow no-break space
    "\u2009": " ",   # thin space
    "\u200b": "",    # zero-width space
    "\u200f": "",    # right-to-left mark
    "\u200e": "",    # left-to-right mark
    "\u2011": "-",   # non-breaking hyphen
    "\u2013": "-",   # en dash
    "\u2014": "-",   # em dash
})


def normalize_digits(text):
    """Rewrite Arabic numerals as Western ones.

    Both knowledge base documents - the Arabic originals included - write every
    number with Western digits: "كود الاشتراك في باقة حكاية ميكسات 180". A
    customer on an Arabic keyboard types ١٨٠, and to the embedding model those
    are unrelated tokens.

    This matters when the number identifies the thing being asked about rather
    than being a quantity to calculate with. Package names, prices and service
    codes are all identifiers, and the chunks holding them are otherwise nearly
    identical - "كود الاشتراك في باقة حكاية ميكسات X" repeated six times - so
    the number is the only signal retrieval can separate them by. Get it wrong
    and the search has nothing left to go on.

    Quantities are unaffected either way: "كلمت ٢٠ دقيقة" retrieves the rate
    table on the strength of its words, and the model reads ٢٠ from the
    question directly.

    Only the search query is normalised. The customer's message is displayed
    and answered exactly as they wrote it.
    """
    return text.translate(DIGIT_TRANSLATION)


def normalize_query(text):
    """Prepare a question for the embedding model.

    Digits and invisible characters both come down to the same thing: two
    strings that read identically to a person can be different tokens to the
    model, and the difference decides whether the right chunk is found.

    Search only. Nothing here changes what the customer sees.
    """
    text = normalize_digits(text)
    text = text.translate(INVISIBLE_TRANSLATION)

    return " ".join(text.split())
