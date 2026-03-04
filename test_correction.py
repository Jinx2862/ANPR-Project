import re

LETTER_CORRECTIONS = {
    '0': 'O',
    '1': 'I',
    '2': 'Z',
    '5': 'S',
    '8': 'B',
    '6': 'G'
}

DIGIT_CORRECTIONS = {
    'O': '0',
    'I': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6'
}

def correct_indian_plate(text):
    if len(text) != 10:
        return text

    char2_as_letter = LETTER_CORRECTIONS.get(text[2], text[2])
    char3_as_letter = LETTER_CORRECTIONS.get(text[3], text[3])
    
    is_bh_series = False
    if char2_as_letter == 'B' and char3_as_letter == 'H':
        is_bh_series = True

    corrected = ""
    if is_bh_series:
        for i, char in enumerate(text):
            if i in [0, 1, 4, 5, 6, 7]: # Digits
                corrected += DIGIT_CORRECTIONS.get(char, char)
            elif i in [2, 3, 8, 9]: # Letters
                corrected += LETTER_CORRECTIONS.get(char, char)
    else:
        for i, char in enumerate(text):
            if i in [0, 1, 4, 5]: # Letters
                corrected += LETTER_CORRECTIONS.get(char, char)
            elif i in [2, 3, 6, 7, 8, 9]: # Digits
                corrected += DIGIT_CORRECTIONS.get(char, char)
            else:
                corrected += char
    return corrected

def test():
    test_cases = [
        # State Series
        ("MHI2AR1234", "MH12AR1234"),
        ("BH12AB1234", "BH12AB1234"), # Now should work!
        ("MH12AR123O", "MH12AR1230"),
        ("MH12AR123B", "MH12AR1238"),
        
        # BH Series
        ("22BH1234AA", "22BH1234AA"),
        ("ZZBHIZ34AA", "22BH1234AA"), # ZZ -> 22, BH stays BH, IZ34 -> 1234
        ("228H1234AA", "22BH1234AA"), # 8 -> B
        ("22BH123400", "22BH1234OO"), # 00 -> OO (letters at end)
    ]

    for input_text, expected in test_cases:
        actual = correct_indian_plate(input_text)
        print(f"Input: {input_text} -> Actual: {actual} | Expected: {expected} | {'PASS' if actual == expected else 'FAIL'}")

if __name__ == "__main__":
    test()
